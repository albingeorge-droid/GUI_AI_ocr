import json
import os
from pathlib import Path
from typing import Any, Dict, List

import boto3
from botocore.config import Config as BotoConfig
from opentelemetry import trace

from utils.ocr.env import load_env
from utils.ocr.tracing import setup_tracing
from utils.ocr.pdf import pdf_to_png_bytes
from utils.ocr.bedrock import bedrock_converse_ocr_page
from utils.ocr.params import load_params, get_ocr_config, get_phoenix_config
from utils.ocr.s3 import list_pdfs_in_prefix, download_s3_key, s3_key_to_local_path


def run_stage_01_ocr_from_s3(
    s3_prefix: str,
    params_path: str = "params.yaml",
    out_dir: str = "artifacts/ocr",
    env_file: str | None = None,
    max_pages: int | None = None,
    debug_creds: bool = False,
) -> None:
    """
    OCR all PDFs under s3://bucket/<s3_prefix>/ and write outputs to out_dir.
    Writes per-PDF JSON + TXT, plus a small manifest.json.
    """
    load_env(env_file)

    params = load_params(params_path)
    ocr_cfg = get_ocr_config(params)
    phoenix_cfg = get_phoenix_config(params)

    # allow env overrides if you want (optional)
    bucket = os.getenv("OCR_S3_BUCKET", ocr_cfg.bucket)
    region = os.getenv("AWS_REGION", ocr_cfg.region) or os.getenv("AWS_DEFAULT_REGION", ocr_cfg.region)

    phoenix_endpoint = os.getenv("PHOENIX_OTLP_ENDPOINT", phoenix_cfg.otlp_endpoint or "") or phoenix_cfg.otlp_endpoint
    if phoenix_endpoint is not None:
        phoenix_endpoint = str(phoenix_endpoint).strip()
        if phoenix_endpoint == "":
            phoenix_endpoint = None

    setup_tracing(service_name=phoenix_cfg.service_name, phoenix_otlp_endpoint=phoenix_endpoint)
    tracer = trace.get_tracer(__name__)

    if debug_creds:
        print("AWS_REGION:", region)
        print("Bucket:", bucket)
        print("Has AWS_ACCESS_KEY_ID:", bool(os.getenv("AWS_ACCESS_KEY_ID")))
        print("Has AWS_SECRET_ACCESS_KEY:", bool(os.getenv("AWS_SECRET_ACCESS_KEY")))
        print("Has AWS_SESSION_TOKEN:", bool(os.getenv("AWS_SESSION_TOKEN")))
        print("PHOENIX_OTLP_ENDPOINT:", phoenix_endpoint)

    s3 = boto3.client("s3", region_name=region, config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"}))
    brt = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"}),
    )

    # Normalize prefix (no leading slash)
    s3_prefix = s3_prefix.lstrip("/")
    if s3_prefix and not s3_prefix.endswith("/"):
        s3_prefix += "/"

    output_root = Path(out_dir) / s3_prefix.strip("/")

    with tracer.start_as_current_span("stage_01_ocr.s3_pipeline") as span:
        span.set_attribute("s3.bucket", bucket)
        span.set_attribute("s3.prefix", s3_prefix)
        span.set_attribute("bedrock.model_id", ocr_cfg.model_id)
        span.set_attribute("aws.region", region)

        pdf_keys = list_pdfs_in_prefix(s3, bucket=bucket, prefix=s3_prefix)
        if not pdf_keys:
            print(f"No PDFs found under s3://{bucket}/{s3_prefix}")
            return

        manifest: List[Dict[str, Any]] = []

        for key in pdf_keys:
            with tracer.start_as_current_span("stage_01_ocr.single_pdf") as pdf_span:
                pdf_span.set_attribute("s3.key", key)

                # download to temp local path inside artifacts for reproducibility
                local_pdf_path = s3_key_to_local_path(str(output_root / "_inputs"), s3_prefix, key)
                download_s3_key(s3, bucket=bucket, key=key, local_path=local_pdf_path)

                # where to write outputs
                base_out = Path(s3_key_to_local_path(str(output_root), s3_prefix, key)).with_suffix("")
                out_json = str(base_out) + ".ocr.json"
                out_txt = str(base_out) + ".ocr.txt"
                Path(out_json).parent.mkdir(parents=True, exist_ok=True)

                pages_png = pdf_to_png_bytes(local_pdf_path, dpi=ocr_cfg.dpi, max_pages=max_pages)

                results: List[Dict[str, Any]] = []
                all_text_lines: List[str] = []

                for i, png_bytes in enumerate(pages_png, start=1):
                    with tracer.start_as_current_span("stage_01_ocr.page") as page_span:
                        page_span.set_attribute("doc.page_number", i)

                        page_text = bedrock_converse_ocr_page(
                            brt=brt,
                            model_id=ocr_cfg.model_id,
                            image_png_bytes=png_bytes,
                            page_index=i,
                            temperature=ocr_cfg.temperature,
                            max_tokens=ocr_cfg.max_tokens,
                            retries=ocr_cfg.retries,
                        )

                        results.append({"page": i, "text": page_text})
                        all_text_lines.append(f"===== PAGE {i} =====\n{page_text}\n")

                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "s3": {"bucket": bucket, "key": key},
                            "model_id": ocr_cfg.model_id,
                            "pages": results,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(all_text_lines))

                manifest.append(
                    {
                        "s3_bucket": bucket,
                        "s3_key": key,
                        "local_pdf": local_pdf_path,
                        "out_json": out_json,
                        "out_txt": out_txt,
                        "pages": len(results),
                    }
                )

        manifest_path = output_root / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"Done.\n- Prefix: s3://{bucket}/{s3_prefix}\n- PDFs: {len(pdf_keys)}\n- Output: {output_root}\n- Manifest: {manifest_path}")
