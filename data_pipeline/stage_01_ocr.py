import json
import os
import sys
import logging
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import boto3
from botocore.config import Config as BotoConfig

from utils.ocr.env import load_env
from utils.ocr.tracing import setup_tracing
from utils.ocr.pdf import pdf_to_png_bytes
from utils.ocr.bedrock import bedrock_converse_ocr_page
from utils.ocr.params import load_params, get_ocr_config, get_phoenix_config
from utils.ocr.s3 import list_pdfs_in_prefix, download_s3_key

def _s3_key_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.NoSuchKey:
        return False
    except Exception as e:
        # head_object returns 404 via ClientError, not NoSuchKey usually
        from botocore.exceptions import ClientError
        if isinstance(e, ClientError) and e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _upload_bytes_to_s3(s3_client, bucket: str, key: str, data: bytes, content_type: str | None = None) -> None:
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    s3_client.put_object(Bucket=bucket, Key=key, Body=data, **extra)


def setup_logger_to_memory(log_level: str = "INFO") -> tuple[logging.Logger, StringIO]:
    """
    Logger that writes to stdout AND an in-memory buffer (so we can upload logs to S3 without local files).
    """
    logger = logging.getLogger("stage_01_ocr")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    buf = StringIO()
    bh = logging.StreamHandler(buf)
    bh.setFormatter(fmt)
    logger.addHandler(bh)

    return logger, buf


def run_stage_01_ocr_from_s3(
    s3_prefix: str,
    params_path: str = "params.yaml",
    env_file: str | None = None,
    max_pages: int | None = None,
    debug_creds: bool = False,
    log_level: str = "INFO",
    upload_log_to_s3: bool = True,
) -> None:
    """
    OCR PDFs under s3://bucket/<s3_prefix>/ and write OCR JSON back to S3:

      <pdf_folder>/ocr/<pdf_stem>.ocr.json

    Uploads:
      <s3_prefix>/manifest.json
      <s3_prefix>/stage_01_ocr.log  (optional, from in-memory buffer)

    Tracing:
      We DO NOT create stage_01_* spans here (to avoid a trace tree).
      Each Bedrock OCR call produces its OWN ROOT span/trace in utils/ocr/bedrock.py.
    """
    load_env(env_file)

    params = load_params(params_path)
    ocr_cfg = get_ocr_config(params)
    phoenix_cfg = get_phoenix_config(params)

    bucket = os.getenv("OCR_S3_BUCKET", ocr_cfg.bucket)
    region = os.getenv("AWS_REGION", ocr_cfg.region) or os.getenv("AWS_DEFAULT_REGION", ocr_cfg.region)

    phoenix_endpoint = os.getenv("PHOENIX_OTLP_ENDPOINT", phoenix_cfg.otlp_endpoint or "") or phoenix_cfg.otlp_endpoint
    if phoenix_endpoint is not None:
        phoenix_endpoint = str(phoenix_endpoint).strip()
        if phoenix_endpoint == "":
            phoenix_endpoint = None

    setup_tracing(service_name=phoenix_cfg.service_name, phoenix_otlp_endpoint=phoenix_endpoint)

    # Normalize prefix
    s3_prefix = s3_prefix.lstrip("/")
    if s3_prefix and not s3_prefix.endswith("/"):
        s3_prefix += "/"

    logger, log_buf = setup_logger_to_memory(log_level=log_level)

    if debug_creds:
        logger.info("AWS_REGION: %s", region)
        logger.info("Bucket: %s", bucket)
        logger.info("Has AWS_ACCESS_KEY_ID: %s", bool(os.getenv("AWS_ACCESS_KEY_ID")))
        logger.info("Has AWS_SECRET_ACCESS_KEY: %s", bool(os.getenv("AWS_SECRET_ACCESS_KEY")))
        logger.info("Has AWS_SESSION_TOKEN: %s", bool(os.getenv("AWS_SESSION_TOKEN")))
        logger.info("PHOENIX_OTLP_ENDPOINT: %s", phoenix_endpoint)

    s3 = boto3.client("s3", region_name=region, config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"}))
    brt = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"}),
    )

    logger.info(
        "Starting Stage 01 OCR | s3://%s/%s | model=%s | region=%s",
        bucket,
        s3_prefix,
        ocr_cfg.model_id,
        region,
    )

    pdf_keys = list_pdfs_in_prefix(s3, bucket=bucket, prefix=s3_prefix)
    if not pdf_keys:
        logger.warning("No PDFs found under s3://%s/%s", bucket, s3_prefix)
        return

    logger.info("Found %d PDFs under s3://%s/%s", len(pdf_keys), bucket, s3_prefix)

    manifest: List[Dict[str, Any]] = []

    for idx, key in enumerate(pdf_keys, start=1):
        try:
            logger.info("[%d/%d] Processing: s3://%s/%s", idx, len(pdf_keys), bucket, key)

            # Compute output S3 key
            rel_key = key[len(s3_prefix):].lstrip("/")  # e.g. 10_CLU_PT-250/10_CLU_PT-250.pdf
            rel_path = Path(rel_key)

            ocr_s3_dir = f"{s3_prefix}{rel_path.parent.as_posix().rstrip('/')}/ocr/"
            out_json_s3_key = f"{ocr_s3_dir}{rel_path.stem}.ocr.json"
            logger.info("S3 output -> %s", out_json_s3_key)


            # ✅ SKIP if OCR already exists
            if _s3_key_exists(s3, bucket=bucket, key=out_json_s3_key):
                logger.info("Skipping (OCR already exists) -> s3://%s/%s", bucket, out_json_s3_key)
                manifest.append(
                    {
                        "s3_bucket": bucket,
                        "input_pdf_s3_key": key,
                        "ocr_json_s3_key": out_json_s3_key,
                        "status": "skipped_exists",
                    }
                )
                continue


            # Short-lived temp file for the PDF (required by pdf2image)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                tmp_pdf_path = tf.name

            try:
                logger.info("Downloading to temp file -> %s", tmp_pdf_path)
                download_s3_key(s3, bucket=bucket, key=key, local_path=tmp_pdf_path)

                pages_png = pdf_to_png_bytes(tmp_pdf_path, dpi=ocr_cfg.dpi, max_pages=max_pages)
                logger.info("Pages to OCR: %d (max_pages=%s)", len(pages_png), max_pages)

                results: List[Dict[str, Any]] = []
                total_pages = len(pages_png)

                for i, png_bytes in enumerate(pages_png, start=1):
                    logger.info("  OCR page %d/%d", i, total_pages)

                    # Each bedrock call will create its own ROOT trace/span in Phoenix
                    page_text = bedrock_converse_ocr_page(
                        brt=brt,
                        model_id=ocr_cfg.model_id,
                        image_png_bytes=png_bytes,
                        page_index=i,
                        temperature=ocr_cfg.temperature,
                        max_tokens=ocr_cfg.max_tokens,
                        retries=ocr_cfg.retries,
                        trace_attrs={
                            "s3.bucket": bucket,
                            "s3.prefix": s3_prefix,
                            "s3.key": key,
                            "ocr.output_s3_key": out_json_s3_key,
                            "doc.page_number": i,
                            "doc.total_pages": total_pages,
                        },
                    )
                    results.append({"page": i, "text": page_text})

                payload = {
                    "s3": {"bucket": bucket, "key": key},
                    "model_id": ocr_cfg.model_id,
                    "pages": results,
                }
                json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

                _upload_bytes_to_s3(
                    s3,
                    bucket=bucket,
                    key=out_json_s3_key,
                    data=json_bytes,
                    content_type="application/json",
                )
                logger.info("Uploaded OCR JSON -> s3://%s/%s", bucket, out_json_s3_key)

                manifest.append(
                    {
                        "s3_bucket": bucket,
                        "input_pdf_s3_key": key,
                        "ocr_json_s3_key": out_json_s3_key,
                        "pages": len(results),
                    }
                )

            finally:
                # delete temp file immediately (no local storage retained)
                try:
                    os.remove(tmp_pdf_path)
                    logger.info("Deleted temp file -> %s", tmp_pdf_path)
                except Exception:
                    logger.warning("Could not delete temp file -> %s", tmp_pdf_path)

        except Exception:
            logger.exception("FAILED processing PDF: s3://%s/%s", bucket, key)
            # continue

    # Upload manifest to S3
    manifest_s3_key = f"{s3_prefix}manifest.json"
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    _upload_bytes_to_s3(s3, bucket=bucket, key=manifest_s3_key, data=manifest_bytes, content_type="application/json")
    logger.info("Uploaded manifest -> s3://%s/%s", bucket, manifest_s3_key)

    # Upload log to S3 (optional) from memory
    if upload_log_to_s3:
        log_s3_key = f"{s3_prefix}stage_01_ocr.log"
        log_bytes = log_buf.getvalue().encode("utf-8")
        _upload_bytes_to_s3(s3, bucket=bucket, key=log_s3_key, data=log_bytes, content_type="text/plain")
        logger.info("Uploaded log -> s3://%s/%s", bucket, log_s3_key)

    logger.info("Done | prefix=s3://%s/%s | pdfs=%d", bucket, s3_prefix, len(pdf_keys))
