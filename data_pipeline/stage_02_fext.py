# data_pipeline/stage_02_fext.py
from __future__ import annotations

import json
import logging
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import boto3
from botocore.config import Config as BotoConfig
from opentelemetry import trace

from utils.ocr.env import load_env
from utils.ocr.tracing import setup_tracing
from utils.ocr.params import load_params, get_ocr_config, get_phoenix_config
from utils.fext import (
    openai_extract_haryana_features,
    HARYANA_CLU_SCHEMA_VERSION,
)

from json import JSONDecodeError

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def _s3_key_exists(s3_client, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _upload_bytes_to_s3(
    s3_client,
    bucket: str,
    key: str,
    data: bytes,
    content_type: str | None = None,
) -> None:
    extra: Dict[str, Any] = {}
    if content_type:
        extra["ContentType"] = content_type
    s3_client.put_object(Bucket=bucket, Key=key, Body=data, **extra)


def setup_logger_to_memory(log_level: str = "INFO") -> tuple[logging.Logger, StringIO]:
    """
    Logger that writes to stdout AND an in-memory buffer (so we can upload logs to S3 without local files).
    """
    logger = logging.getLogger("stage_02_fext")
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


def _load_manifest_from_s3(s3_client, bucket: str, key: str) -> List[Dict[str, Any]]:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read().decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Manifest at s3://{bucket}/{key} is not a list")
    return data


def run_stage_02_fext_from_s3(
    s3_prefix: str,
    params_path: str = "params.yaml",
    env_file: str | None = None,
    debug_creds: bool = False,
    log_level: str = "INFO",
    upload_log_to_s3: bool = True,
    force: bool = False,
    max_workers: int = 5,  # <-- NEW
) -> None:
    """
    Stage 02: Feature extraction for Haryana CLU documents using OpenAI GPT-5-nano.

    Input:
      - Stage 01 OCR outputs in S3 (per-PDF ocr JSON files)
      - Stage 01 manifest under <s3_prefix>/manifest.json

    Output:
      - Per-document feature JSONs:
          <pdf_folder>/fext/<pdf_stem>.features.json
      - Stage 02 manifest:
          <s3_prefix>/manifest_fext.json
      - Optional log file:
          <s3_prefix>/stage_02_fext.log
    """
    # Load env + params
    load_env(env_file)
    params = load_params(params_path)
    ocr_cfg = get_ocr_config(params)
    phoenix_cfg = get_phoenix_config(params)

    bucket = os.getenv("OCR_S3_BUCKET", ocr_cfg.bucket)
    region = os.getenv("AWS_REGION", ocr_cfg.region) or os.getenv(
        "AWS_DEFAULT_REGION", ocr_cfg.region
    )

    phoenix_endpoint = (
        os.getenv("PHOENIX_OTLP_ENDPOINT", phoenix_cfg.otlp_endpoint or "")
        or phoenix_cfg.otlp_endpoint
    )
    if phoenix_endpoint is not None:
        phoenix_endpoint = str(phoenix_endpoint).strip()
        if phoenix_endpoint == "":
            phoenix_endpoint = None

    setup_tracing(service_name=phoenix_cfg.service_name, phoenix_otlp_endpoint=phoenix_endpoint)
    tracer = trace.get_tracer(__name__)

    # Normalize prefix
    s3_prefix = s3_prefix.lstrip("/")
    if s3_prefix and not s3_prefix.endswith("/"):
        s3_prefix += "/"

    logger, log_buf = setup_logger_to_memory(log_level=log_level)

    if debug_creds:
        logger.info("AWS_REGION: %s", region)
        logger.info("Bucket: %s", bucket)
        logger.info("Has AWS_ACCESS_KEY_ID: %s", bool(os.getenv("AWS_ACCESS_KEY_ID")))
        logger.info(
            "Has AWS_SECRET_ACCESS_KEY: %s", bool(os.getenv("AWS_SECRET_ACCESS_KEY"))
        )
        logger.info("Has AWS_SESSION_TOKEN: %s", bool(os.getenv("AWS_SESSION_TOKEN")))
        logger.info("Has OPENAI_API_KEY: %s", bool(os.getenv("OPENAI_API_KEY")))
        logger.info("PHOENIX_OTLP_ENDPOINT: %s", phoenix_endpoint)

    # Only S3 client needed (no Bedrock client)
    s3 = boto3.client(
        "s3",
        region_name=region,
        config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"}),
    )

    logger.info(
        "Starting Stage 02 FEXT (OpenAI) | s3://%s/%s | model=%s | region=%s",
        bucket,
        s3_prefix,
        ocr_cfg.model_id,
        region,
    )

    manifest_key = f"{s3_prefix}manifest.json"
    logger.info("Loading Stage 01 manifest -> s3://%s/%s", bucket, manifest_key)

    manifest = _load_manifest_from_s3(s3, bucket=bucket, key=manifest_key)
    if not manifest:
        logger.warning("Manifest is empty; nothing to process.")
        return

    logger.info("Manifest entries: %d", len(manifest))

    fext_manifest: List[Dict[str, Any]] = []

    logger.info("Using up to %d parallel workers for FEXT", max_workers)

    def process_manifest_entry(idx: int, item: Dict[str, Any]) -> Dict[str, Any] | None:
        pdf_key = item.get("input_pdf_s3_key") or item.get("pdf_s3_key")
        ocr_json_key = item.get("ocr_json_s3_key")

        if not pdf_key or not ocr_json_key:
            logger.warning(
                "[%d/%d] Missing pdf / ocr keys in manifest entry: %s",
                idx,
                len(manifest),
                item,
            )
            return None

        pdf_path = Path(pdf_key)
        base_dir = pdf_path.parent.as_posix().rstrip("/")
        pdf_stem = pdf_path.stem

        fext_dir = f"{base_dir}/fext"
        fext_s3_key = f"{fext_dir}/{pdf_stem}.features.json"

        logger.info(
            "[%d/%d] PDF=%s | OCR=%s | FEXT_OUT=%s",
            idx,
            len(manifest),
            pdf_key,
            ocr_json_key,
            fext_s3_key,
        )

        # Skip if features already exist
        if (not force) and _s3_key_exists(s3, bucket=bucket, key=fext_s3_key):
            logger.info("  Skipping (features already exist) -> s3://%s/%s", bucket, fext_s3_key)
            return {
                "s3_bucket": bucket,
                "input_pdf_s3_key": pdf_key,
                "ocr_json_s3_key": ocr_json_key,
                "fext_json_s3_key": fext_s3_key,
                "status": "skipped_exists",
            }

        try:
            # Load OCR JSON
            obj = s3.get_object(Bucket=bucket, Key=ocr_json_key)
            raw = obj["Body"].read().decode("utf-8")
            ocr_payload = json.loads(raw)
            pages = ocr_payload.get("pages", [])

            page_texts: List[str] = []
            for p in pages:
                t = (p.get("text") or "").strip()
                if t:
                    page_texts.append(t)

            if not page_texts:
                logger.warning("  No OCR text found for %s (pages empty)", pdf_key)
                return {
                    "s3_bucket": bucket,
                    "input_pdf_s3_key": pdf_key,
                    "ocr_json_s3_key": ocr_json_key,
                    "fext_json_s3_key": fext_s3_key,
                    "status": "no_ocr_text",
                }

            combined_text = "\n\n".join(page_texts)

            features = openai_extract_haryana_features(
                model_id=ocr_cfg.model_id,
                ocr_text=combined_text,
                doc_id=pdf_stem,
                temperature=ocr_cfg.temperature,
                max_tokens=ocr_cfg.max_tokens,
                retries=ocr_cfg.retries,
                trace_attrs={
                    "s3.bucket": bucket,
                    "s3.prefix": s3_prefix,
                    "s3.key.pdf": pdf_key,
                    "s3.key.ocr_json": ocr_json_key,
                    "fext.output_s3_key": fext_s3_key,
                },
            )

            out_payload = {
                "schema_version": HARYANA_CLU_SCHEMA_VERSION,
                "model_id": ocr_cfg.model_id,
                "s3": {
                    "bucket": bucket,
                    "pdf_key": pdf_key,
                    "ocr_json_key": ocr_json_key,
                    "fext_json_key": fext_s3_key,
                },
                "features": features,
            }

            out_bytes = json.dumps(out_payload, ensure_ascii=False, indent=2).encode("utf-8")
            _upload_bytes_to_s3(
                s3,
                bucket=bucket,
                key=fext_s3_key,
                data=out_bytes,
                content_type="application/json",
            )
            logger.info("  Uploaded features JSON -> s3://%s/%s", bucket, fext_s3_key)

            return {
                "s3_bucket": bucket,
                "input_pdf_s3_key": pdf_key,
                "ocr_json_s3_key": ocr_json_key,
                "fext_json_s3_key": fext_s3_key,
                "status": "ok",
            }

        except Exception:
            logger.exception("  FAILED feature extraction for PDF: s3://%s/%s", bucket, pdf_key)
            return {
                "s3_bucket": bucket,
                "input_pdf_s3_key": pdf_key,
                "ocr_json_s3_key": ocr_json_key,
                "fext_json_s3_key": fext_s3_key,
                "status": "error",
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_manifest_entry, idx, item): idx
            for idx, item in enumerate(manifest, start=1)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                fext_manifest.append(result)

            # continue to next doc

    # Upload Stage 02 manifest
    fext_manifest_key = f"{s3_prefix}manifest_fext.json"
    fext_manifest_bytes = json.dumps(fext_manifest, ensure_ascii=False, indent=2).encode("utf-8")
    _upload_bytes_to_s3(
        s3,
        bucket=bucket,
        key=fext_manifest_key,
        data=fext_manifest_bytes,
        content_type="application/json",
    )
    logger.info("Uploaded Stage 02 manifest -> s3://%s/%s", bucket, fext_manifest_key)

    # Upload log to S3 (optional) from memory
    if upload_log_to_s3:
        log_s3_key = f"{s3_prefix}stage_02_fext.log"
        log_bytes = log_buf.getvalue().encode("utf-8")
        _upload_bytes_to_s3(
            s3,
            bucket=bucket,
            key=log_s3_key,
            data=log_bytes,
            content_type="text/plain",
        )
        logger.info("Uploaded Stage 02 log -> s3://%s/%s", bucket, log_s3_key)

    logger.info(
        "Stage 02 FEXT Done | prefix=s3://%s/%s | docs=%d",
        bucket,
        s3_prefix,
        len(fext_manifest),
    )