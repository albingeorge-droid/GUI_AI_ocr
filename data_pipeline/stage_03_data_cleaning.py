# data_pipeline/stage_3_data_cleaning.py
from __future__ import annotations

import json
import logging
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config as BotoConfig

from utils.ocr.env import load_env
from utils.ocr.tracing import setup_tracing
from utils.ocr.params import load_params, get_ocr_config, get_phoenix_config

from utils.clean import (
    load_haryana_clean_mapping,
    apply_haryana_csv_overrides,
)
from utils.clean.date_cleaning import clean_clu_permission_date_field
from utils.clean.charges import clean_charge_fields
from utils.clean.name_clean import clean_applicant_name_field


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
    content_type: Optional[str] = None,
) -> None:
    extra: Dict[str, Any] = {}
    if content_type:
        extra["ContentType"] = content_type
    s3_client.put_object(Bucket=bucket, Key=key, Body=data, **extra)


def _setup_logger_to_memory(log_level: str = "INFO") -> tuple[logging.Logger, StringIO]:
    """
    Logger that writes to stdout AND an in-memory buffer (so we can upload logs to S3 without local files).
    """
    logger = logging.getLogger("stage_03_clean")
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


def run_stage_03_clean_from_s3(
    s3_prefix: str,
    params_path: str = "params.yaml",
    env_file: Optional[str] = None,
    clean_mapping_path: str = "utils/clean/csv/Haryana_Clean_Data_final_with fileNames.xlsx",
    debug_creds: bool = False,
    log_level: str = "INFO",
    upload_log_to_s3: bool = True,
    force: bool = False,
) -> None:
    """
    Stage 03: Data cleaning for Haryana CLU features.

    Input:
      - Stage 02 features JSONs in S3 (per-PDF: <pdf_folder>/fext/<pdf_stem>.features.json)
      - Stage 02 manifest: <s3_prefix>/manifest_fext.json
      - Local cleaned mapping CSV/XLSX (clean_mapping_path)

    Output:
      - Per-document cleaned JSONs:
          <pdf_folder>/clean/<pdf_stem>.clean.json
      - Stage 03 manifest:
          <s3_prefix>/manifest_clean.json
      - Optional log file:
          <s3_prefix>/stage_03_clean.log
    """
    # Load env + params
    load_env(env_file)
    params = load_params(params_path)
    ocr_cfg = get_ocr_config(params)
    phoenix_cfg = get_phoenix_config(params)

    bucket = os.getenv("OCR_S3_BUCKET", ocr_cfg.bucket)
    region = (
        os.getenv("AWS_REGION", ocr_cfg.region)
        or os.getenv("AWS_DEFAULT_REGION", ocr_cfg.region)
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

    # Normalize prefix
    s3_prefix = s3_prefix.lstrip("/")
    if s3_prefix and not s3_prefix.endswith("/"):
        s3_prefix += "/"

    logger, log_buf = _setup_logger_to_memory(log_level=log_level)

    if debug_creds:
        logger.info("AWS_REGION: %s", region)
        logger.info("Bucket: %s", bucket)
        logger.info("Has AWS_ACCESS_KEY_ID: %s", bool(os.getenv("AWS_ACCESS_KEY_ID")))
        logger.info(
            "Has AWS_SECRET_ACCESS_KEY: %s", bool(os.getenv("AWS_SECRET_ACCESS_KEY"))
        )
        logger.info("Has AWS_SESSION_TOKEN: %s", bool(os.getenv("AWS_SESSION_TOKEN")))
        logger.info("PHOENIX_OTLP_ENDPOINT: %s", phoenix_endpoint)

    s3 = boto3.client(
        "s3",
        region_name=region,
        config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"}),
    )

    logger.info(
        "Starting Stage 03 CLEAN | s3://%s/%s | model=%s | region=%s",
        bucket,
        s3_prefix,
        ocr_cfg.model_id,
        region,
    )

    # Load CSV/XLSX mapping (local file)
    mapping = load_haryana_clean_mapping(clean_mapping_path)
    logger.info("Loaded clean mapping rows: %d", len(mapping))

    # Load Stage 02 manifest
    manifest_key = f"{s3_prefix}manifest_fext.json"
    logger.info("Loading Stage 02 manifest -> s3://%s/%s", bucket, manifest_key)

    manifest = _load_manifest_from_s3(s3, bucket=bucket, key=manifest_key)
    if not manifest:
        logger.warning("Stage 02 manifest is empty; nothing to process.")
        return

    logger.info("Stage 02 manifest entries: %d", len(manifest))

    clean_manifest: List[Dict[str, Any]] = []

    for idx, item in enumerate(manifest, start=1):
        status = (item.get("status") or "").lower()
        if status not in ("ok", "skipped_exists"):
            logger.info(
                "[%d/%d] Skipping entry with status=%s -> %s",
                idx,
                len(manifest),
                status,
                item,
            )
            continue

        pdf_key = item.get("input_pdf_s3_key") or item.get("pdf_s3_key")
        fext_json_key = item.get("fext_json_s3_key")
        if not pdf_key or not fext_json_key:
            logger.warning(
                "[%d/%d] Missing pdf / fext keys in manifest entry: %s",
                idx,
                len(manifest),
                item,
            )
            continue

        pdf_path = Path(pdf_key)
        base_dir = pdf_path.parent.as_posix().rstrip("/")
        pdf_stem = pdf_path.stem

        clean_dir = f"{base_dir}/clean"
        clean_json_key = f"{clean_dir}/{pdf_stem}.clean.json"

        logger.info(
            "[%d/%d] CLEAN PDF=%s | FEXT=%s | CLEAN_OUT=%s",
            idx,
            len(manifest),
            pdf_key,
            fext_json_key,
            clean_json_key,
        )

        # Skip if cleaned JSON already exists
        if not force and _s3_key_exists(s3, bucket=bucket, key=clean_json_key):
            logger.info(
                "  Skipping (clean JSON already exists) -> s3://%s/%s",
                bucket,
                clean_json_key,
            )
            clean_manifest.append(
                {
                    "s3_bucket": bucket,
                    "input_pdf_s3_key": pdf_key,
                    "fext_json_s3_key": fext_json_key,
                    "clean_json_s3_key": clean_json_key,
                    "mapping_found": True,  # unknown but irrelevant
                    "status": "skipped_exists",
                }
            )
            continue

        try:
            # ---- Load features JSON from S3 ----
            obj = s3.get_object(Bucket=bucket, Key=fext_json_key)
            raw = obj["Body"].read().decode("utf-8")
            fext_payload = json.loads(raw)

            features = fext_payload.get("features") or {}
            if not isinstance(features, dict):
                logger.warning(
                    "  Features is not a dict for %s (got %r); using empty dict.",
                    fext_json_key,
                    type(features),
                )
                features = {}

            # ---- Find matching CSV row by folder or stem ----
            folder_name = pdf_path.parent.name  # e.g. 114_CLU_ST-2326
            mapping_row = (
                mapping.get(folder_name)
                or mapping.get(pdf_stem)
            )

            if mapping_row is None:
                logger.warning(
                    "  No CSV mapping row found for folder='%s' or stem='%s'; "
                    "will copy features as-is.",
                    folder_name,
                    pdf_stem,
                )
                cleaned_features = dict(features)
                mapping_found = False
            else:
                cleaned_features = apply_haryana_csv_overrides(features, mapping_row)
                mapping_found = True

            # 🚿 Run field-level cleaners
            cleaned_features = clean_clu_permission_date_field(cleaned_features)
            cleaned_features = clean_charge_fields(cleaned_features)
            cleaned_features = clean_applicant_name_field(cleaned_features)

            # ---- Build cleaned payload ----
            cleaned_payload = dict(fext_payload)
            cleaned_payload["features"] = cleaned_features


            out_bytes = json.dumps(cleaned_payload, ensure_ascii=False, indent=2).encode(
                "utf-8"
            )

            _upload_bytes_to_s3(
                s3,
                bucket=bucket,
                key=clean_json_key,
                data=out_bytes,
                content_type="application/json",
            )
            logger.info(
                "  Uploaded CLEAN JSON -> s3://%s/%s", bucket, clean_json_key
            )

            clean_manifest.append(
                {
                    "s3_bucket": bucket,
                    "input_pdf_s3_key": pdf_key,
                    "fext_json_s3_key": fext_json_key,
                    "clean_json_s3_key": clean_json_key,
                    "mapping_pdf_name": folder_name,
                    "mapping_found": mapping_found,
                    "status": "ok",
                }
            )

        except Exception:
            logger.exception(
                "  FAILED cleaning for PDF: s3://%s/%s", bucket, pdf_key
            )
            clean_manifest.append(
                {
                    "s3_bucket": bucket,
                    "input_pdf_s3_key": pdf_key,
                    "fext_json_s3_key": fext_json_key,
                    "clean_json_s3_key": clean_json_key,
                    "mapping_pdf_name": folder_name,
                    "mapping_found": False,
                    "status": "error",
                }
            )
            # continue to next doc

    # ---- Upload Stage 03 manifest ----
    clean_manifest_key = f"{s3_prefix}manifest_clean.json"
    clean_manifest_bytes = json.dumps(
        clean_manifest, ensure_ascii=False, indent=2
    ).encode("utf-8")
    _upload_bytes_to_s3(
        s3,
        bucket=bucket,
        key=clean_manifest_key,
        data=clean_manifest_bytes,
        content_type="application/json",
    )
    logger.info(
        "Uploaded Stage 03 manifest -> s3://%s/%s", bucket, clean_manifest_key
    )

    # ---- Upload log to S3 (optional) ----
    if upload_log_to_s3:
        log_s3_key = f"{s3_prefix}stage_03_clean.log"
        log_bytes = log_buf.getvalue().encode("utf-8")
        _upload_bytes_to_s3(
            s3,
            bucket=bucket,
            key=log_s3_key,
            data=log_bytes,
            content_type="text/plain",
        )
        logger.info("Uploaded Stage 03 log -> s3://%s/%s", bucket, log_s3_key)

    logger.info(
        "Stage 03 CLEAN Done | prefix=s3://%s/%s | docs=%d",
        bucket,
        s3_prefix,
        len(clean_manifest),
    )
