#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config as BotoConfig


def extract_row_from_clean_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given the loaded JSON payload (one clean *.clean.json),
    return a flat row dict suitable for CSV.
    """
    s3_info = payload.get("s3", {}) or {}
    features = payload.get("features", {}) or {}

    # pdf_name from s3.pdf_key if present, else fallback to empty string
    pdf_key = s3_info.get("pdf_key") or ""
    pdf_name = Path(pdf_key).stem if pdf_key else ""

    # Flatten terms_and_conditions into a numbered list (multi-line string)
    terms = features.get("terms_and_conditions")
    if isinstance(terms, list):
        cleaned_terms = [str(t).strip() for t in terms if str(t).strip()]
        terms_flat = "\n".join(f"{i+1}. {t}" for i, t in enumerate(cleaned_terms))
    elif terms is None:
        terms_flat = ""
    else:
        terms_flat = str(terms).strip()

    # Flatten khasra_numbers into a single string
    khasra = features.get("khasra_numbers")
    if isinstance(khasra, list):
        khasra_flat = " || ".join(str(k).strip() for k in khasra if str(k).strip())
    elif khasra is None:
        khasra_flat = ""
    else:
        khasra_flat = str(khasra)


    row: Dict[str, Any] = {
        "pdf_name": pdf_name,
        "applicant_name": features.get("applicant_name"),
        "memo_no": features.get("memo_no"),
        "location_controlled_area": features.get("location_controlled_area"),
        "tehsil": features.get("tehsil"),
        "district": features.get("district"),
        "subject": features.get("subject"),
        "khasra_numbers": khasra_flat,
        "purpose": features.get("purpose"),
        "granted_area(sq_mtrs)": features.get("granted_area"),
        "clu_permission_date": features.get("clu_permission_date"),
        "conversion_charges": features.get("conversion_charges"),
        "total_external_development_charges": features.get("total_external_development_charges"),
        "lat": features.get("lat"),
        "long": features.get("long"),
        "terms_and_conditions": terms_flat,
    }

    return row


def list_clean_keys_from_s3(
    s3_client,
    bucket: str,
    prefix: str,
) -> List[str]:
    """
    List all S3 object keys under `prefix` that end with '.clean.json'.
    Uses paginator for safety.
    """
    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj.get("Key")
            if key and key.endswith(".clean.json"):
                keys.append(key)

    return keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Haryana clean JSONs from S3 and convert to a flat CSV."
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket name (e.g. 'guai')",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="S3 prefix under which clean JSONs live (e.g. 'Haryana2/').",
    )
    parser.add_argument(
        "--output-csv",
        default="haryana_clean_from_s3.csv",
        help="Local path to output CSV file.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (optional, else use env / default config).",
    )
    # 🔹 NEW: append flag
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV (no header) if file exists.",
    )

    args = parser.parse_args()

    region = args.region
    s3 = boto3.client(
        "s3",
        region_name=region,
        config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"}),
    )

    # Normalise prefix: no leading '/', optional trailing '/'
    prefix = args.prefix.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    print(f"Listing *.clean.json under s3://{args.bucket}/{prefix} ...")
    keys = list_clean_keys_from_s3(s3, bucket=args.bucket, prefix=prefix)
    print(f"Found {len(keys)} clean JSON files.")

    if not keys:
        print("No clean JSON files found. Exiting.")
        return

    rows: List[Dict[str, Any]] = []

    for i, key in enumerate(keys, start=1):
        try:
            if i % 100 == 0:
                print(f"  Processing {i}/{len(keys)}: {key}")

            obj = s3.get_object(Bucket=args.bucket, Key=key)
            raw = obj["Body"].read().decode("utf-8")
            payload = json.loads(raw)

            row = extract_row_from_clean_payload(payload)
            # If you want the S3 key also captured, uncomment:
            # row["s3_key"] = key
            rows.append(row)

        except Exception as e:
            print(f"[WARN] Failed to process s3://{args.bucket}/{key}: {e}")

    if not rows:
        print("No valid rows extracted. Exiting.")
        return

    # Define column order
    fieldnames = [
        "pdf_name",
        "applicant_name",
        "memo_no",
        "location_controlled_area",
        "tehsil",
        "district",
        "subject",
        "khasra_numbers",
        "purpose",
        "granted_area(sq_mtrs)",
        "clu_permission_date",
        "conversion_charges",
        "total_external_development_charges",
        "lat",
        "long",
        "terms_and_conditions",
    ]

    out_path = Path(args.output_csv)

    # 🔹 NEW: decide mode + header based on --append and file existence
    if args.append and out_path.exists():
        mode = "a"
        write_header = False
        print(f"Appending to existing CSV: {out_path}")
    else:
        mode = "w"
        write_header = True
        print(f"Writing new CSV (or overwriting): {out_path}")

    with out_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

# Example:
# uv run json_to_csv.py --bucket guai --prefix "Haryana2" --region us-east-2 --output-csv Haryana_all.csv
# uv run json_to_csv.py --bucket guai --prefix "Haryana"  --region us-east-2 --output-csv Haryana_all.csv --append
