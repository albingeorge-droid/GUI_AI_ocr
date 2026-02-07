import os
import argparse
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def upload_pdfs(local_dir: Path, bucket: str, top_folder: str) -> None:
    s3 = boto3.client("s3")

    if not local_dir.exists() or not local_dir.is_dir():
        raise ValueError(f"Local directory does not exist or is not a directory: {local_dir}")

    pdfs = sorted([p for p in local_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    if not pdfs:
        print(f"No PDFs found in: {local_dir}")
        return

    print(f"Found {len(pdfs)} PDF(s) in {local_dir}")
    print(f"Uploading to s3://{bucket}/{top_folder}/<pdf_name>/<pdf_file> ...\n")

    for pdf_path in pdfs:
        pdf_filename = pdf_path.name                 # e.g. 1_CLU_FT-216.pdf
        pdf_stem = pdf_path.stem                     # e.g. 1_CLU_FT-216
        s3_key = f"{top_folder}/{pdf_stem}/{pdf_filename}"

        try:
            s3.upload_file(str(pdf_path), bucket, s3_key)
            print(f"✅ Uploaded: {pdf_path}  ->  s3://{bucket}/{s3_key}")
        except (BotoCoreError, ClientError) as e:
            print(f"❌ Failed: {pdf_path}  ->  s3://{bucket}/{s3_key}")
            print(f"   Error: {e}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Upload PDFs to S3 with folder-per-PDF structure.")
    parser.add_argument("--local_dir", required=True, help="Local folder containing PDFs")
    parser.add_argument("--bucket", default="guai", help="S3 bucket name (default: guai)")
    parser.add_argument("--top_folder", required=True, help="Top folder name to create in the bucket")
    args = parser.parse_args()

    upload_pdfs(Path(args.local_dir), args.bucket, args.top_folder)


if __name__ == "__main__":
    main()
# python local_to_aws.py --local_dir "PDFS" --top_folder "Haryana1"
