# uv run main.py ocr --S3-prefix "Haryana"

# .\.venv-phoenix\Scripts\activate
# phoenix serve

import argparse
from data_pipeline.stage_01_ocr import run_stage_01_ocr_from_s3

from data_pipeline.stage_02_fext import run_stage_02_fext_from_s3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py")
    sub = p.add_subparsers(dest="command", required=True)

    ocr = sub.add_parser("ocr", help="Run Stage 01 OCR on PDFs under an S3 prefix")
    ocr.add_argument(
        "--S3-prefix",
        required=True,
        help="Folder/prefix in the S3 bucket (e.g. Haryana/)",
    )
    ocr.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    ocr.add_argument("--env-file", default=None, help="Optional env file path (defaults to .env.local)")
    ocr.add_argument("--max-pages", type=int, default=None, help="Limit pages per PDF (for testing)")
    ocr.add_argument("--debug-creds", action="store_true", help="Print whether AWS env vars are present")

    # Logging
    ocr.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    ocr.add_argument("--log-file", default=None, help="Optional local log file path")
    ocr.add_argument(
        "--upload-log-to-s3",
        action="store_true",
        help="If set, uploads stage_01_ocr.log to S3 under the given prefix",
    )

    # Feature extraction stage
    fext = sub.add_parser("fext", help="Run Stage 02 feature extraction from OCR outputs in S3")
    fext.add_argument("--S3-prefix", required=True, help="Prefix/folder in the S3 bucket (e.g. Haryana/)")
    fext.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    fext.add_argument("--env-file", default=None, help="Optional env file path")
    fext.add_argument("--debug-creds", action="store_true")
    fext.add_argument("--log-level", default="INFO")
    fext.add_argument("--upload-log-to-s3", action="store_true")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ocr":
        run_stage_01_ocr_from_s3(
            s3_prefix=args.S3_prefix,
            params_path=args.params,
            env_file=args.env_file,
            max_pages=args.max_pages,
            debug_creds=args.debug_creds,
            log_level=args.log_level,
            upload_log_to_s3=args.upload_log_to_s3,
        )
    elif args.command == "fext":
        run_stage_02_fext_from_s3(
            s3_prefix=args.S3_prefix,
            params_path=args.params,
            env_file=args.env_file,
            debug_creds=args.debug_creds,
            log_level=args.log_level,
            upload_log_to_s3=args.upload_log_to_s3,
        )
    
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
