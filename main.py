# uv run main.py ocr --S3-prefix "some/folder/"

# .\.venv-phoenix\Scripts\activate
# phoenix serve


import argparse
from data_pipeline.stage_01_ocr import run_stage_01_ocr_from_s3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py")
    sub = p.add_subparsers(dest="command", required=True)

    ocr = sub.add_parser("ocr", help="Run Stage 01 OCR on PDFs under an S3 prefix")
    ocr.add_argument("--S3-prefix", required=True, help="Folder/prefix in the S3 bucket (e.g. Kajaria raw emp doc/)")
    ocr.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    ocr.add_argument("--out-dir", default="artifacts/ocr", help="Local output directory")
    ocr.add_argument("--env-file", default=None, help="Optional env file path (defaults to .env.local)")
    ocr.add_argument("--max-pages", type=int, default=None, help="Limit pages per PDF (for testing)")
    ocr.add_argument("--debug-creds", action="store_true", help="Print whether AWS env vars are present")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ocr":
        run_stage_01_ocr_from_s3(
            s3_prefix=args.S3_prefix,
            params_path=args.params,
            out_dir=args.out_dir,
            env_file=args.env_file,
            max_pages=args.max_pages,
            debug_creds=args.debug_creds,
        )
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
