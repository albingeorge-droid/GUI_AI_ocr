# uv run main.py ocr --S3-prefix "Haryana2" --force

# .\.venv-phoenix\Scripts\activate
# phoenix serve

# to overside the clean json even if it exists, add --force flag:
# uv run main.py clean --S3-prefix "Haryana2" --force

import argparse
from data_pipeline.stage_01_ocr import run_stage_01_ocr_from_s3

from data_pipeline.stage_02_fext import run_stage_02_fext_from_s3

from data_pipeline.stage_03_data_cleaning import run_stage_03_clean_from_s3



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
    ocr.add_argument(
    "--force",
    action="store_true",
    help="If set, re-generate OCR output even if it already exists in S3",
    )


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
    fext.add_argument(
    "--force",
    action="store_true",
    help="If set, re-run feature extraction even if features JSON already exists in S3",
    )


    # Data cleaning stage
    clean_parser = sub.add_parser(
        "clean",
        help="Stage 03: data cleaning for Haryana CLU features",
    )
    clean_parser.add_argument("--S3-prefix", required=True, help="S3 prefix, e.g. 'Haryana'")
    clean_parser.add_argument(
        "--params",
        default="params.yaml",
        help="Path to params.yaml",
    )
    clean_parser.add_argument(
        "--env-file",
        default=None,
        help="Optional env file path (defaults to .env.local)",
    )
    clean_parser.add_argument(
        "--clean-mapping-path",
        default="utils/clean/csv/Haryana_Clean_Data_final_with fileNames.xlsx",
        help="Local path to the cleaned CSV/XLSX mapping file",
    )
    clean_parser.add_argument(
        "--debug-creds",
        action="store_true",
        help="Print whether AWS env vars are present",
    )
    clean_parser.add_argument(
        "--log-level",
        default="INFO",
        help="DEBUG, INFO, WARNING, ERROR",
    )
    clean_parser.add_argument(
        "--upload-log-to-s3",
        action="store_true",
        help="If set, uploads stage_03_clean.log to S3 under the given prefix",
    )
    clean_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-write clean JSON even if it already exists",
    )



    # Pipeline: OCR -> FEXT -> CLEAN
    pipeline = sub.add_parser(
        "pipeline",
        help="Run Stage 01 OCR, then Stage 02 FEXT, then Stage 03 CLEAN (in order)",
    )
    pipeline.add_argument("--S3-prefix", required=True, help="S3 prefix, e.g. 'Haryana2'")
    pipeline.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    pipeline.add_argument("--env-file", default=None, help="Optional env file path")
    pipeline.add_argument("--max-pages", type=int, default=None, help="Limit pages per PDF (testing)")
    pipeline.add_argument("--clean-mapping-path", default="utils/clean/csv/Haryana_Clean_Data_final_with fileNames.xlsx")
    pipeline.add_argument("--debug-creds", action="store_true")
    pipeline.add_argument("--log-level", default="INFO")
    pipeline.add_argument("--upload-log-to-s3", action="store_true")

    # one force flag controls all 3 stages
    pipeline.add_argument(
        "--force",
        action="store_true",
        help="If set, re-run OCR/FEXT/CLEAN even if outputs already exist",
    )



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
            force=args.force,
        )
    elif args.command == "fext":
        run_stage_02_fext_from_s3(
            s3_prefix=args.S3_prefix,
            params_path=args.params,
            env_file=args.env_file,
            debug_creds=args.debug_creds,
            log_level=args.log_level,
            upload_log_to_s3=args.upload_log_to_s3,
            force=args.force,
        )

    elif args.command == "clean":
        run_stage_03_clean_from_s3(
            s3_prefix=args.S3_prefix,
            params_path=args.params,
            env_file=args.env_file,
            clean_mapping_path=args.clean_mapping_path,
            debug_creds=args.debug_creds,
            log_level=args.log_level,
            upload_log_to_s3=args.upload_log_to_s3,
            force=args.force,
        )
    elif args.command == "pipeline":
        # Stage 01: OCR
        run_stage_01_ocr_from_s3(
            s3_prefix=args.S3_prefix,
            params_path=args.params,
            env_file=args.env_file,
            max_pages=args.max_pages,
            debug_creds=args.debug_creds,
            log_level=args.log_level,
            upload_log_to_s3=args.upload_log_to_s3,
            force=args.force,
        )

        # Stage 02: FEXT
        run_stage_02_fext_from_s3(
            s3_prefix=args.S3_prefix,
            params_path=args.params,
            env_file=args.env_file,
            debug_creds=args.debug_creds,
            log_level=args.log_level,
            upload_log_to_s3=args.upload_log_to_s3,
            force=args.force,
        )

        # Stage 03: CLEAN
        run_stage_03_clean_from_s3(
            s3_prefix=args.S3_prefix,
            params_path=args.params,
            env_file=args.env_file,
            clean_mapping_path=args.clean_mapping_path,
            debug_creds=args.debug_creds,
            log_level=args.log_level,
            upload_log_to_s3=args.upload_log_to_s3,
            force=args.force,
        )


    
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
