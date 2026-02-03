# Example usage:
#   python ocr.py --pdf "C:\Users\albin\Documents\GitHub\GUI_AI_ocr\MASTER_PDFS" --recursive
#   python ocr.py --pdf "C:\Users\albin\Documents\GitHub\GUI_AI_ocr\MASTER_PDFS\106_CLU_GN-1953.pdf"

import argparse
import json
import os
import time
from typing import List, Dict, Any
from io import BytesIO
from pathlib import Path

import boto3
from botocore.config import Config
from pdf2image import convert_from_path

from dotenv import load_dotenv

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter


MODEL_OCR_DEFAULT = "us.meta.llama4-scout-17b-instruct-v1:0"


def setup_tracing(service_name: str, phoenix_otlp_endpoint: str | None) -> None:
    """
    Sends OpenTelemetry traces to Phoenix via OTLP gRPC.
    Example endpoint: http://localhost:4317
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    if phoenix_otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=phoenix_otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))


def load_env(env_file: str | None) -> None:
    """
    Load environment variables from a .env file.
    Defaults to .env.local located next to this script, then falls back to current working dir.
    """
    if env_file:
        load_dotenv(env_file, override=False)
        return

    # Prefer .env.local next to the script
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / ".env.local"
    if candidate.exists():
        load_dotenv(candidate.as_posix(), override=False)
        return

    # Fallback: .env.local in current working directory
    load_dotenv(".env.local", override=False)


def pdf_to_png_bytes(pdf_path: str, dpi: int = 250, max_pages: int | None = None) -> List[bytes]:
    images = convert_from_path(pdf_path, dpi=dpi)
    if max_pages is not None:
        images = images[:max_pages]

    png_pages: List[bytes] = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_pages.append(buf.getvalue())
    return png_pages


def bedrock_converse_ocr_page(
    brt,
    model_id: str,
    image_png_bytes: bytes,
    page_index: int,
    temperature: float = 0.0,
    max_tokens: int = 2500,
    retries: int = 6,
) -> str:
    """
    Calls Bedrock converse() with an image + OCR prompt.
    Returns extracted text.
    """
    tracer = trace.get_tracer(__name__)

    prompt_text = (
        "You are an OCR engine. Extract ALL readable text from the provided document image.\n"
        "Rules:\n"
        "- Output ONLY the extracted text (no commentary).\n"
        "- Preserve line breaks and basic structure.\n"
        "- If a word is unclear, keep best guess but do not invent content.\n"
        "- Keep numbers, dates, memo/reference numbers exactly as seen.\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt_text},
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": image_png_bytes},
                    }
                },
            ],
        }
    ]

    last_err = None
    for attempt in range(1, retries + 1):
        with tracer.start_as_current_span("bedrock.converse.ocr") as span:
            span.set_attribute("bedrock.model_id", model_id)
            span.set_attribute("doc.page_index", page_index)
            span.set_attribute("retry.attempt", attempt)

            try:
                resp = brt.converse(
                    modelId=model_id,
                    messages=messages,
                    inferenceConfig={
                        "maxTokens": max_tokens,
                        "temperature": temperature,
                    },
                )

                # Mark this span as an LLM span (Phoenix recognizes this)
                span.set_attribute("openinference.span.kind", "LLM")
                span.set_attribute("llm.model_name", model_id)

                # Bedrock usage fields can vary by model/SDK version; handle safely.
                usage = resp.get("usage", {}) or resp.get("metrics", {}).get("usage", {}) or {}

                # Common Bedrock usage keys (seen across Bedrock APIs)
                input_tokens = (
                    usage.get("inputTokens")
                    or usage.get("input_tokens")
                    or usage.get("promptTokens")
                    or usage.get("prompt_tokens")
                    or 0
                )
                output_tokens = (
                    usage.get("outputTokens")
                    or usage.get("output_tokens")
                    or usage.get("completionTokens")
                    or usage.get("completion_tokens")
                    or 0
                )
                total_tokens = usage.get("totalTokens") or usage.get("total_tokens") or (input_tokens + output_tokens)

                # OpenInference token attributes Phoenix understands
                span.set_attribute("llm.token_count.prompt", int(input_tokens))
                span.set_attribute("llm.token_count.completion", int(output_tokens))
                span.set_attribute("llm.token_count.total", int(total_tokens))

                # (Optional) also expose under generic keys for easier filtering
                span.set_attribute("bedrock.usage.input_tokens", int(input_tokens))
                span.set_attribute("bedrock.usage.output_tokens", int(output_tokens))
                span.set_attribute("bedrock.usage.total_tokens", int(total_tokens))

                out_msg = resp.get("output", {}).get("message", {})
                content_blocks = out_msg.get("content", [])

                texts: List[str] = []
                for block in content_blocks:
                    if "text" in block and block["text"]:
                        texts.append(block["text"])

                return "\n".join(texts).strip()

            except Exception as e:
                last_err = e
                span.record_exception(e)
                sleep_s = min(20, (2 ** (attempt - 1)) * 0.7)
                time.sleep(sleep_s)

    raise RuntimeError(f"Bedrock OCR failed after {retries} attempts. Last error: {last_err}")


def collect_pdf_paths(pdf_or_dir: str, recursive: bool = False) -> List[str]:
    p = Path(pdf_or_dir)

    if p.is_file():
        return [p.as_posix()]

    if not p.is_dir():
        raise FileNotFoundError(f"--pdf path not found: {pdf_or_dir}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    return [x.as_posix() for x in sorted(p.glob(pattern)) if x.is_file()]


def main():
    parser = argparse.ArgumentParser(
        description="OCR PDFs using AWS Bedrock (Llama4 Scout) + Phoenix tracing."
    )
    parser.add_argument("--pdf", required=True, help="Path to a PDF OR a folder containing PDFs")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If --pdf is a folder, search PDFs recursively",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write OCR outputs. Default: next to each PDF.",
    )
    parser.add_argument("--region", default=None, help="AWS region (overrides AWS_REGION env)")
    parser.add_argument("--model_id", default=MODEL_OCR_DEFAULT, help="Bedrock model ID")
    parser.add_argument("--dpi", type=int, default=250, help="PDF->image DPI (higher helps scans)")
    parser.add_argument("--max_pages", type=int, default=None, help="Limit pages for testing")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    parser.add_argument("--max_tokens", type=int, default=2500, help="Max tokens in response per page")
    parser.add_argument(
        "--phoenix-otlp",
        default=None,
        help="Phoenix OTLP gRPC endpoint (overrides PHOENIX_OTLP_ENDPOINT env). Use empty to disable tracing.",
    )
    parser.add_argument("--service-name", default="bedrock-pdf-ocr", help="OTel service name")
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to env file (default: .env.local next to script, else current working dir)",
    )
    parser.add_argument(
        "--debug-creds",
        action="store_true",
        help="Print whether AWS env vars are present (does NOT print secret values).",
    )

    args = parser.parse_args()

    # Load .env.local BEFORE reading env vars / creating boto3 client
    load_env(args.env_file)

    # Region resolution
    region = args.region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

    # Phoenix endpoint resolution
    phoenix_env = os.getenv("PHOENIX_OTLP_ENDPOINT", "http://localhost:4317")
    phoenix_arg = args.phoenix_otlp
    phoenix_endpoint = (phoenix_arg if phoenix_arg is not None else phoenix_env).strip()
    if phoenix_endpoint == "":
        phoenix_endpoint = None  # disables tracing

    setup_tracing(service_name=args.service_name, phoenix_otlp_endpoint=phoenix_endpoint)
    tracer = trace.get_tracer(__name__)

    if args.debug_creds:
        print("AWS_REGION:", region)
        print("Has AWS_ACCESS_KEY_ID:", bool(os.getenv("AWS_ACCESS_KEY_ID")))
        print("Has AWS_SECRET_ACCESS_KEY:", bool(os.getenv("AWS_SECRET_ACCESS_KEY")))
        print("Has AWS_SESSION_TOKEN:", bool(os.getenv("AWS_SESSION_TOKEN")))
        print("PHOENIX_OTLP_ENDPOINT:", phoenix_endpoint)

    # Bedrock Runtime client (boto3 will read creds from env loaded above)
    brt = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(retries={"max_attempts": 10, "mode": "standard"}),
    )

    pdf_list = collect_pdf_paths(args.pdf, recursive=args.recursive)
    if not pdf_list:
        raise SystemExit(f"No PDFs found at: {args.pdf}")

    for pdf_path in pdf_list:
        pdf_path_obj = Path(pdf_path)
        pdf_stem = pdf_path_obj.stem

        # Decide where to put outputs
        if args.out_dir:
            base_dir = Path(args.out_dir)
        else:
            base_dir = pdf_path_obj.parent  # default: next to PDF

        base_dir.mkdir(parents=True, exist_ok=True)

        out_json_path = base_dir / f"{pdf_stem}.ocr.json"
        out_txt_path = base_dir / f"{pdf_stem}.ocr.txt"

        with tracer.start_as_current_span("pdf.ocr.pipeline") as span:
            span.set_attribute("input.pdf", pdf_path)
            span.set_attribute("bedrock.model_id", args.model_id)
            span.set_attribute("aws.region", region)

            pages_png = pdf_to_png_bytes(pdf_path, dpi=args.dpi, max_pages=args.max_pages)

            results: List[Dict[str, Any]] = []
            all_text_lines: List[str] = []

            for i, png_bytes in enumerate(pages_png, start=1):
                with tracer.start_as_current_span("pdf.ocr.page") as page_span:
                    page_span.set_attribute("doc.page_number", i)

                    page_text = bedrock_converse_ocr_page(
                        brt=brt,
                        model_id=args.model_id,
                        image_png_bytes=png_bytes,
                        page_index=i,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )

                    results.append({"page": i, "text": page_text})
                    all_text_lines.append(f"===== PAGE {i} =====\n{page_text}\n")

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"pdf": pdf_path, "model_id": args.model_id, "pages": results},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(all_text_lines))

            print(
                f"Done: {pdf_path}\n"
                f"- JSON: {out_json_path}\n"
                f"- TXT:  {out_txt_path}\n"
                f"- Pages: {len(results)}\n"
            )


if __name__ == "__main__":
    main()
