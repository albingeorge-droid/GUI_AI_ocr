import json
import time
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.context import Context


def bedrock_converse_ocr_page(
    brt: Any,
    model_id: str,
    image_png_bytes: bytes,
    page_index: int,
    temperature: float = 0.0,
    max_tokens: int = 2500,
    retries: int = 6,
    trace_max_output_chars: int = 12000,
    trace_attrs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Calls Bedrock converse() with an image + OCR prompt.
    Returns extracted text.

    Phoenix behavior:
    - This span is created as a ROOT span (new trace) using context=Context().
      So each LLM call appears separately (no trace tree).
    - Stores:
      - input.value (prompt + metadata, no image bytes)
      - output.value (OCR output text, truncated)
      - llm.token_count.prompt/completion/total when available
      - additional trace_attrs (s3.bucket, s3.key, doc.page_number etc.)
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
                {"image": {"format": "png", "source": {"bytes": image_png_bytes}}},
            ],
        }
    ]

    trace_input_payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image", "format": "png", "bytes_len": len(image_png_bytes)},
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        "page_index": page_index,
    }

    last_err: Exception | None = None

    # ✅ ROOT SPAN: new trace per call
    with tracer.start_as_current_span("bedrock.converse.ocr", context=Context()) as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.model_name", model_id)
        span.set_attribute("bedrock.model_id", model_id)
        span.set_attribute("doc.page_index", page_index)

        # attach extra metadata for filtering/searching in Phoenix
        if trace_attrs:
            for k, v in trace_attrs.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    span.set_attribute(k, v)
                else:
                    span.set_attribute(k, json.dumps(v, ensure_ascii=False))

        # prompt visible in Phoenix
        span.set_attribute("input.value", json.dumps(trace_input_payload, ensure_ascii=False))
        span.set_attribute("input.mime_type", "application/json")

        for attempt in range(1, retries + 1):
            span.set_attribute("retry.attempt", attempt)

            try:
                resp = brt.converse(
                    modelId=model_id,
                    messages=messages,
                    inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
                )

                usage = resp.get("usage", {}) or resp.get("metrics", {}).get("usage", {}) or {}

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

                span.set_attribute("llm.token_count.prompt", int(input_tokens))
                span.set_attribute("llm.token_count.completion", int(output_tokens))
                span.set_attribute("llm.token_count.total", int(total_tokens))

                span.set_attribute("bedrock.usage.input_tokens", int(input_tokens))
                span.set_attribute("bedrock.usage.output_tokens", int(output_tokens))
                span.set_attribute("bedrock.usage.total_tokens", int(total_tokens))

                out_msg = resp.get("output", {}).get("message", {})
                content_blocks = out_msg.get("content", [])

                texts: list[str] = []
                for block in content_blocks:
                    if "text" in block and block["text"]:
                        texts.append(block["text"])

                out_text = "\n".join(texts).strip()

                # output visible in Phoenix (truncate if huge)
                if trace_max_output_chars and len(out_text) > trace_max_output_chars:
                    out_for_trace = out_text[:trace_max_output_chars] + "\n\n[TRUNCATED]"
                else:
                    out_for_trace = out_text

                span.set_attribute("output.value", out_for_trace)
                span.set_attribute("output.mime_type", "text/plain")

                return out_text

            except Exception as e:
                last_err = e
                span.record_exception(e)

                sleep_s = min(20, (2 ** (attempt - 1)) * 0.7)
                # optional event so you can see retry behavior
                try:
                    span.add_event("retry", {"attempt": attempt, "sleep_s": float(sleep_s), "error": str(e)})
                except Exception:
                    pass

                time.sleep(sleep_s)

        raise RuntimeError(f"Bedrock OCR failed after {retries} attempts. Last error: {last_err}")
