import time
from typing import Any
from opentelemetry import trace

def bedrock_converse_ocr_page(
    brt: Any,
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
                {"image": {"format": "png", "source": {"bytes": image_png_bytes}}},
            ],
        }
    ]

    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        with tracer.start_as_current_span("bedrock.converse.ocr") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.model_name", model_id)
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

                return "\n".join(texts).strip()

            except Exception as e:
                last_err = e
                span.record_exception(e)
                sleep_s = min(20, (2 ** (attempt - 1)) * 0.7)
                time.sleep(sleep_s)

    raise RuntimeError(f"Bedrock OCR failed after {retries} attempts. Last error: {last_err}")
