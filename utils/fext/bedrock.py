# utils/fext/bedrock.py
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from opentelemetry import trace
from pydantic import ValidationError

from .prompts import build_extraction_user_message
from .schema import HaryanaCLUPlotFeatures, HARYANA_CLU_SCHEMA_VERSION




def _extract_json_snippet(text: str) -> str:
    """
    Best-effort: pull out the JSON object from the model's response.
    Handles cases where the model wraps output in extra text or ```json fences.
    """
    # Strip typical markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        # remove leading/trailing ```...``` if present
        parts = text.split("```")
        # find the first part that looks like JSON
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                text = part
                break

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def bedrock_converse_extract_haryana_features(
    brt: Any,
    model_id: str,
    ocr_text: str,
    doc_id: str,
    temperature: float = 0.0,
    max_tokens: int = 800,
    retries: int = 4,
    max_chars_input: int = 15000,
    trace_max_output_chars: int = 8000,
    trace_attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call Bedrock Llama model to extract Haryana CLU features from OCR text.

    Returns a dict matching HaryanaCLUPlotFeatures.to_dict().
    """
    tracer = trace.get_tracer(__name__)
    last_err: Optional[Exception] = None

    user_text = build_extraction_user_message(ocr_text, max_chars=max_chars_input)

    messages = [
        {
            "role": "user",
            "content": [
                {"text": user_text},
            ],
        }
    ]

    # For tracing: sanitized input payload
    trace_input_payload = {
        "doc_id": doc_id,
        "messages": [{"role": "user", "content": [{"type": "text", "chars": len(user_text)}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }

    for attempt in range(1, retries + 1):
        with tracer.start_as_current_span("bedrock.converse.fext") as span:
            # Mark as LLM span for Phoenix/OpenInference
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.model_name", model_id)
            span.set_attribute("fext.schema_version", HARYANA_CLU_SCHEMA_VERSION)
            span.set_attribute("fext.doc_id", doc_id)
            span.set_attribute("retry.attempt", attempt)

            if trace_attrs:
                for k, v in trace_attrs.items():
                    span.set_attribute(str(k), v)

            # Record input summary
            span.set_attribute("input.value", json.dumps(trace_input_payload, ensure_ascii=False))
            span.set_attribute("input.mime_type", "application/json")

            try:
                resp = brt.converse(
                    modelId=model_id,
                    messages=messages,
                    inferenceConfig={
                        "maxTokens": max_tokens,
                        "temperature": temperature,
                    },
                )

                # Usage metadata (varies by model & SDK)
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
                total_tokens = usage.get("totalTokens") or usage.get("total_tokens") or (
                    input_tokens + output_tokens
                )

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

                raw_output = "\n".join(texts).strip()

                # Record (possibly truncated) output into trace
                if trace_max_output_chars and len(raw_output) > trace_max_output_chars:
                    traced_output = raw_output[:trace_max_output_chars] + "\n\n[TRUNCATED]"
                else:
                    traced_output = raw_output

                span.set_attribute("output.value", traced_output)
                span.set_attribute("output.mime_type", "text/plain")

                # Parse JSON
                # Parse JSON
                snippet = _extract_json_snippet(raw_output)
                parsed = json.loads(snippet)

                # ✅ Pydantic validation
                try:
                    model = HaryanaCLUPlotFeatures.model_validate(parsed)

                except ValidationError as ve:
                    # Attach validation error to trace for easier debugging
                    span.set_attribute("fext.validation_error", ve.json())
                    raise

                features_dict = model.model_dump()
                return features_dict


            except Exception as e:
                last_err = e
                span.record_exception(e)
                # simple exponential backoff
                sleep_s = min(20, (2 ** (attempt - 1)) * 0.7)
                time.sleep(sleep_s)

    raise RuntimeError(f"Feature extraction failed for {doc_id} after {retries} attempts. Last error: {last_err}")
