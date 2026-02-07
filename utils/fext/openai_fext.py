# utils/fext/openai_fext.py
from __future__ import annotations

import json
import os
import time
import logging
from typing import Any, Dict, Optional

from openai import OpenAI
from opentelemetry import trace
from opentelemetry.context import Context
from pydantic import ValidationError

from .prompts import build_extraction_user_message
from .schema import HaryanaCLUPlotFeatures, HARYANA_CLU_SCHEMA_VERSION

_client: OpenAI | None = None
logger = logging.getLogger(__name__)


def get_openai_client() -> OpenAI:
    """
    Lazily create a global OpenAI client after env vars are loaded.
    """
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Set it in .env.local or your environment before running feature extraction."
            )
        _client = OpenAI(api_key=api_key)
    return _client


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
            if part.startswith("json"):
                part = part[4:].strip()  # Remove 'json' prefix
            if part.startswith("{") and part.endswith("}"):
                text = part
                break

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def openai_extract_haryana_features(
    model_id: str,
    ocr_text: str,
    doc_id: str,
    temperature: float = 0.0,
    max_tokens: int = 800,  # Note: Ignored for gpt-5-nano
    retries: int = 4,
    max_chars_input: int = 15000,
    trace_max_output_chars: int = 8000,
    trace_attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call OpenAI GPT-5-nano to extract Haryana CLU features from OCR text.

    Returns a dict matching HaryanaCLUPlotFeatures.model_dump().
    
    IMPORTANT: gpt-5-nano does NOT accept max_completion_tokens!
    """
    client = get_openai_client()
    tracer = trace.get_tracer(__name__)
    last_err: Optional[Exception] = None

    user_text = build_extraction_user_message(ocr_text, max_chars=max_chars_input)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
            ],
        }
    ]

    # For tracing: sanitized input payload
    trace_input_payload = {
        "doc_id": doc_id,
        "messages": [{"role": "user", "content": [{"type": "text", "chars": len(user_text)}]}],
        "model": model_id,
        "temperature": temperature,
    }

    for attempt in range(1, retries + 1):
        with tracer.start_as_current_span("openai.fext", context=Context()) as span:
            # Mark as LLM span for Phoenix/OpenInference
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.model_name", model_id)
            span.set_attribute("fext.schema_version", HARYANA_CLU_SCHEMA_VERSION)
            span.set_attribute("fext.doc_id", doc_id)
            span.set_attribute("retry.attempt", attempt)

            if trace_attrs:
                for k, v in trace_attrs.items():
                    if v is None:
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        span.set_attribute(str(k), v)
                    else:
                        span.set_attribute(str(k), json.dumps(v, ensure_ascii=False))

            # Record input summary
            span.set_attribute("input.value", json.dumps(trace_input_payload, ensure_ascii=False))
            span.set_attribute("input.mime_type", "application/json")

            try:
                logger.info(f"Calling OpenAI API for feature extraction | model={model_id}, doc={doc_id}")

                # ✅ CRITICAL: gpt-5-nano does NOT accept max_completion_tokens!
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    # DO NOT include max_completion_tokens for gpt-5-nano
                )

                logger.debug(f"Feature extraction: Received response from OpenAI for {doc_id}")

                # Extract usage stats
                usage = resp.usage
                if usage:
                    input_tokens = usage.prompt_tokens or 0
                    
                    # Handle reasoning tokens if present
                    reasoning_tokens = 0
                    if hasattr(usage, "completion_tokens_details"):
                        reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0)
                    
                    completion_tokens = usage.completion_tokens or 0
                    output_tokens = completion_tokens - reasoning_tokens
                    total_tokens = usage.total_tokens or (input_tokens + completion_tokens)

                    logger.info(
                        f"{doc_id}: Tokens - input={input_tokens}, output={output_tokens}, "
                        f"reasoning={reasoning_tokens}, total={total_tokens}"
                    )

                    span.set_attribute("llm.token_count.prompt", int(input_tokens))
                    span.set_attribute("llm.token_count.completion", int(output_tokens))
                    span.set_attribute("llm.token_count.total", int(total_tokens))
                    
                    if reasoning_tokens > 0:
                        span.set_attribute("llm.token_count.reasoning", int(reasoning_tokens))

                    span.set_attribute("openai.usage.prompt_tokens", int(input_tokens))
                    span.set_attribute("openai.usage.completion_tokens", int(output_tokens))
                    span.set_attribute("openai.usage.total_tokens", int(total_tokens))

                # Extract message content - ROBUST handling for different response formats
                msg = resp.choices[0].message
                content = None
                
                # Case 1: content is a plain string (most common)
                if isinstance(getattr(msg, "content", None), str):
                    content = msg.content
                    logger.debug(f"{doc_id}: Got string content, length={len(content)}")
                    
                # Case 2: content is a list of parts
                elif isinstance(getattr(msg, "content", None), list):
                    text_parts = []
                    for part in msg.content:
                        if isinstance(part, dict):
                            ptype = part.get("type")
                            if ptype in ("text", "output_text"):
                                text_parts.append(part.get("text", ""))
                        elif hasattr(part, "text"):
                            text_parts.append(part.text)
                    content = "\n".join([t for t in text_parts if t]).strip()
                    logger.debug(f"{doc_id}: Got list content, length={len(content)}")
                
                # Fallback
                if not content:
                    content = getattr(resp, "output_text", None)

                if not content:
                    raise ValueError(f"Empty response from OpenAI for {doc_id}")

                raw_output = content.strip()

                # Record (possibly truncated) output into trace
                if trace_max_output_chars and len(raw_output) > trace_max_output_chars:
                    traced_output = raw_output[:trace_max_output_chars] + "\n\n[TRUNCATED]"
                else:
                    traced_output = raw_output

                span.set_attribute("output.value", traced_output)
                span.set_attribute("output.mime_type", "text/plain")
                span.set_attribute("output.length", len(raw_output))

                # Parse JSON
                snippet = _extract_json_snippet(raw_output)
                parsed = json.loads(snippet)

                # ✅ Pydantic validation
                try:
                    model = HaryanaCLUPlotFeatures.model_validate(parsed)
                except ValidationError as ve:
                    # Attach validation error to trace for easier debugging
                    span.set_attribute("fext.validation_error", ve.json())
                    logger.error(f"{doc_id}: Pydantic validation error: {ve}")
                    raise

                features_dict = model.model_dump()
                logger.info(f"{doc_id}: Successfully extracted features")
                return features_dict

            except Exception as e:
                last_err = e
                span.record_exception(e)
                logger.warning(f"{doc_id}: Attempt {attempt} failed: {e}")
                
                # Simple exponential backoff
                if attempt < retries:
                    sleep_s = min(20, (2 ** (attempt - 1)) * 0.7)
                    logger.info(f"{doc_id}: Retrying in {sleep_s:.1f}s...")
                    time.sleep(sleep_s)

    raise RuntimeError(
        f"Feature extraction failed for {doc_id} after {retries} attempts. Last error: {last_err}"
    )