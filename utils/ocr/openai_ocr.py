from __future__ import annotations

import base64
import json
import os
import logging
from typing import Any, Dict, Optional

from openai import OpenAI
from opentelemetry import trace
from opentelemetry.context import Context

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
                "Set it in .env.local or your environment before running OCR."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def gpt_ocr_page(
    image_png_bytes: bytes,
    model_id: str = "gpt-5-nano",
    max_tokens: int = 4096,  # Note: This parameter is ignored for gpt-5-nano
    page_index: int = 1,
    trace_attrs: Optional[Dict[str, Any]] = None,
    trace_max_output_chars: int = 12000,
) -> str:
    """
    Run OCR on a single PNG page using OpenAI Vision API.
    Returns plain text for that page.
    
    Creates a ROOT span in Phoenix for each call.
    
    IMPORTANT: gpt-5-nano does NOT accept max_completion_tokens parameter!
    """
    client = get_openai_client()
    tracer = trace.get_tracer(__name__)

    base64_image = base64.b64encode(image_png_bytes).decode("utf-8")
    
    # Log image size for debugging
    logger.debug(f"Page {page_index}: Image size = {len(image_png_bytes)} bytes, base64 size = {len(base64_image)} chars")

    prompt = (
        "You are an OCR engine. Read all legible text from this document page and "
        "return it as plain text.\n"
        "- Preserve line breaks when it makes sense.\n"
        "- Do NOT summarize.\n"
        "- Do NOT add any commentary.\n"
        "- Just return the raw text."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                },
            ],
        }
    ]

    # Trace input payload (without base64 image to keep it small)
    trace_input_payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "format": "png", "bytes_len": len(image_png_bytes)},
                ],
            }
        ],
        "model": model_id,
        "page_index": page_index,
    }

    # ✅ ROOT SPAN: new trace per call (matches bedrock.py pattern)
    with tracer.start_as_current_span("openai_ocr_page", context=Context()) as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.model_name", model_id)
        span.set_attribute("openai.model_id", model_id)
        span.set_attribute("doc.page_index", page_index)
        span.set_attribute("image.bytes_len", len(image_png_bytes))

        # Attach extra metadata for filtering/searching in Phoenix
        if trace_attrs:
            for k, v in trace_attrs.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    span.set_attribute(k, v)
                else:
                    span.set_attribute(k, json.dumps(v, ensure_ascii=False))

        # Prompt visible in Phoenix
        span.set_attribute("input.value", json.dumps(trace_input_payload, ensure_ascii=False))
        span.set_attribute("input.mime_type", "application/json")

        try:
            logger.info(f"Calling OpenAI API with model={model_id}, page={page_index}")
            
            # ✅ CRITICAL: gpt-5-nano does NOT accept max_completion_tokens!
            # Only pass model and messages
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                # DO NOT include max_completion_tokens for gpt-5-nano
            )

            logger.debug(f"Page {page_index}: Received response from OpenAI")

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
                    f"Page {page_index}: Tokens - input={input_tokens}, output={output_tokens}, "
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
                logger.debug(f"Page {page_index}: Got string content, length={len(content)}")
                
            # Case 2: content is a list of parts (some models return this)
            elif isinstance(getattr(msg, "content", None), list):
                text_parts = []
                for part in msg.content:
                    if isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype in ("text", "output_text"):
                            text_parts.append(part.get("text", ""))
                    # Handle object-style parts
                    elif hasattr(part, "text"):
                        text_parts.append(part.text)
                content = "\n".join([t for t in text_parts if t]).strip()
                logger.debug(f"Page {page_index}: Got list content with {len(text_parts)} parts, total length={len(content)}")
            
            # Fallback: check for convenience fields
            if not content:
                content = getattr(resp, "output_text", None)
                if content:
                    logger.debug(f"Page {page_index}: Got content from output_text field")

            # Final parsing
            if content:
                out_text = content.strip()
            else:
                out_text = ""
                logger.warning(f"Page {page_index}: No content found in response!")

            # Log if output is empty
            if not out_text:
                logger.error(f"Page {page_index}: EMPTY OCR RESULT! Check API response.")
                logger.error(f"  Response type: {type(resp)}")
                logger.error(f"  Message content type: {type(msg.content)}")
                logger.error(f"  Message content value: {msg.content}")
                span.add_event("empty_output", {
                    "warning": "OCR returned empty text",
                    "content_type": str(type(msg.content)),
                    "content_value": str(msg.content)[:200] if msg.content else "None"
                })
            else:
                logger.info(f"Page {page_index}: Successfully extracted {len(out_text)} characters")

            # Output visible in Phoenix (truncate if huge)
            if trace_max_output_chars and len(out_text) > trace_max_output_chars:
                out_for_trace = out_text[:trace_max_output_chars] + "\n\n[TRUNCATED]"
            else:
                out_for_trace = out_text

            span.set_attribute("output.value", out_for_trace)
            span.set_attribute("output.mime_type", "text/plain")
            span.set_attribute("output.length", len(out_text))

            return out_text

        except Exception as e:
            logger.exception(f"Page {page_index}: OpenAI API error: {e}")
            span.record_exception(e)
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            
            # Re-raise to let caller know
            raise RuntimeError(f"OpenAI OCR failed for page {page_index}: {e}") from e