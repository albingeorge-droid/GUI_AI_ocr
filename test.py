from typing import List, Dict, Any, Optional
import json
import base64
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env.local")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_currency_data_from_pdf(pdf_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Use OpenAI GPT-5-nano Vision to extract currency data from PDF.
    """
    try:
        if not client:
            print("OpenAI client not initialized. Cannot extract currency data.")
            print("Please ensure OPENAI_API_KEY is set in your .env file")
            return None

        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            print(f"PDF file not found: {pdf_file}")
            return None

        print("Extracting currency data using OpenAI GPT-5-nano...")

        # Convert PDF to images and process
        import fitz  # PyMuPDF

        pdf_document = fitz.open(pdf_file)
        all_data = []

        # Process first few pages usually where exchange rates are listed
        for page_num in range(min(20, pdf_document.page_count)):
            page = pdf_document[page_num]

            # Render page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
            image_data = pix.tobytes("png")

            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode("utf-8")

            # Call GPT-5-nano Vision
            try:
                response = client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract the text exactly as it is from this PDF image. Return only the extracted text, with the same formatting and line breaks as closely as possible. Do not add, remove, summarize, or interpret anything.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                )

                # Extract token usage and log details
                usage = response.usage
                reasoning_tokens = (
                    getattr(usage.completion_tokens_details, "reasoning_tokens", 0)
                    if hasattr(usage, "completion_tokens_details")
                    else 0
                )
                output_tokens = usage.completion_tokens - reasoning_tokens

                print(f"  Page {page_num + 1}: Received response from GPT-5-nano")
                print(f"    ├─ Input tokens: {usage.prompt_tokens}")
                print(f"    ├─ Reasoning tokens: {reasoning_tokens}")
                print(f"    ├─ Output tokens: {output_tokens}")
                print(f"    └─ Total tokens: {usage.total_tokens}")

                # Extract text from response (robust to different SDK/model shapes)
                try:
                    content = None
                    msg = response.choices[0].message

                    # Case 1: content is a plain string
                    if isinstance(getattr(msg, "content", None), str):
                        content = msg.content

                    # Case 2: content is a list of parts (dicts)
                    elif isinstance(getattr(msg, "content", None), list):
                        text_parts = []
                        for part in msg.content:
                            if isinstance(part, dict):
                                ptype = part.get("type")
                                if ptype in ("text", "output_text"):
                                    text_parts.append(part.get("text", ""))
                        content = "\n".join([t for t in text_parts if t]).strip()

                    # Fallback: some SDKs expose convenience field(s)
                    if not content:
                        content = getattr(response, "output_text", None)

                    if content:
                        print(f"    ✓ Extracted text from page {page_num + 1}")
                        all_data.append({"page": page_num + 1, "text": content})
                    else:
                        print("    ✗ No text found in response")

                except Exception as e:
                    print(f"  Could not read content from page {page_num + 1}: {e}")

            except Exception as e:
                print(f"Error calling OpenAI API for page {page_num + 1}: {e}")

        pdf_document.close()

        if all_data:
            print(f"Extracted text from {len(all_data)} pages")
            return all_data
        else:
            print("No text extracted")
            return None

    except ImportError:
        print("PyMuPDF not available. Install with: pip install PyMuPDF")
        return None
    except Exception as e:
        print(f"Error extracting currency data: {e}")
        return None


if __name__ == "__main__":
    pdf_path = r"C:\Users\albin\Documents\GitHub\GUI_AI_ocr\CLU_TRY\1_CLU_FT-216.pdf"
    data = extract_currency_data_from_pdf(pdf_path)
    if data:
        print(json.dumps(data, indent=2))
    else:
        print("No data extracted from PDF.")
