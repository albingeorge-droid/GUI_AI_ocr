#!/usr/bin/env python3
"""
Diagnostic script to test OpenAI gpt-5-nano OCR functionality.
This will help identify why OCR is returning empty results.

Usage:
    python diagnose_openai_ocr.py path/to/test.pdf

Or test with a simple test image:
    python diagnose_openai_ocr.py CLU_TRY\1_CLU_FT-216.pdf
"""

import argparse
import base64
import os
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env.local")  


def create_test_image() -> bytes:
    """Create a simple test image with text."""
    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO
    
    # Create white image
    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add text
    text = "This is a test OCR document.\nLine 2: Hello World!\nLine 3: 12345"
    draw.text((50, 50), text, fill='black')
    
    # Convert to PNG bytes
    buf = BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def test_openai_ocr(image_bytes: bytes, model: str = "gpt-5-nano") -> None:
    """Test OCR with detailed logging."""
    
    print(f"\n{'='*60}")
    print(f"Testing OpenAI OCR with model: {model}")
    print(f"{'='*60}\n")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("   Set it with: export OPENAI_API_KEY='sk-...'")
        return
    
    print(f"✓ API Key found: {api_key[:20]}...{api_key[-4:]}")
    print(f"✓ Image size: {len(image_bytes):,} bytes")
    
    # Initialize client
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI client initialized")
    
    # Encode image
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    print(f"✓ Base64 encoded: {len(base64_image):,} characters")
    
    # Prepare request
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
    
    print(f"\n📤 Sending request to OpenAI API...")
    print(f"   Model: {model}")
    print(f"   Max tokens: 4096")
    
    try:
        # Make API call
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=4096,
            messages=messages,
        )
        
        print(f"✓ Response received!")
        
        # Print usage
        if resp.usage:
            print(f"\n📊 Token Usage:")
            print(f"   Input tokens:  {resp.usage.prompt_tokens:,}")
            print(f"   Output tokens: {resp.usage.completion_tokens:,}")
            print(f"   Total tokens:  {resp.usage.total_tokens:,}")
        
        # Extract content
        message = resp.choices[0].message
        content = message.content
        
        print(f"\n📝 Response Details:")
        print(f"   Content type: {type(content)}")
        print(f"   Content is None: {content is None}")
        print(f"   Content is empty string: {content == ''}")
        
        if isinstance(content, str):
            print(f"   Content length: {len(content)} characters")
        elif isinstance(content, list):
            print(f"   Content parts: {len(content)}")
            for idx, part in enumerate(content):
                print(f"      Part {idx}: type={type(part)}, text={getattr(part, 'text', None)}")
        
        # Parse content
        out_text = ""
        
        if isinstance(content, str):
            out_text = content.strip()
        elif isinstance(content, list):
            parts = []
            for c in content:
                text = getattr(c, "text", None)
                if text:
                    parts.append(text)
                elif isinstance(c, dict) and c.get("type") == "text":
                    parts.append(c.get("text", ""))
            out_text = "\n".join(parts).strip()
        elif content is None:
            out_text = ""
        else:
            out_text = str(content).strip()
        
        # Print result
        print(f"\n{'='*60}")
        if out_text:
            print(f"✅ SUCCESS! Extracted {len(out_text)} characters")
            print(f"{'='*60}")
            print(f"\n📄 Extracted Text:\n")
            print(out_text)
        else:
            print(f"❌ FAILURE! OCR returned empty text")
            print(f"{'='*60}")
            print(f"\n🔍 Raw Response Object:")
            print(resp)
            print(f"\n🔍 Raw Content:")
            print(repr(content))
        
    except Exception as e:
        print(f"\n❌ ERROR during API call:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        import traceback
        traceback.print_exc()


def test_pdf_page(pdf_path: Path) -> None:
    """Test OCR on first page of PDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("❌ PyMuPDF not installed. Install with: pip install PyMuPDF")
        return
    
    from io import BytesIO
    
    print(f"📄 Loading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    if doc.page_count == 0:
        print("❌ PDF has no pages!")
        return
    
    print(f"✓ PDF has {doc.page_count} page(s)")
    
    # Render first page
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
    image_bytes = pix.tobytes("png")
    
    print(f"✓ Rendered page 1: {len(image_bytes):,} bytes")
    
    doc.close()
    
    # Test OCR
    test_openai_ocr(image_bytes)


def main():
    parser = argparse.ArgumentParser(description="Diagnose OpenAI OCR issues")
    parser.add_argument("pdf_path", nargs="?", help="Path to PDF file to test")
    parser.add_argument("--test-image", action="store_true", help="Use generated test image")
    parser.add_argument("--model", default="gpt-5-nano", help="Model to use (default: gpt-5-nano)")
    
    args = parser.parse_args()
    
    if args.test_image:
        print("🖼️  Creating test image...")
        image_bytes = create_test_image()
        test_openai_ocr(image_bytes, model=args.model)
    elif args.pdf_path:
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            return
        test_pdf_page(pdf_path)
    else:
        print("❌ Please provide a PDF path or use --test-image")
        parser.print_help()


if __name__ == "__main__":
    main()