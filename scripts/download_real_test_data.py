"""Download real-world test images for OnnxOCR CLI scenarios.

Downloads publicly available test images from various sources:
- PaddleOCR official examples
- Public OCR datasets
- Government sample documents

All images are from public sources and suitable for testing.
"""

from pathlib import Path
import urllib.request
import ssl
import os


# Real-world test images from public sources
REAL_WORLD_IMAGES = {
    # Invoice examples (public samples)
    "invoice_sample.jpg": [
        "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/doc/imgs/12.jpg",
        "https://github.com/PaddlePaddle/PaddleOCR/raw/release/2.7/doc/imgs/12.jpg",
    ],
    
    # General OCR examples with Chinese text
    "chinese_ocr_sample.jpg": [
        "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/doc/imgs/11.jpg",
        "https://github.com/PaddlePaddle/PaddleOCR/raw/release/2.7/doc/imgs/11.jpg",
    ],
    
    # Table example
    "table_sample.jpg": [
        "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/doc/imgs/table.jpg",
        "https://github.com/PaddlePaddle/PaddleOCR/raw/release/2.7/doc/imgs/table.jpg",
    ],
    
    # ID card example (public demo)
    "id_card_sample.jpg": [
        "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/doc/imgs/id_card.jpg",
        "https://github.com/PaddlePaddle/PaddleOCR/raw/release/2.7/doc/imgs/id_card.jpg",
    ],
    
    # Bank card example (public demo)
    "bank_card_sample.jpg": [
        "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/doc/imgs/bank_card.jpg",
        "https://github.com/PaddlePaddle/PaddleOCR/raw/release/2.7/doc/imgs/bank_card.jpg",
    ],
    
    # Business license example
    "business_license_sample.jpg": [
        "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/doc/imgs/business_license.jpg",
        "https://github.com/PaddlePaddle/PaddleOCR/raw/release/2.7/doc/imgs/business_license.jpg",
    ],
    
    # Receipt example
    "receipt_sample.jpg": [
        "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/doc/imgs/receipt.jpg",
        "https://github.com/PaddlePaddle/PaddleOCR/raw/release/2.7/doc/imgs/receipt.jpg",
    ],
    
    # License plate example
    "plate_sample.jpg": [
        "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/doc/imgs/plate.jpg",
        "https://github.com/PaddlePaddle/PaddleOCR/raw/release/2.7/doc/imgs/plate.jpg",
    ],
}


def download_with_fallback(urls: list, output_path: Path, timeout: int = 30) -> bool:
    """Try downloading from multiple URLs with fallback."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as response:
                data = response.read()
                
            # Check if we got valid image data (at least 5KB)
            if len(data) < 5000:
                print(f"  Warning: File too small ({len(data)} bytes), trying next URL...")
                continue
                
            output_path.write_bytes(data)
            print(f"  Downloaded: {output_path.name} ({len(data):,} bytes)")
            return True
            
        except Exception as e:
            print(f"  Failed: {url[:60]}... - {type(e).__name__}")
            continue
    
    return False


def create_placeholder_image(output_path: Path, text: str) -> None:
    """Create a placeholder image with text when download fails."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # If Pillow not available, create a minimal valid JPEG
        output_path.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 100)
        return
    
    width, height = 800, 600
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    
    # Try to use a font
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
    except:
        font = ImageFont.load_default()
    
    # Draw text
    draw.text((50, 50), text, fill="black", font=font)
    draw.text((50, 100), f"Placeholder for: {output_path.name}", fill="gray", font=font)
    draw.text((50, 150), "Download real images with:", fill="gray", font=font)
    draw.text((50, 180), "python scripts/download_real_test_data.py", fill="blue", font=font)
    
    img.save(output_path, "JPEG", quality=95)


def main():
    root = Path(__file__).resolve().parents[1]
    samples_dir = root / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Fix Windows console encoding
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 60)
    print("OnnxOCR Real-World Test Data Downloader")
    print("=" * 60)
    print(f"\nOutput directory: {samples_dir}\n")
    
    success_count = 0
    fail_count = 0
    
    for filename, urls in REAL_WORLD_IMAGES.items():
        output_path = samples_dir / filename
        
        # Skip if file already exists and is valid
        if output_path.exists() and output_path.stat().st_size > 5000:
            print(f"[OK] {filename} (already exists)")
            success_count += 1
            continue
        
        print(f"\nDownloading {filename}...")
        if download_with_fallback(urls, output_path):
            success_count += 1
        else:
            print(f"  Creating placeholder image...")
            create_placeholder_image(output_path, f"Test image: {filename}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {success_count} downloaded, {fail_count} placeholders created")
    print("=" * 60)
    
    if fail_count > 0:
        print("\nNote: Some downloads failed. Placeholder images were created.")
        print("To get real images, you can:")
        print("1. Run this script again with better network connection")
        print("2. Manually download images and place them in data/samples/")
        print("3. Use images from your own test data")
    
    print("\nAvailable test data:")
    for f in sorted(samples_dir.glob("*")):
        size = f.stat().st_size
        if size > 5000:
            status = "[OK]"
        else:
            status = "[PLACEHOLDER]"
        print(f"  {status} {f.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
