from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE_DIR = PROJECT_ROOT / "onnxocr" / "test_images"
OUTPUT_DIR = PROJECT_ROOT / "output"


def ensure_result_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
