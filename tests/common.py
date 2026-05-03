from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE_DIR = PROJECT_ROOT / "onnxocr" / "test_images"
RESULT_DIR = PROJECT_ROOT / "result_img"


def ensure_result_dir() -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR
