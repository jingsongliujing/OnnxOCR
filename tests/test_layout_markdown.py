import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onnxocr.layout_markdown import LayoutMarkdownConverter
from tests.common import TEST_IMAGE_DIR, ensure_result_dir


def run_layout_markdown():
    converter = LayoutMarkdownConverter(
        layout_model_type="pp_doclayoutv2",
        formula_enable=False,
        table_enable=True,
    )
    start = time.time()
    result = converter.convert_file(
        str(TEST_IMAGE_DIR / "layout_cdla.jpg"),
        output_md_path=str(ensure_result_dir() / "test_layout_markdown.md"),
    )
    print("layout markdown total time: {:.3f}".format(time.time() - start))
    print("layout markdown path:", result["markdown_path"])
    print("layout markdown preview:", result["markdown"][:500])
    return result


if __name__ == "__main__":
    run_layout_markdown()
