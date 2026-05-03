from tests.test_general_ocr import run_general_ocr
from tests.test_layout_analysis import run_layout_analysis
from tests.test_layout_markdown import run_layout_markdown
from tests.test_license_plate_ocr import run_plate_ocr
from tests.test_table_ocr import run_table_ocr


def run_all():
    run_general_ocr()
    run_plate_ocr()
    run_table_ocr()
    run_layout_analysis()
    run_layout_markdown()


if __name__ == "__main__":
    run_all()
