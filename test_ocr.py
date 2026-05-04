from tests.test_general_ocr import run_general_ocr
from tests.test_layout_analysis import run_layout_analysis
from tests.test_layout_markdown import run_layout_markdown
from tests.test_license_plate_ocr import run_plate_ocr
from tests.test_table_ocr import run_table_ocr


def run_all():
    # The repository keeps only the PP-OCRv5 general OCR models by default.
    # Extra models for license plate OCR, table recognition, layout analysis,
    # and RapidDoc Markdown export can be downloaded on demand.
    #
    # Mainland China, ModelScope is recommended:
    #   python scripts/download_models.py
    #
    # International users, HuggingFace is recommended:
    #   python scripts/download_models.py --source huggingface
    #
    # HuggingFace mirror in mainland China:
    #   python scripts/download_models.py --source huggingface --hf-endpoint https://hf-mirror.com
    #
    # Model repositories:
    #   https://www.modelscope.cn/models/supersong/onnxocr_model/tree/master/models
    #   https://huggingface.co/jingsongliu/onnxocr_model/tree/main
    run_general_ocr()

    # Uncomment the examples below after downloading the corresponding models.
    # run_plate_ocr()
    # run_table_ocr()
    # run_layout_analysis()
    # run_layout_markdown()


if __name__ == "__main__":
    run_all()
