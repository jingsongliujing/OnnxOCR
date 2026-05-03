from typing import BinaryIO

from onnxocr.rapid_doc.model.pptx.pptx_converter import PptxConverter


def convert_path(file_path: str):
    with open(file_path, "rb") as fh:
        return convert_binary(fh)


def convert_binary(file_binary: BinaryIO):
    converter = PptxConverter()
    converter.convert(file_binary)
    return converter.pages


if __name__ == "__main__":
    print(convert_path(r"D:\file\ppt\OpenClaw_多智能体团队协作流程汇报v2.0.pptx"))
