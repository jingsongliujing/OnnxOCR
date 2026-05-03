import json
from typing import BinaryIO

from onnxocr.rapid_doc.model.xlsx.xlsx_converter import XlsxConverter

def convert_path(file_path: str):
    with open(file_path, "rb") as fh:
        return convert_binary(fh)


def convert_binary(file_binary: BinaryIO):
    converter = XlsxConverter(gap_tolerance=1)
    converter.convert(file_binary)
    return converter.pages

if __name__ == "__main__":
    path = r"D:\file\xlsx\5f39d0a1-c458-4897-bb3e-8e128b140b5a.xlsx"
    content_str = convert_path(path)
    obj = json.loads(content_str)
    print(obj)


