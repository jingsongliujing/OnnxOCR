import typing
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import io
import cv2
import numpy as np

from img2table.document.base import Document
from img2table.document.base.rotation import fix_rotation_image
from img2table.tables.objects.extraction import ExtractedTable

if typing.TYPE_CHECKING:
    from img2table.ocr.base import OCRInstance


@dataclass
class Image(Document):
    # img2table图片传参默认不支持numpy，需要转成BytesIO，这里直接适配支持传入numpy
    src: typing.Union[str, Path, io.BytesIO, bytes, np.ndarray]
    detect_rotation: bool = False

    def validate_src(self, value: typing.Any, **_) -> typing.Union[str, Path, io.BytesIO, bytes, np.ndarray]:
        if not isinstance(value, (str, Path, io.BytesIO, bytes, np.ndarray)):
            raise TypeError(f"Invalid type {type(value)} for src argument")
        return value

    def __post_init__(self) -> None:
        self.pages = None

        super().__post_init__()

    @cached_property
    def images(self) -> list[np.ndarray]:
        if isinstance(self.src, np.ndarray):
            # 支持传入numpy
            img = self.src
        else:
            img = cv2.imdecode(np.frombuffer(self.bytes, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.detect_rotation:
            rotated_img, _ = fix_rotation_image(img=img)
            return [rotated_img]
        return [img]

    def extract_tables(self, ocr: "OCRInstance" = None, implicit_rows: bool = False, implicit_columns: bool = False,
                       borderless_tables: bool = False, min_confidence: int = 50) -> list[ExtractedTable]:
        """
        Extract tables from document
        :param ocr: OCRInstance object used to extract table content
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param implicit_columns: boolean indicating if implicit columns are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: list of extracted tables
        """
        extracted_tables = super().extract_tables(ocr=ocr,
                                                             implicit_rows=implicit_rows,
                                                             implicit_columns=implicit_columns,
                                                             borderless_tables=borderless_tables,
                                                             min_confidence=min_confidence)
        return extracted_tables.get(0)
