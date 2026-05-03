from tempfile import NamedTemporaryFile
import cv2
import numpy as np
import polars as pl
import os
from img2table.document.base import Document
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe

from typing import List

class RapidOcrTable(OCRInstance):
    """
    """
    def __init__(self, ocr_result):
        self.ocr_result = ocr_result

    def hocr(self, image: np.ndarray) -> List:
        """
        Get OCR of an image using Paddle
        :param image: numpy array representing the image
        :return: Paddle OCR result
        """
        with NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_f:
            tmp_file = tmp_f.name
            # Write image to temporary file
            cv2.imwrite(tmp_file, image)

            # Get OCR
            ocr_result = self.ocr_result

        # Remove temporary file
        while os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except PermissionError:
                pass

        # Get result
        # return [(bbox, word, conf) for bbox, word, conf in ocr_result] if ocr_result else []
        return [(bbox, word, conf) for bbox, word, conf in zip(*ocr_result)] if ocr_result else []

    def content(self, document: Document) -> List[List]:
        # Get OCR of all images
        ocrs = [self.hocr(image=image) for image in document.images]

        return ocrs

    def to_ocr_dataframe(self, content: List[List]) -> OCRDataframe:
        """
        Convert hOCR HTML to OCRDataframe object
        :param content: hOCR HTML string
        :return: OCRDataframe object corresponding to content
        """
        # Create list of elements
        list_elements = list()

        for page, ocr_result in enumerate(content):
            word_id = 0
            for word in ocr_result:
                word_id += 1
                dict_word = {
                    "page": page,
                    "class": "ocrx_word",
                    "id": f"word_{page + 1}_{word_id}",
                    "parent": f"word_{page + 1}_{word_id}",
                    "value": word[1],
                    "confidence": round(100 * word[2]),
                    "x1": round(min([edge[0] for edge in word[0]])),
                    "y1": round(min([edge[1] for edge in word[0]])),
                    "x2": round(max([edge[0] for edge in word[0]])),
                    "y2": round(max([edge[1] for edge in word[0]]))
                }

                list_elements.append(dict_word)

        return OCRDataframe(df=pl.DataFrame(list_elements)) if list_elements else None