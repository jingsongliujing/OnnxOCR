# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import copy
from typing import List, Union
import numpy as np
from onnxocr.rapid_doc.model.reading_order.layout_parsing.setting import BLOCK_SETTINGS, REGION_SETTINGS
from onnxocr.rapid_doc.model.reading_order.layout_parsing.utils import get_sub_regions_ocr_res, get_bbox_intersection, \
    calculate_overlap_ratio, calculate_minimum_enclosing_bbox, shrink_supplement_region_bbox

from onnxocr.rapid_doc.model.layout.rapid_layout_self import RapidLayoutInput, ModelType, RapidLayout
from onnxocr.rapid_doc.model.reading_order.layout_parsing.layout_objects import LayoutBlock, LayoutRegion
from onnxocr.rapid_doc.model.reading_order.layout_parsing.setting import BLOCK_LABEL_MAP
from onnxocr.rapid_doc.model.reading_order.layout_parsing.utils import update_region_box, caculate_bbox_area, \
    remove_overlap_blocks
from onnxocr.rapid_doc.model.reading_order.layout_parsing.xycut_enhanced import xycut_enhanced


def sort_layout_parsing_blocks(
        layout_parsing_page: LayoutRegion
) -> List[LayoutBlock]:
    layout_parsing_regions = xycut_enhanced(layout_parsing_page)
    parsing_res_list = []
    for region in layout_parsing_regions:
        layout_parsing_blocks = xycut_enhanced(region)
        parsing_res_list.extend(layout_parsing_blocks)

    return parsing_res_list


def standardized_data(
        image: list,
        region_det_res,
        layout_det_res,
        overall_ocr_res,
        text_rec_score_thresh: Union[float, None] = None,
) -> list:
    """
    Retrieves the layout parsing result based on the layout detection result, OCR result, and other recognition results.
    Args:
        image (list): The input image.
        overall_ocr_res: An object containing the overall OCR results, including detected text boxes and recognized text. The structure is expected to have:
            - "input_img": The image on which OCR was performed.
            - "dt_boxes": A list of detected text box coordinates.
            - "rec_texts": A list of recognized text corresponding to the detected boxes.

        layout_det_res: An object containing the layout detection results, including detected layout boxes and their labels. The structure is expected to have:
            - "boxes": A list of dictionaries with keys "coordinate" for box coordinates and "block_label" for the type of content.
        text_rec_score_thresh (Optional[float], optional): The score threshold for text recognition. Defaults to None.
    Returns:
        list: A list of dictionaries representing the layout parsing result.
    """

    matched_ocr_dict = {}
    region_to_block_map = {}
    block_to_ocr_map = {}
    object_boxes = []
    footnote_list = []
    paragraph_title_list = []
    bottom_text_y_max = 0
    max_block_area = 0.0
    doc_title_num = 0

    base_region_bbox = [65535, 65535, 0, 0]
    layout_det_res = remove_overlap_blocks(
        layout_det_res,
        threshold=0.5,
        smaller=True,
    )

    # match layout boxes and ocr boxes and get some information for layout_order_config
    for box_idx, box_info in enumerate(layout_det_res["boxes"]):
        box = box_info["coordinate"]
        label = box_info["label"].lower()
        object_boxes.append(box)
        _, _, _, y2 = box

        # update the region box and max_block_area according to the layout boxes
        base_region_bbox = update_region_box(box, base_region_bbox)
        max_block_area = max(max_block_area, caculate_bbox_area(box))

        # update_layout_order_config_block_index(layout_order_config, label, box_idx)

        # set the label of footnote to text, when it is above the text boxes
        if label == "footnote":
            footnote_list.append(box_idx)
        elif label == "paragraph_title":
            paragraph_title_list.append(box_idx)
        if label == "text":
            bottom_text_y_max = max(y2, bottom_text_y_max)
        if label == "doc_title":
            doc_title_num += 1

        if label not in ["formula", "table", "seal"]:
            _, matched_idxes = get_sub_regions_ocr_res(
                overall_ocr_res, [box], return_match_idx=True
            )
            block_to_ocr_map[box_idx] = matched_idxes
            for matched_idx in matched_idxes:
                if matched_ocr_dict.get(matched_idx, None) is None:
                    matched_ocr_dict[matched_idx] = [box_idx]
                else:
                    matched_ocr_dict[matched_idx].append(box_idx)

    # fix the footnote label
    for footnote_idx in footnote_list:
        if (
                layout_det_res["boxes"][footnote_idx]["coordinate"][3]
                < bottom_text_y_max
        ):
            layout_det_res["boxes"][footnote_idx]["label"] = "text"

    # check if there is only one paragraph title and without doc_title
    only_one_paragraph_title = len(paragraph_title_list) == 1 and doc_title_num == 0
    if only_one_paragraph_title:
        paragraph_title_block_area = caculate_bbox_area(
            layout_det_res["boxes"][paragraph_title_list[0]]["coordinate"]
        )
        title_area_max_block_threshold = BLOCK_SETTINGS.get(
            "title_conversion_area_ratio_threshold", 0.3
        )
        if (
                paragraph_title_block_area
                > max_block_area * title_area_max_block_threshold
        ):
            layout_det_res["boxes"][paragraph_title_list[0]]["label"] = "doc_title"

    # Replace the OCR information of the hurdles.
    for overall_ocr_idx, layout_box_ids in matched_ocr_dict.items():
        if len(layout_box_ids) > 1:
            matched_no = 0
            overall_ocr_box = copy.deepcopy(
                overall_ocr_res["rec_boxes"][overall_ocr_idx]
            )
            overall_ocr_dt_poly = copy.deepcopy(
                overall_ocr_res["dt_polys"][overall_ocr_idx]
            )
            for box_idx in layout_box_ids:
                layout_box = layout_det_res["boxes"][box_idx]["coordinate"]
                crop_box = get_bbox_intersection(overall_ocr_box, layout_box)
                for ocr_idx in block_to_ocr_map[box_idx]:
                    ocr_box = overall_ocr_res["rec_boxes"][ocr_idx]
                    iou = calculate_overlap_ratio(ocr_box, crop_box, "small")
                    if iou > 0.8:
                        overall_ocr_res["rec_texts"][ocr_idx] = ""
                # x1, y1, x2, y2 = [int(i) for i in crop_box]
                # crop_img = np.array(image)[y1:y2, x1:x2]
                #crop_img_rec_res = list(text_rec_model([crop_img]))[0]
                crop_img_dt_poly = get_bbox_intersection(
                    overall_ocr_dt_poly, layout_box, return_format="poly"
                )
                #crop_img_rec_score = crop_img_rec_res["rec_score"]
                #crop_img_rec_text = crop_img_rec_res["rec_text"]
                crop_img_rec_score = 0
                crop_img_rec_text = "-"
                text_rec_score_thresh = (
                    text_rec_score_thresh
                    if text_rec_score_thresh is not None
                    # else (self.general_ocr_pipeline.text_rec_score_thresh)
                    else (0)
                )
                if crop_img_rec_score >= text_rec_score_thresh:
                    matched_no += 1
                    if matched_no == 1:
                        # the first matched ocr be replaced by the first matched layout box
                        overall_ocr_res["dt_polys"][
                            overall_ocr_idx
                        ] = crop_img_dt_poly
                        overall_ocr_res["rec_boxes"][overall_ocr_idx] = crop_box
                        overall_ocr_res["rec_polys"][
                            overall_ocr_idx
                        ] = crop_img_dt_poly
                        overall_ocr_res["rec_scores"][
                            overall_ocr_idx
                        ] = crop_img_rec_score
                        overall_ocr_res["rec_texts"][
                            overall_ocr_idx
                        ] = crop_img_rec_text
                    else:
                        # the other matched ocr be appended to the overall ocr result
                        overall_ocr_res["dt_polys"].append(crop_img_dt_poly)
                        if len(overall_ocr_res["rec_boxes"]) == 0:
                            overall_ocr_res["rec_boxes"] = np.array([crop_box])
                        else:
                            overall_ocr_res["rec_boxes"] = np.vstack(
                                (overall_ocr_res["rec_boxes"], crop_box)
                            )
                        overall_ocr_res["rec_polys"].append(crop_img_dt_poly)
                        overall_ocr_res["rec_scores"].append(crop_img_rec_score)
                        overall_ocr_res["rec_texts"].append(crop_img_rec_text)
                        overall_ocr_res["rec_labels"].append("text")
                        block_to_ocr_map[box_idx].remove(overall_ocr_idx)
                        block_to_ocr_map[box_idx].append(
                            len(overall_ocr_res["rec_texts"]) - 1
                        )

    # when there is no layout detection result but there is ocr result, convert ocr detection result to layout detection result
    if len(layout_det_res["boxes"]) == 0 and len(overall_ocr_res["rec_boxes"]) > 0:
        for idx, ocr_rec_box in enumerate(overall_ocr_res["rec_boxes"]):
            base_region_bbox = update_region_box(ocr_rec_box, base_region_bbox)
            layout_det_res["boxes"].append(
                {
                    "label": "text",
                    "coordinate": ocr_rec_box,
                    "score": overall_ocr_res["rec_scores"][idx],
                }
            )
            block_to_ocr_map[idx] = [idx]

    mask_labels = (
            BLOCK_LABEL_MAP.get("unordered_labels", [])
            + BLOCK_LABEL_MAP.get("header_labels", [])
            + BLOCK_LABEL_MAP.get("footer_labels", [])
    )
    block_bboxes = [box["coordinate"] for box in layout_det_res["boxes"]]
    region_det_res["boxes"] = sorted(
        region_det_res["boxes"],
        key=lambda item: caculate_bbox_area(item["coordinate"]),
    )
    if len(region_det_res["boxes"]) == 0:
        region_det_res["boxes"] = [
            {
                "coordinate": base_region_bbox,
                "label": "SupplementaryRegion",
                "score": 1,
            }
        ]
        region_to_block_map[0] = range(len(block_bboxes))
    else:
        block_idxes_set = set(range(len(block_bboxes)))
        # match block to region
        for region_idx, region_info in enumerate(region_det_res["boxes"]):
            matched_idxes = []
            region_to_block_map[region_idx] = []
            region_bbox = region_info["coordinate"]
            for block_idx in block_idxes_set:
                if layout_det_res["boxes"][block_idx]["label"] in mask_labels:
                    continue
                overlap_ratio = calculate_overlap_ratio(
                    region_bbox, block_bboxes[block_idx], mode="small"
                )
                if overlap_ratio > REGION_SETTINGS.get(
                        "match_block_overlap_ratio_threshold", 0.8
                ):
                    matched_idxes.append(block_idx)
            old_region_bbox_matched_idxes = []
            if len(matched_idxes) > 0:
                while len(old_region_bbox_matched_idxes) != len(matched_idxes):
                    old_region_bbox_matched_idxes = copy.deepcopy(matched_idxes)
                    matched_idxes = []
                    matched_bboxes = [
                        block_bboxes[idx] for idx in old_region_bbox_matched_idxes
                    ]
                    new_region_bbox = calculate_minimum_enclosing_bbox(
                        matched_bboxes
                    )
                    for block_idx in block_idxes_set:
                        if (
                                layout_det_res["boxes"][block_idx]["label"]
                                in mask_labels
                        ):
                            continue
                        overlap_ratio = calculate_overlap_ratio(
                            new_region_bbox, block_bboxes[block_idx], mode="small"
                        )
                        if overlap_ratio > REGION_SETTINGS.get(
                                "match_block_overlap_ratio_threshold", 0.8
                        ):
                            matched_idxes.append(block_idx)
                for block_idx in matched_idxes:
                    block_idxes_set.remove(block_idx)
                region_to_block_map[region_idx] = matched_idxes
                region_det_res["boxes"][region_idx]["coordinate"] = new_region_bbox
        # Supplement region when there is no matched block
        while len(block_idxes_set) > 0:
            unmatched_bboxes = [block_bboxes[idx] for idx in block_idxes_set]
            if len(unmatched_bboxes) == 0:
                break
            supplement_region_bbox = calculate_minimum_enclosing_bbox(
                unmatched_bboxes
            )
            matched_idxes = []
            # check if the new region bbox is overlapped with other region bbox, if have, then shrink the new region bbox
            for region_idx, region_info in enumerate(region_det_res["boxes"]):
                if len(region_to_block_map[region_idx]) == 0:
                    continue
                region_bbox = region_info["coordinate"]
                overlap_ratio = calculate_overlap_ratio(
                    supplement_region_bbox, region_bbox
                )
                if overlap_ratio > 0:
                    supplement_region_bbox, matched_idxes = (
                        shrink_supplement_region_bbox(
                            supplement_region_bbox,
                            region_bbox,
                            image.shape[1],
                            image.shape[0],
                            block_idxes_set,
                            block_bboxes,
                        )
                    )

            matched_idxes = [
                idx
                for idx in matched_idxes
                if layout_det_res["boxes"][idx]["label"] not in mask_labels
            ]
            if len(matched_idxes) == 0:
                matched_idxes = [
                    idx
                    for idx in block_idxes_set
                    if layout_det_res["boxes"][idx]["label"] not in mask_labels
                ]
                if len(matched_idxes) == 0:
                    break
            matched_bboxes = [block_bboxes[idx] for idx in matched_idxes]
            supplement_region_bbox = calculate_minimum_enclosing_bbox(
                matched_bboxes
            )
            region_idx = len(region_det_res["boxes"])
            region_to_block_map[region_idx] = list(matched_idxes)
            for block_idx in matched_idxes:
                block_idxes_set.remove(block_idx)
            region_det_res["boxes"].append(
                {
                    "coordinate": supplement_region_bbox,
                    "label": "SupplementaryRegion",
                    "score": 1,
                }
            )

        mask_idxes = [
            idx
            for idx in range(len(layout_det_res["boxes"]))
            if layout_det_res["boxes"][idx]["label"] in mask_labels
        ]
        for idx in mask_idxes:
            bbox = layout_det_res["boxes"][idx]["coordinate"]
            region_idx = len(region_det_res["boxes"])
            region_to_block_map[region_idx] = [idx]
            region_det_res["boxes"].append(
                {
                    "coordinate": bbox,
                    "label": "SupplementaryRegion",
                    "score": 1,
                }
            )

    region_block_ocr_idx_map = dict(
        region_to_block_map=region_to_block_map,
        block_to_ocr_map=block_to_ocr_map,
    )

    return region_block_ocr_idx_map, region_det_res, layout_det_res


def get_layout_parsing_objects(
        image: list,
        region_block_ocr_idx_map: dict,
        region_det_res,
        overall_ocr_res,
        layout_det_res,
        text_rec_score_thresh: Union[float, None] = None,
) -> list:
    """
    Extract structured information from OCR and layout detection results.

    Args:
        image (list): The input image.
        overall_ocr_res: An object containing the overall OCR results, including detected text boxes and recognized text. The structure is expected to have:
            - "input_img": The image on which OCR was performed.
            - "dt_boxes": A list of detected text box coordinates.
            - "rec_texts": A list of recognized text corresponding to the detected boxes.

        layout_det_res: An object containing the layout detection results, including detected layout boxes and their labels. The structure is expected to have:
            - "boxes": A list of dictionaries with keys "coordinate" for box coordinates and "block_label" for the type of content.
        text_rec_score_thresh (Union[float, None]): The minimum score required for a recognized character to be considered valid. If None, use the default value specified during initialization. Default is None.

    Returns:
        list: A list of structured boxes where each item is a dictionary containing:
            - "block_label": The label of the content (e.g., 'table', 'chart', 'image').
            - The label as a key with either table HTML or image data and text.
            - "block_bbox": The coordinates of the layout box.
    """
    layout_parsing_blocks: List[LayoutBlock] = []

    for box_idx, box_info in enumerate(layout_det_res["boxes"]):

        label = box_info["label"]
        block_bbox = box_info["coordinate"]
        rec_res = {"boxes": [], "rec_texts": [], "rec_labels": []}

        block = LayoutBlock(label=label, bbox=block_bbox)

        if label == "formula":
            _, ocr_idx_list = get_sub_regions_ocr_res(
                overall_ocr_res, [block_bbox], return_match_idx=True
            )
            region_block_ocr_idx_map["block_to_ocr_map"][box_idx] = ocr_idx_list
        else:
            ocr_idx_list = region_block_ocr_idx_map["block_to_ocr_map"].get(
                box_idx, []
            )
        for box_no in ocr_idx_list:
            rec_res["boxes"].append(overall_ocr_res["rec_boxes"][box_no])
            rec_res["rec_texts"].append(
                overall_ocr_res["rec_texts"][box_no],
            )
            rec_res["rec_labels"].append(
                overall_ocr_res["rec_labels"][box_no],
            )
        block.update_text_content(
            image=image,
            ocr_rec_res=rec_res,
            text_rec_model=None,
            text_rec_score_thresh=text_rec_score_thresh,
        )

        if (
                label
                in ["seal", "table", "formula", "chart"]
                + BLOCK_LABEL_MAP["image_labels"]
        ):
            block.image = {"path": 'img_path', "img": 'img'}

        layout_parsing_blocks.append(block)

    page_region_bbox = [65535, 65535, 0, 0]
    layout_parsing_regions: List[LayoutRegion] = []
    for region_idx, region_info in enumerate(region_det_res["boxes"]):
        region_bbox = np.array(region_info["coordinate"]).astype("int")
        region_blocks = [
            layout_parsing_blocks[idx]
            for idx in region_block_ocr_idx_map["region_to_block_map"][region_idx]
        ]
        if region_blocks:
            page_region_bbox = update_region_box(region_bbox, page_region_bbox)
            region = LayoutRegion(bbox=region_bbox, blocks=region_blocks)
            layout_parsing_regions.append(region)

    layout_parsing_page = LayoutRegion(
        bbox=np.array(page_region_bbox).astype("int"), blocks=layout_parsing_regions
    )

    return layout_parsing_page

def get_layout_parsing_res(
        image: list,
        region_det_res,
        layout_det_res,
        overall_ocr_res,
    ) -> list:
        """
        Retrieves the layout parsing result based on the layout detection result, OCR result, and other recognition results.
        Args:
            image (list): The input image.
            layout_det_res: The detection result containing the layout information of the document.
            overall_ocr_res: The overall OCR result containing text information.
        Returns:
            list: A list of dictionaries representing the layout parsing result.
        """

        # Standardize data
        region_block_ocr_idx_map, region_det_res, layout_det_res = (
            standardized_data(
                image=image,
                region_det_res=region_det_res,
                layout_det_res=layout_det_res,
                overall_ocr_res=overall_ocr_res,
                text_rec_score_thresh=0,
            )
        )

        # Format layout parsing block
        layout_parsing_page = get_layout_parsing_objects(
            image=image,
            region_block_ocr_idx_map=region_block_ocr_idx_map,
            region_det_res=region_det_res,
            overall_ocr_res=overall_ocr_res,
            layout_det_res=layout_det_res,
            text_rec_score_thresh=0,
        )

        parsing_res_list = sort_layout_parsing_blocks(layout_parsing_page)

        index = 1
        for block in parsing_res_list:
            if block.label in BLOCK_LABEL_MAP["visualize_index_labels"]:
                block.order_index = index
                index += 1

        return parsing_res_list
