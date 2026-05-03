import os
from copy import deepcopy

from loguru import logger
from onnxocr.rapid_doc.utils.table_merge import merge_table
from ..model.layout.rapid_layout_self.model_handler.pp_doclayout.post_process import calculate_polygon_overlap_ratio, \
    calculate_bbox_area
from ..model.reading_order.utils import calculate_overlap_ratio
from ..utils.boxbase import is_in
from ..utils.span_pre_proc import txt_in_ori_image


def cross_page_table_merge(pdf_info: list[dict]):
    """Merge tables that span across multiple pages in a PDF document.

    Args:
        pdf_info (list[dict]): A list of dictionaries containing information about each page in the PDF.

    Returns:
        None
    """
    is_merge_table = os.getenv('MINERU_TABLE_MERGE_ENABLE', 'true')
    if is_merge_table.lower() in ['true', '1', 'yes']:
        merge_table(pdf_info)
    elif is_merge_table.lower() in ['false', '0', 'no']:
        pass
    else:
        logger.warning(f'unknown MINERU_TABLE_MERGE_ENABLE config: {is_merge_table}, pass')
        pass

def remove_layout_in_ori_images(images_layout_res, pdf_dict_list, scale_list):
    """
    移除落在原始图片区域内的 layout_res，
    对于确实替换掉内容的图片区域，添加一条新的 'image' 类型检测结果。
    若图片区域中含有文本（背景图），则跳过该图片。
    """
    for index, layout_res in enumerate(images_layout_res):
        ori_image_list = pdf_dict_list[index].get('ori_image_list')
        scale = scale_list[index]

        if not ori_image_list:
            continue

        # ✅ 过滤掉包含文本的“背景图”区域
        valid_ori_images = []
        for ori in ori_image_list:
            if txt_in_ori_image(pdf_dict_list[index], ori['bbox']):
                # 含文本，不视为纯图片，跳过
                continue
            valid_ori_images.append(ori)

        if not valid_ori_images:
            # 没有纯图片，跳过
            images_layout_res[index] = layout_res
            continue

        # ✅ 计算所有原图 bbox（已缩放）
        scaled_ori_bboxes = [
            [
                ori['bbox'][0] * scale,
                ori['bbox'][1] * scale,
                ori['bbox'][2] * scale,
                ori['bbox'][3] * scale
            ]
            for ori in valid_ori_images
        ]

        filtered_layout_res = []
        replaced_ori_bboxes = set()  # 记录被用来替换的图片区域下标

        for res in layout_res:
            # 保留 category_id==2 的区域
            if res['category_id'] == 2:
                filtered_layout_res.append(res)
                continue

            x1, y1, x2, y2 = res['poly'][0], res['poly'][1], res['poly'][4], res['poly'][5]
            res_bbox = [int(x1), int(y1), int(x2), int(y2)]

            # 检查是否落入某个图片区域
            matched_idx = None
            for idx, ori_bbox in enumerate(scaled_ori_bboxes):
                if is_in(res_bbox, ori_bbox):
                    matched_idx = idx
                    break

            if matched_idx is not None:
                replaced_ori_bboxes.add(matched_idx)  # 记录该图片区域确实发生替换
                continue  # 删除该 layout_res
            filtered_layout_res.append(res)

        # ✅ 只添加被替换掉的图片区域
        for idx in replaced_ori_bboxes:
            xmin, ymin, xmax, ymax = map(int, scaled_ori_bboxes[idx])
            image_res = {
                "category_id": 3,
                "original_label": "image",
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": 1.0,
            }
            filtered_layout_res.append(image_res)

        images_layout_res[index] = filtered_layout_res

    return images_layout_res


def filter_overlap_boxes(layout_det_res, use_custom_ocr):
    """
    Remove overlapping boxes from layout detection results based on a given overlap ratio.

    Args:
        layout_det_res: Layout detection result dict containing a 'boxes' list.

    Returns:
       Filtered dict with overlapping boxes removed.
    """
    layout_det_res_filtered = deepcopy(layout_det_res)
    boxes = [
        box for box in layout_det_res_filtered if box["original_label"] != "reference"
    ]
    dropped_indexes = set()

    for i in range(len(boxes)):
        coordinate_i = [boxes[i]["poly"][0], boxes[i]["poly"][1], boxes[i]["poly"][4], boxes[i]["poly"][5]]
        x1, y1, x2, y2 = coordinate_i
        w, h = x2 - x1, y2 - y1
        if w < 6 or h < 6:
            dropped_indexes.add(i)
        for j in range(i + 1, len(boxes)):
            if i in dropped_indexes or j in dropped_indexes:
                continue
            coordinate_j = [boxes[j]["poly"][0], boxes[j]["poly"][1], boxes[j]["poly"][4], boxes[j]["poly"][5]]
            overlap_ratio = calculate_overlap_ratio(
                coordinate_i, coordinate_j, "small"
            )
            if (
                boxes[i]["original_label"] == "inline_formula"
                or boxes[j]["original_label"] == "inline_formula"
            ):
                if not use_custom_ocr:
                    continue
                if overlap_ratio > 0.5:
                    if boxes[i]["original_label"] == "inline_formula":
                        dropped_indexes.add(i)
                    if boxes[j]["original_label"] == "inline_formula":
                        dropped_indexes.add(j)
                    continue
            if overlap_ratio > 0.7:
                if boxes[i].get("polygon_points"):
                    poly_overlap_ratio = calculate_polygon_overlap_ratio(
                        boxes[i]["polygon_points"], boxes[j]["polygon_points"], "small"
                    )
                    if poly_overlap_ratio < 0.7:
                        continue
                box_area_i = calculate_bbox_area(coordinate_i)
                box_area_j = calculate_bbox_area(coordinate_j)
                if {boxes[i]["original_label"], boxes[j]["original_label"]} & {
                    "image",
                    # "table",
                    "seal",
                    "chart",
                } and boxes[i]["original_label"] != boxes[j]["original_label"]:
                    continue
                if box_area_i >= box_area_j:
                    dropped_indexes.add(j)
                else:
                    dropped_indexes.add(i)
    layout_det_res_filtered = [
        box for idx, box in enumerate(boxes) if idx not in dropped_indexes
    ]
    return layout_det_res_filtered