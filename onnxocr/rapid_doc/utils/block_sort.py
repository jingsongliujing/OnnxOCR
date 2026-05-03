# Copyright (c) Opendatalab. All rights reserved.
import copy
import statistics

import numpy as np
from loguru import logger

from onnxocr.rapid_doc.model.reading_order.layout_parsing.setting import blocktype_to_sort_label
from onnxocr.rapid_doc.model.reading_order.layout_parsing.xycut_plus_v3 import get_layout_parsing_res
from onnxocr.rapid_doc.utils.enum_class import BlockType, ContentType
from onnxocr.rapid_doc.model.reading_order.xycut_plus import xycut_plus_sort
from onnxocr.rapid_doc.utils.ocr_utils import bbox_to_points


def sort_blocks_by_bbox(blocks, page_w, page_h, footnote_blocks, page_pil_img):

    """获取所有line并计算正文line的高度"""
    line_height = get_line_height(blocks)

    """向blocks添加lines"""
    add_lines_to_blocks(blocks, page_w, page_h, line_height, footnote_blocks)

    """使用 xycut-plus 对blocks进行排序"""
    blocks = sort_blocks_by_xycut_plus(blocks, page_pil_img)

    """将image和table的block还原回group形式参与后续流程"""
    blocks = revert_group_blocks(blocks)

    """重排block"""
    sorted_blocks = sorted(blocks, key=lambda b: b['index'])

    """block内重排(img和table的block内多个caption或footnote的排序)"""
    for block in sorted_blocks:
        if block['type'] in [BlockType.IMAGE, BlockType.TABLE]:
            block['blocks'] = sorted(block['blocks'], key=lambda b: b['index'])

    return sorted_blocks


def get_line_height(blocks):
    page_line_height_list = []
    for block in blocks:
        if block['type'] in [
            BlockType.TEXT, BlockType.TITLE,
            BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE,
            BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE
        ]:
            for line in block['lines']:
                bbox = line['bbox']
                page_line_height_list.append(int(bbox[3] - bbox[1]))
    if len(page_line_height_list) > 0:
        return statistics.median(page_line_height_list)
    else:
        return 10


def add_lines_to_blocks(fix_blocks, page_w, page_h, line_height, footnote_blocks):
    page_line_list = []

    def add_lines_to_block(b):
        line_bboxes = insert_lines_into_block(b['bbox'], line_height, page_w, page_h)
        b['lines'] = []
        for line_bbox in line_bboxes:
            b['lines'].append({'bbox': line_bbox, 'spans': []})
        page_line_list.extend(line_bboxes)

    for block in fix_blocks:
        if block['type'] in [
            BlockType.TEXT, BlockType.TITLE,
            BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE,
            BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE
        ]:
            if len(block['lines']) == 0:
                add_lines_to_block(block)
            elif block['type'] in [BlockType.TITLE] and len(block['lines']) == 1 and (block['bbox'][3] - block['bbox'][1]) > line_height * 2:
                block['real_lines'] = copy.deepcopy(block['lines'])
                add_lines_to_block(block)
            else:
                for line in block['lines']:
                    bbox = line['bbox']
                    page_line_list.append(bbox)
        elif block['type'] in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.INTERLINE_EQUATION]:
            block['real_lines'] = copy.deepcopy(block['lines'])
            add_lines_to_block(block)

    for block in footnote_blocks:
        footnote_block = {'bbox': block[:4]}
        add_lines_to_block(footnote_block)


def insert_lines_into_block(block_bbox, line_height, page_w, page_h):
    # block_bbox是一个元组(x0, y0, x1, y1)，其中(x0, y0)是左下角坐标，(x1, y1)是右上角坐标
    x0, y0, x1, y1 = block_bbox

    block_height = y1 - y0
    block_weight = x1 - x0

    # 如果block高度小于n行正文，则直接返回block的bbox
    if line_height * 2 < block_height:
        if (
            block_height > page_h * 0.25 and page_w * 0.5 > block_weight > page_w * 0.25
        ):  # 可能是双列结构，可以切细点
            lines = int(block_height / line_height)
        else:
            # 如果block的宽度超过0.4页面宽度，则将block分成3行(是一种复杂布局，图不能切的太细)
            if block_weight > page_w * 0.4:
                lines = 3
            elif block_weight > page_w * 0.25:  # （可能是三列结构，也切细点）
                lines = int(block_height / line_height)
            else:  # 判断长宽比
                if block_height / block_weight > 1.2:  # 细长的不分
                    return [[x0, y0, x1, y1]]
                else:  # 不细长的还是分成两行
                    lines = 2

        line_height = (y1 - y0) / lines

        # 确定从哪个y位置开始绘制线条
        current_y = y0

        # 用于存储线条的位置信息[(x0, y), ...]
        lines_positions = []

        for i in range(lines):
            lines_positions.append([x0, current_y, x1, current_y + line_height])
            current_y += line_height
        return lines_positions

    else:
        return [[x0, y0, x1, y1]]

def extract_block_original_order(block):
    order = block.get('original_order')
    if order is not None and order >= 0:
        return order
    orders = []
    for line in block.get('lines', []):
        for span in line.get('spans', []):
            order = span.get('original_order')
            if order is not None and order >= 0:
                orders.append(order)
    return min(orders) if orders else -1

def sort_blocks_by_xycut_plus(fix_blocks, page_pil_img):
    for block in fix_blocks:
        # 如果block['bbox']任意值小于0，将其置为0
        block['bbox'] = [max(0, x) for x in block['bbox']]
        # 删除图表body block中的虚拟line信息, 并用real_lines信息回填
        if block['type'] in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.TITLE, BlockType.INTERLINE_EQUATION]:
            if 'real_lines' in block:
                block['virtual_lines'] = copy.deepcopy(block['lines'])
                block['lines'] = copy.deepcopy(block['real_lines'])
                del block['real_lines']
    try:
        # 判断fix_blocks里面有original_order>=0的，说明版面模型自带阅读顺序，直接使用版面的阅读顺序
        block_orders = [
            extract_block_original_order(block)
            for block in fix_blocks
        ]
        has_original_order = any(order >= 0 for order in block_orders)
        if has_original_order:
            for block, order in zip(fix_blocks, block_orders):
                block['index'] = order if order >= 0 else len(fix_blocks)
            sorted_blocks = sorted(fix_blocks, key=lambda b: b['index'])
            line_index = 1
            for block in sorted_blocks:
                for line in block.get('lines', []):
                    line['index'] = line_index
                    line_index += 1
            return fix_blocks
    except Exception as e:
        logger.exception(e)

    page_image = np.array(page_pil_img)
    block_bboxes = []
    layout_det_res = []
    rec_labels, rec_texts, rec_boxes, rec_polys, rec_scores, dt_polys = [],[],[],[],[],[]
    for block in fix_blocks:
        block_bboxes.append(block['bbox'])
        # 统计 lines 中的 spans 的 original_label 出现次数
        label_counter = {}
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                if span['type'] == ContentType.TEXT:
                    np_bbox = np.array(span['bbox'], dtype=np.float32)
                    points = bbox_to_points(span['bbox'])
                    rec_scores.append(span['score'])
                    rec_boxes.append(np_bbox)
                    rec_labels.append(span['type'])
                    rec_texts.append(span['content'])
                    dt_polys.append(points)
                    rec_polys.append(points)
                original_label = span.get('original_label')
                if original_label:
                    label_counter[original_label] = label_counter.get(original_label, 0) + 1
        # 取出现最多的 original_label
        if label_counter:
            most_common_label = max(label_counter, key=label_counter.get)
        else:
            # 如果没有任何 original_label，就退回使用 blocktype_to_sort_label
            most_common_label = blocktype_to_sort_label.get(block['type'], 'unknown')
        layout_det_res.append({
            "coordinate": block['bbox'],
            "label": most_common_label,
            "score": 1.0,
        })


    try:
        # 使用 xycut-plus-v3进行排序
        layout_det_res = {'boxes': layout_det_res}
        region_det_res = {'boxes': []}
        overall_ocr_res = {
            'rec_labels': rec_labels,
            'rec_texts': rec_texts,
            'rec_boxes': np.array(rec_boxes, dtype=np.float32),
            'rec_polys': rec_polys,
            'rec_scores': rec_scores,
            'dt_polys': dt_polys,
        }
        parsing_res_list = get_layout_parsing_res(
            page_image,
            region_det_res=region_det_res,
            layout_det_res=layout_det_res,
            overall_ocr_res=overall_ocr_res,
        )
        index_to_order = {parsing_res.index: order for order, parsing_res in enumerate(parsing_res_list)}
        for i, block in enumerate(fix_blocks):
            block['index'] = index_to_order.get(i, len(fix_blocks))
    except Exception as e:
        logger.exception(e)
        # 使用 xycut-plus 进行排序
        sorted_indices = xycut_plus_sort(block_bboxes)
        sorted_boxes = [block_bboxes[i] for i in sorted_indices]
        for i, block in enumerate(fix_blocks):
            block['index'] = sorted_boxes.index(block['bbox'])

    # 生成line index
    sorted_blocks = sorted(fix_blocks, key=lambda b: b['index'])
    line_inedx = 1
    for block in sorted_blocks:
        for line in block['lines']:
            line['index'] = line_inedx
            line_inedx += 1

    return fix_blocks


def revert_group_blocks(blocks):
    image_groups = {}
    table_groups = {}
    new_blocks = []
    for block in blocks:
        if block['type'] in [BlockType.IMAGE_BODY, BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE]:
            group_id = block['group_id']
            if group_id not in image_groups:
                image_groups[group_id] = []
            image_groups[group_id].append(block)
        elif block['type'] in [BlockType.TABLE_BODY, BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE]:
            group_id = block['group_id']
            if group_id not in table_groups:
                table_groups[group_id] = []
            table_groups[group_id].append(block)
        else:
            new_blocks.append(block)

    for group_id, blocks in image_groups.items():
        new_blocks.append(process_block_list(blocks, BlockType.IMAGE_BODY, BlockType.IMAGE))

    for group_id, blocks in table_groups.items():
        new_blocks.append(process_block_list(blocks, BlockType.TABLE_BODY, BlockType.TABLE))

    return new_blocks


def process_block_list(blocks, body_type, block_type):
    indices = [block['index'] for block in blocks]
    median_index = statistics.median(indices)

    body_block = next((block for block in blocks if block.get('type') == body_type), None)
    body_bbox = body_block['bbox'] if body_block else []
    polygon_points = body_block.get('polygon_points') if body_block else None

    result = {
        'type': block_type,
        'bbox': body_bbox,
        'blocks': blocks,
        'index': median_index,
    }
    if polygon_points:
        result['polygon_points'] = polygon_points
    return result