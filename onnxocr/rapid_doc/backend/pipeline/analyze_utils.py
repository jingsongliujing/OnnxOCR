# Copyright (c) RapidAI. All rights reserved.
"""
批量分析模块
"""
import copy
from typing import List, Dict
from collections import defaultdict

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from .model_init import AtomModelSingleton
from .model_list import AtomicModel
from ...utils.boxbase import rotate_table_image
from ...utils.enum_class import CategoryId
from ...utils.model_utils import crop_img
from ...utils.ocr_utils import (
    merge_det_boxes, update_det_boxes, sorted_boxes, get_rotate_crop_image,
    get_adjusted_mfdetrec_res, get_ocr_result_list, OcrConfidence, get_ocr_result_list_table
)
from ...utils.span_pre_proc import (
    txt_spans_extract, txt_spans_bbox_extract,
    txt_most_angle_extract_table, extract_table_fill_image
)

# =================================== OCR-det ===================================
def _extract_text_from_pdf(
        ocr_res_all_page: List[Dict],
        pdf_dict_list: List[Dict],
        scale_list: List[float]
):
    """从 PDF 中提取文本框"""
    ocr_res_grouped = {}
    for x in ocr_res_all_page:
        ocr_res_grouped.setdefault(x["page_idx"], []).append(x)

    total_texts = sum(len(texts) for texts in ocr_res_grouped.values())

    with tqdm(total=total_texts, desc="PDF-det Predict") as pbar:
        for page_idx, text_list in ocr_res_grouped.items():
            page_dict = pdf_dict_list[page_idx] if text_list else {}
            scale = scale_list[page_idx] if text_list else 1.0

            for ocr_res_dict in text_list:
                if ocr_res_dict['ocr_enable']:
                    continue
                if page_dict.get("rotate_label") in ["90", "180", "270"]:
                    ocr_res_dict['ocr_enable'] = True
                    continue

                for res in ocr_res_dict['ocr_res_list']:
                    new_image, useful_list = crop_img(
                        res, ocr_res_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )

                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_dict['single_page_mfdetrec_res'] + ocr_res_dict['checkbox_res'],
                        useful_list
                    )

                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                    ocr_res = txt_spans_bbox_extract(
                        page_dict, res, mfd_res=adjusted_mfdetrec_res,
                        scale=scale, useful_list=useful_list
                    )

                    if ocr_res:
                        ocr_result_list = get_ocr_result_list(
                            ocr_res, useful_list, ocr_res_dict['ocr_enable'],
                            bgr_image, ocr_res_dict['lang'],
                            res['original_label'], res['original_order']
                        )
                        ocr_res_dict['layout_res'].extend(ocr_result_list)

                pbar.update(1)


def _run_ocr_det_batch(
        ocr_res_all_page: List[Dict],
        atom_model_manager: AtomModelSingleton,
        ocr_config
):
    """批量 OCR 检测"""
    # OCR 配置
    use_det_mode = ocr_config.get("use_det_mode", "auto")
    ocr_det_base_batch_size = ocr_config.get("Det.rec_batch_num", 1)

    all_cropped_info = []

    for ocr_res_dict in ocr_res_all_page:
        for res in ocr_res_dict['ocr_res_list']:
            ocr_enable = ocr_res_dict['ocr_enable']

            if not ocr_res_dict['ocr_enable']:
                if res.get('need_ocr_det'):
                    ocr_enable = True
                elif use_det_mode == 'txt' or (use_det_mode != 'ocr' and not res.get('need_ocr_det')):
                    continue

            res.pop('need_ocr_det', None)

            new_image, useful_list = crop_img(
                res, ocr_res_dict['np_img'], crop_paste_x=50, crop_paste_y=50
            )

            adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                ocr_res_dict['single_page_mfdetrec_res'] + ocr_res_dict['checkbox_res'],
                useful_list
            )

            bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

            all_cropped_info.append((
                bgr_image, useful_list, ocr_res_dict, res,
                adjusted_mfdetrec_res, ocr_res_dict['lang'], ocr_enable
            ))

    if not all_cropped_info:
        return

    # 按语言分组
    lang_groups = defaultdict(list)
    for info in all_cropped_info:
        lang_groups[info[5]].append(info)

    RESOLUTION_GROUP_STRIDE = 64

    for lang, lang_crop_list in lang_groups.items():
        if not lang_crop_list:
            continue

        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            det_db_box_thresh=0.3,
            lang=lang,
            ocr_config=ocr_config,
        )

        # 按分辨率分组
        resolution_groups = defaultdict(list)
        for info in lang_crop_list:
            h, w = info[0].shape[:2]
            target_h = ((h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
            target_w = ((w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
            resolution_groups[(target_h, target_w)].append(info)

        # 批量处理
        for (target_h, target_w), group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det {lang}"):
            batch_images = []
            for info in group_crops:
                img = info[0]
                h, w = img.shape[:2]
                padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                padded_img[:h, :w] = img
                batch_images.append(padded_img)

            det_batch_size = min(len(batch_images), ocr_det_base_batch_size)
            batch_results = ocr_model.det_batch_predict(batch_images, det_batch_size)

            for info, (dt_boxes, _) in zip(group_crops, batch_results):
                bgr_image, useful_list, ocr_res_dict, res, adjusted_mfdetrec_res, _lang, ocr_enable = info

                if dt_boxes is not None and len(dt_boxes) > 0:
                    dt_boxes_sorted = sorted_boxes(dt_boxes)
                    dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []

                    dt_boxes_final = (
                        update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                        if dt_boxes_merged and adjusted_mfdetrec_res
                        else dt_boxes_merged
                    )

                    if dt_boxes_final:
                        ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]
                        ocr_result_list = get_ocr_result_list(
                            ocr_res, useful_list, ocr_enable, bgr_image,
                            _lang, res['original_label'], res['original_order']
                        )
                        ocr_res_dict['layout_res'].extend(ocr_result_list)


# =================================== OCR-rec ===================================
def _run_ocr_rec_postprocess(images_layout_res: List[List[Dict]], ocr_config):
    """OCR rec 后处理"""
    atom_model_manager = AtomModelSingleton()

    need_ocr_by_lang = {}
    img_crop_by_lang = {}

    for layout_res in images_layout_res:
        for item in layout_res:
            if item['category_id'] == CategoryId.OcrText:
                if 'np_img' in item and 'lang' in item:
                    lang = item['lang']

                    if lang not in need_ocr_by_lang:
                        need_ocr_by_lang[lang] = []
                        img_crop_by_lang[lang] = []

                    need_ocr_by_lang[lang].append(item)
                    img_crop_by_lang[lang].append(item['np_img'])

                    item.pop('np_img')
                    item.pop('lang')

    if not img_crop_by_lang:
        return

    for lang, img_crop_list in img_crop_by_lang.items():
        if not img_crop_list:
            continue

        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            det_db_box_thresh=0.3,
            lang=lang,
            ocr_config=ocr_config,
        )

        ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]

        assert len(ocr_res_list) == len(need_ocr_by_lang[lang])

        for item, (ocr_text, ocr_score) in zip(need_ocr_by_lang[lang], ocr_res_list):
            item['text'] = ocr_text
            item['score'] = float(f"{ocr_score:.3f}")

            if ocr_score < OcrConfidence.min_confidence:
                item['category_id'] = CategoryId.LowScoreText
            else:
                # 特殊字符过滤
                bbox = [item['poly'][0], item['poly'][1], item['poly'][4], item['poly'][5]]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                special_texts = ['（204号', '（20', '（2', '（2号', '（20号', '号', '（204']
                if ocr_text in special_texts and ocr_score < 0.8 and width < height:
                    item['category_id'] = CategoryId.LowScoreText

# =================================== OCR-rec ===================================
def _process_single_table(
        table_res_dict: Dict,
        page_dict: Dict,
        scale: float,
        atom_model_manager: AtomModelSingleton,
        table_config,
        ocr_config,
):
    """处理单个表格"""
    # 表格配置
    table_force_ocr = table_config.get("force_ocr", False)
    skip_text_in_image = table_config.get("skip_text_in_image", True)
    use_img2table = table_config.get("use_img2table", False)
    table_use_word_box = table_config.get("use_word_box", True)
    table_formula_enable = table_config.get("table_formula_enable", True)
    table_image_enable = table_config.get("table_image_enable", True)
    table_extract_original_image = table_config.get("extract_original_image", False)


    _lang = table_res_dict['lang']
    useful_list = table_res_dict['useful_list']

    adjusted_mfdetrec_res = None
    if table_formula_enable:
        adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
            table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res'],
            useful_list, return_text=True
        )

    ocr_config_clean = None
    if ocr_config is not None:
        ocr_config_clean = copy.deepcopy(ocr_config)
        ocr_config_clean.pop("custom_model", None)
    ocr_model = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang=_lang,
        ocr_config=ocr_config_clean,
        enable_merge_det_boxes=False,
    )

    # 获取表格文本框
    bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
    det_res = ocr_model.ocr(bgr_image, mfd_res=adjusted_mfdetrec_res, rec=False)[0]

    angles = []
    rotate_label = "0"
    pdf_not_rotate = page_dict.get("rotate_label") not in ["90", "180", "270"]
    if pdf_not_rotate:
        # 检测文字旋转
        rotate_label, angles = txt_most_angle_extract_table(page_dict, table_res_dict, scale=scale)
    if not angles:
        # 如果没有文本的角度，使用模型判断是否旋转
        img_orientation_cls_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.ImgOrientationCls,
        )
        rotate_label = img_orientation_cls_model.predict(bgr_image, det_res)
    if rotate_label in ["90", "270"]:
        rotate_table_image(table_res_dict, rotate_label)
        # 旋转后的表格需要重新获取文本框
        bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
        det_res = ocr_model.ocr(bgr_image, mfd_res=adjusted_mfdetrec_res, rec=False)[0]

    ocr_result = []

    # 尝试从 PDF 提取文本
    if (not table_force_ocr and not table_res_dict['ocr_enable'] and rotate_label == "0" and pdf_not_rotate):
        ocr_result = _extract_table_text_from_pdf(
            table_res_dict, page_dict, scale, det_res, useful_list, table_use_word_box
        )

    # 如果提取失败，使用 OCR
    if not ocr_result and det_res:
        ocr_result = _run_table_ocr(ocr_model, bgr_image, det_res, table_use_word_box)

    # 表格识别
    table_model = atom_model_manager.get_atom_model(
        atom_model_name='table',
        lang=_lang,
        ocr_config=ocr_config,
        table_config=table_config,
    )

    fill_image_res = []
    if table_image_enable:
        if not pdf_not_rotate:
            table_extract_original_image = False
        fill_image_res = extract_table_fill_image(
            page_dict, table_res_dict, scale, table_extract_original_image
        )

    table_res_dict['table_res'].pop('layout_image_list', None)

    html_code = table_model.predict(
        table_res_dict['table_img'], ocr_result,
        fill_image_res, adjusted_mfdetrec_res,
        skip_text_in_image, use_img2table
    )

    if html_code and '<table>' in html_code and '</table>' in html_code:
        start_index = html_code.find('<table>')
        end_index = html_code.rfind('</table>') + len('</table>')
        table_res_dict['table_res']['html'] = html_code[start_index:end_index]

        # 保存公式和图片位置信息
        formula_boxes = [
            t["bbox"] for t in table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res']
            if "bbox" in t
        ]
        if formula_boxes:
            table_res_dict['table_res']['formula_boxes'] = [
                [int(coord / scale) for coord in bbox] for bbox in formula_boxes
            ]

        img_boxes = [t["ori_bbox"] for t in fill_image_res if "bbox" in t]
        if img_boxes:
            table_res_dict['table_res']['img_boxes'] = [
                [int(coord / scale) for coord in bbox] for bbox in img_boxes
            ]
    else:
        logger.warning('table recognition processing fails')

def _extract_table_text_from_pdf(
        table_res_dict: Dict,
        page_dict: Dict,
        scale: float,
        det_res: List,
        useful_list: List,
        table_use_word_box
) -> List:
    """从 PDF 中提取表格文本"""
    if not det_res:
        return []

    try:
        ocr_spans = get_ocr_result_list_table(det_res, useful_list, scale)
        poly = table_res_dict['table_res']['poly']
        table_bboxes = [[
            int(poly[0] / scale), int(poly[1] / scale),
            int(poly[4] / scale), int(poly[5] / scale),
            None, None, None, 'text', None, None, None, None, 1
        ]]

        txt_spans_extract(
            page_dict, ocr_spans, table_res_dict['table_img'], scale,
            table_bboxes, [], return_word_box=table_use_word_box,
            useful_list=table_res_dict['useful_list']
        )

        if table_use_word_box:
            filtered = [
                (w[2], w[0], w[1])
                for item in ocr_spans
                for group in [item.get('word_result')]
                if group
                for w in group
                if w and w[2] != ""
            ]
        else:
            filtered = [
                [item['ori_bbox'], item['content'], item['score']]
                for item in ocr_spans if item.get('content')
            ]

        return [list(x) for x in zip(*filtered)] if filtered else []
    except Exception:
        logger.warning('table ocr_result get from pdf error')
        return []

def _run_table_ocr(
        ocr_model,
        bgr_image: np.ndarray,
        det_res: List,
        table_use_word_box,
) -> List:
    """执行表格 OCR"""
    rec_img_list = []
    for dt_box in det_res:
        rec_img_list.append({
            "cropped_img": get_rotate_crop_image(bgr_image, np.asarray(dt_box, dtype=np.float32)),
            "dt_box": np.asarray(dt_box, dtype=np.float32),
        })

    cropped_img_list = [item["cropped_img"] for item in rec_img_list]
    ocr_res_list = ocr_model.ocr(
        cropped_img_list, det=False, tqdm_enable=False,
        return_word_box=table_use_word_box, ori_img=bgr_image, dt_boxes=det_res
    )[0]

    ocr_result = []
    for img_dict, ocr_res in zip(rec_img_list, ocr_res_list):
        if table_use_word_box:
            ocr_result.extend([
                [word_result[2], word_result[0], word_result[1]]
                for word_result in ocr_res[2]
            ])
        else:
            ocr_result.append([img_dict["dt_box"], ocr_res[0], ocr_res[1]])

    return [list(x) for x in zip(*ocr_result)] if ocr_result else []
