# Copyright (c) RapidAI. All rights reserved.
"""
MagicModel - 页面模型数据处理类

处理版面检测结果，提取各类区块和 spans 信息。
"""
import math
from typing import List, Tuple, Dict

from onnxocr.rapid_doc.utils.boxbase import (
    bbox_relative_pos, calculate_iou, bbox_distance, get_minbox_if_overlap_by_ratio
)
from onnxocr.rapid_doc.utils.enum_class import CategoryId, ContentType
from onnxocr.rapid_doc.utils.magic_model_utils import tie_up_category_by_distance_v3, reduct_overlap


class MagicModel:
    """页面模型数据处理类"""
    
    # 置信度阈值
    LOW_CONFIDENCE_THRESHOLD = 0.05
    HIGH_IOU_THRESHOLD = 0.9
    
    def __init__(self, page_model_info: dict, scale: float):
        """
        初始化 MagicModel
        
        Args:
            page_model_info: 页面模型信息
            scale: 缩放比例
        """
        self.__page_model_info = page_model_info
        self.__scale = scale
        
        # 预处理步骤
        self.__fix_axis()
        self.__fix_by_remove_low_confidence()
        self.__fix_by_remove_high_iou_and_low_confidence()
        self.__fix_footnote()
        self.__fix_by_remove_overlap_image_table_body()
    
    def __fix_axis(self):
        """转换坐标系统：poly/polygon_points 从图像像素 → 页面坐标（与 bbox 一致）"""
        need_remove_list = []
        layout_dets = self.__page_model_info['layout_dets']
        scale = self.__scale

        for layout_det in layout_dets:
            x0, y0, _, _, x1, y1, _, _ = layout_det['poly']
            bbox = [
                math.floor(x0 / scale * 100) / 100,
                math.floor(y0 / scale * 100) / 100,
                math.floor(x1 / scale * 100) / 100,
                math.floor(y1 / scale * 100) / 100,
            ]
            layout_det['bbox'] = bbox

            # polygon_points 与 poly 同源（图像像素），需同一 scale 转到页面坐标
            polygon_points = layout_det.get('polygon_points')
            if polygon_points is not None and len(polygon_points) >= 3:
                layout_det['polygon_points'] = [
                    [round(x / scale, 2), round(y / scale, 2)]
                    for x, y in polygon_points
                ]

            if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                need_remove_list.append(layout_det)

        for item in need_remove_list:
            layout_dets.remove(item)
    
    def __fix_by_remove_low_confidence(self):
        """移除低置信度的区块"""
        need_remove_list = []
        layout_dets = self.__page_model_info['layout_dets']
        
        for layout_det in layout_dets:
            if layout_det['score'] <= self.LOW_CONFIDENCE_THRESHOLD:
                need_remove_list.append(layout_det)
        
        for item in need_remove_list:
            layout_dets.remove(item)
    
    def __fix_by_remove_high_iou_and_low_confidence(self):
        """移除高 IOU 且低置信度的区块"""
        need_remove_list = []
        relevant_categories = [
            CategoryId.Title, CategoryId.Text, CategoryId.ImageBody,
            CategoryId.ImageCaption, CategoryId.TableBody, CategoryId.TableCaption,
            CategoryId.TableFootnote, CategoryId.InterlineEquation_Layout,
            CategoryId.InterlineEquationNumber_Layout,
        ]
        
        layout_dets = [
            x for x in self.__page_model_info['layout_dets']
            if x['category_id'] in relevant_categories
        ]
        
        for i in range(len(layout_dets)):
            for j in range(i + 1, len(layout_dets)):
                det1, det2 = layout_dets[i], layout_dets[j]
                
                if calculate_iou(det1['bbox'], det2['bbox']) > self.HIGH_IOU_THRESHOLD:
                    det_to_remove = det1 if det1['score'] < det2['score'] else det2
                    
                    if det_to_remove not in need_remove_list:
                        need_remove_list.append(det_to_remove)
        
        for item in need_remove_list:
            self.__page_model_info['layout_dets'].remove(item)
    
    def __fix_footnote(self):
        """修正 footnote 类型"""
        footnotes = []
        figures = []
        tables = []
        
        for obj in self.__page_model_info['layout_dets']:
            if obj['category_id'] == CategoryId.TableFootnote:
                footnotes.append(obj)
            elif obj['category_id'] == CategoryId.ImageBody:
                figures.append(obj)
            elif obj['category_id'] == CategoryId.TableBody:
                tables.append(obj)
        
        if not footnotes or not figures:
            return
        
        dis_figure_footnote = {}
        dis_table_footnote = {}
        
        for i, footnote in enumerate(footnotes):
            for figure in figures:
                pos_flag_count = sum(
                    1 if x else 0
                    for x in bbox_relative_pos(footnote['bbox'], figure['bbox'])
                )
                if pos_flag_count > 1:
                    continue
                
                dis_figure_footnote[i] = min(
                    self._bbox_distance(figure['bbox'], footnote['bbox']),
                    dis_figure_footnote.get(i, float('inf')),
                )
            
            for table in tables:
                pos_flag_count = sum(
                    1 if x else 0
                    for x in bbox_relative_pos(footnote['bbox'], table['bbox'])
                )
                if pos_flag_count > 1:
                    continue
                
                dis_table_footnote[i] = min(
                    self._bbox_distance(table['bbox'], footnote['bbox']),
                    dis_table_footnote.get(i, float('inf')),
                )
        
        for i, footnote in enumerate(footnotes):
            if i in dis_figure_footnote:
                if dis_table_footnote.get(i, float('inf')) > dis_figure_footnote[i]:
                    footnote['category_id'] = CategoryId.ImageFootnote
    
    def __fix_by_remove_overlap_image_table_body(self):
        """处理重叠的 image_body 和 table_body"""
        need_remove_list = []
        layout_dets = self.__page_model_info['layout_dets']
        
        image_blocks = [x for x in layout_dets if x['category_id'] == CategoryId.ImageBody]
        table_blocks = [x for x in layout_dets if x['category_id'] == CategoryId.TableBody]
        
        def process_overlapping_blocks(blocks):
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    block1, block2 = blocks[i], blocks[j]
                    overlap_box = get_minbox_if_overlap_by_ratio(
                        block1['bbox'], block2['bbox'], 0.8
                    )
                    
                    if overlap_box is not None:
                        area1 = (block1['bbox'][2] - block1['bbox'][0]) * (block1['bbox'][3] - block1['bbox'][1])
                        area2 = (block2['bbox'][2] - block2['bbox'][0]) * (block2['bbox'][3] - block2['bbox'][1])
                        
                        if area1 <= area2:
                            small_block, large_block = block1, block2
                        else:
                            small_block, large_block = block2, block1
                        
                        if small_block not in need_remove_list:
                            # 扩展大区块边界
                            x1, y1, x2, y2 = large_block['bbox']
                            sx1, sy1, sx2, sy2 = small_block['bbox']
                            large_block['bbox'] = [
                                min(x1, sx1), min(y1, sy1),
                                max(x2, sx2), max(y2, sy2)
                            ]
                            need_remove_list.append(small_block)
        
        process_overlapping_blocks(image_blocks)
        process_overlapping_blocks(table_blocks)
        
        for item in need_remove_list:
            if item in layout_dets:
                layout_dets.remove(item)
    
    def _bbox_distance(self, bbox1, bbox2) -> float:
        """计算两个 bbox 之间的距离"""
        left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)
        flags = [left, right, bottom, top]
        count = sum(1 if v else 0 for v in flags)
        
        if count > 1:
            return float('inf')
        
        if left or right:
            l1 = bbox1[3] - bbox1[1]
            l2 = bbox2[3] - bbox2[1]
        else:
            l1 = bbox1[2] - bbox1[0]
            l2 = bbox2[2] - bbox2[0]
        
        if l2 > l1 and (l2 - l1) / l1 > 0.3:
            return float('inf')
        
        return bbox_distance(bbox1, bbox2)
    
    def __tie_up_category_by_distance_v3(self, subject_category_id, object_category_id):
        """根据距离关联主体和客体类别"""
        
        def get_subjects():
            return reduct_overlap([
                {
                    'bbox': x['bbox'],
                    'score': x['score'],
                    'original_label': x.get('original_label'),
                    'original_order': x.get('original_order'),
                    'polygon_points': x.get('polygon_points'),
                }
                for x in self.__page_model_info['layout_dets']
                if x['category_id'] == subject_category_id
            ])
        
        def get_objects():
            return reduct_overlap([
                {
                    'bbox': x['bbox'],
                    'score': x['score'],
                    'original_label': x.get('original_label'),
                    'original_order': x.get('original_order'),
                    'polygon_points': x.get('polygon_points'),
                }
                for x in self.__page_model_info['layout_dets']
                if x['category_id'] == object_category_id
            ])
        
        return tie_up_category_by_distance_v3(get_subjects, get_objects)
    
    def get_imgs(self) -> List[Dict]:
        """获取图像区块"""
        with_captions = self.__tie_up_category_by_distance_v3(
            CategoryId.ImageBody, CategoryId.ImageCaption
        )
        with_footnotes = self.__tie_up_category_by_distance_v3(
            CategoryId.ImageBody, CategoryId.ImageFootnote
        )
        
        ret = []
        for v in with_captions:
            record = {
                'image_body': v['sub_bbox'],
                'image_caption_list': v['obj_bboxes'],
            }
            filter_idx = v['sub_idx']
            d = next(filter(lambda x: x['sub_idx'] == filter_idx, with_footnotes))
            record['image_footnote_list'] = d['obj_bboxes']
            ret.append(record)
        
        return ret
    
    def get_tables(self) -> List[Dict]:
        """获取表格区块"""
        with_captions = self.__tie_up_category_by_distance_v3(
            CategoryId.TableBody, CategoryId.TableCaption
        )
        with_footnotes = self.__tie_up_category_by_distance_v3(
            CategoryId.TableBody, CategoryId.TableFootnote
        )
        
        ret = []
        for v in with_captions:
            record = {
                'table_body': v['sub_bbox'],
                'table_caption_list': v['obj_bboxes'],
            }
            filter_idx = v['sub_idx']
            d = next(filter(lambda x: x['sub_idx'] == filter_idx, with_footnotes))
            record['table_footnote_list'] = d['obj_bboxes']
            ret.append(record)
        
        return ret
    
    def get_equations(self) -> Tuple[List, List, List]:
        """获取公式区块"""
        inline_equations = self.__get_blocks_by_type(
            CategoryId.InlineEquation, ['latex']
        )
        interline_equations = self.__get_blocks_by_type(
            CategoryId.InterlineEquation_YOLO, ['latex']
        )
        interline_equation_blocks = self.__get_blocks_by_type(
            CategoryId.InterlineEquation_Layout
        )
        
        return inline_equations, interline_equations, interline_equation_blocks
    
    def get_discarded(self) -> List[Dict]:
        """获取废弃区块"""
        return self.__get_blocks_by_type(CategoryId.Abandon)
    
    def get_text_blocks(self) -> List[Dict]:
        """获取文本区块"""
        return self.__get_blocks_by_type(CategoryId.Text)
    
    def get_title_blocks(self) -> List[Dict]:
        """获取标题区块"""
        return self.__get_blocks_by_type(CategoryId.Title)
    
    def get_all_spans(self) -> List[Dict]:
        """
        获取所有 spans
        
        返回图像、表格、公式、复选框等类型的 span
        """
        all_spans = []
        layout_dets = self.__page_model_info['layout_dets']
        
        allow_category_ids = [
            CategoryId.ImageBody,
            CategoryId.TableBody,
            CategoryId.InlineEquation,
            CategoryId.InterlineEquation_YOLO,
            CategoryId.OcrText,
            CategoryId.CheckBox,
        ]
        
        for layout_det in layout_dets:
            category_id = layout_det['category_id']
            
            if category_id not in allow_category_ids:
                continue
            
            # 跳过 VL OCR 结果，由 get_vl_ocr_spans 单独处理
            if layout_det.get('vl_ocr'):
                continue
            
            span = {
                'bbox': layout_det['bbox'],
                'score': layout_det['score'],
                'original_label': layout_det.get('original_label'),
                'original_order': layout_det.get('original_order'),
                'polygon_points': layout_det.get('polygon_points'),
            }
            
            if category_id == CategoryId.ImageBody:
                span['type'] = ContentType.IMAGE
                if 'seal' == layout_det.get("original_label"):
                    span['content'] = layout_det.get('text')
            elif category_id == CategoryId.TableBody:
                latex = layout_det.get('latex')
                html = layout_det.get('html')
                
                if latex:
                    span['latex'] = latex
                elif html:
                    span['html'] = html
                    if layout_det.get('latex_boxes'):
                        span['latex_boxes'] = layout_det.get('latex_boxes')
                    elif layout_det.get('img_boxes'):
                        span['img_boxes'] = layout_det.get('img_boxes')
                
                span['type'] = ContentType.TABLE
            elif category_id == CategoryId.InlineEquation:
                span['content'] = layout_det.get('latex') or ''
                span['type'] = ContentType.INLINE_EQUATION
            elif category_id in [CategoryId.InterlineEquation_Layout, CategoryId.InterlineEquation_YOLO]:
                span['content'] = layout_det.get('latex') or ''
                span['type'] = ContentType.INTERLINE_EQUATION
            elif category_id == CategoryId.CheckBox:
                span['content'] = layout_det.get('checkbox') or ''
                span['type'] = ContentType.CHECKBOX
            elif category_id == CategoryId.OcrText:
                span['content'] = layout_det['text']
                span['type'] = ContentType.TEXT
            
            all_spans.append(span)
        
        return self._remove_duplicate_spans(all_spans)
    
    def get_vl_ocr_spans(self) -> List[Dict]:
        """
        获取 VL OCR 识别的 spans
        
        VL OCR 模式下返回的是区域级别的多行文本
        """
        vl_ocr_spans = []
        layout_dets = self.__page_model_info['layout_dets']
        
        for layout_det in layout_dets:
            if not layout_det.get('vl_ocr'):
                continue
            
            # VL OCR 的结果
            text = layout_det.get('text', '')
            if not text:
                continue
            
            span = {
                'bbox': layout_det['bbox'],
                'score': layout_det.get('score', 0.95),
                'content': text,
                'type': ContentType.TEXT,
                'vl_ocr': True,
                'original_label': layout_det.get('original_label'),
                'original_order': layout_det.get('original_order'),
                'polygon_points': layout_det.get('polygon_points'),
            }
            
            vl_ocr_spans.append(span)
        
        return vl_ocr_spans
    
    def __get_blocks_by_type(self, category_type: int, extra_cols: List[str] = None) -> List[Dict]:
        """根据类型获取区块"""
        if extra_cols is None:
            extra_cols = []
        
        blocks = []
        layout_dets = self.__page_model_info.get('layout_dets', [])
        
        for item in layout_dets:
            if item.get('category_id') == category_type:
                block = {
                    'bbox': item.get('bbox'),
                    'original_label': item.get('original_label'),
                    'original_order': item.get('original_order'),
                    'polygon_points': item.get('polygon_points'),
                    'score': item.get('score'),
                }
                
                for col in extra_cols:
                    block[col] = item.get(col)
                
                blocks.append(block)
        
        return blocks
    
    @staticmethod
    def _remove_duplicate_spans(spans: List[Dict]) -> List[Dict]:
        """移除重复的 spans"""
        seen = []
        unique_spans = []
        
        for span in spans:
            if span not in seen:
                seen.append(span)
                unique_spans.append(span)
        
        return unique_spans
