# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import List, Tuple

import numpy as np

from ...utils.typings import ModelType
from ..utils import wrap_with_html_struct


class TableLabelDecode:
    def __init__(self, dict_character, cfg, merge_no_span_structure=True):
        if merge_no_span_structure:
            if "<td></td>" not in dict_character:
                dict_character.append("<td></td>")
            if "<td>" in dict_character:
                dict_character.remove("<td>")

        dict_character = self.add_special_char(dict_character)
        self.char_to_index = {}
        for i, char in enumerate(dict_character):
            self.char_to_index[char] = i

        self.character = dict_character
        self.td_token = ["<td>", "<td", "<td></td>"]
        self.cfg = cfg

    def __call__(
        self,
        bbox_preds: np.ndarray,
        structure_probs: np.ndarray,
        shape_list: np.ndarray,
        ori_imgs: np.ndarray,
    ):
        result = self.decode(bbox_preds, structure_probs, shape_list, ori_imgs)
        return result

    def decode(
        self,
        bbox_preds: np.ndarray,
        structure_probs: np.ndarray,
        shape_list: np.ndarray,
        ori_imgs: np.ndarray,
    ) -> Tuple[List[Tuple[List[str], float]], List[np.ndarray]]:
        """convert text-label into text-index."""
        ignored_tokens = self.get_ignored_tokens()
        end_idx = self.char_to_index[self.end_str]

        structure_idx = structure_probs.argmax(axis=2)
        structure_probs = structure_probs.max(axis=2)

        table_structs, cell_bboxes = [], []
        batch_size = len(structure_idx)
        for batch_idx in range(batch_size):
            structure_list, bbox_list, score_list = [], [], []
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                if idx > 0 and char_idx == end_idx:
                    break

                if char_idx in ignored_tokens:
                    continue

                text = self.character[char_idx]
                if text in self.td_token:
                    bbox = bbox_preds[batch_idx, idx]
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)

                structure_list.append(text)
                score_list.append(structure_probs[batch_idx, idx])

            bboxes = self.normalize_bboxes(bbox_list, ori_imgs[batch_idx])
            cell_bboxes.append(bboxes)

            table_structs.append(
                (wrap_with_html_struct(structure_list), float(np.mean(score_list)))
            )
        return table_structs, cell_bboxes

    def _bbox_decode(self, bbox, shape):
        h, w = shape[:2]
        bbox[0::2] *= w
        bbox[1::2] *= h
        return bbox

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            return np.array(self.char_to_index[self.beg_str])

        if beg_or_end == "end":
            return np.array(self.char_to_index[self.end_str])

        raise TypeError(f"unsupport type {beg_or_end} in get_beg_end_flag_idx")

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def normalize_bboxes(self, bbox_list, ori_imgs):
        cell_bboxes = np.array(bbox_list)
        if self.cfg["model_type"] == ModelType.SLANETPLUS:
            cell_bboxes = self.rescale_cell_bboxes(ori_imgs, cell_bboxes)
        cell_bboxes = self.filter_blank_bbox(cell_bboxes)
        return cell_bboxes

    def rescale_cell_bboxes(
        self, img: np.ndarray, cell_bboxes: np.ndarray
    ) -> np.ndarray:
        h, w = img.shape[:2]
        resized = 488
        ratio = min(resized / h, resized / w)
        w_ratio = resized / (w * ratio)
        h_ratio = resized / (h * ratio)
        cell_bboxes[:, 0::2] *= w_ratio
        cell_bboxes[:, 1::2] *= h_ratio
        return cell_bboxes

    @staticmethod
    def filter_blank_bbox(cell_bboxes: np.ndarray) -> np.ndarray:
        # 过滤掉占位的bbox
        mask = ~np.all(cell_bboxes == 0, axis=1)
        return cell_bboxes[mask]
