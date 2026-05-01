# -*- encoding: utf-8 -*-
import re
from typing import Any, Dict, List

import numpy as np
import torch

from ...inference_engine.base import get_engine
from ...utils import EngineType
from ..utils import wrap_with_html_struct
from .consts import (
    BBOX_TOKENS,
    EOS_TOKEN,
    MAX_SEQ_LEN,
    TASK_TOKENS,
    VALID_HTML_BBOX_TOKENS,
)
from .post_process import rescale_bboxes
from .pre_process import TablePreprocess


class UniTableStructure:
    def __init__(self, cfg: Dict[str, Any]):
        if cfg["engine_type"] is None:
            cfg["engine_type"] = EngineType.TORCH
        self.model = get_engine(cfg["engine_type"])(cfg)

        self.encoder = self.model.encoder
        self.device = self.model.device

        self.vocab = self.model.vocab

        self.token_white_list = [
            self.vocab.token_to_id(i) for i in VALID_HTML_BBOX_TOKENS
        ]

        self.bbox_token_ids = set(self.vocab.token_to_id(i) for i in BBOX_TOKENS)
        self.bbox_close_html_token = self.vocab.token_to_id("]</td>")

        self.prefix_token_id = self.vocab.token_to_id("[html+bbox]")

        self.eos_id = self.vocab.token_to_id(EOS_TOKEN)

        self.context = (
            torch.tensor([self.prefix_token_id], dtype=torch.int32)
            .repeat(1, 1)
            .to(self.device)
        )
        self.eos_id_tensor = torch.tensor(self.eos_id, dtype=torch.int32).to(
            self.device
        )

        self.max_seq_len = MAX_SEQ_LEN

        self.decoder = self.model.decoder

        self.preprocess_op = TablePreprocess(self.device)

    def __call__(self, imgs: List[np.ndarray]):
        img_batch, ori_shapes = self.preprocess_op(imgs)
        memory_batch = self.encoder(img_batch)

        struct_list, total_bboxes = [], []
        for i, memory in enumerate(memory_batch):
            self.decoder.setup_caches(
                max_batch_size=1,
                max_seq_length=self.max_seq_len,
                dtype=torch.float32,
                device=self.device,
            )

            context = self.loop_decode(
                self.context, self.eos_id_tensor, memory[None, ...]
            )
            bboxes, html_tokens = self.decode_tokens(context)

            ori_h, ori_w = ori_shapes[i]
            one_bboxes = rescale_bboxes(ori_h, ori_w, bboxes)
            total_bboxes.append(one_bboxes)

            one_struct = wrap_with_html_struct(html_tokens)
            struct_list.append((one_struct, 1.0))
        return struct_list, total_bboxes

    def loop_decode(self, context, eos_id_tensor, memory):
        box_token_count = 0
        for _ in range(self.max_seq_len):
            eos_flag = (context == eos_id_tensor).any(dim=1)
            if torch.all(eos_flag):
                break

            next_tokens = self.decoder(memory, context)
            if next_tokens[0] in self.bbox_token_ids:
                box_token_count += 1
                if box_token_count > 4:
                    next_tokens = torch.tensor(
                        [self.bbox_close_html_token], dtype=torch.int32
                    )
                    box_token_count = 0
            context = torch.cat([context, next_tokens], dim=1)
        return context

    def decode_tokens(self, context):
        pred_html = context[0]
        pred_html = pred_html.detach().cpu().numpy()
        pred_html = self.vocab.decode(pred_html, skip_special_tokens=False)
        seq = pred_html.split("<eos>")[0]
        token_black_list = ["<eos>", "<pad>", *TASK_TOKENS]
        for i in token_black_list:
            seq = seq.replace(i, "")

        tr_pattern = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
        td_pattern = re.compile(r"<td(.*?)>(.*?)</td>", re.DOTALL)
        bbox_pattern = re.compile(r"\[ bbox-(\d+) bbox-(\d+) bbox-(\d+) bbox-(\d+) \]")

        decoded_list, bbox_coords = [], []

        # 查找所有的 <tr> 标签
        for tr_match in tr_pattern.finditer(pred_html):
            tr_content = tr_match.group(1)
            decoded_list.append("<tr>")

            # 查找所有的 <td> 标签
            for td_match in td_pattern.finditer(tr_content):
                td_attrs = td_match.group(1).strip()
                td_content = td_match.group(2).strip()
                if td_attrs:
                    decoded_list.append("<td")
                    # 可能同时存在行列合并，需要都添加
                    attrs_list = td_attrs.split()
                    for attr in attrs_list:
                        decoded_list.append(" " + attr)
                    decoded_list.append(">")
                    decoded_list.append("</td>")
                else:
                    decoded_list.append("<td></td>")

                # 查找 bbox 坐标
                bbox_match = bbox_pattern.search(td_content)
                if bbox_match:
                    xmin, ymin, xmax, ymax = map(int, bbox_match.groups())
                    # 将坐标转换为从左上角开始顺时针到左下角的点的坐标
                    coords = np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
                    bbox_coords.append(coords)
                else:
                    # 填充占位的bbox，保证后续流程统一
                    bbox_coords.append(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
            decoded_list.append("</tr>")

        bbox_coords_array = np.array(bbox_coords).astype(np.float32)
        return bbox_coords_array, decoded_list
