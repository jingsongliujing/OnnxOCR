import logging
import traceback
import numpy as np
from typing import List, Dict, Any
from .main import TSRUnetStructurer
from .table_recover import TableRecover
from .utils.utils_table_recover import (
    match_ocr_cell,
    plot_html_table,
    box_4_2_poly_to_box_4_1,
    sorted_ocr_boxes,
    gather_ocr_list_by_row,
)

class UnetTableRecognition:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.table_structure = TSRUnetStructurer(self.cfg)
        self.table_recover = TableRecover()

    def __call__(
        self, ori_imgs: List[np.ndarray], ocr_results,
        **kwargs,
    ):
        need_ocr = True
        col_threshold = 15
        row_threshold = 10
        if kwargs:
            need_ocr = kwargs.get("need_ocr", True)
            col_threshold = kwargs.get("col_threshold", 15)
            row_threshold = kwargs.get("row_threshold", 10)

        pred_htmls, cell_bboxes, logic_points_list = [], [], []
        for i, img in enumerate(ori_imgs):

            ocr_result = list(
                zip(ocr_results[i][0], ocr_results[i][1], ocr_results[i][2])
            )
            polygons, rotated_polygons = self.table_structure(img, **kwargs)
            if polygons is None:
                pred_htmls.append("")
                cell_bboxes.append([])
                logic_points_list.append([])
                # logging.warning("polygons is None.")
                continue

            try:
                table_res, logi_points = self.table_recover(
                    rotated_polygons, row_threshold, col_threshold
                )
                # 将坐标由逆时针转为顺时针方向，后续处理与无线表格对齐
                polygons[:, 1, :], polygons[:, 3, :] = (
                    polygons[:, 3, :].copy(),
                    polygons[:, 1, :].copy(),
                )
                if not need_ocr:
                    sorted_polygons, idx_list = sorted_ocr_boxes(
                        [box_4_2_poly_to_box_4_1(box) for box in polygons]
                    )
                    pred_htmls.append("")
                    cell_bboxes.append(sorted_polygons)
                    logic_points_list.append(logi_points[idx_list])
                    continue

                cell_box_det_map, not_match_orc_boxes = match_ocr_cell(ocr_result, polygons)
                # 如果有识别框没有ocr结果，直接进行rec补充
                cell_box_det_map = self.fill_blank_rec(img, polygons, cell_box_det_map)
                # 转换为中间格式，修正识别框坐标,将物理识别框，逻辑识别框，ocr识别框整合为dict，方便后续处理
                t_rec_ocr_list = self.transform_res(cell_box_det_map, polygons, logi_points)
                # 将每个单元格中的ocr识别结果排序和同行合并，输出的html能完整保留文字的换行格式
                t_rec_ocr_list = self.sort_and_gather_ocr_res(t_rec_ocr_list)
                # cell_box_map =
                logi_points = [t_box_ocr["t_logic_box"] for t_box_ocr in t_rec_ocr_list]
                cell_box_det_map = {
                    i: [ocr_box_and_text[1] for ocr_box_and_text in t_box_ocr["t_ocr_res"]]
                    for i, t_box_ocr in enumerate(t_rec_ocr_list)
                }
                pred_html = plot_html_table(logi_points, cell_box_det_map)
                polygons = np.array(polygons).reshape(-1, 8)
                logi_points = np.array(logi_points)

                pred_htmls.append(pred_html)
                cell_bboxes.append(polygons)
                logic_points_list.append(logi_points)

            except Exception:
                logging.warning(traceback.format_exc())
                pred_htmls.append("")
                cell_bboxes.append([])
                logic_points_list.append([])

        return pred_htmls, cell_bboxes, logic_points_list

    def transform_res(
        self,
        cell_box_det_map: Dict[int, List[any]],
        polygons: np.ndarray,
        logi_points: List[np.ndarray],
    ) -> List[Dict[str, any]]:
        res = []
        for i in range(len(polygons)):
            ocr_res_list = cell_box_det_map.get(i)
            if not ocr_res_list:
                continue
            xmin = min([ocr_box[0][0][0] for ocr_box in ocr_res_list])
            ymin = min([ocr_box[0][0][1] for ocr_box in ocr_res_list])
            xmax = max([ocr_box[0][2][0] for ocr_box in ocr_res_list])
            ymax = max([ocr_box[0][2][1] for ocr_box in ocr_res_list])
            dict_res = {
                # xmin,xmax,ymin,ymax
                "t_box": [xmin, ymin, xmax, ymax],
                # row_start,row_end,col_start,col_end
                "t_logic_box": logi_points[i].tolist(),
                # [[xmin,xmax,ymin,ymax], text]
                "t_ocr_res": [
                    [box_4_2_poly_to_box_4_1(ocr_det[0]), ocr_det[1]]
                    for ocr_det in ocr_res_list
                ],
            }
            res.append(dict_res)
        return res

    def sort_and_gather_ocr_res(self, res):
        for i, dict_res in enumerate(res):
            _, sorted_idx = sorted_ocr_boxes(
                [ocr_det[0] for ocr_det in dict_res["t_ocr_res"]], threhold=0.3
            )
            dict_res["t_ocr_res"] = [dict_res["t_ocr_res"][i] for i in sorted_idx]
            dict_res["t_ocr_res"] = gather_ocr_list_by_row(
                dict_res["t_ocr_res"], threhold=0.3
            )
        return res

    def fill_blank_rec(
        self,
        img: np.ndarray,
        sorted_polygons: np.ndarray,
        cell_box_map: Dict[int, List[str]],
    ) -> Dict[int, List[Any]]:
        """找到poly对应为空的框，尝试将直接将poly框直接送到识别中"""
        for i in range(sorted_polygons.shape[0]):
            if cell_box_map.get(i):
                continue
            box = sorted_polygons[i]
            cell_box_map[i] = [[box, "", 1]]
            continue
        return cell_box_map
