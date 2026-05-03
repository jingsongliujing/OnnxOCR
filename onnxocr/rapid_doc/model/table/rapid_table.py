import cv2
import numpy as np
from loguru import logger

from onnxocr.rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import inline_left_delimiter, inline_right_delimiter
from onnxocr.rapid_doc.model.table.rapid_table_self.table_cls import TableCls
from onnxocr.rapid_doc.model.table.rapid_table_self import ModelType, RapidTable, RapidTableInput, EngineType
from onnxocr.rapid_doc.model.table.utils import select_best_table_model
from onnxocr.rapid_doc.utils.boxbase import is_in
from onnxocr.rapid_doc.utils.config_reader import get_device
from onnxocr.rapid_doc.utils.ocr_utils import points_to_bbox, bbox_to_points


class RapidTableModel(object):
    def __init__(self, ocr_engine, table_config=None):
        if table_config is None:
            table_config = {}
        device = get_device()
        engine_cfg = None
        if device.startswith('cuda'):
            device_id = int(device.split(':')[1]) if ':' in device else 0  # GPU 编号
            engine_cfg = {'use_cuda': True, "cuda_ep_cfg.device_id": device_id}
        elif device.startswith('npu'):
            device_id = int(device.split(':')[1]) if ':' in device else 0  # npu 编号
            engine_cfg = {'use_cann': True, "cann_ep_cfg.device_id": device_id}
        engine_cfg = engine_cfg or {}
        # 如果传入了 engine_cfg，则覆盖参数
        if table_config.get('engine_cfg'):
            engine_cfg = table_config.get('engine_cfg')
        self.use_compare_table = table_config.get('use_compare_table') if table_config else False
        self.model_type = table_config.get("model_type", ModelType.UNET_SLANET_PLUS)
        self.ocr_engine = ocr_engine

        self.engine_type = table_config.get('engine_type') if table_config else None

        if self.model_type == ModelType.UNET_SLANET_PLUS:
            cls_input_args = RapidTableInput(model_type=table_config.get("cls.model_type", ModelType.Q_CLS), engine_type=self.engine_type,
                                            model_dir_or_path=table_config.get("cls.model_dir_or_path"),
                                            engine_cfg=engine_cfg, use_ocr=False)
            self.table_cls = TableCls(cls_input_args)
            wired_input_args = RapidTableInput(model_type=ModelType.UNET, engine_type=self.engine_type,
                                               model_dir_or_path=table_config.get("unet.model_dir_or_path"),
                                               engine_cfg=engine_cfg, use_ocr=False)
            self.wired_table_model = RapidTable(wired_input_args)
            wireless_input_args = RapidTableInput(model_type=ModelType.SLANETPLUS, engine_type=self.engine_type,
                                                  model_dir_or_path=table_config.get("slanet_plus.model_dir_or_path"),
                                                  engine_cfg=engine_cfg, use_ocr=False)
            self.wireless_table_model = RapidTable(wireless_input_args)
        elif self.model_type == ModelType.UNET_UNITABLE:
            raise ValueError("UNET_UNITABLE requires a Torch backend and is not shipped in OnnxOCR.")
        else:
            if self.model_type == ModelType.UNITABLE:
                raise ValueError("UNITABLE requires a Torch backend and is not shipped in OnnxOCR.")
            input_args = RapidTableInput(model_type=self.model_type, engine_type=self.engine_type,
                                         model_dir_or_path=table_config.get("model_dir_or_path"),
                                         engine_cfg=engine_cfg, use_ocr=False)
            self.table_model = RapidTable(input_args)

    def batch_predict(self, images: list, ocr_result=None, fill_image_res=None, mfd_res=None, skip_text_in_image=True, use_img2table=False) -> list[str]:
        results = []
        for image in images:
            res = self.predict(
                image=image,
                ocr_result=ocr_result,
                fill_image_res=fill_image_res,
                mfd_res=mfd_res,
                skip_text_in_image=skip_text_in_image,
                use_img2table=use_img2table
            )
            results.append(res)
        return results

    def predict(self, image, ocr_result=None, fill_image_res=None, mfd_res=None, skip_text_in_image=True, use_img2table=False):
        bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        # First check the overall image aspect ratio (height/width)
        img_height, img_width = bgr_image.shape[:2]
        img_aspect_ratio = img_height / img_width if img_width > 0 else 1.0
        img_is_portrait = img_aspect_ratio > 1.2

        if img_is_portrait:

            det_res = self.ocr_engine.ocr(bgr_image, rec=False)[0]
            # Check if table is rotated by analyzing text box aspect ratios
            is_rotated = False
            if det_res:
                vertical_count = 0

                for box_ocr_res in det_res:
                    p1, p2, p3, p4 = box_ocr_res

                    # Calculate width and height
                    width = p3[0] - p1[0]
                    height = p3[1] - p1[1]

                    aspect_ratio = width / height if height > 0 else 1.0

                    # Count vertical vs horizontal text boxes
                    if aspect_ratio < 0.8:  # Taller than wide - vertical text
                        vertical_count += 1
                    # elif aspect_ratio > 1.2:  # Wider than tall - horizontal text
                    #     horizontal_count += 1

                # If we have more vertical text boxes than horizontal ones,
                # and vertical ones are significant, table might be rotated
                if vertical_count >= len(det_res) * 0.3:
                    is_rotated = True

                # logger.debug(f"Text orientation analysis: vertical={vertical_count}, det_res={len(det_res)}, rotated={is_rotated}")

            # Rotate image if necessary
            if is_rotated:
                # logger.debug("Table appears to be in portrait orientation, rotating 90 degrees clockwise")
                image = cv2.rotate(np.asarray(image), cv2.ROTATE_90_CLOCKWISE)
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Continue with OCR on potentially rotated image
        if not ocr_result:
            ocr_result = self.ocr_engine.ocr(bgr_image, mfd_res=mfd_res)[0]
            if ocr_result:
                ocr_result = [list(x) for x in zip(*[[item[0], item[1][0], item[1][1]] for item in ocr_result])]
            else:
                ocr_result = None

        if not ocr_result:
            return None
        # 把图片结果，添加到ocr_result里。uuid作为占位符，后面保存图片时替换
        if fill_image_res:
            for fill_image in fill_image_res:
                bbox = points_to_bbox(fill_image['ocr_bbox'])
                cv2.rectangle(bgr_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), thickness=-1) # 填白图像区域，防止表格识别被影响
                ocr_result[0].append(fill_image['ocr_bbox'])
                ocr_result[1].append(fill_image['uuid'])
                ocr_result[2].append(1)
                if skip_text_in_image:
                    # 找出所有 OCR 框在图片框内的下标
                    delete_indices = []
                    for idx, ocr in enumerate(ocr_result[0][:-1]):  # 排除刚添加的图片框自身
                        if is_in(points_to_bbox(ocr), points_to_bbox(fill_image['ocr_bbox'])):
                            delete_indices.append(idx)
                    # 按逆序删除，防止下标错位
                    for idx in sorted(delete_indices, reverse=True):
                        del ocr_result[0][idx]
                        del ocr_result[1][idx]
                        del ocr_result[2][idx]
        # 表格内的公式填充
        if mfd_res:
            for mfd in mfd_res:
                if mfd.get('latex'):
                    ocr_result[1].append(f"{inline_left_delimiter}{mfd['latex']}{inline_right_delimiter}")
                elif mfd.get('checkbox'):
                    ocr_result[1].append(mfd['checkbox'])
                else:
                    continue
                ocr_result[0].append(bbox_to_points(mfd['bbox']))
                ocr_result[2].append(1)

        """开始识别表格"""
        cls = None
        """使用 img2table 识别"""
        if use_img2table:
            try:
                from onnxocr.rapid_doc.model.table.img2table_self.image import Image
                from onnxocr.rapid_doc.model.table.img2table_self.RapidOcrTable import RapidOcrTable

                cls, elasp = self.table_cls(image)
                cls = cls[0]
                if cls == "wired":
                    borderless_tables = False
                else:
                    borderless_tables = True
                opencv_ocr = RapidOcrTable(ocr_result)
                doc = Image(src=bgr_image)
                extracted_tables = doc.extract_tables(
                    ocr=opencv_ocr,
                    implicit_rows=False,
                    implicit_columns=False,
                    borderless_tables=borderless_tables,
                    min_confidence=50
                )
                if extracted_tables:
                    # print(f"img2table detected {len(extracted_tables)} tables")
                    html_code = "<html><body>" + extracted_tables[0].html + "</body></html>"
                    return html_code
            except ImportError:
                raise ValueError(
                    "Could not import img2table python package. "
                    "Please install it with `pip install img2table`."
                )
            except Exception as e:
                logger.exception(e)

        """使用 rapid_table_self 识别"""
        try:
            ocr_result = [ocr_result]
            bgr_image = [bgr_image]

            if self.model_type == ModelType.UNET_SLANET_PLUS or self.model_type == ModelType.UNET_UNITABLE:
                if not cls:
                    cls, elasp = self.table_cls(bgr_image)
                    cls = cls[0]
                if cls == "wired":
                    wired_pred = self.wired_table_model(bgr_image, ocr_result).pred_htmls
                    wired_html_code = wired_pred[0] if len(wired_pred) > 0 else None
                    if self.use_compare_table:
                        wireless_pred = self.wireless_table_model(bgr_image, ocr_result).pred_htmls
                        wireless_html_code = wireless_pred[0] if len(wireless_pred) > 0 else None
                        html_code = select_best_table_model(ocr_result[0], wired_html_code, wireless_html_code)
                    else:
                        html_code = wired_html_code
                else:  # wireless
                    html = self.wireless_table_model(bgr_image, ocr_result).pred_htmls
                    html_code = html[0] if len(html) > 0 else None
            else:
                html_code = self.table_model(bgr_image, ocr_result).pred_htmls[0]
            return html_code
        except Exception as e:
            logger.exception(e)
            return None
