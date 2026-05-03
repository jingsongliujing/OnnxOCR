import cv2

from onnxocr.rapid_doc.model.layout.rapid_layout_self import ModelType, RapidLayout, RapidLayoutInput
from onnxocr.rapid_doc.utils.config_reader import get_device
from onnxocr.rapid_doc.utils.enum_class import CategoryId
from onnxocr.rapid_doc.utils.boxbase import calculate_iou

class RapidLayoutModel(object):
    def __init__(self, layout_config=None):
        cfg = RapidLayoutInput(model_type=ModelType.PP_DOCLAYOUTV2)

        device = get_device()
        if device.startswith('cuda'):
            device_id = int(device.split(':')[1]) if ':' in device else 0  # GPU 编号
            engine_cfg = {'use_cuda': True, "cuda_ep_cfg.device_id": device_id}
            cfg.engine_cfg = engine_cfg
        elif device.startswith('npu'):
            device_id = int(device.split(':')[1]) if ':' in device else 0  # npu 编号
            engine_cfg = {'use_cann': True, "cann_ep_cfg.device_id": device_id}
            cfg.engine_cfg = engine_cfg
            cfg.model_type = ModelType.DOCLAYOUT_DOCSTRUCTBENCH
            cfg.conf_thresh = 0.2
        # 如果传入了 layout_config，则用传入配置覆盖默认配置
        if layout_config is not None:
            if layout_config.get("model_type"):
                cfg.model_type = layout_config.get("model_type")
            if layout_config.get("layout_shape_mode"):
                cfg.layout_shape_mode = layout_config.get("layout_shape_mode")
            if not layout_config.get("conf_thresh"):
                if cfg.model_type == ModelType.PP_DOCLAYOUT_S:
                    # S可能存在部分漏检，自动调低阈值
                    cfg.conf_thresh = 0.2
                elif cfg.model_type == ModelType.DOCLAYOUT_DOCSTRUCTBENCH:
                    # 可能存在部分漏检，自动调低阈值
                    cfg.conf_thresh = 0.2
            # 遍历字典，把传入配置设置到 default_cfg 对象中
            for key, value in layout_config.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
                    setattr(cfg, key, value)
        layout_config = layout_config or {}
        self.markdown_ignore_labels = layout_config.get("markdown_ignore_labels",
                                                        ["number", "footnote", "header", "header_image", "footer", "footer_image", "aside_text",])
        self.pp_doclayout_cls_dict, self.pp_doclayout_plus_cls_dict, self.pp_doclayoutv2_cls_dict \
            = get_cls_dicts(self.markdown_ignore_labels)
        self.model = RapidLayout(cfg=cfg)
        self.model_type = cfg.model_type
        self.doclayout_yolo_list = ['title', 'plain text', 'abandon', 'figure', 'figure_caption',
                                    'table', 'table_caption', 'table_footnote', 'isolate_formula', 'formula_caption',
                                    '10', '11', '12', 'inline_formula', 'isolated_formula', 'ocr_text']

    def predict(self, image):
        return self.batch_predict(images=[image], batch_size=1)[0]

    def batch_predict(self, images: list, batch_size: int, dpi=200) -> list:
        images_layout_res = []
        processed_images = []
        scales = []

        # 判断是否需要缩放到144 DPI
        for img in images:
            h, w = img.shape[:2]
            # 以A4纸200DPI为基准，判断是否太大
            if max(h, w) > 2200:
                scale = 144 / dpi
                resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                scale = 1.0
                resized = img
            # import uuid
            # cv2.imwrite(rf"C:\ocr\img\test\output-images\{uuid.uuid4().hex}.png", resized)
            processed_images.append(resized)
            scales.append(scale)

        all_results = self.model(img_contents=processed_images, batch_size=batch_size, tqdm_enable=True)
        for img_idx, results in enumerate(all_results):
            # import uuid
            # results.vis(f"output-PP_DOCLAYOUT/{uuid.uuid4().hex}__{img_idx}.png")
            layout_res = []
            boxes, scores, class_names = results.boxes, results.scores, results.class_names
            orders = results.orders if results.orders is not None else [-1] * len(boxes)
            polygon_pointses = results.polygon_points if results.polygon_points is not None else [None] * len(boxes)
            scale = scales[img_idx]
            restore_scale = 1.0 / scale
            temp_results = []
            for xyxy, polygon_points, conf, cla, order in zip(boxes, polygon_pointses, scores, class_names, orders):
                xmin, ymin, xmax, ymax = [round(float(p), 2) for p in xyxy]
                # xmin, ymin, xmax, ymax = [p for p in xyxy]
                if self.model_type == ModelType.PP_DOCLAYOUT_PLUS_L:
                    category_id = self.pp_doclayout_plus_cls_dict[cla]
                elif self.model_type in [ModelType.PP_DOCLAYOUTV2, ModelType.PP_DOCLAYOUTV3]:
                    category_id = self.pp_doclayoutv2_cls_dict[cla]
                elif self.model_type == ModelType.DOCLAYOUT_DOCSTRUCTBENCH:
                    if cla == 'isolate_formula':
                        category_id = 14
                    else:
                        category_id = self.doclayout_yolo_list.index(cla)
                else:
                    category_id = self.pp_doclayout_cls_dict[cla]
                temp_results.append({
                    "category_id": category_id,
                    "original_label": cla,
                    "original_order": order,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "polygon_points": polygon_points,
                    "score": round(float(conf), 3)
                })

            if self.model_type not in [ModelType.PP_DOCLAYOUTV2, ModelType.PP_DOCLAYOUTV3]:
                # 行内公式判断
                temp_results = self.check_inline_formula(temp_results)

            for item in temp_results:
                xmin, ymin, xmax, ymax = item["bbox"]
                xmin *= restore_scale
                ymin *= restore_scale
                xmax *= restore_scale
                ymax *= restore_scale
                polygon_points = item["polygon_points"]
                if polygon_points is not None:
                    polygon_points = [
                        [float(x * restore_scale), float(y * restore_scale)]
                        for x, y in polygon_points
                    ]
                layout_res.append({
                    "category_id": item["category_id"],
                    "original_label": item["original_label"],
                    "original_order": item["original_order"],
                    "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    "polygon_points": polygon_points,
                    "score": item["score"],
                })
            images_layout_res.append(layout_res)
        return images_layout_res

    def check_inline_formula(self, temp_results):
        """
        判断行内公式（公式框大部分被文字框包含）
        """
        for item in temp_results:
            if item["category_id"] == CategoryId.InterlineEquation_YOLO:  # 公式初始默认为行间公式
                for other in temp_results:
                    if other["category_id"] == CategoryId.Text:  # plain text
                        if is_contained(item["bbox"], other["bbox"]):
                            # 如果公式框大部分被文字框包含，则修改为行内公式
                            item["category_id"] = CategoryId.InlineEquation
                            break
        return temp_results

def is_contained(box1, box2, thresh=0.9):
    """
    如果是行内公式，必然会和周围的OCR文本框 IoU大于一定阈值
    """
    return calculate_iou(box1, box2) >= thresh


def get_cls_dicts(markdown_ignore_labels):
    """
    如果是行内公式，必然会和周围的OCR文本框 IoU大于一定阈值
    """
    # PP-DocLayout-L、PP-DocLayout-M、PP-DocLayout-S 23个常见的类别
    pp_doclayout_cls_dict = {
        "paragraph_title": CategoryId.Title,
        "image": CategoryId.ImageBody,
        "text": CategoryId.Text,
        "number": CategoryId.Text,  # Abandon
        "abstract": CategoryId.Text,
        "content": CategoryId.Text,
        "figure_title": CategoryId.Text,
        "formula": CategoryId.InterlineEquation_YOLO,
        "table": CategoryId.TableBody,
        "table_title": CategoryId.TableCaption,
        "reference": CategoryId.Text,
        "doc_title": CategoryId.Title,
        "footnote": CategoryId.Text,  # Abandon
        "header": CategoryId.Text,  # Abandon
        "algorithm": CategoryId.Text,
        "footer": CategoryId.Text,  # Abandon
        "seal": CategoryId.ImageBody,
        "chart_title": CategoryId.ImageCaption,
        "chart": CategoryId.ImageBody,
        "formula_number": CategoryId.InterlineEquationNumber_Layout,
        "header_image": CategoryId.ImageBody,  # Abandon
        "footer_image": CategoryId.ImageBody,  # Abandon
        "aside_text": CategoryId.Text,  # Abandon
    }
    pp_doclayout_cls_dict = {
        k: (CategoryId.Abandon if k in markdown_ignore_labels else v)
        for k, v in pp_doclayout_cls_dict.items()
    }

    # PP-DocLayout_plus-L 20个常见的类别
    pp_doclayout_plus_cls_dict = {
        "paragraph_title": CategoryId.Title,
        "image": CategoryId.ImageBody,
        "text": CategoryId.Text,
        "number": CategoryId.Text,  # Abandon
        "abstract": CategoryId.Text,
        "content": CategoryId.Text,
        "figure_title": CategoryId.Text,
        "formula": CategoryId.InterlineEquation_YOLO,
        "table": CategoryId.TableBody,
        "reference": CategoryId.Text,
        "doc_title": CategoryId.Title,
        "footnote": CategoryId.Text,  # Abandon
        "header": CategoryId.Text,  # Abandon
        "algorithm": CategoryId.Text,
        "footer": CategoryId.Text,  # Abandon
        "seal": CategoryId.ImageBody,
        "chart": CategoryId.ImageBody,
        "formula_number": CategoryId.InterlineEquationNumber_Layout,
        "aside_text": CategoryId.Text,  # Abandon
        "reference_content": CategoryId.Text,
    }
    pp_doclayout_plus_cls_dict = {
        k: (CategoryId.Abandon if k in markdown_ignore_labels else v)
        for k, v in pp_doclayout_plus_cls_dict.items()
    }

    # PP-DocLayoutV2 25个常见的类别
    pp_doclayoutv2_cls_dict = {
        "abstract": CategoryId.Text,
        "algorithm": CategoryId.Text,
        "aside_text": CategoryId.Text,  # Abandon
        "chart": CategoryId.ImageBody,
        "content": CategoryId.Text,
        "display_formula": CategoryId.InterlineEquation_YOLO,  # 行间公式
        "doc_title": CategoryId.Title,
        "figure_title": CategoryId.Text,
        "footer": CategoryId.Text,  # Abandon
        "footer_image": CategoryId.ImageBody,  # Abandon
        "footnote": CategoryId.Text,  # Abandon
        "formula_number": CategoryId.InterlineEquationNumber_Layout,
        "header": CategoryId.Text,
        "header_image": CategoryId.ImageBody,  # Abandon
        "image": CategoryId.ImageBody,
        "inline_formula": CategoryId.InlineEquation,  # 行内公式
        "number": CategoryId.Text,  # Abandon
        "paragraph_title": CategoryId.Title,
        "reference": CategoryId.Text,
        "reference_content": CategoryId.Text,
        "seal": CategoryId.ImageBody,
        "table": CategoryId.TableBody,
        "text": CategoryId.Text,
        "vertical_text": CategoryId.Text,
        "vision_footnote": CategoryId.Text,  # Abandon
    }
    pp_doclayoutv2_cls_dict = {
        k: (CategoryId.Abandon if k in markdown_ignore_labels else v)
        for k, v in pp_doclayoutv2_cls_dict.items()
    }

    return pp_doclayout_cls_dict, pp_doclayout_plus_cls_dict, pp_doclayoutv2_cls_dict

if __name__ == '__main__':

    # pytorch_paddle_ocr = RapidLayoutModel()
    # lay = RapidLayoutModel()
    # img = cv2.imread("C:\ocr\img\page_6.png")
    # # img = cv2.imread("D:\\file\\text-pdf\\img\\defout1.png")
    # aa = lay.batch_predict([img], 1)
    # print(aa)

    # r"C:\ocr\models\ppmodel\layout\PP-DocLayout-M\openvino\pp_doclayout_m.xml"

    cfg = RapidLayoutInput(model_type=ModelType.PP_DOCLAYOUTV2)
    # engine_cfg = {'use_cuda': True, "cuda_ep_cfg.device_id": 0, "cuda_ep_cfg.gpu_mem_limit": 2 * 1024 * 1024 * 1024,}
    # cfg.engine_cfg = engine_cfg
    model = RapidLayout(cfg=cfg)

    all_results = model(img_contents=[r"D:\file\text-pdf\images\vl1.57.png"])

    print(all_results)
    all_results[0].vis(r"layout_vis.png")
    # all_results[0].img = cv2.imread(r"D:\file\text-pdf\images\vl1.57.png")
    # img_list = all_results[0].crop()
    # for i, img in enumerate(img_list):
    #     cv2.imwrite(f"C:\ocr\img\layout_crop\img{i}.png", img)