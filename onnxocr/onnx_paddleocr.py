import argparse
import time

from .layout_recognition import LayoutRecognizer
from .license_plate import LicensePlateRecognizer
from .predict_system import TextSystem
from .table_recognition import TableRecognizer
from .utils import draw_ocr
from .utils import infer_args as init_args
from .visualization import (
    save_layout_visualization,
    save_plate_visualization,
    save_table_visualization,
)


class ONNXPaddleOcr(TextSystem):
    def __init__(self, **kwargs):
        self.use_plate_recognition = kwargs.pop("use_plate_recognition", False)
        self.use_table_recognition = kwargs.pop("use_table_recognition", False)
        self.use_layout_analysis = kwargs.pop("use_layout_analysis", False)
        self.plate_min_score = kwargs.pop("plate_min_score", 0.4)
        self.plate_iou_thresh = kwargs.pop("plate_iou_thresh", 0.5)
        plate_detect_model_path = kwargs.pop("plate_detect_model_path", None)
        plate_rec_model_path = kwargs.pop("plate_rec_model_path", None)
        plate_providers = kwargs.pop("plate_providers", None)
        table_model_type = kwargs.pop("table_model_type", "slanet_plus")
        table_model_path = kwargs.pop("table_model_path", None)
        table_engine_cfg = kwargs.pop("table_engine_cfg", None)
        layout_model_type = kwargs.pop("layout_model_type", "pp_layout_cdla")
        layout_model_path = kwargs.pop("layout_model_path", None)
        layout_engine_cfg = kwargs.pop("layout_engine_cfg", None)
        layout_conf_thresh = kwargs.pop("layout_conf_thresh", 0.5)
        layout_iou_thresh = kwargs.pop("layout_iou_thresh", 0.5)

        active_modes = [
            self.use_plate_recognition,
            self.use_table_recognition,
            self.use_layout_analysis,
        ]
        if sum(bool(mode) for mode in active_modes) > 1:
            raise ValueError(
                "use_plate_recognition, use_table_recognition and use_layout_analysis cannot be enabled together."
            )

        if self.use_plate_recognition:
            self.plate_recognizer = LicensePlateRecognizer(
                detect_model_path=plate_detect_model_path,
                rec_model_path=plate_rec_model_path,
                providers=plate_providers,
            )
            return

        if self.use_layout_analysis:
            self.layout_recognizer = LayoutRecognizer(
                model_type=layout_model_type,
                model_path=layout_model_path,
                engine_cfg=layout_engine_cfg,
                conf_thresh=layout_conf_thresh,
                iou_thresh=layout_iou_thresh,
            )
            return

        parser = init_args()
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        params = argparse.Namespace(**inference_args_dict)

        params.rec_image_shape = "3, 48, 320"
        params.__dict__.update(**kwargs)

        super().__init__(params)
        self.table_recognizer = None
        if self.use_table_recognition:
            self.table_recognizer = TableRecognizer(
                model_type=table_model_type,
                model_path=table_model_path,
                engine_cfg=table_engine_cfg,
            )

    def ocr(self, img, det=True, rec=True, cls=True, plate_min_score=None, plate_iou_thresh=None):
        if self.use_plate_recognition:
            return self.plate_recognizer.recognize(
                img,
                min_score=self.plate_min_score if plate_min_score is None else plate_min_score,
                iou_thresh=self.plate_iou_thresh if plate_iou_thresh is None else plate_iou_thresh,
            )

        if self.use_layout_analysis:
            return self.layout_recognizer.recognize(img)

        if self.use_table_recognition:
            ocr_result = self._general_ocr(img, det=True, rec=True, cls=cls)
            return self.table_recognizer.recognize(img, ocr_result[0])

        return self._general_ocr(img, det=det, rec=rec, cls=cls)

    def _general_ocr(self, img, det=True, rec=True, cls=True):
        if cls is True and self.use_angle_cls is False:
            print(
                "Since the angle classifier is not initialized, the angle classifier will not be used during the forward process"
            )

        if det and rec:
            ocr_res = []
            dt_boxes, rec_res = self.__call__(img, cls)
            tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
            ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            dt_boxes = self.text_detector(img)
            tmp_res = [box.tolist() for box in dt_boxes]
            ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []

            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls and cls:
                img, cls_res_tmp = self.text_classifier(img)
                if not rec:
                    cls_res.append(cls_res_tmp)
            rec_res = self.text_recognizer(img)
            ocr_res.append(rec_res)

            if not rec:
                return cls_res
            return ocr_res


def sav2Img(org_img, result, name="draw_ocr.jpg"):
    from PIL import Image

    result = result[0]
    image = org_img[:, :, ::-1]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores)
    im_show = Image.fromarray(im_show)
    im_show.save(name)


def sav2PlateImg(org_img, result, name="draw_plate.jpg"):
    save_plate_visualization(org_img, result, name)


def sav2TableImg(org_img, result, name="draw_table.jpg", show_logic=False):
    save_table_visualization(org_img, result, name, show_logic=show_logic)


def sav2LayoutImg(org_img, result, name="draw_layout.jpg"):
    save_layout_visualization(org_img, result, name)


if __name__ == "__main__":
    import cv2

    model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)

    img = cv2.imread(
        "/data2/liujingsong3/fiber_box/test/img/20230531230052008263304.jpg"
    )
    s = time.time()
    result = model.ocr(img)
    e = time.time()
    print("total time: {:.3f}".format(e - s))
    print("result:", result)
    for box in result[0]:
        print(box)

    sav2Img(img, result)
