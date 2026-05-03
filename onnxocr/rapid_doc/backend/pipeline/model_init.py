import copy

from loguru import logger

from .model_list import AtomicModel
from ...model.custom import CustomBaseModel
from ...model.layout.rapid_layout import RapidLayoutModel
from ...model.formula.rapid_formula_model import RapidFormulaModel
from ...model.ocr.rapid_ocr import RapidOcrModel
from ...model.orientation.rapid_orientation_model import RapidOrientationModel
from ...model.table.rapid_table import RapidTableModel
from ...utils.hash_utils import make_hashable

def table_model_init(lang=None, ocr_config=None, table_config=None):
    ocr_config_clean = None
    if ocr_config is not None:
        ocr_config_clean = copy.deepcopy(ocr_config)
        ocr_config_clean.pop("custom_model", None)
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang=lang,
        ocr_config=ocr_config_clean,
        enable_merge_det_boxes=False
    )
    table_model = RapidTableModel(ocr_engine, table_config)
    return table_model


def img_orientation_cls_model_init():
    cls_model = RapidOrientationModel()
    return cls_model

def formula_model_init(formula_config=None):
    model = RapidFormulaModel(formula_config)
    return model


def layout_model_init(layout_config=None):
    model = RapidLayoutModel(layout_config)
    return model

def ocr_model_init(det_db_box_thresh=0.3, lang=None, ocr_config=None, det_db_unclip_ratio=1.8, enable_merge_det_boxes=True, is_seal=False):
    model = RapidOcrModel(
            det_db_box_thresh=det_db_box_thresh,
            lang=lang,
            ocr_config=ocr_config,
            use_dilation=True,
            det_db_unclip_ratio=det_db_unclip_ratio,
            enable_merge_det_boxes=enable_merge_det_boxes,
            is_seal=is_seal)
    return model


class AtomModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_atom_model(self, atom_model_name: str, **kwargs):
        if atom_model_name in [AtomicModel.Layout]:
            key = (atom_model_name, make_hashable(kwargs.get('layout_config', None)))
        elif atom_model_name in [AtomicModel.OCR]:
            key = (
                atom_model_name,
                make_hashable(kwargs.get('ocr_config', None)),
                kwargs.get('det_db_box_thresh', 0.3),
                kwargs.get('lang'),
                kwargs.get('det_db_unclip_ratio', 1.8),
                kwargs.get('enable_merge_det_boxes', True),
                kwargs.get('is_seal', True),
            )
        elif atom_model_name in [AtomicModel.Table]:
            key = (atom_model_name, make_hashable(kwargs.get('table_config', None)))
        elif atom_model_name in [AtomicModel.FORMULA]:
            key = (atom_model_name, make_hashable(kwargs.get('formula_config', None)))
        else:
            key = atom_model_name

        if key not in self._models:
            self._models[key] = atom_model_init(model_name=atom_model_name, **kwargs)
        return self._models[key]

def atom_model_init(model_name: str, **kwargs):
    atom_model = None
    if model_name == AtomicModel.Layout:
        atom_model = layout_model_init(
            kwargs.get('layout_config'),
        )
    elif model_name == AtomicModel.FORMULA:
        atom_model = (kwargs.get('formula_config') or {}).get('custom_model')
        if not isinstance(atom_model, CustomBaseModel):
            atom_model = formula_model_init(
                kwargs.get('formula_config'),
            )
    elif model_name == AtomicModel.OCR:
        atom_model = (kwargs.get('ocr_config') or {}).get('custom_model')
        if not isinstance(atom_model, CustomBaseModel):
            atom_model = ocr_model_init(
                kwargs.get('det_db_box_thresh', 0.3),
                kwargs.get('lang'),
                kwargs.get('ocr_config'),
                kwargs.get('det_db_unclip_ratio', 1.8),
                kwargs.get('enable_merge_det_boxes', True),
                kwargs.get('is_seal', False),
            )
    elif model_name == AtomicModel.Table:
        atom_model = (kwargs.get('table_config') or {}).get('custom_model')
        if not isinstance(atom_model, CustomBaseModel):
            atom_model = table_model_init(
                kwargs.get('lang'),
                kwargs.get('ocr_config'),
                kwargs.get('table_config'),
            )
    elif model_name == AtomicModel.ImgOrientationCls:
        atom_model = img_orientation_cls_model_init()
    else:
        logger.error('model name not allow')
        exit(1)

    if atom_model is None:
        logger.error('model init failed')
        exit(1)
    else:
        return atom_model


class MineruPipelineModel:
    def __init__(self, **kwargs):
        self.layout_config = kwargs.get('layout_config')
        self.ocr_config = kwargs.get('ocr_config')
        self.formula_config = kwargs.get('formula_config')
        self.apply_formula = self.formula_config.get('enable', True)
        self.table_config = kwargs.get('table_config')
        self.apply_table = self.table_config.get('enable', True)
        self.lang = kwargs.get('lang', None)
        self.device = kwargs.get('device', 'cpu')
        logger.info(
            'DocAnalysis init, this may take some times......'
        )
        atom_model_manager = AtomModelSingleton()

        # 初始化layout模型
        self.layout_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.Layout,
            device=self.device,
            layout_config=self.layout_config,
        )

        if self.apply_formula:
            # 初始化公式解析模型
            self.formula_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.FORMULA,
                device=self.device,
                formula_config=self.formula_config,
            )

        # 初始化ocr
        self.ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            det_db_box_thresh=0.3,
            lang=self.lang,
            ocr_config=self.ocr_config,
        )
        # init table model
        if self.apply_table:
            self.table_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Table,
                lang=self.lang,
                ocr_config=self.ocr_config,
                table_config=self.table_config,
            )

        logger.info('DocAnalysis init done!')