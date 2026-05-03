from onnxocr.rapid_doc.model.formula.rapid_formula_self import (
    EngineType,
    ModelType,
    RapidFormula,
    RapidFormulaInput,
)


class RapidFormulaModel:
    def __init__(self, formula_config=None):
        cfg = RapidFormulaInput(
            model_type=ModelType.PP_FORMULANET_PLUS_M,
            engine_type=EngineType.ONNXRUNTIME,
        )
        if formula_config is not None:
            for key, value in formula_config.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        self.latex_engine = RapidFormula(cfg=cfg)

    def predict(self, image):
        return self.batch_predict(images=[image], batch_size=1)[0]

    def batch_predict(self, images: list, batch_size) -> list[str]:
        all_results = self.latex_engine(
            img_contents=images,
            batch_size=batch_size,
            tqdm_enable=True,
        )
        return [result.rec_formula for result in all_results]
