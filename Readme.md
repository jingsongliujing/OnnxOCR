# OnnxOCR

If this project helps you, please consider giving it a **Star**.

![onnx_logo](onnxocr/test_images/onnxocr_logo.png)

**A high-performance multilingual OCR project based on ONNXRuntime**

![GitHub stars](https://img.shields.io/github/stars/jingsongliujing/OnnxOCR?style=social)
![GitHub forks](https://img.shields.io/github/forks/jingsongliujing/OnnxOCR?style=social)
![GitHub license](https://img.shields.io/github/license/jingsongliujing/OnnxOCR)
![Python Version](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)
![AtomGit Star](https://atomgit.com/OnnxOCR/OnnxOCR/star/badge.svg)

English | [简体中文](./Readme_cn.md)

## Version Updates

- **2026.05.15**
  1. Added an Agent-oriented `onnocr` / `onnxocr` vertical OCR CLI for listing scenarios, inspecting schemas, and running image recognition.
  2. Added verified scenarios for train tickets, exam papers, license plates, table structuring, and image-to-Markdown. Privacy-sensitive ID card and bank card templates are available as candidates.
  3. The CLI supports both short commands such as `onnocr list` and explicit Agent module calls such as `onnocr onnxocr.skill_cli list`.
  4. Added standard `SkillInput` / `SkillOutput` contracts, a registry, template-based field extraction, docs, examples, and lightweight tests.

- **2026.05.01**
  1. Added ONNX license plate detection and recognition.
  2. Added RapidTable-based ONNX table recognition.
  3. Added RapidLayout-based Chinese and English layout analysis.
  4. Added RapidDoc-based document layout analysis and Markdown export.
  5. Added `/plate`, `/table`, `/layout`, `/layout_markdown`, and related HTTP endpoints.

- **2025.05.21**
  1. Added PP-OCRv5 models, supporting Simplified Chinese, Traditional Chinese, Chinese Pinyin, English, and Japanese in one model.
  2. Improved overall recognition accuracy compared with PP-OCRv4.
  3. Recognition accuracy is consistent with PaddleOCR 3.0.

## Core Advantages

1. **Deep learning framework free**: a general OCR project ready for deployment.
2. **Cross-architecture support**: PaddleOCR-converted ONNX models can run on ARM and x86 devices.
3. **Unified inference engine**: all ONNX models create ONNXRuntime sessions through `onnxocr/inference_engine.py`.
4. **Multilingual support**: one model supports 5 text types.
5. **Source-level integration**: `rapid_layout`, `rapid_table`, and `rapid_doc` live under the `onnxocr/` package, with no dependency on `rapidocr==3.4.3` or `rapid-orientation`.
6. **Hardware adaptation friendly**: downstream vendors can adapt GPU/NPU providers by modifying the unified inference engine.
7. **Extensible Agent CLI**: industry-specific OCR templates can be called directly by Codex, Claude Code, and other local Agent tools without changing the core ONNX inference code.

## Environment Setup

```bash
python>=3.8
git clone https://github.com/jingsongliujing/OnnxOCR.git
cd OnnxOCR
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

Notes:

- By default, the repository only includes the PP-OCRv5 general OCR model files required by `tests/test_general_ocr.py`.
- Extra models for license plate recognition, table recognition, layout analysis, orientation classification, and RapidDoc Markdown export are large and should be downloaded on demand. For international users, [HuggingFace](https://huggingface.co/jingsongliu/onnxocr_model/tree/main) is recommended.
- Larger PP-OCRv5 Server ONNX models can also be downloaded separately and used to replace det/rec models under `onnxocr/models/ppocrv5/`.

## Model Download

Extra models are hosted on [HuggingFace: jingsongliu/onnxocr_model](https://huggingface.co/jingsongliu/onnxocr_model/tree/main). International users are recommended to download from HuggingFace:

```bash
python scripts/download_models.py --source huggingface
```

The core HuggingFace API is:

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download("jingsongliu/onnxocr_model")
```

For users in mainland China, ModelScope remains the default and recommended source:

```bash
python scripts/download_models.py
```

ModelScope repository: [supersong/onnxocr_model](https://www.modelscope.cn/models/supersong/onnxocr_model/tree/master/models).

The script copies the repository `models/` directory into local `onnxocr/models/` and checks required optional files such as `onnxocr/models/rapid_doc/layout/pp_doclayoutv2.onnx`.

To check local models only:

```bash
python scripts/download_models.py --check-only
```

## One-Click Run

```bash
python test_ocr.py
```

`test_ocr.py` runs only general OCR by default. The optional examples are commented out; uncomment them after downloading the corresponding models with `python scripts/download_models.py`.

Feature-specific tests:

```bash
python tests/test_general_ocr.py
python tests/test_license_plate_ocr.py
python tests/test_table_ocr.py
python tests/test_layout_analysis.py
python tests/test_layout_markdown.py
```

Generated files are written to `result_img/`, which is ignored by git.

## General OCR

```python
import cv2
from onnxocr.onnx_paddleocr import ONNXPaddleOcr

img = cv2.imread("onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg")
model = ONNXPaddleOcr(use_angle_cls=False, use_gpu=False)
result = model.ocr(img)
print(result)
```

## License Plate Recognition

License plate recognition is integrated into `ONNXPaddleOcr` as an optional mode. Existing general OCR usage is unchanged.

```python
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2PlateImg

plate_model = ONNXPaddleOcr(
    use_angle_cls=True,
    use_gpu=False,
    use_plate_recognition=True,
    plate_min_score=0.4,
)
plate_result = plate_model.ocr(img)
sav2PlateImg(img, plate_result, name="./result_img/test_plate_vis.jpg")
```

Model files:

```text
onnxocr/models/license_plate/car_plate_detect.onnx
onnxocr/models/license_plate/plate_rec.onnx
```

## Table Recognition

Table recognition is integrated from RapidTable. It reuses general OCR detection/recognition results, restores table structure, and outputs HTML, cell boxes, and logical row/column coordinates.

```python
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2TableImg

table_model = ONNXPaddleOcr(
    use_angle_cls=True,
    use_gpu=False,
    use_table_recognition=True,
    table_model_type="slanet_plus",
)
table_result = table_model.ocr(img)
print(table_result["html"])
sav2TableImg(img, table_result, name="./result_img/test_table_vis.jpg")
```

Model files:

```text
onnxocr/models/table/slanet-plus.onnx
onnxocr/models/table/ch_ppstructure_mobile_v2_SLANet.onnx
onnxocr/models/table/en_ppstructure_mobile_v2_SLANet.onnx
```

## Chinese / English Layout Analysis

Layout analysis is integrated from RapidLayout. It locates document elements such as titles, text blocks, tables, figures, headers, and footers.

```python
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2LayoutImg

layout_model = ONNXPaddleOcr(
    use_gpu=False,
    use_layout_analysis=True,
    layout_model_type="pp_layout_cdla",
)
layout_result = layout_model.ocr(img)
sav2LayoutImg(img, layout_result, name="./result_img/test_layout_vis.jpg")
```

For English layout analysis, set `layout_model_type` to `pp_layout_publaynet`.

Model files:

```text
onnxocr/models/layout/layout_cdla.onnx
onnxocr/models/layout/layout_publaynet.onnx
```

## Document To Markdown

Document-to-Markdown export is integrated from RapidDoc. It analyzes titles, paragraphs, tables, figures, and other layout elements, then saves the result as a Markdown file.

```python
from onnxocr.layout_markdown import LayoutMarkdownConverter

converter = LayoutMarkdownConverter(
    layout_model_type="pp_doclayoutv2",
    formula_enable=False,
    table_enable=True,
)
result = converter.convert_file(
    "onnxocr/test_images/layout_cdla.jpg",
    output_md_path="./result_img/test_layout_markdown.md",
)
print(result["markdown_path"])
```

RapidDoc model files:

```text
onnxocr/models/rapid_doc/layout/pp_doclayoutv2.onnx
onnxocr/models/rapid_doc/table/q_cls.onnx
onnxocr/models/rapid_doc/table/unet.onnx
onnxocr/models/rapid_doc/table/slanet-plus.onnx
```

## Inference Engine Adaptation

General OCR, license plate recognition, table recognition, layout analysis, and RapidDoc document parsing all create ONNXRuntime sessions through `onnxocr/inference_engine.py`.

To adapt a downstream GPU/NPU provider, start from:

```python
from onnxocr.inference_engine import create_session
```

Main extension points:

- `create_session(model_path, providers=None, use_gpu=False, gpu_id=0, sess_options=None)`
- `build_providers(use_gpu=False, gpu_id=0, providers=None)`
- `build_providers_from_engine_cfg(engine_cfg)`
- `ProviderConfig(engine_cfg)`

Only `onnxocr/inference_engine.py` imports `onnxruntime` directly. Feature modules do not call ONNXRuntime APIs directly.

## API Service

Start service:

```bash
python app-service.py
```

Main endpoints:

- `/ocr`: general OCR.
- `/plate`: license plate recognition.
- `/table`: table recognition.
- `/layout`: layout analysis.
- `/layout_markdown`: image/PDF to Markdown.

WebUI:

```bash
python webui.py
```

## Docker Image

```bash
docker build -t ocr-service .
docker run -itd --name onnxocr-service -p 5006:5005 ocr-service
```

## Project Layout

```text
onnxocr/
  inference_engine.py        # single ONNXRuntime entry
  onnx_paddleocr.py          # public user API
  skill_cli.py               # vertical OCR skill CLI
  skills/                    # extensible OCR skill runtime
  predict_det.py             # general OCR detection
  predict_rec.py             # general OCR recognition
  orientation.py             # local RapidOrientation ONNX adapter
  license_plate.py           # license plate OCR
  table_recognition.py       # table recognition wrapper
  layout_recognition.py      # layout analysis wrapper
  layout_markdown.py         # RapidDoc Markdown wrapper
  rapid_layout/              # source-level RapidLayout ONNX integration
  rapid_table/               # source-level RapidTable ONNX integration
  rapid_doc/                 # source-level RapidDoc ONNX integration
  models/                    # local ONNX models
tests/                       # feature-specific tests
```

## Vertical OCR CLI

OnnxOCR includes an Agent-oriented vertical OCR CLI for fixed industry templates. It keeps scenarios discoverable, schemas stable, and commands easy for Claude Code, Codex, and other Agent tools to call without custom integration.

The CLI layer is intentionally thin:

- ONNX inference, detection, recognition, table OCR, and license plate OCR stay in the existing OnnxOCR engine.
- Each vertical scenario owns template selection, field extraction, normalization, validation, confidence, and export mapping.
- New industries can be added by registering a new template factory without changing the core model code.

Default scenarios verified with real images:

| Skill ID | Chinese-first scenario | Output |
| --- | --- | --- |
| `transport.train_ticket` | 火车票 OCR | ticket number, stations, train number, departure time, seat, price |
| `education.exam_paper` | 试卷信息 OCR | title, grade, subject, time limit, total score |
| `vehicle.plate` | 车牌识别 OCR | plate list and raw plate model output |
| `table.structure` | 表格结构化 OCR | HTML, cell boxes, logical row/column points |
| `document.image_to_markdown` | 图片转 Markdown | Markdown path, assets directory, preview text |

Candidate scenarios are available with `--candidates`. They stay out of the default registry until public or authorized anonymized samples validate the extraction quality.

| Skill ID | Chinese-first scenario | Output |
| --- | --- | --- |
| `agriculture.quality_inspection` | 农产品质检单 OCR | product, batch, inspection date, result, supplier |
| `agriculture.traceability_label` | 农产品溯源标签 OCR | trace code, product, origin, producer, harvest date |
| `agriculture.plant_protection_record` | 植保作业记录 OCR | crop, pesticide, dosage, operator, operation date |
| `oa.reimbursement` | 企业报销单 OCR | applicant, department, amount, date, purpose |
| `finance.invoice` | 发票 OCR | invoice code, invoice number, date, buyer, seller, amount |
| `finance.bank_card` | 银行卡 OCR | card number, bank name, card network/type, valid thru |
| `identity.id_card` | 中国公民身份证 OCR | name, gender, nation, birth date, address, ID number |
| `business.license` | 营业执照 OCR | credit code, company name, type, legal representative, address, date |
| `legal.contract_key_info` | 合同关键信息 OCR | contract number, parties, amount, sign date |
| `government.red_head_document` | 红头文件 OCR | document number, issuer, title, date |
| `logistics.inbound_order` | 物流仓储入库单 OCR | order number, sender, receiver, product, quantity |
| `logistics.express_waybill` | 快递面单 OCR | waybill number, sender, receiver, phone, address, company |
| `medical.lab_report` | 检验报告 OCR | patient, report number, department, item, report date |
| `transport.taxi_invoice` | 出租车票 OCR | invoice code, invoice number, info code, company, time, amount |
| `vehicle.driving_license` | 行驶证 OCR | plate number, owner, vehicle type, VIN, register date |
| `vehicle.driver_license` | 驾驶证 OCR | name, license number, vehicle class, valid period |
| `document.pdf_to_markdown` | PDF 转 Markdown | Markdown path, assets directory, preview text |

CLI:

```bash
pip install -e .

onnocr list
onnocr list --candidates
onnocr schema transport.train_ticket
onnocr run transport.train_ticket data/samples/scid_train_ticket.jpg --pretty
```

Agent-compatible explicit module style:

```bash
onnocr onnxocr.skill_cli list
onnocr onnxocr.skill_cli run vehicle.plate onnxocr/test_images/license_plate_single_blue.jpg --pretty
```

Real Chinese-image smoke test:

```bash
onnocr run education.exam_paper onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg --pretty
```

It should extract `title`, `grade`, `subject`, `time_limit`, and `total_score`.

Agent tools:

1. Ask Claude Code, Codex, or another local Agent to read [AGENTS.md](AGENTS.md) and [skills/industry-ocr/SKILL.md](skills/industry-ocr/SKILL.md).
2. The Agent can use `list` to choose a scenario, `schema` to inspect fields, and `run` to execute OCR on an image.
3. If fields are empty but `raw_text` has useful OCR text, extend `FieldSpec` labels, regex patterns, or add a new template.

Python API:

```python
from onnxocr.skills import OnnxOCREngine, SkillInput, create_default_registry

engine = OnnxOCREngine()
registry = create_default_registry()
skill = registry.get("transport.train_ticket", engine)
result = skill.run(SkillInput(image_path="sample.jpg"))
print(result.to_dict())
```

Standard output:

```json
{
  "skill_id": "transport.train_ticket",
  "fields": {
    "from_station": "鹤壁东站",
    "to_station": "郑州东站",
    "train_no": "G1289",
    "departure_time": "2018年01月05日17:41",
    "price": "59.5元"
  },
  "confidence": 0.85
}
```

Why not just use a large model Agent?

- Large model vision is excellent for open-ended understanding, but vertical OCR needs stable fields, reproducible behavior, confidence, audit trails, and batch-friendly costs.
- OnnxOCR Skill can run offline with ONNXRuntime, including private networks, edge devices, and localized hardware environments.
- The best workflow is composable: OnnxOCR Skill produces deterministic JSON, then the Agent performs review, correction, routing, system entry, or business reasoning.

Notes:

- Skill extraction quality depends on the underlying OCR result. If the document does not match the selected template, required fields are returned as `null` with low confidence.
- Built-in skills are Chinese-first for domestic contributors. Common English aliases are kept for compatibility and open-source examples.
- See [AGENTS.md](AGENTS.md), [docs/skills.md](docs/skills.md), and [skills/industry-ocr/SKILL.md](skills/industry-ocr/SKILL.md) for Agent usage and extension guidelines.

## Effect Demonstration

| Example 1 | Example 2 |
|-----------|-----------|
| ![](result_img/r1.png) | ![](result_img/r2.png) |

| Example 3 | Example 4 |
|-----------|-----------|
| ![](result_img/r3.png) | ![](result_img/draw_ocr4.jpg) |

## Contact & Communication

### OnnxOCR Community

![WeChat Group](onnxocr/test_images/微信群.jpg)

![QQ Group](onnxocr/test_images/QQ群.jpg)

## Acknowledgments

Thanks to [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for technical support and model references.

Thanks to the [RapidAI](https://github.com/RapidAI) open-source community, including [RapidTable](https://github.com/RapidAI/RapidTable), [RapidLayout](https://github.com/RapidAI/RapidLayout), [RapidDoc](https://github.com/RapidAI/RapidDoc), and [RapidOrientation](https://github.com/RapidAI/RapidOrientation), for excellent models, code, and engineering references.

## Open Source & Donations

If you recognize this project, you can support it via Alipay or WeChat Pay.

<img src="onnxocr/test_images/weixin_pay.jpg" alt="WeChat Pay" width="200">
<img src="onnxocr/test_images/zhifubao_pay.jpg" alt="Alipay" width="200">

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jingsongliujing/OnnxOCR&type=Date)](https://star-history.com/#jingsongliujing/OnnxOCR&Date)

## Contribution Guidelines

Issues and Pull Requests are welcome. For the vertical OCR CLI, the goal is not to claim every document type at once. The goal is to turn each real industry scenario into a stable capability that is installable, callable, testable, and auditable.

### OnnxOCR CLI Contribution Guide

Recommended contribution areas:

- **Add vertical scenarios**: agriculture quality forms, traceability labels, express waybills, business licenses, vehicle licenses, driver licenses, contract key fields, medical lab reports, and similar fixed templates.
- **Improve candidate scenarios**: add real anonymized samples, field rules, unit tests, and smoke-test records for templates listed behind `--candidates`.
- **Improve field extraction**: add Chinese labels, English aliases, regex rules, and normalization for dates, amounts, IDs, card numbers, and document numbers.
- **Improve Agent ergonomics**: make `onnocr list`, `schema`, and `run` easier for Claude Code, Codex, and other local Agent tools to call.
- **Add real evaluations**: provide public or authorized anonymized samples, then record which fields passed, which failed, and whether the scenario can move into the default registry.

When adding a CLI scenario, follow this workflow:

1. Add a template factory in `onnxocr/skills/builtin/industry.py` with a Chinese-first name, description, and domain tags.
2. Define fields with `FieldSpec`. Use English `snake_case` field IDs, Chinese-first labels, and common English aliases when useful.
3. Register the template in `onnxocr/skills/registry.py`. Keep it in `create_candidate_registry()` until real-sample validation passes.
4. Add fake OCR-line unit tests in `tests/test_skills.py` so the field rules can be regression-tested without large model files.
5. Add public or authorized anonymized samples under `data/samples/`, and document their source and purpose in `data/samples/README.md`.
6. Run a real OCR CLI smoke test and record the result in `docs/skill_evaluation.md`.
7. Update the scenario tables and examples in `Readme_cn.md`, `Readme.md`, and `docs/skills.md`.

Local verification:

```bash
pip install -e .
onnocr list
onnocr list --candidates
onnocr schema <skill_id> --candidates
onnocr run <skill_id> <image_path> --pretty --candidates
python -B -m pytest tests/test_skills.py -p no:cacheprovider
```

A candidate scenario can move into the default registry only when:

- It has at least one public or authorized anonymized real sample. Do not commit customer documents or private sensitive data.
- Core required fields are extracted reliably on the real sample, and known failures are documented.
- Unit tests and CLI smoke tests pass.
- The scenario boundary is clear; do not turn one template into a vague “universal OCR” rule set.
- The docs explain what the scenario is good for and what it is not good for.

Privacy and data rules:

- Do not submit real sensitive data such as ID cards, bank cards, contracts, medical reports, or express waybills.
- For sensitive scenarios, use public samples, official examples, or authorized anonymized samples.
- Synthetic samples are useful for engineering smoke tests, but they are not enough to enable a scenario by default.

Design principles:

- Agent-friendly CLI: stable commands, JSON output, and clear errors.
- Chinese-first, English-compatible: easy for Chinese contributors while still useful to open-source users.
- Small templates first: one scenario should solve one clear problem.
- Composable workflow: OnnxOCR CLI produces deterministic structured JSON; large model Agents handle review, correction, data entry, and business routing.
