# OnnxOCR

如果项目对您有帮助，欢迎点击右上角 **Star** 支持！

![onnx_logo](onnxocr/test_images/onnxocr_logo.png)

**基于 ONNXRuntime 的高性能多语种 OCR 工程**

![GitHub stars](https://img.shields.io/github/stars/jingsongliujing/OnnxOCR?style=social)
![GitHub forks](https://img.shields.io/github/forks/jingsongliujing/OnnxOCR?style=social)
![GitHub license](https://img.shields.io/github/license/jingsongliujing/OnnxOCR)
![Python Version](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)
![AtomGit Star](https://atomgit.com/OnnxOCR/OnnxOCR/star/badge.svg)

[English](./Readme.md) | 简体中文

## 版本更新

- **2026.05.15**
  1. 新增面向 Agent 的 `onnocr` / `onnxocr` 垂直 OCR CLI，支持查看场景、查看 Schema、运行图片识别。
  2. 内置火车票、试卷、车牌、表格结构化、图片转 Markdown 等已验证场景，身份证和银行卡等隐私/金融场景先作为候选模板提供。
  3. CLI 兼容 `onnocr onnxocr.skill_cli ...` 显式模块写法，也支持更短的 `onnocr list`、`onnocr run ...`。
  4. 新增标准 `SkillInput` / `SkillOutput` 契约、注册中心、模板字段抽取、开发文档、示例和轻量测试。

- **2026.05.01**
  1. 新增 ONNX 车牌检测与车牌号识别能力。
  2. 新增基于 RapidTable 的 ONNX 表格识别能力。
  3. 新增基于 RapidLayout 的中英文版面分析能力。
  4. 新增基于 RapidDoc 的文档版面分析与 Markdown 导出能力。
  5. `ONNXPaddleOcr` 新增 `use_plate_recognition`、`use_table_recognition`、`use_layout_analysis` 参数，默认均为 `False`，原有通用 OCR 调用方式不受影响。
  6. 新增 `/plate`、`/table`、`/layout`、`/layout_markdown` 等 HTTP 接口。

- **2025.05.21**
  1. 新增 PP-OCRv5 模型，单模型支持简体中文、繁体中文、中文拼音、英文、日文。
  2. 整体识别精度相比 PP-OCRv4 提升。
  3. 识别效果与 PaddleOCR 3.0 保持一致。

## 核心优势

1. **脱离深度学习训练框架**：可直接用于部署的通用 OCR 工程。
2. **跨架构支持**：基于 PaddleOCR 转换的 ONNX 模型，可部署在 ARM 和 x86 架构设备上。
3. **统一推理引擎**：项目内 ONNX 模型统一通过 `onnxocr/inference_engine.py` 创建 ONNXRuntime Session。
4. **多语种支持**：单模型支持 5 种文字类型。
5. **国产化适配友好**：下游厂商适配 GPU/NPU 时，优先修改统一推理引擎即可。
6. **Agent CLI 可扩展**：行业模板、字段抽取、校验和导出映射独立于底层 ONNX 推理代码，可被 Codex、Claude Code 等工具直接调用。

## 环境安装

```bash
python>=3.8
git clone https://github.com/jingsongliujing/OnnxOCR.git
cd OnnxOCR
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

说明：

- 仓库默认只包含 `tests/test_general_ocr.py` 所需的 PP-OCRv5 通用 OCR 模型。
- 车牌识别、表格识别、版面分析、方向分类、RapidDoc Markdown 导出等扩展模型较大，国内优先从 [ModelScope](https://www.modelscope.cn/models/supersong/onnxocr_model/tree/master/models) 下载，也可以使用 [HuggingFace](https://huggingface.co/jingsongliu/onnxocr_model/tree/main) 或国内镜像站 `https://hf-mirror.com`。
- PP-OCRv5 Server ONNX 模型也可下载后替换 `onnxocr/models/ppocrv5/` 下的 det/rec 模型。

## 模型下载

扩展模型统一托管在 ModelScope 和 HuggingFace。国内网络建议优先使用 [ModelScope models 目录](https://www.modelscope.cn/models/supersong/onnxocr_model/tree/master/models)：

```bash
python scripts/download_models.py
```

等价的 ModelScope 核心代码如下：

```python
from modelscope import snapshot_download

model_dir = snapshot_download("supersong/onnxocr_model")
```

如果希望从 HuggingFace 下载，可使用：

```bash
python scripts/download_models.py --source huggingface
```

HuggingFace 模型地址：[jingsongliu/onnxocr_model](https://huggingface.co/jingsongliu/onnxocr_model/tree/main)。

国内访问 HuggingFace 较慢时，可使用镜像站 `hf-mirror.com`：

```bash
python scripts/download_models.py --source huggingface --hf-endpoint https://hf-mirror.com
```

脚本会把模型仓库中的 `models/` 目录同步到本地 `onnxocr/models/`，并检查 RapidDoc 所需的 `onnxocr/models/rapid_doc/layout/pp_doclayoutv2.onnx` 等关键文件是否存在。

只检查本地模型是否齐全：

```bash
python scripts/download_models.py --check-only
```

## 一键运行

```bash
python test_ocr.py
```

`test_ocr.py` 默认只运行通用 OCR。车牌识别、表格识别、版面分析、RapidDoc Markdown 导出示例已在文件中注释，下载对应模型后按需取消注释即可。

独立测试文件：

```bash
python tests/test_general_ocr.py
python tests/test_license_plate_ocr.py
python tests/test_table_ocr.py
python tests/test_layout_analysis.py
python tests/test_layout_markdown.py
```

测试输出默认写入 `result_img/`，该目录已加入 `.gitignore`。

## 通用 OCR

```python
import cv2
from onnxocr.onnx_paddleocr import ONNXPaddleOcr

img = cv2.imread("onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg")
model = ONNXPaddleOcr(use_angle_cls=False, use_gpu=False)
result = model.ocr(img)
print(result)
```

## 车牌识别

车牌识别作为可选模式融合到 `ONNXPaddleOcr` 中，默认仍使用原来的通用 OCR 流程。

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

模型文件：

```text
onnxocr/models/license_plate/car_plate_detect.onnx
onnxocr/models/license_plate/plate_rec.onnx
```

## 表格识别

表格识别集成自 RapidTable，复用通用 OCR 的文字检测和识别结果，再进行表格结构还原，输出 HTML、单元格框和逻辑行列坐标。

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

模型文件：

```text
onnxocr/models/table/slanet-plus.onnx
onnxocr/models/table/ch_ppstructure_mobile_v2_SLANet.onnx
onnxocr/models/table/en_ppstructure_mobile_v2_SLANet.onnx
```

## 中英文版面分析

版面分析集成自 RapidLayout，用于定位文档图像中的标题、正文、表格、图片、页眉页脚等元素。

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

英文版面模型可将 `layout_model_type` 设置为 `pp_layout_publaynet`。

模型文件：

```text
onnxocr/models/layout/layout_cdla.onnx
onnxocr/models/layout/layout_publaynet.onnx
```

## 文档转 Markdown

文档转 Markdown 集成自 RapidDoc，支持根据版面分析结果识别标题、段落、表格和图片等内容，并保存为 Markdown 文件。

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

RapidDoc 模型文件：

```text
onnxocr/models/rapid_doc/layout/pp_doclayoutv2.onnx
onnxocr/models/rapid_doc/table/q_cls.onnx
onnxocr/models/rapid_doc/table/unet.onnx
onnxocr/models/rapid_doc/table/slanet-plus.onnx
```

## 推理引擎适配

项目中的通用 OCR、车牌识别、表格识别、版面分析和 RapidDoc 文档解析都通过 `onnxocr/inference_engine.py` 创建 ONNXRuntime Session。

如果需要适配下游厂商 GPU/NPU，通常只需要修改：

```python
from onnxocr.inference_engine import create_session
```

核心扩展点：

- `create_session(model_path, providers=None, use_gpu=False, gpu_id=0, sess_options=None)`
- `build_providers(use_gpu=False, gpu_id=0, providers=None)`
- `build_providers_from_engine_cfg(engine_cfg)`
- `ProviderConfig(engine_cfg)`

当前项目只有 `onnxocr/inference_engine.py` 直接 `import onnxruntime`，业务模块不直接依赖 ONNXRuntime API。

## API 服务

启动服务：

```bash
python app-service.py
```

主要接口：

- `/ocr`：通用 OCR。
- `/plate`：车牌识别。
- `/table`：表格识别。
- `/layout`：版面分析。
- `/layout_markdown`：图片或 PDF 转 Markdown。

WebUI：

```bash
python webui.py
```

## Docker 镜像环境

```bash
docker build -t ocr-service .
docker run -itd --name onnxocr-service -p 5006:5005 ocr-service
```

## 代码结构

```text
onnxocr/
  inference_engine.py        # 唯一 ONNXRuntime 入口
  onnx_paddleocr.py          # 用户统一调用入口
  skill_cli.py               # 垂直 OCR Skill 命令行入口
  skills/                    # 可扩展 OCR Skill 运行时
  predict_det.py             # 通用 OCR 检测
  predict_rec.py             # 通用 OCR 识别
  orientation.py             # 本地 RapidOrientation ONNX 适配
  license_plate.py           # 车牌识别
  table_recognition.py       # 表格识别封装
  layout_recognition.py      # 版面分析封装
  layout_markdown.py         # RapidDoc Markdown 导出封装
  rapid_layout/              # RapidLayout 源码级 ONNX 集成
  rapid_table/               # RapidTable 源码级 ONNX 集成
  rapid_doc/                 # RapidDoc 源码级 ONNX 集成
  models/                    # 本地 ONNX 模型
tests/                       # 独立功能测试
```

## 垂直行业 OCR CLI

OnnxOCR 现在提供面向 Agent 的垂直 OCR CLI。它把固定行业模板做成可发现、可测试、可组合的命令行能力，让 Claude Code、Codex 等 Agent 工具无需额外适配即可调用 OnnxOCR。

CLI 层保持轻量：

- ONNX 推理、文本检测识别、表格识别、车牌识别仍由现有 OnnxOCR 引擎负责。
- 每个垂直场景只负责模板选择、字段抽取、字段归一化、校验、置信度和导出映射。
- 新行业只需要新增模板工厂并注册，不需要改动底层模型推理代码。

默认启用场景（真实图片烟测通过）：

| Skill ID | 场景 | 输出字段 |
| --- | --- | --- |
| `transport.train_ticket` | 火车票 OCR | 票号、出发站、到达站、车次、发车时间、座位、席别、票价 |
| `education.exam_paper` | 试卷信息 OCR | 标题、年级、科目、考试时间、满分 |
| `vehicle.plate` | 车牌识别 OCR | 车牌列表和原始车牌模型结果 |
| `table.structure` | 表格结构化 OCR | HTML、单元格框、逻辑行列坐标 |
| `document.image_to_markdown` | 图片转 Markdown | Markdown 文件、资源目录、预览文本 |

候选场景（需通过 `--candidates` 查看，等待公开/授权脱敏真实样例验证后再默认启用）：

| Skill ID | 场景 | 输出字段 |
| --- | --- | --- |
| `agriculture.quality_inspection` | 农产品质检单 OCR | 产品、批次、检测日期、结果、供应商 |
| `agriculture.traceability_label` | 农产品溯源标签 OCR | 追溯码、产品、产地、生产者、采摘日期 |
| `agriculture.plant_protection_record` | 植保作业记录 OCR | 作物、农药、用量、作业人、作业日期 |
| `oa.reimbursement` | 企业 OA 报销单、行政单据 | 报销人、部门、金额、日期、用途 |
| `finance.invoice` | 发票 OCR | 发票代码、发票号码、开票日期、购方、销方、金额 |
| `finance.bank_card` | 银行卡 OCR | 卡号、银行名称、卡组织、有效期 |
| `identity.id_card` | 中国公民身份证 OCR | 姓名、性别、民族、出生日期、住址、身份证号 |
| `business.license` | 营业执照 OCR | 统一社会信用代码、名称、类型、法定代表人、住所、成立日期 |
| `legal.contract_key_info` | 合同关键信息 OCR | 合同编号、甲方、乙方、金额、签署日期 |
| `government.red_head_document` | 红头文件 OCR | 文号、发文机关、标题、日期 |
| `logistics.inbound_order` | 物流仓储入库单 OCR | 单号、发货方、收货方、货品、数量 |
| `logistics.express_waybill` | 快递面单 OCR | 运单号、寄件人、收件人、电话、地址、快递公司 |
| `medical.lab_report` | 检验报告 OCR | 姓名、报告单号、科室、检验项目、报告日期 |
| `transport.taxi_invoice` | 出租车票 OCR | 发票代码、号码、信息码、公司、上下车时间、金额 |
| `vehicle.driving_license` | 行驶证 OCR | 号牌号码、所有人、车辆类型、VIN、注册日期 |
| `vehicle.driver_license` | 驾驶证 OCR | 姓名、证号、准驾车型、有效期 |
| `document.pdf_to_markdown` | PDF 转 Markdown | Markdown 文件、资源目录、预览文本 |

命令行：

```bash
pip install -e .

onnocr list
onnocr list --candidates
onnocr schema transport.train_ticket
onnocr run transport.train_ticket data/samples/scid_train_ticket.jpg --pretty
```

兼容 Agent 显式模块写法：

```bash
onnocr onnxocr.skill_cli list
onnocr onnxocr.skill_cli run vehicle.plate onnxocr/test_images/license_plate_single_blue.jpg --pretty
```

真实中文图片烟测：

```bash
onnocr run education.exam_paper onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg --pretty
```

预期能抽取 `title`、`grade`、`subject`、`time_limit`、`total_score` 等字段。

在 Agent 工具中使用：

1. 让 Claude Code、Codex 等工具先读取 [AGENTS.md](AGENTS.md) 和 [skills/industry-ocr/SKILL.md](skills/industry-ocr/SKILL.md)。
2. Agent 用 `list` 选择场景，用 `schema` 查看字段，用 `run` 对图片执行结构化 OCR。
3. 如果字段为空但 `raw_text` 有内容，优先补充 `FieldSpec` 标签、正则或新增行业模板。

Python API：

```python
from onnxocr.skills import OnnxOCREngine, SkillInput, create_default_registry

engine = OnnxOCREngine()
registry = create_default_registry()
skill = registry.get("transport.train_ticket", engine)
result = skill.run(SkillInput(image_path="sample.jpg"))
print(result.to_dict())
```

标准输出示例：

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

说明：

- Skill 抽取质量依赖底层 OCR 识别质量；如果图片不符合当前模板，必填字段会返回 `null`，整体置信度会较低。
- 内置 Skill 中文优先，便于国内贡献者阅读、讨论和补充真实行业模板；英文标签只是兼容别名，用于开源样例和跨境业务。
- 大模型 Agent 适合开放式理解；OnnxOCR Skill 更适合批量、离线、低成本、可复现、可审计的结构化 OCR。推荐组合方式是：Skill 产出稳定 JSON，Agent 负责复核、纠错、录入和业务流转。
- Agent 使用方式见 [AGENTS.md](AGENTS.md)，扩展方式见 [docs/skills.md](docs/skills.md) 和 [skills/industry-ocr/SKILL.md](skills/industry-ocr/SKILL.md)。

## 效果展示

| 示例 1 | 示例 2 |
|--------|--------|
| ![](result_img/r1.png) | ![](result_img/r2.png) |

| 示例 3 | 示例 4 |
|--------|--------|
| ![](result_img/r3.png) | ![](result_img/draw_ocr4.jpg) |

## 联系与交流

### OnnxOCR 交流群

![微信群](onnxocr/test_images/微信群.jpg)

![QQ群](onnxocr/test_images/QQ群.jpg)

## 致谢

非常感谢 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 提供的技术支持和模型参考。

非常感谢 [RapidAI](https://github.com/RapidAI) 开源社区，以及其中 [RapidTable](https://github.com/RapidAI/RapidTable)、[RapidLayout](https://github.com/RapidAI/RapidLayout)、[RapidDoc](https://github.com/RapidAI/RapidDoc)、[RapidOrientation](https://github.com/RapidAI/RapidOrientation) 等项目提供的优秀模型、代码和工程参考。

## 开源与捐赠

如果您认可本项目，可以通过支付宝或微信进行支持。

<img src="onnxocr/test_images/weixin_pay.jpg" alt="微信支付" width="200">
<img src="onnxocr/test_images/zhifubao_pay.jpg" alt="支付宝" width="200">

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=jingsongliujing/OnnxOCR&type=Date)](https://star-history.com/#jingsongliujing/OnnxOCR&Date)

## 贡献指南

欢迎提交 Issues 和 Pull Requests，共同改进项目。垂直行业 OCR CLI 的目标不是把所有场景一次性做满，而是把每个真实行业场景做成可安装、可调用、可测试、可复核的稳定能力。

### OnnxOCR CLI 贡献指南

推荐贡献方向：

- **新增垂直场景**：例如农产品质检单、溯源标签、快递面单、营业执照、行驶证、驾驶证、合同关键字段、医疗检验报告等。
- **完善候选场景**：给 `--candidates` 中已有模板补充真实脱敏样例、字段规则、单元测试和烟测记录。
- **增强字段抽取**：补充中文标签、英文别名、正则规则、日期/金额/证件号等归一化逻辑。
- **改进 Agent 体验**：让 `onnocr list`、`schema`、`run` 的输出更适合 Claude Code、Codex 等工具读取和调用。
- **补充真实评估**：新增公开或授权脱敏样例，并记录哪些字段成功、哪些字段失败、是否能提升为默认场景。

新增一个 CLI 场景时，请按以下步骤提交：

1. 在 `onnxocr/skills/builtin/industry.py` 新增模板工厂函数，使用中文名称、中文说明和领域标签。
2. 使用 `FieldSpec` 定义字段，字段 ID 使用英文 `snake_case`，字段标签中文优先，可补常见英文别名。
3. 在 `onnxocr/skills/registry.py` 注册。未通过真实样例验证前，只注册到 `create_candidate_registry()`。
4. 在 `tests/test_skills.py` 添加不依赖模型文件的假 OCR 行测试，确保字段规则可回归。
5. 准备公开或授权脱敏样例，放入 `data/samples/`，并在 `data/samples/README.md` 说明来源和用途。
6. 用真实 OCR 引擎跑 CLI 烟测，并把结果写入 `docs/skill_evaluation.md`。
7. 更新 `Readme_cn.md`、`Readme.md`、`docs/skills.md` 中的场景表和调用示例。

本地验证命令：

```bash
pip install -e .
onnocr list
onnocr list --candidates
onnocr schema <skill_id> --candidates
onnocr run <skill_id> <image_path> --pretty --candidates
python -B -m pytest tests/test_skills.py -p no:cacheprovider
```

候选场景提升为默认场景的标准：

- 至少有一个公开或授权脱敏的真实样例，不接受客户真实隐私数据直接入库。
- 核心必填字段在真实样例上能够稳定抽取，失败字段和原因有记录。
- 单元测试和 CLI 烟测均通过。
- 场景边界清晰，不把一个模板写成“万能 OCR”。
- 文档中说明适合处理什么、不适合处理什么。

隐私和数据要求：

- 不提交身份证、银行卡、合同、医疗报告、快递面单等真实敏感数据。
- 如需贡献此类场景，请使用公开样张、官方示例或经过授权的脱敏样例。
- 合成样例可以用于工程链路测试，但不能单独作为默认启用的依据。

设计原则：

- CLI 对 Agent 友好：命令稳定、输出 JSON、错误信息清楚。
- 中文优先，英文兼容：方便国内开发者贡献，同时不影响开源用户使用。
- 小模板优先：一个场景解决一个明确问题，避免把规则堆成不可维护的大模板。
- 可组合：OnnxOCR CLI 负责确定性 OCR 和结构化 JSON，大模型 Agent 负责复核、纠错、录入和业务流转。
