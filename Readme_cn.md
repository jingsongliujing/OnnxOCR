# OnnxOCR

如果项目对您有帮助，欢迎点击右上角 **Star** 支持！

![onnx_logo](onnxocr/test_images/onnxocr_logo.png)

**基于 ONNXRuntime 的高性能多语种 OCR 工程**

![GitHub stars](https://img.shields.io/github/stars/jingsongliujing/OnnxOCR?style=social)
![GitHub forks](https://img.shields.io/github/forks/jingsongliujing/OnnxOCR?style=social)
![GitHub license](https://img.shields.io/github/license/jingsongliujing/OnnxOCR)
![Python Version](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)
![AtomGit Star](https://atomgit.com/OnnxOCR/OnnxOCR/star/badge.svg)

[English](./Readme.md) | 简体中文 | [日本語](./Readme_ja.md)

## 版本更新

- **2026.06.11**
  1. 默认通用 OCR 升级为官方 PP-OCRv6 Medium ONNX 模型。
  2. 集成官方 PP-OCRv6 tiny/small/medium 检测与识别 ONNX 模型，以及 `ppocrv6_dict.txt`、`ppocrv6_tiny_dict.txt` 字典。
  3. PP-OCRv6 检测后处理参数对齐官方 `inference.yml`。
  4. 新增 `scripts/benchmark_ppocr_versions.py`，用于多图对比 PP-OCRv5 与 PP-OCRv6 速度。

- **2026.05.27**
  1. 新增 OCR + Qwen3.5-2B ONNX 信息抽取使用方法。
  2. 新增 `onnxocr.qwen35_2b` 包内模块，用于 Qwen3.5-2B ONNX 模型下载、校验和纯 Python 推理。
  3. 新增 `examples/id_card_extract_with_qwen.py` 端到端示例：先用 OnnxOCR 做全量文字识别，再用 Qwen3.5-2B 抽取身份证结构化字段。
  4. Qwen3.5-2B ONNX 使用独立 ModelScope 仓库：[supersong/qwen2bonnx](https://www.modelscope.cn/models/supersong/qwen2bonnx/tree/master/models)。

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
4. **多语种支持**：默认 PP-OCRv6 medium/small 单模型支持 50 种语言，tiny 支持 49 种语言。
5. **国产化适配友好**：下游厂商适配 GPU/NPU 时，优先修改统一推理引擎即可。

## 环境安装

```bash
python>=3.8
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

说明：

- 仓库默认使用官方 PP-OCRv6 medium ONNX 通用 OCR 模型。
- 车牌识别、表格识别、版面分析、方向分类、RapidDoc Markdown 导出等扩展模型较大，国内优先从 [ModelScope](https://www.modelscope.cn/models/supersong/onnxocr_model/tree/master/models) 下载，也可以使用 [HuggingFace](https://huggingface.co/jingsongliu/onnxocr_model/tree/main) 或国内镜像站 `https://hf-mirror.com`。
- PP-OCRv6 tiny/small/medium ONNX 模型位于 `onnxocr/models/ppocrv6/`，可通过 `ONNXPaddleOcr(ocr_model_size="small")` 或 `ocr_model_size="tiny"` 切换。

## 模型下载

默认下载源为 PaddleX 官方 BOS，包含 PP-OCRv6 tiny/small/medium 检测与识别 ONNX 模型：

```bash
python scripts/download_models.py
```

如需继续从 ModelScope 下载历史扩展模型，可使用 `--source modelscope`。

如果希望从 HuggingFace 下载，可使用：

```bash
python scripts/download_models.py --source huggingface
```

HuggingFace 模型地址：[jingsongliu/onnxocr_model](https://huggingface.co/jingsongliu/onnxocr_model/tree/main)。

国内访问 HuggingFace 较慢时，可使用镜像站 `hf-mirror.com`：

```bash
python scripts/download_models.py --source huggingface --hf-endpoint https://hf-mirror.com
```

脚本会把模型同步到本地 `onnxocr/models/`，并检查 PP-OCRv6 与 RapidDoc 所需的关键文件是否存在。

只检查本地模型是否齐全：

```bash
python scripts/download_models.py --check-only
```

### Qwen3.5-2B 信息抽取基座

OnnxOCR 也可以在 OCR 之后接入本地 Qwen3.5-2B ONNX 模型，作为信息抽取基座使用。典型流程是：

- 先用 OnnxOCR 对图片做全量文字识别，得到文本和文本框。
- 再把 OCR 文本交给 Qwen3.5-2B ONNX，抽取为结构化 JSON。
- OCR 和抽取都在本地通过 Python + ONNXRuntime 运行，不依赖 JS。

Qwen3.5-2B ONNX 使用独立模型仓库，和车牌、表格、版面分析等可选 OCR 模型不是同一个下载地址：

- ModelScope：[supersong/qwen2bonnx](https://www.modelscope.cn/models/supersong/qwen2bonnx/tree/master/models)
- 本地目录：`onnxocr/models/qwen_2b`

准备 Qwen3.5-2B ONNX 模型：

```bash
python examples/qwen35_2b_onnx.py download --variant q4
python examples/qwen35_2b_onnx.py verify
```

模型默认放在：

```text
onnxocr/models/qwen_2b
```

纯 Python 文本/图文推理冒烟测试：

```bash
python examples/qwen35_2b_onnx.py run-python --prompt "你好，简单介绍一下你自己。"
python examples/qwen35_2b_onnx.py run-python --image onnxocr/models/qwen_2b/images/demo.jpeg --prompt "用一句话描述这张图片。"
```

也可以直接调用包内 API：

```python
from onnxocr.qwen35_2b import Qwen35ONNX

qwen = Qwen35ONNX("onnxocr/models/qwen_2b")
text, stopped = qwen.generate("从下面 OCR 文本中抽取字段：...")
print(text)
```

扩展新的场景抽取模板：

```python
from onnxocr.qwen35_2b import ExtractionTemplate, QwenScenarioExtractor

invoice_template = ExtractionTemplate(
    name="invoice_basic",
    description="发票 OCR",
    fields=["发票号码", "销售方", "购买方", "价税合计", "开票日期"],
    rules=[
        "只根据 OCR 文本抽取，不要编造。",
        "缺失或不确定的字段填空字符串。",
        "只输出 JSON，不要解释。",
    ],
)

extractor = QwenScenarioExtractor(model_dir="onnxocr/models/qwen_2b")
fields, raw = extractor.extract(ocr_text, template=invoice_template)
print(fields)
```

基础包不内置任何垂直场景提示词。请把场景模板放在你的应用代码或 `examples/` 中，再传给 `QwenScenarioExtractor`。你可以通过定义字段和规则，继续扩展发票、营业执照、合同、快递面单、医疗报告、质检单等垂直 OCR 场景。

## 一键运行通用 OCR

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

## PP-OCRv5 / PP-OCRv6 速度对比

本仓库提供本地基准脚本，用于对比 PP-OCRv5 与 PP-OCRv6 tiny/small/medium 在同一批图片上的端到端 OCR 速度。脚本会对每张图片先预热 1 次，再计时多次取平均。

```bash
python scripts/benchmark_ppocr_versions.py --repeats 2
```

本机 CPUExecutionProvider 实测结果如下，完整输出见 `output/benchmarks/ppocr_versions_benchmark.md` 与 `output/benchmarks/ppocr_versions_benchmark.json`。不同 CPU、ONNXRuntime 版本、线程数和图片尺寸会影响绝对耗时。

| 模型 | 总平均耗时(s) | 单图平均耗时(s) | 识别行数 |
| --- | ---: | ---: | ---: |
| PP-OCRv5 | 7.292 | 0.911 | 250 |
| PP-OCRv6 tiny | 4.231 | 0.529 | 249 |
| PP-OCRv6 small | 7.506 | 0.938 | 249 |
| PP-OCRv6 medium | 26.124 | 3.266 | 242 |

| 图片 | 尺寸 | PP-OCRv5(s) | v6 tiny(s) | v6 small(s) | v6 medium(s) | tiny/v5 | small/v5 | medium/v5 | v5 行数 | tiny 行数 | small 行数 | medium 行数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 715873facf064583b44ef28295126fa7.jpg | 1920x2560 | 2.471 | 1.453 | 2.673 | 8.789 | 0.59x | 1.08x | 3.56x | 72 | 72 | 71 | 64 |
| 12.jpg | 720x1150 | 0.473 | 0.228 | 0.338 | 1.559 | 0.48x | 0.71x | 3.30x | 4 | 5 | 4 | 4 |
| 1.jpg | 720x1150 | 0.351 | 0.173 | 0.313 | 1.131 | 0.49x | 0.89x | 3.22x | 2 | 2 | 2 | 2 |
| french_0.jpg | 692x1024 | 0.403 | 0.214 | 0.437 | 1.221 | 0.53x | 1.08x | 3.03x | 6 | 9 | 8 | 6 |
| japan_2.jpg | 1536x839 | 1.203 | 0.687 | 1.041 | 3.907 | 0.57x | 0.87x | 3.25x | 50 | 49 | 55 | 54 |
| weixin_pay.jpg | 1263x1719 | 0.333 | 0.462 | 0.676 | 2.270 | 1.38x | 2.03x | 6.81x | 3 | 3 | 3 | 3 |
| table.jpg | 371x293 | 1.123 | 0.534 | 1.133 | 4.138 | 0.48x | 1.01x | 3.68x | 81 | 74 | 73 | 75 |
| 00006737.jpg | 896x528 | 0.936 | 0.480 | 0.896 | 3.109 | 0.51x | 0.96x | 3.32x | 32 | 35 | 33 | 34 |

说明：当前 CPU ONNXRuntime 下，PP-OCRv6 tiny 平均快于 PP-OCRv5，PP-OCRv6 small 与 PP-OCRv5 接近，PP-OCRv6 medium 因模型更大明显更慢。PP-OCRv6 medium/small 的优势主要在官方评测中的多语种覆盖和综合精度；对速度敏感时可优先使用 `ONNXPaddleOcr(ocr_model_size="tiny")` 或 `ocr_model_size="small"`。

## 通用 OCR

```python
import cv2
from onnxocr.onnx_paddleocr import ONNXPaddleOcr

img = cv2.imread("onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg")
model = ONNXPaddleOcr(use_angle_cls=False, use_gpu=False)
result = model.ocr(img)
print(result)
```

## OCR + Qwen 信息抽取

`examples/id_card_extract_with_qwen.py` 是一个最小端到端垂直场景示例。可复用的 Qwen 推理和通用模板抽取代码位于 `onnxocr.qwen35_2b`，身份证字段和提示词规则只放在该示例脚本中。

默认示例：

```bash
python examples/id_card_extract_with_qwen.py
```

指定图片路径：

```bash
python examples/id_card_extract_with_qwen.py --image "onnxocr/test_images/8f113149-ff64-4c9f-8dc0-e34100365aa4.jpg"
```

输出文件：

```text
result_img/id_card_front_qwen_extract.json
```

脚本默认不会在控制台打印身份证敏感字段，只会写入本地 JSON。仅在可信本地环境中使用 `--show-sensitive` 查看抽取明细。

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

`app-service.py` 是用于部署的 JSON API 服务，`webui.py` 是浏览器可视化服务。两者分离后，项目更容易部署、测试和维护。

启动 API 服务：

```bash
python app-service.py
```

环境变量：

- `ONNXOCR_PORT`：服务端口，默认 `5005`。
- `ONNXOCR_USE_GPU`：设为 `1` / `true` 时启用 GPU provider。
- `ONNXOCR_DEBUG`：设为 `1` / `true` 时启用 Flask debug。
- `ONNXOCR_OUTPUT_DIR`：Markdown 和资源输出目录，默认 `result_img`。
- `ONNXOCR_MAX_UPLOAD_MB`：最大上传大小，单位 MB，默认 `200`。

主要接口支持 JSON base64 图片，也支持 multipart 文件上传：

- `/health`：服务健康检查。
- `/ocr`：通用 OCR。
- `/plate`：车牌识别。
- `/table`：表格识别。
- `/layout`：版面分析。
- `/layout_markdown`：图片或 PDF 转 Markdown。

JSON 示例：

```bash
curl -X POST http://127.0.0.1:5005/ocr \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"<base64-image>\"}"
```

multipart 示例：

```bash
curl -X POST http://127.0.0.1:5005/ocr \
  -F "image=@onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg"
```

启动 WebUI：

```bash
python webui.py
```

## Docker 镜像环境

```bash
docker build -t ocr-service .
docker run -itd --name onnxocr-service -p 5006:5005 ocr-service
```

Docker 构建通过 `.dockerignore` 排除缓存、运行输出和可选大模型文件。车牌、表格、版面分析、RapidDoc 等扩展模型可以在镜像内按需下载，也可以运行时挂载。

## 代码结构

```text
app-service.py                 # 可部署 JSON API 服务
webui.py                       # 浏览器可视化服务
test_ocr.py                    # 本地一键 OCR 示例
requirements.txt               # 运行依赖
Dockerfile                     # API 服务容器镜像
.dockerignore                  # 排除缓存、输出和大模型文件
Readme.md                      # 英文文档
Readme_cn.md                   # 中文文档
Readme_ja.md                   # 日文文档
onnxocr/
  inference_engine.py        # 唯一 ONNXRuntime 入口
  onnx_paddleocr.py          # 用户统一调用入口
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
scripts/
  download_models.py         # 可选模型下载/检查脚本
tests/                       # 独立功能测试和 API 烟测
```

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

欢迎提交 Issues 和 Pull Requests，共同改进项目。为了保证开源用户按文档操作时不踩坑，请遵守以下约定：

1. API-only 能力放在 `app-service.py`，浏览器 WebUI 能力放在 `webui.py`。
2. 不提交 `result_img/`、`results/`、`uploads/` 或本地缓存目录中的生成文件。
3. 不提交客户真实单据、身份证、银行卡、医疗记录、合同、快递面单、生产密钥等敏感数据。
4. 测试和文档样例只使用公开样例、官方示例或经过授权的脱敏样例。
5. 如果功能依赖可选模型文件，请在 PR 中说明具体模型文件和下载命令。
6. README 中出现的命令必须能在当前仓库中找到对应实现，不要写未实现的入口。

提交前建议检查：

```bash
python -B -m pytest tests/test_app_service.py -p no:cacheprovider
python tests/test_general_ocr.py
python tests/test_license_plate_ocr.py
python tests/test_table_ocr.py
python tests/test_layout_analysis.py
python tests/test_layout_markdown.py
```

部分测试依赖可选模型文件。如果本地无法运行，请在 PR 中说明缺少哪些模型文件。
