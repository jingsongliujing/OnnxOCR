# OnnxOCR

如果项目对您有帮助，欢迎点击右上角 **Star** 支持！

![onnx_logo](onnxocr/test_images/onnxocr_logo.png)

**基于 ONNXRuntime 的高性能多语种 OCR 工程**

![GitHub stars](https://img.shields.io/github/stars/jingsongliujing/OnnxOCR?style=social)
![GitHub forks](https://img.shields.io/github/forks/jingsongliujing/OnnxOCR?style=social)
![GitHub license](https://img.shields.io/github/license/jingsongliujing/OnnxOCR)
![Python Version](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)

[English](./Readme.md) | 简体中文

## 版本更新

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

## 环境安装

```bash
python>=3.8
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

欢迎提交 Issues 和 Pull Requests，共同改进项目。
