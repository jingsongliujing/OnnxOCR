# OnnxOCR

**High-performance multilingual OCR based on ONNXRuntime**

![onnx_logo](assets/logos/onnxocr_logo.png)

![GitHub stars](https://img.shields.io/github/stars/jingsongliujing/OnnxOCR?style=social)
![GitHub forks](https://img.shields.io/github/forks/jingsongliujing/OnnxOCR?style=social)
![GitHub license](https://img.shields.io/github/license/jingsongliujing/OnnxOCR)
![Python Version](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)

English | [简体中文](./Readme_cn.md)

---

## Overview

OnnxOCR is a **ready-to-use, framework-free** OCR engineering project. Built on PaddleOCR's ONNX-converted models, it supports deployment on ARM/x86 devices for document digitization, receipt processing, and ID recognition.

**Core Value:** Converts PaddleOCR's excellent models to ONNX format, maintaining full recognition capability while significantly reducing deployment complexity.

---

## Features

| Feature | Description | Extra Models Required |
|---------|-------------|----------------------|
| **General OCR** | Multi-language text detection & recognition (Chinese, English, Japanese) | ❌ Built-in |
| **License Plate** | Chinese license plate detection & recognition | ✅ Download |
| **Table Recognition** | Table structuring with HTML/coordinates output | ✅ Download |
| **Layout Analysis** | Document element detection (title/text/table/image) | ✅ Download |
| **Document to Markdown** | Document/Image/PDF to Markdown conversion | ✅ Download |
| **Vertical OCR CLI** | Structured extraction for tickets, IDs, bank cards, etc. | ❌ Built-in |

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/jingsongliujing/OnnxOCR.git
cd OnnxOCR

# Install dependencies
pip install -r requirements.txt
```

### 2. First OCR Example

```python
import cv2
from onnxocr.onnx_paddleocr import ONNXPaddleOcr

# Initialize model
model = ONNXPaddleOcr(use_angle_cls=False, use_gpu=False)

# Read image and recognize
img = cv2.imread("onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg")
result = model.ocr(img)
print(result)
```

Or run the test directly:

```bash
python test_ocr.py
```

### 3. Use Vertical OCR CLI (Optional)

```bash
# Install CLI
pip install -e .

# List supported scenarios
onnocr list

# Recognize train ticket
onnocr run transport.train_ticket <image_path> --pretty
```

---

## Feature Details

### General OCR

Multi-language text detection and recognition supporting Chinese, English, Japanese, and more.

```python
import cv2
from onnxocr.onnx_paddleocr import ONNXPaddleOcr

img = cv2.imread("test.jpg")
model = ONNXPaddleOcr(use_angle_cls=False, use_gpu=False)
result = model.ocr(img)
```

### License Plate Recognition

Integrated license plate detection and recognition for Chinese plates (blue, green, yellow).

```python
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2PlateImg

plate_model = ONNXPaddleOcr(
    use_angle_cls=True,
    use_gpu=False,
    use_plate_recognition=True,
    plate_min_score=0.4,
)
plate_result = plate_model.ocr(img)
sav2PlateImg(img, plate_result, name="./output/plate_vis.jpg")
```

**Model files:**
```
onnxocr/models/license_plate/car_plate_detect.onnx
onnxocr/models/license_plate/plate_rec.onnx
```

### Table Recognition

Based on RapidTable, converts table images to structured HTML and cell coordinates.

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
sav2TableImg(img, table_result, name="./output/table_vis.jpg")
```

**Model files:**
```
onnxocr/models/table/slanet-plus.onnx
onnxocr/models/table/ch_ppstructure_mobile_v2_SLANet.onnx
```

### Layout Analysis

Based on RapidLayout, locates document elements like titles, text blocks, tables, and images.

```python
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2LayoutImg

layout_model = ONNXPaddleOcr(
    use_gpu=False,
    use_layout_analysis=True,
    layout_model_type="pp_layout_cdla",
)
layout_result = layout_model.ocr(img)
sav2LayoutImg(img, layout_result, name="./output/layout_vis.jpg")
```

**Model files:**
```
onnxocr/models/layout/layout_cdla.onnx
onnxocr/models/layout/layout_publaynet.onnx
```

### Document to Markdown

Based on RapidDoc, converts document images/PDFs to structured Markdown files.

```python
from onnxocr.layout_markdown import LayoutMarkdownConverter

converter = LayoutMarkdownConverter(
    layout_model_type="pp_doclayoutv2",
    formula_enable=False,
    table_enable=True,
)
result = converter.convert_file(
    "onnxocr/test_images/layout_cdla.jpg",
    output_md_path="./output/test_markdown.md",
)
print(result["markdown_path"])
```

**Model files:**
```
onnxocr/models/rapid_doc/layout/pp_doclayoutv2.onnx
onnxocr/models/rapid_doc/table/slanet-plus.onnx
```

---

## Vertical OCR CLI

OnnxOCR provides an Agent-oriented vertical OCR CLI for structured field extraction.

### Supported Scenarios

**Default scenarios (verified):**

| Scenario | Skill ID | Extracted Fields |
|----------|----------|------------------|
| Exam Paper | `education.exam_paper` | Title, grade, subject, time limit, total score |
| ID Card | `identity.id_card` | Name, gender, nation, birth date, address, ID number |
| Bank Card | `finance.bank_card` | Card number, bank name, valid thru, card type |
| Invoice | `finance.invoice` | Invoice code, invoice number, date, buyer, seller, amount |
| License Plate | `vehicle.plate` | Plate number |
| Table | `table.structure` | HTML, cell coordinates |
| Image to Markdown | `document.image_to_markdown` | Markdown file path |

**Candidate scenarios (need verification):**

Train ticket, taxi invoice, business license, contract, driving license, driver license, express waybill, lab report, etc. Use `onnocr list --candidates` for the full list.

### CLI Usage

```bash
# Install
pip install -e .

# List scenarios
onnocr list
onnocr list --candidates

# View field schema
onnocr schema finance.invoice

# Recognize image
onnocr run finance.invoice <image_path> --pretty

# Real test
onnocr run education.exam_paper onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg --pretty
onnocr run finance.invoice data/samples/invoice_sample.jpg --pretty
```

### Python API

```python
from onnxocr.cli_runtime import OnnxOCREngine, CLIInput, create_default_registry

engine = OnnxOCREngine()
registry = create_default_registry()
cli = registry.get("finance.invoice", engine)
result = cli.run(CLIInput(image_path="sample.jpg"))
print(result.to_dict())
```

**Output example:**

```json
{
  "cli_id": "finance.invoice",
  "fields": {
    "invoice_no": "12345678",
    "invoice_date": "2026-05-15",
    "buyer": "北京示例科技有限公司",
    "amount": "1280.50"
  },
  "confidence": 0.85
}
```

---

## Model Download

Only general OCR models are included by default. Download extension models separately:

```bash
# From HuggingFace (recommended for international users)
python scripts/download_models.py --source huggingface

# From ModelScope (recommended for users in China)
python scripts/download_models.py

# Check local models
python scripts/download_models.py --check-only
```

**Download sources:**
- HuggingFace: https://huggingface.co/jingsongliu/onnxocr_model
- ModelScope: https://www.modelscope.cn/models/supersong/onnxocr_model

---

## API Service

### Start HTTP Service

```bash
python app-service.py
```

**Endpoints:**
- `/ocr` - General OCR
- `/plate` - License plate recognition
- `/table` - Table recognition
- `/layout` - Layout analysis
- `/layout_markdown` - Document to Markdown

### Start WebUI

```bash
python webui.py
```

---

## Docker Deployment

```bash
docker build -t ocr-service .
docker run -itd --name onnxocr-service -p 5006:5005 ocr-service
```

---

## Effect Demonstration

| Example 1 | Example 2 |
|-----------|-----------|
| ![](assets/demo/r1.png) | ![](assets/demo/r2.png) |

| Example 3 | Example 4 |
|-----------|-----------|
| ![](assets/demo/r3.png) | ![](assets/demo/draw_ocr4.jpg) |

---

## Inference Engine Adaptation

All models create ONNXRuntime sessions through `onnxocr/inference_engine.py`. To adapt GPU/NPU providers:

```python
from onnxocr.inference_engine import create_session
```

**Key extension points:**
- `create_session()` - Create inference session
- `build_providers()` - Build inference providers

---

## Project Structure

```
onnxocr/
├── inference_engine.py      # ONNXRuntime entry point
├── onnx_paddleocr.py        # User API entry
├── skill_cli.py             # Vertical OCR CLI
├── cli_runtime/            # CLI runtime
├── predict_det.py           # Text detection
├── predict_rec.py           # Text recognition
├── license_plate.py         # License plate OCR
├── table_recognition.py     # Table recognition
├── layout_recognition.py    # Layout analysis
├── layout_markdown.py       # Document to Markdown
├── models/                  # ONNX models
└── tests/                   # Test files
```

---

## Version History

**2026.05.15**
- Added vertical OCR CLI for structured extraction (train tickets, exam papers, ID cards, bank cards)
- Added Agent integration support

**2026.05.01**
- Added license plate, table recognition, layout analysis, document to Markdown
- Added HTTP API endpoints

**2025.05.21**
- Added PP-OCRv5 models with multi-language support
- Improved recognition accuracy

---

## Contributing

Issues and Pull Requests are welcome!

**Recommended contributions:**
- Add new vertical industry scenarios
- Validate candidate scenarios with real samples
- Improve field extraction rules
- Enhance Agent integration experience

See [docs/skills.md](docs/skills.md) for detailed guidelines.

---

## Community

| WeChat Group | QQ Group |
|--------------|----------|
| ![WeChat](assets/social/微信群.jpg) | ![QQ](assets/social/QQ群.jpg) |

---

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Technical support and model references
- [RapidAI](https://github.com/RapidAI) - RapidTable, RapidLayout, RapidDoc and other excellent projects

---

## License

This project is licensed under the MIT License.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jingsongliujing/OnnxOCR&type=Date)](https://star-history.com/#jingsongliujing/OnnxOCR&Date)

---

**If this project helps you, please consider giving it a Star!**
