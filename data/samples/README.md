# 测试数据目录

本目录包含用于 OnnxOCR CLI 场景测试的真实图片数据。

## 数据来源

| 文件 | 来源 | 说明 |
|------|------|------|
| `chinese_ocr_sample.jpg` | PaddleOCR 官方示例 | 中文 OCR 测试图 |
| `invoice_sample.jpg` | PaddleOCR 官方示例 | 发票测试图 |
| `id_card_sample.jpg` | PaddleOCR 官方示例 | 身份证测试图 |
| `bank_card_sample.jpg` | PaddleOCR 官方示例 | 银行卡测试图 |
| `business_license_sample.jpg` | PaddleOCR 官方示例 | 营业执照测试图 |
| `table_sample.jpg` | PaddleOCR 官方示例 | 表格测试图 |
| `plate_sample.jpg` | PaddleOCR 官方示例 | 车牌测试图 |
| `receipt_sample.jpg` | PaddleOCR 官方示例 | 收据测试图 |
| `scid_train_ticket.jpg` | SCID 数据集 | 火车票测试图 |
| `scid_taxi_invoice.jpg` | SCID 数据集 | 出租车票测试图 |

## 下载测试数据

```bash
# 下载真实测试数据
python scripts/download_real_test_data.py
```

## 使用说明

这些数据用于 CLI 场景的功能测试和验证。

## 默认场景测试命令

```bash
# 试卷识别
onnocr run education.exam_paper onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg --pretty

# 身份证识别
onnocr run identity.id_card data/samples/id_card_sample.jpg --pretty

# 银行卡识别
onnocr run finance.bank_card data/samples/bank_card_sample.jpg --pretty

# 发票识别
onnocr run finance.invoice data/samples/invoice_sample.jpg --pretty

# 车牌识别
onnocr run vehicle.plate data/samples/plate_sample.jpg --pretty

# 表格识别
onnocr run table.structure data/samples/table_sample.jpg --pretty

# 图片转 Markdown
onnocr run document.image_to_markdown onnxocr/test_images/layout_cdla.jpg --pretty
```

## 候选场景测试命令

```bash
# 火车票识别
onnocr run transport.train_ticket data/samples/scid_train_ticket.jpg --pretty --candidates

# 出租车票识别
onnocr run transport.taxi_invoice data/samples/scid_taxi_invoice.jpg --pretty --candidates

# 营业执照识别
onnocr run business.license data/samples/business_license_sample.jpg --pretty --candidates
```

## 评估结论

详见 `docs/skill_evaluation.md`。
