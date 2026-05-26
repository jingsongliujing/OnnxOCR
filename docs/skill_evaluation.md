# Skill 真实样例评估记录

评估原则：只有真实图片烟测效果稳定的 CLI 场景才进入默认注册表；效果不稳定或缺少公开样例的模板放入候选注册表。

## 数据来源

- SCID 中文票据样例页：https://davar-lab.github.io/dataset/scid.html
- 下载样例：
  - `data/samples/scid_train_ticket.jpg`
  - `data/samples/scid_taxi_invoice.jpg`
- 仓库内置中文试卷图片：
  - `onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg`
- PaddleOCR 官方示例图片：
  - `data/samples/invoice_sample.jpg`
  - `data/samples/id_card_sample.jpg`
  - `data/samples/bank_card_sample.jpg`
  - `data/samples/business_license_sample.jpg`
  - `data/samples/plate_sample.jpg`
  - `data/samples/table_sample.jpg`

## 默认启用

| CLI 场景 | 样例 | 结果 |
| --- | --- | --- |
| `education.exam_paper` | `onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg` | 成功抽取标题、年级、科目、考试时间、满分 |
| `identity.id_card` | `data/samples/id_card_sample.jpg` | 成功抽取姓名、性别、出生日期、住址、身份证号 |
| `finance.bank_card` | `data/samples/bank_card_sample.jpg` | 成功抽取卡号、银行名称、有效期、卡类型 |
| `finance.invoice` | `data/samples/invoice_sample.jpg` | 成功抽取发票代码、发票号码、开票日期、购买方、金额 |
| `vehicle.plate` | `data/samples/plate_sample.jpg` | 成功抽取车牌号 |
| `table.structure` | `data/samples/table_sample.jpg` | 成功输出 HTML、单元格框和逻辑行列坐标 |
| `document.image_to_markdown` | `onnxocr/test_images/layout_cdla.jpg` | 成功生成 Markdown 文件和图片资源目录 |

## 暂不默认启用

| Skill ID | 原因 |
| --- | --- |
| `transport.train_ticket` | 火车票已电子化，纸质票据场景价值降低，放入候选 |
| `transport.taxi_invoice` | 能抽发票代码、号码、信息码、公司和上下车时间，但金额被底层 OCR 误识别，暂放候选 |
| `agriculture.*` | 缺少公开真实农产品单据样例验证 |
| `legal.contract_key_info` | 缺少公开真实合同样例验证 |
| `government.red_head_document` | 缺少公开真实公文样例验证 |
| `logistics.inbound_order` | 和火车票/发票等票据混用效果差，需单独找入库单/面单样例验证 |
| `medical.lab_report` | 涉及隐私和医疗数据，需脱敏公开样例后再验证 |
| `business.license` | 缺少可确认授权的公开营业执照样例，未进入默认 |
| `logistics.express_waybill` | 涉及地址/电话隐私，当前只保留脱敏/仿真样例单元测试，未进入默认 |
| `vehicle.driving_license` / `vehicle.driver_license` | 缺少可确认授权的公开证照样例，未进入默认 |
| `document.pdf_to_markdown` / PPT / DOCX | 图片转 Markdown 已验证；PDF/PPT/DOCX 还需独立样例和 Office 转换环境验证 |

## 复现命令

```bash
onnocr list
onnocr list --candidates
onnocr run education.exam_paper onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg --pretty
onnocr run identity.id_card data/samples/id_card_sample.jpg --pretty
onnocr run finance.bank_card data/samples/bank_card_sample.jpg --pretty
onnocr run finance.invoice data/samples/invoice_sample.jpg --pretty
onnocr run vehicle.plate data/samples/plate_sample.jpg --pretty
onnocr run table.structure data/samples/table_sample.jpg --pretty
onnocr run document.image_to_markdown onnxocr/test_images/layout_cdla.jpg --pretty
onnocr run transport.train_ticket data/samples/scid_train_ticket.jpg --pretty --candidates
onnocr run transport.taxi_invoice data/samples/scid_taxi_invoice.jpg --pretty --candidates
onnocr run business.license data/samples/business_license_sample.jpg --pretty --candidates
```

