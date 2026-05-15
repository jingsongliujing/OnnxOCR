# Skill 真实样例评估记录

评估原则：只有真实图片烟测效果稳定的 Skill 才进入默认注册表；效果不稳定或缺少公开样例的模板放入候选注册表。

## 数据来源

- SCID 中文票据样例页：https://davar-lab.github.io/dataset/scid.html
- 下载样例：
  - `data/samples/scid_train_ticket.jpg`
  - `data/samples/scid_taxi_invoice.jpg`
- 仓库内置中文试卷图片：
  - `onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg`
- 完全虚构的脱敏合成样例，仅用于候选场景工程链路烟测：
  - `data/samples/synthetic_prc_id_card_front.png`
  - `data/samples/synthetic_bank_card.png`

## 默认启用

| Skill ID | 样例 | 结果 |
| --- | --- | --- |
| `transport.train_ticket` | `data/samples/scid_train_ticket.jpg` | 成功抽取票号、出发站、到达站、车次、发车时间、席别、座位、票价 |
| `education.exam_paper` | `onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg` | 成功抽取标题、年级、科目、考试时间、满分 |
| `vehicle.plate` | `onnxocr/test_images/license_plate_single_blue.jpg` | 成功抽取 `浙B2V9L7`、`辽DU4356` |
| `table.structure` | `onnxocr/test_images/table.jpg` | 成功输出 HTML、单元格框和逻辑行列坐标 |
| `document.image_to_markdown` | `onnxocr/test_images/layout_cdla.jpg` | 成功生成 Markdown 文件和图片资源目录 |

## 暂不默认启用

| Skill ID | 原因 |
| --- | --- |
| `transport.taxi_invoice` | 能抽发票代码、号码、信息码、公司和上下车时间，但金额被底层 OCR 误识别，暂放候选 |
| `agriculture.*` | 缺少公开真实农产品单据样例验证 |
| `finance.invoice` | 通用发票模板过粗，需按增值税专票、普票、电子发票继续细分并验证 |
| `legal.contract_key_info` | 缺少公开真实合同样例验证 |
| `government.red_head_document` | 缺少公开真实公文样例验证 |
| `logistics.inbound_order` | 和火车票/发票等票据混用效果差，需单独找入库单/面单样例验证 |
| `medical.lab_report` | 涉及隐私和医疗数据，需脱敏公开样例后再验证 |
| `identity.id_card` | 涉及个人隐私；合成样例可抽取出生日期和身份证号，但姓名、性别依赖真实中文识别质量，未进入默认 |
| `finance.bank_card` | 涉及金融隐私；合成样例可抽取完整卡号和有效期，但缺少可复核真实样例，未进入默认 |
| `business.license` | 缺少可确认授权的公开营业执照样例，未进入默认 |
| `logistics.express_waybill` | 涉及地址/电话隐私，当前只保留脱敏/仿真样例单元测试，未进入默认 |
| `vehicle.driving_license` / `vehicle.driver_license` | 缺少可确认授权的公开证照样例，未进入默认 |
| `document.pdf_to_markdown` / PPT / DOCX | 图片转 Markdown 已验证；PDF/PPT/DOCX 还需独立样例和 Office 转换环境验证 |

## 复现命令

```bash
onnocr list
onnocr list --candidates
onnocr run transport.train_ticket data/samples/scid_train_ticket.jpg --pretty
onnocr run education.exam_paper onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg --pretty
onnocr run vehicle.plate onnxocr/test_images/license_plate_single_blue.jpg --pretty
onnocr run table.structure onnxocr/test_images/table.jpg --pretty
onnocr run document.image_to_markdown onnxocr/test_images/layout_cdla.jpg --pretty
onnocr run transport.taxi_invoice data/samples/scid_taxi_invoice.jpg --pretty --candidates
onnocr run identity.id_card data/samples/synthetic_prc_id_card_front.png --pretty --candidates
onnocr run finance.bank_card data/samples/synthetic_bank_card.png --pretty --candidates
```

