# OnnxOCR 垂直行业 CLI 开发指南

OnnxOCR 提供一层可扩展的垂直 OCR CLI。它面向固定行业模板，不替代通用 OCR，而是在通用 OCR 之上补齐模板选择、字段抽取、归一化、校验、置信度和导出映射。

设计目标是让能力可发现、可安装、可测试，并能被 Claude Code、Codex 等 Agent 工具通过命令行直接调用。

本项目中文优先，英文兼容：

- Skill 名称、说明、领域标签、贡献文档以中文为主。
- 字段 ID 使用英文 snake_case，便于 JSON、数据库、API 和下游系统消费。
- 常见英文标签作为兼容别名，方便开源用户和跨境业务测试。

## 目录结构

```text
AGENTS.md                       # Agent 进入仓库后的操作指南
skills/
  README.md                     # Skill 索引
  industry-ocr/SKILL.md         # Agent 可读的中文 Skill 说明
onnxocr/
  skill_cli.py                  # Skill 命令行入口
  skills/
    base.py                     # 标准输入输出模型和基类
    engine.py                   # ONNXPaddleOcr 懒加载适配器
    extractors.py               # 标签和正则抽取工具
    registry.py                 # Skill 注册中心
    template_skill.py           # 模板型 OCR Skill
    builtin/industry.py         # 内置垂直行业 Skill
tests/test_skills.py            # 不依赖大模型的 Skill 单元测试
```

## 默认启用场景

默认启用的 Skill 必须通过真实图片烟测。

| Skill ID | 中文场景 | 主要字段 |
| --- | --- | --- |
| `transport.train_ticket` | 火车票 OCR | 票号、出发站、到达站、车次、发车时间、座位、席别、票价 |
| `education.exam_paper` | 试卷信息 OCR | 标题、年级、科目、考试时间、满分 |
| `vehicle.plate` | 车牌识别 OCR | 车牌列表、原始模型结果 |
| `table.structure` | 表格结构化 OCR | HTML、单元格框、逻辑行列坐标 |
| `document.image_to_markdown` | 图片转 Markdown | Markdown 文件、资源目录、预览文本 |

## 候选 Skill

候选 Skill 需要通过 `--candidates` 查看和运行，不能视为默认可用能力。

| Skill ID | 中文场景 | 主要字段 |
| --- | --- | --- |
| `agriculture.quality_inspection` | 农产品质检单 OCR | 产品、批次、检测日期、结果、供应商 |
| `agriculture.traceability_label` | 农产品溯源标签 OCR | 追溯码、产品、产地、生产者、采摘日期 |
| `agriculture.plant_protection_record` | 植保作业记录 OCR | 作物、农药、用量、作业人、作业日期 |
| `oa.reimbursement` | 企业报销单 OCR | 报销人、部门、金额、日期、用途 |
| `finance.invoice` | 发票 OCR | 发票代码、发票号码、开票日期、购方、销方、金额 |
| `finance.bank_card` | 银行卡 OCR | 卡号、银行名称、卡组织/卡类型、有效期 |
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

## 命令行

```bash
onnocr list
onnocr list --candidates
onnocr schema transport.train_ticket
onnocr run transport.train_ticket data/samples/scid_train_ticket.jpg --pretty
```

仓库内置中文图片烟测：

```bash
onnocr run education.exam_paper onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg --pretty
```

## Python API

```python
from onnxocr.skills import OnnxOCREngine, SkillInput, create_default_registry

engine = OnnxOCREngine()
registry = create_default_registry()
skill = registry.get("transport.train_ticket", engine)
result = skill.run(SkillInput(image_path="data/samples/scid_train_ticket.jpg"))
print(result.to_dict())
```

## 输出契约

所有 Skill 都接收 `SkillInput`，返回 `SkillOutput`。

`SkillInput` 支持：

- `image`：OpenCV 解码后的图片。
- `image_path`：本地图片路径。
- `ocr_lines`：预先计算好的 OCR 行，便于测试和上游流水线复用。
- `options`：Skill 自定义选项。

`SkillOutput` 包含：

- `skill_id`：稳定 ID，例如 `logistics.inbound_order`。
- `skill_name`：中文名称。
- `fields`：最终结构化字段。
- `field_results`：字段值、置信度、来源文本、位置框。
- `raw_text`：抽取所用 OCR 文本。
- `confidence`：Skill 级置信度。
- `metadata`：版本、领域标签、可选原始输出。

## 和直接用大模型 Agent 的区别

大模型视觉 Agent 很强，但垂直 OCR Skill 解决的是工程化问题：

- **可复现**：规则、字段、置信度可固定，适合批量和回归测试。
- **可离线**：ONNXRuntime 可在内网、边缘设备和国产化环境运行。
- **低成本**：大批量单据不必全部调用大模型视觉接口。
- **可审计**：每个字段可以追踪来源文本和置信度。
- **可维护**：字段 Schema、校验和导出映射在代码中沉淀。
- **可组合**：Skill 做确定性结构化，大模型做纠错、解释、流程编排。

推荐架构：

```text
图片/PDF -> OnnxOCR -> 垂直 Skill -> 结构化 JSON -> Agent 复核/录入/业务流转
```

## 新增 Skill

1. 在 `onnxocr/skills/builtin/industry.py` 新增工厂函数。
2. 使用 `TemplateSpec` 填写中文名称、中文说明和领域标签。
3. 使用 `FieldSpec` 定义字段：
   - 字段 ID 用英文 snake_case。
   - 字段标签中文优先，必要时补充英文别名。
   - 身份证号、单号、日期、金额等用正则约束。
   - 核心业务字段设置 `required=True`。
4. 在 `onnxocr/skills/registry.py` 注册。
5. 在 `tests/test_skills.py` 添加不依赖模型文件的假 OCR 行测试。
6. 在 `docs/skills.md`、README 表格和评估记录中补充场景。
7. 只有通过公开或授权脱敏真实样例烟测后，才能从候选注册表提升到默认注册表。

## 质量标准

- 不提交客户真实单据、隐私图片、生产密钥。
- 不把行业规则写进底层 ONNX 推理模块。
- 模板不匹配时返回低置信度和空字段，而不是直接异常。
- 可选大模型文件缺失时，单元测试仍应能跑。
- 新增场景至少有一个中文样例测试。

