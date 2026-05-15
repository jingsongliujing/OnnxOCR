# OnnxOCR 垂直行业 OCR CLI

## 什么时候使用

当用户需要从固定行业单据、证照、表格截图中抽取稳定字段，并希望得到可复现的 JSON 输出时，使用本 CLI。

典型请求：

- “识别这张农产品质检单，提取产品、批次、日期、检测结果。”
- “把这张报销单结构化成 JSON。”
- “识别发票号码、开票日期和价税合计。”
- “把纸质表格转成结构化数据。”
- “批量跑某类入库单，输出统一字段。”

如果用户只是想让大模型描述图片内容，或者做开放式问答，可以不用本 CLI。

## 设计理念

本项目为 Agent 原生调用设计：能力要可发现、可安装、可测试、可复用，并且能被 Claude Code、Codex 等 Agent 工具直接通过命令行调用。

本 CLI 中文优先，英文兼容：

- 场景名称、说明、领域标签以中文为主，方便国内开发者贡献。
- 字段别名保留常见英文写法，方便英文样例、跨境业务和开源用户测试。
- 文档中明确说明英文只是兼容层，不是贡献门槛。

## 安装

在 OnnxOCR 仓库中安装依赖：

```bash
pip install -r requirements.txt
```

如需车牌、表格、版面分析、RapidDoc 等可选模型：

```bash
python scripts/download_models.py
```

如果只是测试字段抽取逻辑，可以直接运行单元测试，不需要下载大模型：

```bash
python -B -m pytest tests/test_skills.py -p no:cacheprovider
```

## Agent 调用

Agent 应优先使用命令行入口：

```bash
onnocr list
onnocr list --candidates
onnocr schema transport.train_ticket
onnocr run transport.train_ticket data/samples/scid_train_ticket.jpg --pretty
```

仓库内置真实中文图片烟测：

```bash
onnocr run education.exam_paper onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg --pretty
```

Python 调用：

```python
from onnxocr.skills import OnnxOCREngine, SkillInput, create_default_registry

engine = OnnxOCREngine()
registry = create_default_registry()
skill = registry.get("transport.train_ticket", engine)
result = skill.run(SkillInput(image_path="data/samples/scid_train_ticket.jpg"))
print(result.to_dict())
```

## 默认启用场景

默认启用的场景必须通过真实图片烟测。

| Skill ID | 中文场景 | 已验证样例 |
| --- | --- | --- |
| `transport.train_ticket` | 火车票 OCR | SCID 火车票样例 |
| `education.exam_paper` | 试卷信息 OCR | 仓库内置中文试卷图片 |
| `vehicle.plate` | 车牌识别 OCR | 仓库内置车牌测试图 |
| `table.structure` | 表格结构化 OCR | 仓库内置表格测试图 |
| `document.image_to_markdown` | 图片转 Markdown | 仓库内置版面分析图片 |

候选场景可通过 `--candidates` 查看和运行。候选场景不是默认承诺能力，贡献和提升规则见 `docs/skills.md` 与 `docs/skill_evaluation.md`：

```bash
onnocr list --candidates
onnocr run transport.taxi_invoice data/samples/scid_taxi_invoice.jpg --pretty --candidates
```

## 输出契约

每个 Skill 返回：

- `skill_id`：稳定 Skill ID。
- `skill_name`：中文 Skill 名称。
- `fields`：最终结构化字段。
- `field_results`：每个字段的值、置信度、来源文本和位置框。
- `raw_text`：底层 OCR 文本。
- `confidence`：Skill 级置信度。
- `metadata`：领域标签、版本和可选原始输出。

示例：

```json
{
  "skill_id": "transport.train_ticket",
  "skill_name": "火车票 OCR",
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

## 为什么不直接用大模型 Agent

大模型 Agent 很适合开放式理解，但垂直 OCR Skill 有不同价值：

- **可复现**：同一模型、同一模板、同一图片输出更稳定，方便批处理和回归测试。
- **可离线部署**：ONNXRuntime 可在内网、边缘设备、国产化环境中运行。
- **成本可控**：批量单据不必每张都调用大模型视觉接口。
- **可审计**：字段来自哪一行、置信度多少、模板规则是什么，都能追踪。
- **可工程化**：字段 Schema、校验、导出映射和单元测试都在代码里。
- **可叠加大模型**：Skill 负责确定性 OCR 和结构化，大模型负责纠错、解释、流程编排和异常处理。

推荐组合方式：先用 OnnxOCR Skill 生成结构化 JSON，再让大模型 Agent 做字段校验、业务判断、系统录入或人工复核建议。

## 新增 Skill 规范

1. 在 `onnxocr/skills/builtin/industry.py` 新增工厂函数。
2. 用 `TemplateSpec` 描述中文名称、中文说明、领域标签。
3. 用 `FieldSpec` 添加字段标签、正则、是否必填和归一化规则。
4. 在 `onnxocr/skills/registry.py` 注册。
5. 在 `tests/test_skills.py` 添加假 OCR 行测试。
6. 文档中补充场景说明和字段意义。

字段命名建议使用英文 snake_case，字段标签和文档使用中文优先。这样既方便国内贡献，也方便 JSON 被下游系统消费。

