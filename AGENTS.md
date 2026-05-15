# OnnxOCR Agent 使用指南

本仓库提供可被 Claude Code、Codex 等 Agent 工具直接调用的垂直 OCR CLI。

## 快速判断

当用户要处理固定行业单据、证照、表格或希望输出稳定 JSON 字段时，优先使用 CLI：

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

真实评估记录见 `docs/skill_evaluation.md`。默认 `list` 只展示真实烟测通过的场景；`--candidates` 才展示候选模板。

如果用户只是想看图片内容或做开放式理解，可以直接使用大模型视觉能力；如果用户需要批量处理、字段稳定、离线部署、可审计和可复现，使用 OnnxOCR CLI。

## 推荐工作流

1. 用 `onnocr list` 查看可用场景。
2. 用 `schema` 查看字段定义和中英文字段别名。
3. 选择最接近的场景运行图片。
4. 如果字段为空但 `raw_text` 有内容，说明模板不匹配或字段别名不足，应该扩展 `FieldSpec`。
5. 新增场景时，先写 `tests/test_skills.py` 的假 OCR 行测试，再注册模板。

## CLI 目录

- `skills/industry-ocr/SKILL.md`：给 Agent 阅读的中文优先 CLI 使用说明。
- `docs/skills.md`：给开发者阅读的扩展文档。
- `onnxocr/skills/builtin/industry.py`：内置垂直行业模板。
- `onnxocr/skills/registry.py`：模板注册中心。

## 约束

- 不要把客户真实单据、个人隐私图片、生产密钥提交到仓库。
- 不要把行业规则写进底层 ONNX 推理模块，行业规则应留在 CLI 模板层。
- 表格、车牌等可选能力需要先下载对应模型。

