# 真实样例数据

本目录用于本地烟测，不建议放入客户真实单据。

当前下载的公开样例来自 SCID 中文票据数据集页面：

- https://davar-lab.github.io/dataset/scid.html
- `scid_train_ticket.jpg`
- `scid_taxi_invoice.jpg`

本目录还包含两张完全虚构的脱敏合成样例，用于验证候选 CLI 场景的工程链路，不代表真实样例验证已经通过：

- `synthetic_prc_id_card_front.png`
- `synthetic_bank_card.png`

复现命令：

```bash
onnocr run transport.train_ticket data/samples/scid_train_ticket.jpg --pretty
onnocr run transport.taxi_invoice data/samples/scid_taxi_invoice.jpg --pretty --candidates
onnocr run identity.id_card data/samples/synthetic_prc_id_card_front.png --pretty --candidates
onnocr run finance.bank_card data/samples/synthetic_bank_card.png --pretty --candidates
```

评估结论见 `docs/skill_evaluation.md`。

