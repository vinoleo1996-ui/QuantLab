# P5_MATCHED：同换手随机滞后对照（模型线的生死判据，2026-10-17 23:55 CST）

## 问题

`P4_LOWTURN` 的 GBDT_HYST 相对 POOL50 的 +2.27pp 把两件事混在一起：GBDT 分数与
"进出缓冲带"规则。POOL50 是日历换仓（1.30x），HYST 是 entry 10% / exit 50%（2.95x），
所以这 +2.27pp 不能归因于模型。外部 review 要求做匹配对照。

## 设计

同一条流水线、同一个选择器 `select_hysteresis`、同一 entry/exit 比例（10%/50%）、
同样 50 只、同一行业上限（按 mandate 25% 权重推导）、同一 800k 股数账本、同样成本，
**唯一区别**：排序用固定种子随机（`random_order(seed)`），不用 GBDT。

- 相位 0/5/10/15 × 种子 1/2 = 8 本；成本 1 倍与 2 倍都跑。
- 对照：`output/p4_lowturn_20261013/ledger/nav_GBDT_HYST_p{phase}_cost{cost}.csv`（同相位）。
- 统计量：月度收益差（HYST − 随机）在 4 个相位上池化，Newey-West t（lag 3）。

## 门槛

- **GBDT_HYST 必须同时满足**：成本 1 倍年化中位数高于随机对照 ≥ +1.0pp，且 NW t ≥ 2.0。
- 否则模型线关闭，"低换手增益"归给缓冲带规则本身（方案甲的经济学解释不变，alpha 归零）。

## 判定影响

- 通过：模型线作为 alpha 候选进入前向，GBDT_HYST 成为股票腿候选。
- 不通过：关闭模型线（P4_ALPHA/P4_LOWTURN 一并记为规则增益，不是 alpha）；
  P5_BASE 维持 POOL50_G97 随机 50 只。

`promotion_approved=false`。证据：`output/p5_matched_20261017/report.json`。
