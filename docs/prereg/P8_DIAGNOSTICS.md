# P8 零成本诊断：解禁否决、行业条件反转、权重/集中度

登记时间：2026-10-18 18:55 CST。诊断不计 confirmatory、不进 DSR 家族；不产生策略候选。

## D1 解禁否决（风险移除，不是 alpha）

- 事件：`share_float_pit`（按"股"）中 `float_date ∈ (d, d+90]` 且 `float_ratio ≥ 5%`，
  只在 `available_at ≤ d` 时可见。
- 样本日：每 20 个 session 一个；池 = 后 40% ADV 池。
- 统计：同日池内"未来 90 日有解禁"vs"无解禁"的 20 日前向收益差，NW t（滞后 20）。
- 判定：覆盖率 <8% → `no_action`（截面太薄）；否则 |t|≥2 且暴露组更差才记"值得否决层"。

## D2 行业条件反转

- 样本日：每 20 个 session；池内每个行业取成员过去 20 日收益均值，分高/低三分位。
- 统计：高三分位成员 vs 低三分位成员的 20 日前向收益差，NW t。
- 判定：|t|≥2 且 ex-2025 同号 → 记"值得独立假设"；否则关闭。

## D3 权重/集中度

- 样本日：每 20 个 session；用 `scores.parquet` 的 GBDT 分数取前 50（分数仅作排序，
  不声称 alpha，P5_MATCHED 已判其 t=1.03）。
- 比较：分数加权 vs 等权（同为前 50）、前 30/40/50 等权 vs 池 EW 的 20 日前向收益差。
- 只报告数字与 NW t，**不跑账本、不作候选**；若只提高同一倾斜暴露，明确标注"口径非增益"。

## 输出

`scripts/p8_zero_cost_diag.py` → `output/p8_zero_cost_diag_20261018/report.json`；
`promotion_approved=false`。
