# P6：财务质量由排序键升级为入池硬闸门（Stage A，零成本池级）

登记时间：2026-10-18 17:30 CST（由 `docs/prereg/drafts/P6_MULTI_CRITERIA_POOL_DRAFT.md` 转正）。
登记后不可修改；更正只追加。

## 问题

把财务档（最新已知净利 ≥0 且 0<PB≤3）从**排序键**改为**入池硬闸门**，是否能在不改变
POOL50 其余规则（50 只、97% gross、行业 12、20 日网格、相位 0/5/10/15、种子 1/2）的前提下，
提高池的收益或降低回撤？

与 P2_QUALITY_DIVIDEND 的区别：P2 是全市场质量+股息优选且财务非 PIT；本试验只在
后 40% ADV 池内做一个**排除门**（不可投诉的尾部剔除），不是优选倾斜，也不使用股息。

## 数据与 PIT

- 财务：`data/cloud/financials_pit/bundle_20261012/part-all-all.parquet`，
  可见时点 = `max(available_at, ann_date 18:00)`，复用 `Fundamentals`；取每日最新已知版本。
- 估值：panel `20261013T023232Z` 的 `pb_mrq`（当日收盘可知）。
- 缺失财务或缺失 PB 一律**不合规**（fail-closed），并报告覆盖率；覆盖率 <70% 的日子记
  `insufficient_data`。

## 门与池

- 合规 = 最新已知 `n_income_attr_p ≥ 0` 且 `0 < pb_mrq ≤ 3`。
- 池：`pool_on(..., band='relative', exclude_st=True)` 的同一分位规则，但只对合规名字
  计算 bottom-40%；BJ 排除。为可复现，Stage A 用向量化实现，并与 `pool_on` 在 20 个
  随机交易日抽查重叠率（要求 ≥99%）。

## Stage A（零成本，不建账本）

- 两个等权日频组合：合规池 EW vs 未加门池 EW（均用 `adj_close`，前一日成员决定当日收益）。
- 主统计量：配对日超额（合规 − 未加门）的 Newey-West t，滞后 20。
- 门槛（全部满足才登记 Stage B）：
  1. NW **t ≥ 2.0**；
  2. ex-2025 段与全样本**同号**；
  3. 合规覆盖率中位数 ≥70%；
  4. 合规池的 Calmar 不低于未加门池（按 MANDATE_V2 同时报 Calmar 与 MDD）。
- 报告：两池的 CAGR/MDD/Calmar、逐年、日均换名比例、门剔除比例。

## 结果改变的决策

- 通过：登记 Stage B（16 本账，整手/成本1/2/停牌证据），并按 50 只随机合规池对照。
- 不通过：关闭该门，不再调 PB/净利阈值。
- `promotion_approved=false`。

## 源码

`scripts/p6_pool_gate_stage_a.py`；输出 `output/p6_pool_gate_stage_a_20261018/`。
