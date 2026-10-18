# P8_PEAD_SUE：池内盈余漂移（Stage A，零成本）

登记时间：2026-10-18 17:05 CST。登记后不可修改；更正只追加在文末。

## 问题

在后 40% ADV 池内，业绩公告后的盈余惊喜是否产生可交易的漂移？
与已证伪的 P4_ALPHA（池内截面 ML 打分）不同：本假设赚的是**公告后时间维度**的钱，
不是同一截面上排序的钱；因此不复用 GBDT 分数，也不以 P4_ALPHA 的失败作为先验否证。

## 数据与 PIT

- `data/cloud/financials_pit/20261012`（bundle），字段 `ann_date`、`end_date`、
  `report_type`、`update_flag`、`n_income_attr_p`、`available_at`。
- 可见时点 = `max(available_at, 自身 ann_date 18:00)`，复用
  `quantlab/alpha/features.py` 的 `Fundamentals`（同刻多版本取 `update_flag=0` 原始披露）。
- 同一 `(code, end)` 取**最早可见**版本作为当时已知值；后续重述不回溯。
- 面板：`data/smallcap_v4/20261013T023232Z`（可信 base，字段到 2026-10-12）；
  Stage A 样本 2018-01-02..2026-09-22。

## 信号

- 季度净利 `X`（`n_income_attr_p`，累计值；同一 `end` 季度同比可比）。
- 季节差 `d_t = X_t − X_{t−4q}`；`SUE_t = d_t / sd(d_{t−8..t−1})`，要求 ≥4 个历史季节差且 sd>0。
- 剔除：ST（事件日 `is_st=1`）、金融类（`sw_l1` 银行/非银）、`报告期 end` 不是季末的行。
- 事件日 = 可见时点（18:00）之后的第一个交易日为信号日 `s`；在 `s` 收盘后决策。

## 组合（Stage A 只做零成本 IC/超额检验，不建账本）

- 池：`s` 日的后 40% ADV 池（`universe.pool_on(..., band='relative', exclude_st=True)`），
  且排除 BJ。
- 当日事件集合 = 可见时点落在 `(s−5 交易日, s]` 且属于池的股票；按 SUE 取**前四分位**。
- 持有：信号日收盘入场，持有 20 个交易日（主）；60 日为声明变体（登记时冻结，DSR 计入两者）。
- 组合日收益 = 当日处于持有期的前四分位事件股票等权 `adj_close` 收益；
  对照 = 当日池内全部股票等权 `adj_close` 收益。两者逐日配对。

## 统计与门槛（Stage A）

- 主统计量：日超额（事件组合 − 池 EW）的 Newey-West t，滞后 20。
- 门槛（全部满足才进 Stage B）：
  1. 配对 NW **t ≥ 2.0**；
  2. 事件 ≥ **200** 个、覆盖 ≥ **5** 个财年；
  3. ex-2025 段日超额均值与全样本**同号**；
  4. 两个持有期变体的 BH-FDR **q ≤ 0.10**；
  5. `deflated_sharpe`（`registered_trials=2`）**DSR > 0**。
- 对照：同事件日低惊喜腿（后四分位）与同池随机 50 只；两者都必须弱于高惊喜腿。
- 数据/执行门槛不通过 → `insufficient_data`，不硬跑。

## 结果改变的决策

- 通过：登记 Stage B（16 本账，整手/成本1/2/停牌证据），B 不过不跑 C。
- 不通过：关闭 PEAD 族，不调 SUE 公式、不换持有期、不重跑。
- `promotion_approved=false`。

## 源码

新增 `scripts/p8_pead_stage_a.py`；复用 `quantlab/alpha/features.py` 的
`Fundamentals`、`quantlab/smallcap/universe.py` 的 `rolling_features/pool_on`、
`quantlab/research_statistics.py` 的 `newey_west/deflated_sharpe`、
`quantlab/research_trials.py` 的 `register/record_outcome`。
输出 `output/p8_pead_stage_a_20261018/`。
