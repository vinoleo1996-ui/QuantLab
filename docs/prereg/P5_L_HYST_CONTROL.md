# 预注册：GBDT_HYST 的 matched-turnover 随机对照（P5 归因诊断）

（2026-10-17 23:10 CST 登记。登记后不可修改；更正只追加在文末并记入台账。）

## 依据

- `P4_LOWTURN` 的 `GBDT_HYST` 在成本 1 倍下年化 14.38%、回撤 −33.55%，年单边换手 2.9472 倍，但对 `POOL50` 的 Newey-West t 只有 0.95，未过门槛（`output/p4_lowturn_20261013/report.json`）。
- 它相对随机 50 只的年化优势，可能来自两处：GBDT 排序包含信息，或“进 10%/出 50%”的滞后规则本身降低了换手和成本。现有证据不能把两者分开。
- 本诊断只回答“在相同换手和相同进出规则下，GBDT 排序是否仍优于随机排序”，不提出新策略，不改变 `P4_LOWTURN` 的既有判定。

## 已看过的信息（如实披露）

- 已看到 `GBDT_HYST` 的 14.38%/12.16%、换手 2.9472 倍和 NW t=0.95；对照的随机种子、持久性和门槛在看到这些结果之后才定。因此这是归因对照，不是独立确认，不能用来单独批准策略。
- 随机对照不使用任何收益、回撤或成交数据来选题；只用计划层面的持仓替换率校准持久性参数。

## 计数

新族 `P5_l`，1 次诊断：`HYST_CONTROL`，`confirm_uses=0`。不计入正式策略试验数；DSR 的试验数仍只统计 `role='trial'` 的正式试验。台账单独登记，结果只用于归因。

## 规则（与 `P4_LOWTURN` 对齐）

- 股票池、24 个特征、逐年滚动训练得到的 GBDT 分数、每 20 个交易日调仓、相位 0/5/10/15、50 只、每个申万一级行业最多 12 只、等权合计 90%、80 万、账本与成本，全部与 `docs/prereg/P4_LOWTURN.md` 相同。
- 进出规则：`quantlab/alpha/portfolio.py:select_hysteresis`，entry_frac=0.10、exit_frac=0.50，不搜索。
- 处理组：用修复后的 `quantlab/p4_ledger.py` 重跑 `GBDT_HYST` 的四个相位，成本 1/2；不直接使用修复前的冻结账本，避免退市清算差异混入对照。
- 对照组：把 GBDT 分数整体替换为每只股票的随机分数，其余逐位相同。随机分数为
  `z[c,t] = rho * z[c,t-1] + sqrt(1-rho^2) * eps[c,t]`，
  `eps` 为独立标准正态，按 seed 生成；只使用排序，不使用分数大小。
- 持久性 `rho` 只在 `{0.90, 0.95, 0.98, 0.99}` 中选：用相位 0、seed 1–2 的**计划持仓替换率中位数**，选与 `GBDT_HYST` 的 0.2339048 最接近的一个；平局取更大的 `rho`。校准只看计划成员，不看任何收益。
- 对照路径：相位 0/5/10/15 × seed 1/2，共 8 条；成本 1/2 各跑一遍。
- 样本：2018-01-02 至 2026-09-22；子段 2022-01-04 至 2026-09-22。

## 统计与门槛

- 主统计量：每个相位，把 `GBDT_HYST` 的日收益减去该相位两条随机路径的日中位收益，得到日超额；对相位 0/5/10/15 的日超额取等权平均。
- Newey-West t 值，滞后 20；双侧。
- 循环块 bootstrap，块长 20，10,000 次，双侧 p 值。
- 同时报告：两组的成本 1/2 年化中位数、最大回撤中位数、实际替换率、换手、费用与滑点；随机组的 8 条路径分布。
- 判定：
  1. 数据质量或执行门槛不通过：记 `insufficient_data`，不作显著性结论。
  2. `|NW t| >= 1.96` 且 `bootstrap p <= 0.05`：记 `significant`，说明 GBDT 排序在 matched turnover 下仍有增量；只允许据此再登记新的采纳试验，不直接改策略。
  3. 否则：记 `not_significant`，`GBDT_HYST` 的相对优势不能归因于 GBDT 排序；`P4_LOWTURN` 维持 `no_improvement`，不再调 entry/exit 或随机种子重跑。

## 结果改变的决策

- `significant`：开新的预注册讨论是否把 GBDT 排序用于股票组合；本诊断本身不改变基线。
- `not_significant`：关闭“GBDT 排序带来额外 alpha”的假设，保留 `POOL50`/G97 研究基线；不把 14.38% 当作模型选股能力的证据。
- `insufficient_data`：只修数据/执行，不重选随机种子。
- `promotion_approved=false`。

## 源码与数据身份

- 计划与账本：`scripts/p5_l_hyst_control.py`（运行前落盘）、`quantlab/alpha/portfolio.py:select_hysteresis`、`quantlab/p4_ledger.py`。
- 数据：`output/p4_alpha_20261012/scores.parquet`、panel `20261006T152412Z`、`docs/prereg/P4_LOWTURN.md` 的相位与成本定义。
- 输出：`output/p5_l_hyst_control_20261018/`，不覆盖任何已有账本。

## 结果与更正 1（2026-10-17 23:47 CST）

- 两组账本各 12/12 完成。校准选中 `rho=0.90`，其计划替换率中位数 **0.118**，处理组为 **0.235**；
  预注册的 `rho` 网格没有一条能接近目标，因此这不是同换手对照。按预注册不扩网格、不换种子、
  不为匹配而重跑；只能把它当作偏低换手的保守对照，不能当作 matched-turnover 证据。
- 判定 **`insufficient_data`**：数据质量门槛为 false（处理与对照的 `stale_held_max_days` 均为 451），
  账户恒等式检查通过。按预注册，数据质量不通过时不得作显著性结论。
- 描述性统计（不作结论）：处理组成本1/2 年化中位 15.75%/13.37%、MDD −38.48%/−38.63%；
  随机对照 12.88%/11.41%、MDD −31.60%/−32.08%。相位平均超额 NW t=1.534，
  block bootstrap p=0.128，四个相位 t 均小于 1.96。
- 因为数据质量失败且换手未匹配，既不能宣称“GBDT 排序有增量”，也不能宣称“没有增量”；
  `P4_LOWTURN` 维持 `no_improvement`，不采纳、不改基线、不重跑。证据
  `output/p5_l_hyst_control_20261018/report.json`，台账 `P5_l/HYST_CONTROL#outcome1`。
  `promotion_approved=false`。
