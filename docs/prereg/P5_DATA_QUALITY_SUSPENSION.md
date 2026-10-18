# 预注册：P5 数据质量——停牌 vs 数据缺口、ETF 缺失日

（2026-10-18 CST 登记。先登记规则，再改代码；登记后不可修改，更正只追加在文末。）

## 范围

- 适用对象：`P5_BASE` 整手账复评（`scripts/p5_base_lot_replay.py`，2018-01-02 至 2026-09-22）。
- 只改数据质量分类、审计字段和门槛判定；不改信号、目标、成本、权重、打新或样本区间。
- 冻结目录 `output/p5_base_lot_20261018/` 不覆盖；复评写入 `output/p5_base_lot_20261019/`。

## 证据源

- **股票主证据（本会话可验证）**：`data/local_supplement/raw/<code>_daily.csv` 及其
  `<code>_daily.json` sidecar。必须经 `quantlab/data/local_adapter.py:verified_csv`
  校验 sidecar `sha256` 与文件字节一致；字段至少包含
  `date/tradestatus/volume/amount/close/preclose`。
- **`suspend_d`**：计划记录中数据负责人已在 promax 验证 `sz.000616`
  2018-01-24..2018-07-24 与本地 `tradestatus` 一致。本会话没有 promax 凭据
  （`PCD_API_KEY` 未设置），因此 `suspend_d` 本地缓存不存在时**不伪造该证据**，
  以 raw cache 四条件为主证据；`suspend_d` 只在本地有缓存时作为交叉验证。
- **覆盖披露**：本地 raw cache 有 5,419 个代码；panel 有 6,080 个代码，
  661 个代码无 raw cache。held 且发生 stale 的代码若缺 raw，按 fail closed
  归 `unexplained`；仅“曾经持有”但从未 stale 的代码不因缺 raw 单独判失败。
- **ETF**：`data/smallcap_assets/raw/` 的腾讯上游 JSON（每年三次独立归档）
  与 2026-10-18 现场查询的东方财富公开日线（无凭据）。两者对
  `sh511260` 2018-08-17、`sh513100` 2022-01-13 均无 bar；两日均为交易日、
  非季末、前后一日均有 bar，且 ETF 袖当日无成交。

## 股票停牌判定（run 级）

一个 held stale run（连续 replay session 中持有且无有效收盘价）只有在**每个
session 同时满足**以下条件时才归为 `suspended`：

1. raw cache 有该 session 的记录，且 sidecar 完整性校验通过；
2. `tradestatus == 0`；
3. `volume == 0` 且 `amount == 0`；
4. `close == preclose`（绝对误差 ≤ 1e-6，且两者均非空）；
5. run 的最后一个 session 严格早于该代码的 `delist_date`（无退市记录视为满足）；
6. run 完全落在 sidecar 声明的 `requested_start..requested_end` 内。

任一条不满足（两源冲突、raw 文件缺失、单行缺失、四条件不满足、覆盖不完整、
跨退市日），**整个 run** 归 `unexplained`，fail closed。

- 5 个 session 的阈值不变；不因分类放宽。
- `stale_held_max_days` 保留为全部 stale run 的最大长度；
  另记 `suspended_held_max_days` 和 `unexplained_stale_held_max_days`。
- `suspension_coverage_complete` 为真当且仅当每个 stale held session
  都有经校验的 raw 证据行；缺行或缺文件即为假。
- 退市清算仍由 ledger 处理；跨退市日的 held stale run 不归停牌，
  按本规则记 `unexplained`（若存在）。

## ETF 缺失日判定

一个 ETF 缺价日只有在**全部满足**时才归 `stale_mark`：

1. 不是季末/调仓日，且 ETF 袖当日没有成交；
2. 腾讯 raw 上游 JSON 与东方财富公开日线均无该 bar；
3. 单日孤立：前一交易日和后一交易日都有 bar；
4. 无公司行动/折算导致的价格断点。

不满足任一条，或缺失落在调仓/成交日，归 `blocked`，不得用 ffill 价格成交。
非调仓日允许按最后收盘盯市，但必须记 `etf_missing_dates`、
`etf_missing_on_rebalance`、`etf_stale_mark_days`。

## 门槛

- `data_quality_ok`：account checks 通过、无 missing panel、
  `unexplained_stale_held_max_days <= 5`、`suspension_coverage_complete`、
  无未结算退市、无 ETF `blocked`。
- `worst_path_ok = min(blended_mdd) >= -0.25`，成本 1 倍和 2 倍分别检查。
- `gates_all_pass = data_quality_ok and arithmetic_pass and worst_path_ok`。
- `P5_BASE` 是诊断/复现，不是独立确认；状态只能是 `no_improvement`
  或 `insufficient_data`，**永远不能是 `passed`**；`promotion_approved=false`。

## 结果改变的决策

- 数据分类完整且仅 worst-path 失败：记 `no_improvement`，不进入纸面前向。
- 仍有 `unexplained` 或 ETF `blocked`：记 `insufficient_data`，10-30 前向记
  `blocked`，不硬生成订单。
- 不调门槛、不换样本、不调参救结果；复评 NAV 若与冻结不一致，单独列出差异，
  不静默覆盖历史。

## 复评一致性

- 分类只影响审计字段和门槛，不应影响成交；32 个 path（8 路径 × 2 资本 × 2 成本）
  的 CAGR/MDD 应与冻结一致（容差 1e-10）。
- 19 笔退市清算、missing panel 0 应保持不变。
- ETF 去 ffill 成交后若 NAV 改变，逐格列出；已知两日不是调仓日，预期 NAV 不变。

## 更正

### 更正 1（2026-10-18 CST）：停牌行 volume/amount 的 NaN 编码

原始规则第 3 条字面要求 `volume == 0 且 amount == 0`。实现后对 5,419 个 raw
文件做编码审计发现：`tradestatus == 0` 的 319,510 行中有 14,987 行（4.69%）
把零成交编码为 `volume`/`amount` 均为 `NaN`（二者成对出现，无单边缺失）；
这些行的 `close == preclose`，且 319,501 行 OHLC 四值完全相等。若按字面
拒绝 NaN，会把真实停牌误判为“当日有成交”，使整段 run 变成 `unexplained`
（p10_s2 冒烟中 sh.600723、sh.603318、sz.002627、sh.600193 即因此触发）。

澄清后的第 3 条：`volume` 与 `amount` 各自为 0 或为空（NaN）即满足。
其余条件不变：`tradestatus == 0` 仍是必要条件，`close == preclose` 仍要求
两者非空且误差 ≤1e-6。`tradestatus == 1` 的行永远不判停牌，即使 volume 为空
（全库仅 1 例：sh.600656 2016-05-12）。缺失 volume/amount 本身不构成停牌证据。

此更正只修正零值的存储编码，不改变阈值、样本、信号、成本或门槛，也不基于
复评绩效；它发生在正式 32-path 复评登记之前。原规则文本与哈希保留于本文件
上半部分，本文件最终 sha256 作为复评报告记录的有效预注册哈希。

### 更正 2（2026-10-18 CST）：补充证据缓存覆盖到复评终点

主 raw cache 的 sidecar 声明区间为 2014-01-02..2026-06-30（2026-09-24 抓取），
而复评终点为 2026-09-22。若 held stale run 落在 2026-07-01 之后，主缓存只能
判 `outside_coverage`，会把真实停牌误判为证据不足。smoke 复核发现两类缺口：

- 主缓存完全无 daily 文件的 held 代码（9 个）；
- 主缓存到 2026-06-30 截止、但 2026-07-01..09-22 仍有 held stale 的代码。

为使覆盖与复评窗口一致，2026-10-18 用同一 baostock 公共接口、同一字段、
同一 `adjustflag=3` 抓取 19 个相关代码的 2014-01-02..2026-09-22 日线，写入
独立目录 `data/local_supplement/raw_ext_20261018/`，每个文件都有 sidecar
（`requested_start/requested_end/rows/sha256/fetched_at`）并经 `verified_csv`
校验。主目录 `data/local_supplement/raw/` 未被修改。

解析顺序：补充缓存优先，主缓存兜底；补充文件存在但校验失败时 **不** 回退到
主缓存，按 `unexplained` fail closed。10 个代码与主缓存重叠区间共 24,949 行，
逐字段字符串比对 0 处不一致；审计见
`output/p5_base_lot_20261019/suspension_evidence_extension.json`。

该更正只补证据覆盖，不改变停牌四条件、5 日阈值、样本、信号、成本或门槛。
2026-07-16..22 的 sh.688237、2026-07-10..16 的 sz.002677、2018-04-12 的
sz.000655 均由补充缓存确认为 `tradestatus=0` 且平盘；东方财富公开日线同样
显示这些日期无 bar、前后交易日有 bar，仅作旁证，不作为主证据。
