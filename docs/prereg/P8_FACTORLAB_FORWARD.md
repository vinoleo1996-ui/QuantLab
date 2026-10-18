# P8_FACTORLAB_FORWARD：FactorLab 候选首次真跑（历史筛选 + 前向流初始化）

登记时间：2026-10-18 17:45 CST。登记后不可修改。

## 问题

`output/factor_lab/candidates.jsonl` 的 10 条 LLM 候选自 2026-10-12 起一直停在 `proposed`，
本仓唯一自带在线多重检验的因子发现通道从未真正运行。本试验只做两件事：
(1) 用冻结的 `referee.screen` 对 10 条候选做历史筛选（2015–2021 screen，2022–2026 holdout）；
(2) 用已有前向数据初始化 `ForwardStream`，不修改 DSL、不新增候选。

## 口径

- 面板 `20261013T023232Z`；池 = 后 40% ADV 池（与策略一致）。
- 标签 = 未来 20 个 session 的开盘到开盘收益（`open` 已按复权尺度换算）。
- 观测日 = 每 20 个 session 一个（非重叠），与 `referee.screen` 的 `observation_days` 一致。
- 库相关：用现有价量因子库（ret20/vol20/turn20/amihud/close_to_high/hl_range）计算最大相关。
- 候选清单与公式在登记时冻结，sha256 记录在报告里。

## 门槛

- 历史筛选沿用 `referee.screen` 的冻结门槛：|screen IC|≥0.02、screen t≥3.0、
  holdout 同号且 t≥2.0、与现有库最大相关 <0.5。
- 通过者只获得"进入前向流"的资格；前向 admission 仍由 `online_ebh` 决定（α=0.10、slots=100）。
- 前向数据不足时如实记 `forward_pending`，不得用历史数据冒充前向。

## 结果改变的决策

- 有候选通过历史筛选：写入前向流并按日更新；不直接进账本。
- 全部不通过：关闭这批候选，不重跑 DSL、不改标签或池。
- 与已证伪族同构（Amihud/TURN/IVOL/REV/ATURN/MAX/TREND）的候选即使过筛也标 `same_family`，不作候选。
- `promotion_approved=false`。

## 源码

`scripts/p8_factorlab_forward_run.py`；输出 `output/p8_factorlab_forward_20261018/`。
