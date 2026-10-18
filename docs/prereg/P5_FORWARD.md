# 预注册：P5_BASE 纸面前向

（2026-10-17 CST 登记。登记后不可修改；只记录，不实盘。）

## 对象

`P5_BASE`：股票 65% × `POOL50_G97` 账本（97% 仓位）+ 35% G0 袖（黄金 518880 13.125 / 十年国债 511260 13.125 / 纳指 513100 8.75，3:3:2，季末再平衡）+ 23% 灾难线；80 万；打新 1.5% 封顶。

## 起点与频率

- 2026-10-30 首次快照（月末收盘），此后每月末。
- 记录：组合净值、股票/ETF 权重、与 `ALLOC60` 的累计差、当月最大回撤。
- 数据：当日收盘权重与净值；不回溯修改。

## 停止条件

- 6 个月：`P5_BASE` 累计落后 `ALLOC60` 超过 3pp → 停，记为前向反证。
- 否则继续记到 12 个月，再按 12% / 25% / 10% 三门槛重判。

## 边界

- 只记录，不实盘；`promotion_approved=false`。
- 不改股票规则、袖权重、灾难线；任何改动都是新试验。
- 回测证据：`output/p5_g0_20261017/report.json`（13.19% / −22.78% / 含打新 14.69%）。

## 源码

`quantlab/p4_alloc.py` 的 `allocate` + G0 `target_weights`；`scripts/p4_alloc.py`；股票账本 `output/p4_gross_20261013/ledger/POOL50_G97`。

## 更正一（2026-10-17 20:35 CST，转基线 + 撤销灾难线）

- 组合层主线从旧 `ALLOC60` 切换到 `P5_BASE`：股票 65% × `POOL50_G97` 账本 + 35% G0 袖（黄金 518880 13.125 / 十年国债 511260 13.125 / 纳指 513100 8.75，3:3:2，季末再平衡）。
- **撤销 23% 灾难线**：实测 8 条路径 0 触发，规则惰性；改为不设事后回撤砍仓。证据 `output/p5_g0_20261017/report.json` 的 `G97_65_noline_q`（13.19% / −22.78% / 成本 2 倍 12.33%，含打新 14.69 / 13.83）。
- QDII 溢价暂停的溢出额度按 1:1 分给黄金与十年国债——下一版代码实现；当前版全额给国债，差异 <0.1pp。
- 首次快照 2026-10-30，此后每月末；停止条件不变（6 个月累计落后旧 `ALLOC60` 超过 3pp 记前向反证）。

## 更正二（2026-10-17 22:45 CST，快照管线落地）

- 快照入口改为 `scripts/p5_snapshot.py --as-of YYYY-MM-DD --capital 800000`；旧 `scripts/p4_forward.py` 只服务已撤销的 P4_ALLOC，现于 CLI 显式退出，不再用于 P5_BASE。
- 快照状态：`output/p5_forward/state.json`；历史：`output/p5_forward/history.csv`；单期：`output/p5_forward/<as_of>/snapshot.json`。`--dry-run` 只写 `output/p5_forward/dryrun/<as_of>/`，不推进状态。
- 若请求日尚无面板数据，快照截断到最后一个有数据的交易日，并同时记录 `requested_as_of` 与 `as_of`；不得用缺失数据生成订单或收益。
- 订单执行：下一交易日开盘价 + 30bp 滑点，整手买入、佣金/印花税、现金非负；`status` 只能为 `paper_simulated` 或 `broker_confirmed`。券商成交记录须用 `--holdings/--cash/--broker-confirmed` 导入，与纸面记录分开。
- 对照 `ALLOC60` 使用冻结的 `output/p4_alpha_20261012/ledger/nav_POOL50_p0_s1_cost1.csv`、60% 股票袖、月度再平衡；配置哈希写入快照，便于复核。
- 2026-09-22 dry-run 仅用于管线验证（46 只股票订单、2 只 ETF、4 只低于 1 手、QDII 暂停），不作为前向观测；管线测试期间写入的 pre-start 状态已移入 `output/p5_forward/_discarded_dryrun_20261017/`，不代表已开始的记录。
- 首次真实快照仍为 2026-10-30，若届时数据未更新则记 `blocked`，不伪造。

## 更正三（2026-10-18 13:00 CST，重放式前向引擎与 blocked 门禁）

- **引擎改为重放**：每次快照从 2018-01-02 重放两条袖——股票袖 `p4_ledger.replay`（52 万、97% gross、冻结 `POOL50_G97_p0_s1` 计划）+ ETF 袖 `run_g0_lot_sleeve`（28 万、整手、3:3:2、季末再平衡）——再按冻结规则季度混合。不再用 raw close 盯市或增量状态推算净值。
- **状态分离**：纸面 `output/p5_forward/state_paper.json`，券商 `output/p5_forward/state_broker.json`；历史 `history.csv`，单期 `output/p5_forward/<as_of>/snapshot.json`。状态只记录历史、配置哈希和重放账本引用，不再作为定价输入。
- **blocked 取代截断**：更正二“截断到最后一个有数据的交易日”作废。`requested_as_of` 超过生产账本最后完整交易日、或不是交易日时，状态只能为 `blocked`：写 snapshot、`orders=[]`、`nav=null`，不推进 state，不生成收益。数据质量门禁失败（无法解释的停牌、缺 panel、未结算退市、ETF 再平衡日缺价）同样 `blocked`。
- **订单口径**：快照只报账本中仍为 `pending/partial` 且剩余股数大于零的订单，执行日为下一交易日开盘；已成交、已过期、已撤销不重复报单。本期快照不预扣现金/股份，下一期快照重放到执行日时由账本成交并计入 `filled_now`。订单股数为 52 万股票袖口径，组合权重按重放混合后的 scale 报告，不把袖内股数当作 80 万账户的整手指令。
- **parity**：2026-09-22 重放 `POOL50_G97_p0_s1` 得 **13.5248% / −23.77%**，与冻结 `output/p5_base_lot_20261018/runs/cap800000_cost1.csv` 的 ΔCAGR **8.3e-15pp**、ΔNAV **0.0**，容差 |ΔCAGR| ≤0.20pp、|ΔNAV| ≤0.20%；证据 `output/p5_forward/parity/2026-09-22/snapshot.json`。该对照不是独立验证。
- **10-22 / 10-30**：两者均不是 `p0_s1` 的 20 日网格日；phase 0 最后信号 2026-09-28，下一次 2026-11-02。`--prestart` 只做日历检查（`output/p5_forward/prestart/<date>/snapshot.json`），数据未覆盖时 `data_status=not_available`，不产生订单。
- **未来网格**：prestart 的未来日期用工作日近似并标 `calendar_provisional=true`，只用于显示 `next_grid`，不得据此下单或记收益。


## 更正四（2026-10-18 16:10 CST，季末基准、对照腿与两处门禁）

- **季末基准固定为全历史日历**：前向引擎调用 `run_g0_lot_sleeve`/`run_one`/`A.allocate` 时
  一律传 `basis='calendar'` + `dataset.calendar`，季末日不再随 `--as-of` 移动。
  parity 因此改为对齐 calendar 口径的 `output/p5_base_lot_20261018_r1/`
  （**13.5248666% / −23.7699395% / NAV 2,418,168.3196**，ΔNAV 0.0），
  同时保留 window 口径的 `20261019` 作历史参考并标 `quarter_basis`。
- **信号日只报订单来源的日程**：快照的 `last_grid/next_grid/signal_today` 改由
  `frozen_schedule` 的 `days[phase//5::4]` 给出；`decision_grid(dataset.calendar,…)`
  降级为交叉校验，两者不一致时写 `calendar_mismatch` 并判 `blocked`，
  不再出现“报一个日、下另一个日的单”。
- **六个月对照腿更正**：`alloc60_series` 原先用重放出来的 97% gross `POOL50_G97` 股票腿
  构造 `ALLOC60`，等于用被测组合自己当对照。现按更正二第 43 行回到预注册定义：
  股票腿取冻结 90% gross `output/p4_alpha_20261012/ledger/nav_POOL50_p0_s1_cost1.csv`
  （sha256 记入快照 `alloc60.config.frozen_equity_sha256`）、60% 敞口、月度再平衡，
  月度日用 `basis='calendar'`。该更正只影响停止条件对照，不改变组合本身规则。
- **blocked 再加两道防线**：`run()` 在 `status=blocked` 时强制 `orders=[]` 并跳过六个月
  停止检查（不拿 null NAV 去算相对收益）；`nav` 在 blocked 时一律为 `null`。
- **退市断供 fail-closed**：持仓名若无退市记录且断供段在窗口末仍未结束（≥20 session），
  按 `open_ended_supply_gap` 归为无法解释，`data_quality_ok=false` → 快照 `blocked`。
  本会话不再靠“原始数据显示一直停牌”默认放行。回归 `tests/test_stale_gate.py`。
