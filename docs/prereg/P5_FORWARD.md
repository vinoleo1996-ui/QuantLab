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
