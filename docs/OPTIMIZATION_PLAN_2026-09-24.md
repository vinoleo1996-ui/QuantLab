# QuantLab 全仓优化计划

> 日期：2026-09-24 ｜ 基线：`8851cda`（第一份扩展实现）+ 26 项未提交改动
> 结论先行：主路径已收敛（单一 opportunity 策略 + 全链路 blocked_data 门禁），唯一瓶颈是**数据**；本计划其余部分是与数据采购**并行**的安全网和外围加固。任何绕过数据的收益声明都不在计划内。

---

## 0. 现状裁定

已完成（不再重复投入）：

| 项 | 状态 | 依据 |
|---|---|---|
| 双轨配权（eq 强制满仓 / expected 双轨） | 已移除 | `strategy/allocation.py` 仅 `ineq`，无 `expected` 即 `forecast_required` |
| 旧 repaired_legacy / random 主动策略 | 已移除 | `backtest/engine.py` 默认 `opportunity`，旧因子预计算删除 |
| 账本税费与公司行动 | 已补 | `backtest/ledger.py` 股息税分批、tax_policy 三态、非 distribution 阻断 |
| 文档卖点与代码矛盾 | 已收敛 | review/GitHub/旧策略移入 `docs/archive/`，README 明确“无已证明年化” |
| 运行身份/可复现 | 已增强 | `report/runs.py` git_commit/dirty + lock 哈希 + atomic 写 + 指纹碰撞校验 |
| 研究门禁 | 已建 | `research.py` sqlite 注册 + purge_labels + promotion_gate 强制 EvidenceBundle |

未解决（本计划的目标）：

- **数据**：`data/research/manifest.json` 不存在，`data/bench`、`fundamentals`、`us_stock` 空，公开缓存无 `open_status/available_at/revision` → 一切正式入口 `blocked_data`。
- **测试安全网**：F01/F04/F06 反例仍在 `scripts/` 未进 `tests/`；`approve_resume`、前向断档恢复、回测 vs 前向 NAV 相等、末日 pending 均无测试。
- **信号**：`cli.py` 仍自比恒持有；根 `signals.json/monitor.json` 孤儿，`run_id` 未统一。
- **参数**：约 40 个字段处于 “reserved_reject_overrides” 或反向 “读了但 ACTIVE 禁改”。
- **外围**：看板死按钮/空专家/无基准、daily_pulse 断裂、实时 fallback 丢已成功行、反爬/代理/pickle 安全、forward 断点死锁。

---

## 一、阶段 A：先冻结（1 天，零风险，必须最先）

| # | 任务 | 验收 |
|---|---|---|
| A1 | 提交 26 项未提交改动（docs/archive + engine/opportunity 清理） | `git status` 干净，`runs.py` provenance 中 `git_dirty=False` |
| A2 | 锁 live 依赖：`requirements.txt` / `requirements-data.txt` 全 `==` 并纳入 lock | `provenance()` 的 `dependency_lock_sha256` 非 None |
| A3 | 基线测试跑绿并固化 | `MPLCONFIGDIR=/private/tmp/quantlab-mpl .venv2/bin/python -m unittest discover -s tests` 全过；结果存档 `output/upgrade_20260924/` |

> 理由：当前工作区是脏的，任何后续实验无法归因。先提交冻结，再动数据和代码。

---

## 二、阶段 B：并行安全网（1 周，不等数据，与 C 并行）

| # | 任务 | 文件 | 验收 |
|---|---|---|---|
| B1 | F01（全期权重 NaN→同分）/F04（换仓收益错记）/F06（末日同日）确定性反例从 `scripts/review_audit.py` 搬进 tests | 新建 `tests/test_regressions_f*.py` | unittest 可独立运行，红→绿锁定 |
| B2 | 补行为测试：`approve_resume` 人工恢复、forward 断档恢复、末日 pending、回测 vs 前向逐日 NAV 相等 | `tests/test_ledger.py`、`test_forward_v3.py`、`test_replay.py` | 各至少 1 条反例 |
| B3 | 修信号差分：期初账本权重 vs 目标权重求差，pending/partial 全输出；落盘 `runs/<run_id>/signals.json` 并更新 latest；删根孤儿文件或标记 deprecated | `cli.py`、`portfolio/current.py` | 构造“涨/跌/持有”三单，信号非全持有 |
| B4 | 清理摆设参数：真读的 `weight_max_dev`（portfolio.py）、`regime_vol_quantile`（risk.py）加回 ACTIVE；死参标 deprecated 并从 STRATEGY 卖点删除 | `config_validation.py`、`STRATEGY.md` | `scripts/audit_parameters.py --check` 全绿 |

> 理由：B1/B2 防止后续改动破坏已修复的账本/配权；B3/B4 是不需要数据即可修复的已知正确性缺陷。

---

## 三、阶段 C：P1 数据（关键路径，免费优先，2-4 周）

> 2026-09-24 修订：免费源实核后，主路径改为 **CNEquity 数据湖（免费骨干）+ a-stock-data（免费补充/独立交叉）+ BaoStock（第三交叉）**；Tushare ~500 元降级为“最终抽样对账”备选，仅在免费多源仍无法消歧时再买。详见 `DATA_PROVIDERS_2026-09-24.md`「免费优先落地路径」。契约落点：`scripts/import_research_dataset.py` 生成规范化目录 → `audit_manifest` 验收。

| # | 任务 | 免费来源 | 验收 |
|---|---|---|---|
| C0 | 出口探针：`pip install cnequity; cne init --profile demo; cne sources probe` | CNEquity（TDX 主源，大陆出口更稳，海外配代理） | demo 湖 ~25s 出真实日线，probe 无致命失败 |
| C1 | 建湖：`cne init --profile full`（约 3h，全市场日线 2016 起）+ `cne backfill daily_bars --start 2014-01-01` + `cne backfill trading_status`（过夜一次性，~10-11h） | CNEquity：instruments / trading_calendar / daily_bars / trading_status(停复牌+ST) / adj_factors / corporate_actions / delisting_events | 湖落 `data/cnequity/`（与 `data/a_stock` 旧死缓存隔离）；保留退市股 |
| C2 | 独立交叉核验（与 CNEquity/BaoStock 对拍）：腾讯日K、通达信盘后包、上市/退市日、申万行业变迁史、估值历史 | a-stock-data §1.2/§1.3/§1.6/§6.5/§6.6/§6.7（V3.10.0，零鉴权）；现役 `collect_free_research.py`（BaoStock 预注册 cohort）继续跑 | 困难样本（退市/合并/长期停牌/送转/分红税档/上市初期）三方一致或差异有据 |
| C3 | 按 `import_research_dataset.py` 格式产规范化目录：manifest（`universe_coverage='historical'`、`actions_coverage` 按核验结果、`price_basis`、`as_of`）+ panels + corporate_actions + benchmark + calendar | 上表全部 | `load_dataset` + `audit_manifest` 通过；两池：**先上市资格+成交额池**（不需要历史市值），市值池等 §6.5 退市覆盖核验后再开 |
| C4 | 时间四分离：CNEquity 行级 `source/data_version/fetched_at` 对齐 `available_at/revision`；公告只有日期时按交易日历保守延迟；抓取时间不得冒充历史可知时间 | CNEquity 溯源字段 | 逐行校验通过，修订缺口明示 |
| C5 | 基准：CNEquity `index_bars` 沪深300/中证500，身份=指数非ETF；`contracts.py` 基准校验过 | CNEquity + a-stock-data §12 | `cli backtest` 不再 `benchmark missing` |
| C6 | 开盘状态：规则推导 `open_status='derived'`（昨收 × 涨跌停制度表 + open 价），`qualification.py:84` 已支持；不用成交额反推 | — | `ledger.trade_permission` 不误拒；qualification 在 warnings 层标注 `opening_capacity_unverified` |

明确不做（本阶段）：美股、分钟线、公告模块（`enable_announcements` 保持 False）、ETF 穿透、开盘容量校准、财务披露时间（首代机会模块不用财报，冻结 fundamentals/announcements）。全部等数据验收之后。

---

## 四、阶段 D：P2 一次冻结研究（数据过关后 1-2 周）

| # | 任务 | 验收 |
|---|---|---|
| D1 | 端到端：`cli research --manifest` 跑通机会策略 + 4 对照（买入持有 / 月等权 / 波动等权 / 宽基指数ETF） | 无 `blocked_data`；产出 `summary.json` |
| D2 | 随机对照 1000 次（种子 20260924）+ 36/6/6 向前滚动 + purge_labels + block20/60 CI | `promotion_gate` 明确输出 “unproven” 或进入晋升门槛 |
| D3 | 7 角色 `EvidenceBundle` 正式落盘（accounting/temporal/data/stress/oos/execution/forward），`evidence.verify_bundle` 校验通过 | `promotion_approved` 不再仅是 False 占位 |
| D4 | 收益来源拆解：池 / 暴露 / 排序 / 配权 / 执行 各自贡献 | 一页拆解表；结论可能是“未证明有效” |

> 主指标：扣费后日超额收益均值年化；同时展示相对财富、回撤、成本、换手、未成交、数据覆盖。至少三个不重叠测试段，至少 2/3 扣费增量 >0；剔除单一行业/单只/少数最佳交易日后仍稳。

---

## 五、阶段 E：外围加固（D 之后或低优先并行）

| # | 任务 | 文件 |
|---|---|---|
| E1 | 看板：sector 死按钮、清仓→sell 映射、experts 空、基准线、持仓三列 `-` | `webapp/app.py`、`portfolio/current.py` |
| E2 | `daily_pulse` 幂等写 + 固定宇宙 + 专家链路 + 失败计数不掩盖 | `scripts/daily_pulse.py` |
| E3 | 实时 fallback 按 code 合并（不复写已成功行）+ 多源溯源 + 429/Retry-After 分支 + 接通 `max_retry` | `data/realtime.py`、`sources.py` |
| E4 | 去 UA 伪装（industry/eastmoney/sina）、代理注入鉴权、pickle 反序列化收敛 | `factors/industry.py`、`webapp/app.py` |
| E5 | forward：恢复路径接 `manual_recovery_required`、零成交不得通过对账、前向强制 `usable()`/`research_interpretable` | `forward.py`、`mandate.py` |
| E6 | 依赖/安全：UA 指纹移除后重验反爬；`curl_cffi` 仅作备源；无密钥外泄 | `requirements*.txt` |

---

## 六、不做清单（明确出界）

- 新增模型 / 代理 / 角色扮演交易员 / LGBM / LLM 配仓 / 无限挖因子
- 扩展 GitHub 参考项目调研（已归档 `docs/archive/`）
- 美股双市场、公告增强、分钟级执行校准
- 任何“年化 XX%”对外口径（当前无证据；旧 14-29% 与诊断 0% 均不是有效业绩）

---

## 七、总时间线（人力假设 1-2 人）

```
第 0 周    A 冻结 + B 安全网启动
第 1-6 周  C 数据采购与导入（关键路径；B 并行完成）
第 7-8 周  D 一次冻结研究 → 出“有效 / 未证明有效”裁定
第 9 周起  E 外围加固 + 按 D 结论决定算法去留
```

---

## 八、成功标准与退出条件

- **成功**：一份可信数据 + 一次完整可复现研究 + 收益来源拆解，三者齐备，才谈“策略是否保留”。
- **诚实默认**：任何一步证据不足，结论为“未证明有效”，不拿旧数字或代理数字充数。
- **冻结门**：进入六冻结版前向验证前，必须先过 D3 的 EvidenceBundle 与 D2 的随机对照；开发日志/历史补跑/合成夹具不计入六个月冻结验证。