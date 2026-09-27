# Expert selection validation: approved implementation plan

Source authority: user approved the complete plan in this chat on 2026-09-27; subsequent review adds a same-reference-universe offline random diagnostic before nonlinear training, not a significance-as-equivalence rule.

## Frozen scope
60w CNY, at most50 stocks,90% target, original20-session dates,150 retention cutoff,15 industry slots, original financial tiers/PB/ADV/risk permission. Shared6157c64 ledger,30bp slippage, source costs and lots. No production-policy change, no combined A+S arm. Parent dataset6a53f151a1e545857f6925173891ac91a1e06a88bb0de8a7ef34ac6fcd9b7deb.

## Tasks
1. Freeze protocol; review the supplied critique and run a same-reference-pool offline diagnostic using existing router targets. Draw without replacement BEFORE examining outcomes. Unknown outcomes remain unknown. Whole-pool and financial-tier/industry matched controls;1000 deterministic baskets. Compare20/60 session forward price-return proxies with calendar20/60-session blocks. No account or net-alpha claim. Crossing zero means inconclusive, not beta proven. Inspect actual source for reviewer claims.
2. Build shared features: preserve archived2020-2026H1 values; reconstruct early monthly factors for36 matured training cohorts from2014 warmup; add completed2026July/August monthly snapshots through9/22. Verify common history and label endpoints. Training is chronological only; no random time split or training-in predictions supplied to router.
3. Implement and test S: unified within-financial-tier expert ranking before150 retention and admission; missing score positions unchanged. All other selection parameters and prior target semantics fixed. Compare with refreshed R on two periods.
4. Implement A1/A2: same11 factors, monthly last36 matured cohorts,20-session forward price-return ranks;6 LightGBM models x100 rounds,lr.05,15 leaves,min leaf100,L2=1,bagging.8,seed20260927. A1 fixed features/date-balanced samples; A2 learning-curve weighting alpha1=alpha2=decay=1,bins10/5,ratios.8/.7/.6/.5/.4; within-training-date shuffles and date-normalized weights. Equal submodel means. No hyperparameter scan. Entire decision falls back to R if the new expert is unavailable. Retain router12-36 cohorts,state andridge.01.
5. Run frozen changed accounts only; reuse target-identical prior accounts and50/100 references. Windows2022-2025 and new2025 account through2026-09-22.10k paired calendar block bootstrap20/60, family-adjusted R/50/100 contrasts plus A2-A1. Record overlapping windows, year results, profitability, DD,recovery,turnover,cost,attribution,unknown data. Increment candidates get cost2,70%liquidity reduction,5-session locks hitting exits,and delay.
6. Independent tests/account/statistical verification; canonical delivery and append-only continuity update; commit only owned source/test/docs, no auto-trade or promotion. Record failures and all unresolved gates. At least6months frozen forward still required.

## 已完成的账户结果与取舍

2026-09-27实际完成R/S/A1/A2两个期间的共同费用账户；R长段按完整身份复用，其余仅回放改变的组。两个期间分别是2022—2025，以及2025年重新投入60万元至2026-09-22；两段重叠，不是独立样本外。

| 组别 | 2022—2025净利润 | 自然日年化 | 2025初至2026-09-22净利润 | 本轮取舍 |
|---|---:|---:|---:|---|
| R五专家路由 | 303,322.46元 | 10.80% | 124,246.67元 | 保留共同基线 |
| S统一保留排序 | 214,557.26元 | 7.97% | 103,722.12元 | 关闭冻结配置 |
| A1普通GBDT专家 | 315,194.02元 | 11.16% | 105,935.57元 | 仅保留研究线索 |
| A2 DoubleEnsemble适配 | 283,459.45元 | 10.19% | 96,881.73元 | 关闭冻结配置 |
| 50只盈利参考 | 642,497.88元 | 20.02% | 84,669.40元 | 保留强制对照 |
| 100只盈利参考 | 533,913.68元 | 17.30% | 128,133.27元 | 保留独立宽篮对照 |

S提高换手但两段减少净利润；A2两段均不及A1，复杂机制没有显示账户优势。A1长段略增、近期下降，且相对R只有2024/2025两年有增量。全部预登记比较族均无正的置信区间下界，没有候选达到替换盈利参考的门槛。训练外预测有互补信息线索，不等于头部或扣费Alpha。

A1两期间的8个注册压力账户已完成并独立复核。长段成本翻倍仍盈利176,583.11元，但回撤27.32%超预算；两个容量降70%情景均未命中实际约束，不计通过。两段锁仓及延迟情景真实命中、原退出完成。压力总体未通过，不能因个别情景盈利而批准采用。

新增算法、训练及账户研究已经接入主目录；生产默认策略和权限保持原研究身份。完整数据/协议/模型/目标/账户/统计/压力与独立复核入口：`output/expert_selection_20260927/REVIEW.md`；原信号、股数和预计费用观察：同目录`OBSERVATION.md`。

### 复核与限制

- 最终完整测试52模块、344项通过，0失败/错误/跳过，源码哈希未变。入口为`python scripts/run_isolated_tests.py --output <新的验收目录>`；它逐模块隔离生产与冻结测试，仍拒绝真实混用，空测试不能报绿。此前失败日志保留于本轮输出目录。
- 各stage保存来源哈希、源码副本和完整产物哈希；原运行工作区保留于`/Users/agiuser/.codex/worktrees/expert-selection-validation/QuantLab`，协议绝对路径依赖它，暂勿归档。账本仍是`output/frozen_source/6157c64`。
- 财务历史数值修订及`available_at`、部分行业与公司行动仍未认证；全部绩效为条件性研究，无采用许可，无自动交易。
- 主研究runner的未认证部分结果恢复缺陷尚未工程修复，本轮通过首次目录为空、不中断重启、完整账户原哈希复用的规则避免触发，独立核验确认遵守。未来无人值守恢复必须先发布独立模拟完成checkpoint；不能仅自算现有pickle哈希后相信它。压力runner已经实现独立checkpoint。
- 后处理或报告错误不重跑账户；源文件哈希不符时不修改旧协议迁就当前文件。需要审计时使用保存的源码与独立核验器。
- 研究依赖本机已有冻结数据和历史辅助脚本，单纯克隆源码不等于取得完整市场数据。本轮未清理历史output或其他聊天修改。
