> 归档材料：实现状态对应8851cda或文中注明的旧快照，不是当前开发规范。当前默认入口及计划以../STRATEGY.md和../IMPLEMENTATION_V3_2026-09-24.md为准。

# 2026 GitHub量化项目：排名证据与机制研究

核验日期：2026-09-24。此文件替代“今年增星TOP10已确认”的任何旧表述。

## 1. 排名裁定

**尚未取得可验证的全体候选2026年净增Star TOP10。** 已检索204个候选，人工纳入50个直接相关项目。搜索覆盖量化研究、股票数据、回测、交易与投资工具；没有声称穷尽GitHub，更未按收益筛选。

目标指标为：同一截止时点累计Star − 2026-01-01累计Star。当前总Star、创建年份筛选、WatchEvent新增数、当前仍在Star列表中的加入日期，均不能替代旧项目的完整年初累计快照。

| 路径 | 实测 | 为什么不能用于最终排名 |
|---|---|---|
| OSS Insight历史接口 | 明示degraded，2026-05-01起严重缺失；更早也有缺失警告 | 事件派生数只是下界 |
| ClickHouse公开github_events | 最大时间2026-07-02 22:59:59 | 未覆盖研究截止日；WatchEvent月计数异常下降，不能假定完整 |
| GitHub带时间stargazers接口 | HTTP401 | 未获得数据；即使获得当前Star列表，也不能重建已取消Star的年初存量 |
| 年初网页存档 | 查询超时 | 未取得可核验基线 |

证据保存在 `output/upgrade_20260924/github_2026/`：`candidate_metadata.json`、`oss_qlib.json`、`clickhouse.json`、`archive_monthly.json`、`ranking_audit.json`。可离线运行 `python scripts/github_growth_audit.py` 复核分类。

要结束这项阻塞，需要覆盖候选集的可信年初累计快照与同一截止日累计快照，或能审计完整增减事件的数据。不能因为榜单暂缺就把原十个项目重新冠名TOP10。

## 2. 补充发现：所检索候选中，2026新建项目的当前Star前十

**此表仅排序2026新建候选，不包含老项目的年内净增排名。** 新建项目年初存量为零，表中数量可表示从创建至抓取时点的净增；无法证明它们超过所有老项目。数量取同一批归档元数据，不混用后续刷新值。

| 序号 | 项目 | 创建日期UTC | 归档Star |
|---|---|---|---|
| 1 | [ZhuLinsen/daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis) | 2026-01-10 | 65,556 |
| 2 | [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) | 2026-04-01 | 33,939 |
| 3 | [simonlin1212/a-stock-data](https://github.com/simonlin1212/a-stock-data) | 2026-05-11 | 10,289 |
| 4 | [TraderAlice/OpenAlice](https://github.com/TraderAlice/OpenAlice) | 2026-02-18 | 7,138 |
| 5 | [tradesdontlie/tradingview-mcp](https://github.com/tradesdontlie/tradingview-mcp) | 2026-03-29 | 6,633 |
| 6 | [shy3130/tick-stock-panel](https://github.com/shy3130/tick-stock-panel) | 2026-06-18 | 4,967 |
| 7 | [muxuuu/serenity-skill](https://github.com/muxuuu/serenity-skill) | 2026-05-04 | 4,017 |
| 8 | [HiThink-Tech/Financial-API](https://github.com/HiThink-Tech/Financial-API) | 2026-06-09 | 3,748 |
| 9 | [simonlin1212/TradingAgents-astock](https://github.com/simonlin1212/TradingAgents-astock) | 2026-05-13 | 3,539 |
| 10 | [alsk1992/CloddsBot](https://github.com/alsk1992/CloddsBot) | 2026-01-26 | 2,841 |

前两项与原清单重叠，后八项加入观察集。“相关”包括数据/执行基础设施，不能称它们均有可移植的选股算法。补取 tradingview-mcp、tick-stock-panel、TradingAgents-astock、CloddsBot 源码的两次只读请求自动审批超时；对这些项目目前仅有仓库元数据，不做源码级算法判断。

## 3. 原十项目机制的实际落点

以下是机制借鉴，没有复制其源码或引入交易代理依赖。

| 项目 | 本次代码落点 | 完成边界 |
|---|---|---|
| LEAN | `backtest/ledger.py`、`tax.py`：现金/股数/权益/成交与税费 | 复杂公司行动阻断；实盘对账未验收 |
| Freqtrade | 截断重算、未来数据、ETF披露时点、预热长度与前向一致性测试 | 已测固定样本；不是全市场无泄露证明 |
| Qlib | `pipeline.py`、Forecast、模型JSON、版本化输入输出、purged walk-forward | 合成工程流程已贯通；真实研究待数据 |
| RD-Agent | SQLite预登记、配置去重、最多12配置、失败留档 | 中断研究套件不能无条件自动续跑，见实施记录 |
| OpenBB | 多源公开缓存、字段来源、用途验收、覆盖证据 | 公共样本不等于全量PIT；供应商许可仍需确认 |
| TradingAgents | `data/announcements.py`原文偏移/数字/单位/披露时间验真 | 未启用公告Alpha与LLM自动交易 |
| Vibe-Trading | 明确交易规则、Mandate、未知证据阻断 | 规则输入必须历史可得 |
| ai-hedge-fund | 模型与投资权限分离，预测不能改账户约束 | 不借用人物角色或宣传业绩 |
| daily_stock_analysis | 日线新鲜度、失败状态、共同决策摘要 | 公告采集、运营监控未完整上线 |
| AI-Trader | 持久模拟账本、日志哈希、幂等、缺失日恢复、人工成交核对 | 恢复记录标注重建；无六个月冻结验证 |

## 4. 新项目：读到的机制、采用条件与拒绝的默认值

| 项目 / 固定提交 | 证据层级与机制 | QuantLab决策 | 许可证与维护成本 |
|---|---|---|---|
| [NautilusTrader](https://github.com/nautechsystems/nautilus_trader/blob/3a2b765f3a6ef9148f6245e2d55e8350a6c94006/crates/execution/src/models/fill.rs) `3a2b765` | 源码：可分离的成交/滑点模型、种子和概率、部分成交及竞争/成交量模型 | 借鉴执行模型分离；不采用无限流动性默认值，不凭空设定排队成功率 | LGPL-3.0；若整引擎集成会增加Rust/事件模型成本。本轮不依赖 |
| [AKQuant](https://github.com/akfamily/akquant/blob/a4796129f3da26bfaf7c7a21aa8aee736126b3a0/src/risk/stock.rs) `a479612` | 源码：可卖数量/T+1；`python/akquant/backtest/fill_mode.py`把成交价基准和时序分开 | 转为确定性执行契约与回归测试；不能假定支持每种ETF | MIT；移植具体代码仍需逐文件检查 |
| [QuantDinger](https://github.com/OpenByteInc/QuantDinger/blob/9f47e5efbcdcf59fbdccb561554a14c87c419823/backend_api_python/app/services/live_trading/account_risk.py) `9f47e5e` | 源码：拟下单后的账户暴露与成本检查 | Ledger在真实现金和既有锁仓上检查每笔买入，整手后再验证 | Apache-2.0；不引入其杠杆/永续合约功能 |
| [OpenAlice](https://github.com/TraderAlice/OpenAlice/blob/fc8a669097e817efdf347eb15d69e9e6629e3257/services/uta/src/domain/trading/guards/guard-pipeline.ts) `fc8a669` | 源码：guard读取上下文，dispatcher处理账户；空guard链可绕过 | 采用独立权限边界；QuantLab缺规则拒绝，不照搬空链放行 | AGPL-3.0；仅机制参考 |
| [FinceptTerminal](https://github.com/Fincept-Corporation/FinceptTerminal/tree/b7d850b49dc033bb133e6e5d2476444ac5c422b1) `b7d850b` | 源码：回测供应商能力及标准化结果接口 | 用途分层验收；该证据未显示新Alpha算法 | GitHub为NOASSERTION，许可未澄清前不复制；终端整合成本高 |
| [a-stock-data](https://github.com/simonlin1212/a-stock-data/tree/2e0ae6383c649b2bc5f68d3bc430d357f1c59ae7) `2e0ae63` | README：多公开接口、分钟/竞价/财务/资讯 | 加入数据样本候选；需逐字段查历史时间与退市覆盖 | Apache-2.0代码；数据授权单独核验，中等适配成本 |
| [Financial-API](https://github.com/HiThink-Tech/Financial-API/tree/3bca7805a4127ece8d81961917e740d2effac6ec) `3bca780` | README：同花顺API、竞价、公司行动、批量及DuckDB访问 | 优先评估执行校准与公司行动样本；API密钥与授权未取得 | MIT代码不等于免费数据；采购另议 |
| [free-stockdb](https://github.com/hello245m/free-stockdb/tree/74a26af765acef4f1c6b703d12eb25493fa05c0d) `74a26af` | README：压缩增量、断点下载、分钟及复权数据 | 借鉴下载恢复与版本清单；不以qfq反推原始成交和分红 | MIT代码；供应商历史/数据许可未核验 |
| [serenity-skill](https://github.com/muxuuu/serenity-skill/tree/7175becb67cccdeae1ae03cbb8428fe1954b892d) `7175bec` | README：供应链瓶颈→订单→经营兑现的研究线索 | 转为公告阶段预登记假设：经营改善扣除行业/规模/趋势后是否还有净增量 | MIT；事实提取需原文位置、版本和披露时间，人工核验成本高 |
| [QuantaAlpha](https://github.com/QuantaAlpha/QuantaAlpha/tree/b7ceb27b1001261d7a95b209a963664ae1f8ab23) `b7ceb27` | README：进化式LLM因子搜索 | 只吸收实验谱系和去重；拒绝无限因子搜索、测试集反复择优 | GitHub许可未知；不引入依赖 |

源码/README保存在相应 `owner__repo/` 目录及 `nautilus_fill.txt`、`alice_guard.txt`。README声称的能力未自动升级为源码证明、数据覆盖证明或收益证明。

## 5. 我们保留的研究主线

核心预测仍是一个价量机会模块：过去120日市场/行业残差暴露、跳过5日的20/60/120日残差趋势，配合历史行业扩散和上涨质量，用训练数据校准到下一开盘起20日总收益。风险侧独立控制同跌群组、流动性、锁仓和真实可用现金。

新项目对我们最有价值的增量是**成交假设可审计、拟交易后账户约束、数据覆盖可验证**；公告阶段才检验供应链/经营改善的信息增量。没有采用“热门项目都有的同一买入名单”。同质化风险不能靠换因子名消除，仍要看持仓行业/相关性、退出压力、扣费增量及前向记录。

榜单未完成是明确的外部证据缺口；不影响修复已经复现的账户/时序BUG，也不构成扩大模型搜索预算的理由。
