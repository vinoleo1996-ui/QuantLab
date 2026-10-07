# 数据供应商与开源数据核验

核验日期：2026-09-24。本页是当前P1数据工作记录；无采购、无账号授权、无新增策略。证据文件位于 `output/data_feasibility_20260924/`，早期样本保留在 `output/upgrade_20260924/provider_samples/`。

## 裁定

“免费源一用就是幸存者与前视、不付费就只能blocked”不成立。能否用于研究取决于证券范围、字段历史含义、事件完整性和执行假设。免费源可贡献有效历史数据；截至本次核验，仍未取得能通过当前机会模型全部要求的完整真实数据集。

付费购买也不能直接通过qualification。抽样对账验证字段含义与数值；完整覆盖还须核对证券范围、时间区间、分页、缺行和公司行动。多个接口若复用同一底层源，不构成独立交叉验证。

## 免费优先落地：CNEquity + a-stock-data 实核（2026-09-24 补充）

针对“哪些免费项目能免费用起来”，本轮实核两个活跃项目（README/数据集目录/端点清单级核验，未整包运行；接口可用性以 C0 探针为准）：

| 项目 | 协议/许可 | 能直接填哪张契约表 | 核验边界 |
|---|---|---|---|
| [CNEquity](https://github.com/rootSunc/CNEquity)（PyPI `cnequity`，2026 活跃） | Apache-2.0，免注册免 token，自托管 Parquet 湖 | `instruments`(证券主数据)、`trading_calendar`、`trading_status`(停复牌/ST)、`daily_bars`(全市场日线)、`index_bars`、`adj_factors`(复权因子)、`corporate_actions`(公司行为)、`delisting_events`、`index_constituents`(历史成分，回填 cni)、`industry_members`(行业成分，回填 sw)、`valuation_metrics`；行级 `source/data_version/fetched_at` 溯源；**保留退市股**（自身演示：用今日名单回看 2016-2021 等权收益 5.9%→12.0%，与本仓发现的幸存者偏差一致） | 全市场日线默认 2016 起（`--profile full` 约 3h，可 `backfill daily_bars --start 2001`）；历史 ST 需一次 10-11h 回填；退市股复权/历史成分 as-of 语义须实测；主源 TDX，海外出口可能需代理 |
| [a-stock-data](https://github.com/simonlin1212/a-stock-data)（V3.10.0 2026-09-22，87 端点/34 源，零鉴权） | Apache-2.0 | §1.2 腾讯日K(前后复权)、§1.3 通达信盘后包(当日全市场日线含成交额)、§1.6 新浪复权因子、§6.5 估值历史(2016+，含换手/停牌/ST，**可当市值池免费替代**)、§6.6 上市/退市日(唯一零鉴权退市日源)、§6.7 申万行业变迁史(仅代码)、§6.8 ST名单 | §12 指数成分只给最近快照，**明确无历史时点回填**，不能用于成分池；mootdx 行情命令 2026-09 起失效(#52)，K线已改腾讯/盘后包；估值历史不支持北交所 |
| BaoStock（现役 `collect_free_research.py` 底座） | 免注册 | 原始日线/日历/分红/isST/停牌/上市退市日期；2014-01-02 历史名单 2970 条含 98 只后来退市证券 | 退市行情需权限审核（实测 10002007/Broken pipe），不能作唯一后端 |

**结论修正（相对早前“4 项硬缺口”）：**
- 历史指数成分：**部分推翻**——CNEquity `index_constituents` 回填 cni 声称提供 2016+ 历史成分，待实测 as-of 语义；若通过则成分池可开。
- 历史市值：a-stock-data §6.5 免费替代 Tushare `daily_basic`（2016+）；退市覆盖待核，市值池在核验后开，上市资格+成交额池不依赖它。
- 开盘状态：`qualification.py:84` 已接受 `open_status='derived'`——规则推导（昨收×涨跌停制度表+open 价）合法，不是用成交额反推。
- 披露时间：首代机会模块（残差趋势/行业扩散/上涨质量）不用财报，冻结 fundamentals/announcements 即可，非首代门禁。
- 真硬缺口收窄为：**退市股复权/公司行动完整性**。CNEquity 保留退市股但 delisted 复权待实测；若缺口仍在，用不复权价+事件缺失 `data_issue` 标注+末价结算，结果标 `research_unvalidated`，不宣 qualified。

**落地顺序（并入 OPTIMIZATION_PLAN 阶段 C）：** C0 探针(demo 25s) → C1 `cne init --profile full`+backfill(2014 起+历史ST 过夜) → C2 a-stock-data 独立对拍（腾讯K线/盘后包/§6.6/§6.7/§6.5）→ C3 按 `import_research_dataset.py` 产规范化目录两池 → C4 溯源对齐 → C5 基准 → C6 open_status=derived。Tushare ~500 元仅在免费多源无法消歧时作抽样对账备选。

## 实际取得的证据

| 数据 | 实际结果 | 能证明什么 / 不能证明什么 |
|---|---|---|
| 上交所终止上市名单 | 本次公开接口返回159条，含证券代码、LIST_DATE、DELIST_DATE | 官方免费渠道提供终止上市证券；尚未核验全沪深范围和退市结算收益 |
| 与已有2014-01-02名单交叉核对 | BaoStock2970条中有98只后来终止上市的沪市证券，如600005、600068 | 该免费历史名单保留了部分后来退出的证券；不是今日存续名单直接回放 |
| 名称字段核验 | 2014名单出现“退市银鸽”等后来名称 | code_name不是可靠历史名称；禁止从名称推断历史ST。历史ST须独立日级字段/事件证据 |
| 早期免费价量/权益样本 | 单股与沪深300原始日线各3096行，贵州茅台2024分红2行 | 已有真实原始行情与现金分红案例；未证明所有退市证券、送转、配股与合并均完整 |
| 本次BaoStock新增退市行情 | 首次登录报10002007/Broken pipe；一次拟重试因自动权限审核超时未执行 | 本轮未取得新增退市前日线。审核超时不代表安全拒绝，不把文档能力当取数成功 |
| 申万官方行业变迁文件 | 找到官方XLS地址；本次证书链失败，系统curl验证证书重试返回508 | 有明确候选来源，本轮未下载成功，未确认覆盖与历史修订语义 |

实际交叉核验：`output/data_feasibility_20260924/actual_sample_checks.json`。原始JSON、CSV和交叉表在`official/`；`sample_lineage.json`保存输入哈希。此次没有生成正式`data/research/manifest.json`。

官方退市页面：[上交所终止上市公司](https://www.sse.com.cn/assortment/stock/list/delisting/)。AKShare接口把DELIST_DATE映射成“暂停上市日期”，本次保留交易所原始字段名，避免误读。

## 优先使用哪些项目

以下是用途优先级，不是Star排名。已获取固定提交、README及相关接口源代码；未安装第三方Skill或执行其嵌入指令。

| 来源/项目 | 可直接借鉴或接入的部分 | 核验边界 | 代码许可 |
|---|---|---|---|
| [BaoStock](https://www.baostock.com/) | 原始日线、日级ST/停牌、上市退市日期、历史证券列表、现金分红/送转日期 | 官方文档有ipoDate/outDate/status和日期查询；逐证券历史完整性、行业修订和连接稳定性仍需实测 | 数据使用条款另核 |
| [AKShare](https://github.com/akfamily/akshare/tree/0191689d57c667b7c7a198fd0cf97316837ef311) | 交易所退市名单、巨潮行业变更、分红等独立字段适配器 | 是采集库；不保证所有接口今天可用或历史完整。巨潮变更日期不是自动等同于首次披露时间 | MIT |
| [a-stock-data](https://github.com/simonlin1212/a-stock-data/tree/2e0ae6383c649b2bc5f68d3bc430d357f1c59ae7) | 免费来源定位，尤其申万行业变迁、通达信盘后文件 | 主要为Markdown内嵌Python。sw_industry_as_of只筛start_date，忽略update_date和修订语义，不能接受“自动消除前视”的说法；不整包引入 | Apache-2.0 |
| [HiThink Financial-API](https://github.com/HiThink-Tech/Financial-API/tree/3bca7805a4127ece8d81961917e740d2effac6ec) | 官方统一接口、日线及集合竞价候选、marketdb本地落盘 | 需要API Key；README说明公开渠道不提供分钟/tick/L2和公告原文。历史退市/行业变更覆盖未实测 | MIT |
| [free-stockdb](https://github.com/hello245m/free-stockdb/tree/74a26af765acef4f1c6b703d12eb25493fa05c0d) | 校验后增量同步、本地压缩和查询机制 | 源码不内置数据地址，sync_url.txt仅注释示例；它没有自动提供一份完整免费数据库。现阶段不替换本仓存储 | MIT |
| [mootdx](https://github.com/mootdx/mootdx/tree/e99ae34382d970c68654c6d17c45512e728f130d) | 通达信文件读取、除权信息解析，作为独立对账候选 | 最新推送记录2024-07；a-stock-data报告2026-09公开K线命令失效，本次未独立测其服务。不能作为唯一后端 | MIT |
| [Qlib](https://github.com/microsoft/qlib/tree/be725493eb1a6bbb42bf11b37aa7669f59610ff1) / [社区investment_data](https://github.com/chenditc/investment_data) | 数据结构、版本与研究流程参考；社区包可供对照 | Qlib README称官方数据暂不可用并指向社区源；社区更新依赖Tushare。加工后的特征包不能替代原始成交价和现金权益账本 | Qlib MIT；社区数据授权另核 |
| [vnpy_rqdata](https://github.com/vnpy/vnpy_rqdata/tree/cf55c3194bb93fe36fb15679ffa280723485fcdc) | RQData认证和行情适配方式参考 | 不是免费RQData授权；本仓无需引入整个交易框架 | MIT |

开源代码许可不代表底层数据可商用或再分发。GitHub库文件与官方文档已存档；Google本次只返回JS跳转壳，没有可用搜索结果，结论来自官方文档、源码和实测样本。

## 付费应买具体缺口

| 来源 | 官方已列出的字段 | 仍需验证 |
|---|---|---|
| [Tushare daily_basic](https://tushare.pro/document/2?doc_id=32) | 按日市值/流通股本，至少2000积分，5000积分无总量限制；更新时间15–17点 | 历史股本是否事后回填、全市场分页、退市覆盖；积分不是人民币报价 |
| [Tushare index_member_all](https://tushare.pro/document/2?doc_id=335) | 行业in_date/out_date，至少2000积分 | 默认is_new=Y，历史研究须取历史记录；纳入/剔除日期仍需区分首次公布与生效 |
| [Tushare dividend](https://tushare.pro/document/2?doc_id=103) | 实施公告日、登记/除权/到账/红股上市日、税前分红；至少2000积分；每日20–21点更新 | 不可假设18点已通过该API取得当天更新；预案与实施须区分，税后字段不能代替本账户持有期税 |
| [Tushare suspend_d](https://tushare.pro/document/2?doc_id=214) | 停复牌日期、suspend_timing日内区间；至少2000积分 | 日内区间覆盖、空值语义、临停与恢复时段；须以样本验证开盘状态重建 |
| [RQData股票文档](https://www.ricequant.com/doc/rqdata/python/stock-mod.html) | 历史行业date查询、证券状态变动、分红、分钟、PIT市值字段market_cap_3等 | 历史版本/覆盖/授权报价。is_suspended说明是全天停牌，单独不足以代表开盘交易状态 |

优先对比Tushare所需接口套餐与RQData同一组困难样本，哪个填补实际缺口更完整、维护成本更低再决定。聚宽可列备选，但本轮未实测，不列为已验证替代。Wind/Choice可作独立对账；少量抽样不证明底层全量数据完整。当前A股/境内ETF任务不扩展采购美股数据。

## 门禁与可知时间应怎样处理

1. `available_at`是统一语义，不要求供应商原生使用这个字段名。公告只有日期时按真实交易日历保守延迟；行情可以登记有依据的公开可用规则及缓冲，但必须把“推定可用时间”与“实测发布时间/供应商入库时间”区分。当前抓取时间不得冒充历史可知时间，无法解决的修订仍留缺口。
2. `open_status`可从可靠停复牌区间、交易所状态或开盘逐笔/分钟证据映射。全日tradestatus=1不足以证明09:30有成交，不能把全天成交额非零直接改写为开盘可交易。
3. 开盘价格/成交记录可作为已提交订单的执行回放结果；它们不能流入此前的选股和订单决策。是否实际有能力以该价格成交需要独立容量校准。
4. 我们的`qualification.py`已把开盘容量和公告模块作为独立未验证状态，不因尚无公告分钟时间或开盘容量证明就拒绝全部价量研究。当前直接阻断原因仍是缺经过核验的manifest及其必需覆盖证据。不能为出收益数字伪造证明文件。
5. 当前机会模型需要历史行业；通用收益研究不必购买所有公告/PIT财务。上市资格+成交额池可不使用历史市值，但不得用今日市值补齐。历史指数成分仅在研究协议以该指数成分定义池时才必需。
6. 未支持的配股、合并、零碎股及退市结算也是账本缺口，购买数据本身不会修复；按明确事件补齐处理后才能验收相关绩效，不能事后删除这些证券。

## 下一步执行顺序

- 先建立沪深证券生命周期并集：历史列表、官方退市、上市/代码变更互核，逐字段区分当前名称和历史状态。落点：CNEquity `instruments`+`delisting_events` × a-stock-data §6.6 × `historical_securities_20140102.csv`。
- 对该范围抓原始价、日级ST/停牌、公司行动；困难样本包括退市/合并、长期停牌、送转、分红税档及上市初期，不能只验证存续蓝筹。落点：CNEquity `daily_bars`/`trading_status`/`corporate_actions`，a-stock-data §1.2/§1.3/§6.5/§6.7 对拍。
- 验证行业变迁的生效与公布语义；官方免费文件不通或历史修订不清时，用同一困难样本评估付费来源。落点：a-stock-data §6.7 + CNEquity `industry_members` 回填 sw。
- 先证明一份最小真实数据包可以正确记账，再扩全历史池和一次冻结研究。无需引入新交易框架、代理或其他模型。落点：`scripts/import_research_dataset.py` → `audit_manifest`。

本轮仅推进P1字段核验与真实样本。正式机会策略、全量1000次随机对照、真实OOS和六个月冻结前向证据仍未取得；任何来源均不保证策略收益。
