# 可回放数据集契约 v2

这是外部真实数据导入接口。现有公开缓存不满足完整契约，系统不会生成虚假的历史披露时间或原始价格来通过门禁。

## 清单

入口`data/research/manifest.json`。每个文件引用为`{"path":"相对路径.parquet","sha256":"文件的SHA256"}`，不允许路径越出数据集根目录。输入文件变更必须生成新版本和新清单。

清单必需：

- `source`、`revision`、`as_of`：来源、版本、回放截至日。
- `price_basis: "raw"`、`currency: "CNY"`、`amount_unit: "CNY"`、`turnover_unit: "fraction"`或`"percent"`。全数据集统一单位。
- `universe_coverage: "historical"`、`actions_coverage: "complete"`：必须有供应商覆盖依据，字段声明本身不等于独立验证。
- `benchmark_type: "index"`或`"equity_etf"`。指数价格序列与可交易ETF净收益不是同一口径。
- `panels: {"证券ID": 文件引用}`、`instruments`、`calendar`、`benchmark`。
- 可选`actions`、`announcements`、`exposures`。宣称公司行动完整但提供空表，意味着供应商确认该区间无公司行动，而不是未取得数据。

## 逐证券日线

Parquet的DatetimeIndex或`event_time`列为交易日期。至少包含：

`open, close, amount, turnover, available_at, source, revision, quality_status, open_status, open_status_available_at`

- `quality_status="ok"`需要完成字段校验和单位核对。公开抓取默认标记`unverified_public_feed`。
- `available_at`为当条收盘数据实际可知时间；每日研究决策时点为中国时间18:00。
- `open_status`为`trading / halted / unknown`，其时间必须不晚于当日09:30；不能由全天成交额反推开盘状态。
- 有价格限制时提供`limit_up, limit_down`，或交易所除权口径`reference_price`，结合当时有效的`limit_ratio`和`price_tick`计算。
- 不能将复权行情的`open/close`填写为原始价。分红、送转另计，不与复权重复。

## 历史证券记录

每行是当时可知的一版完整证券状态：

`instrument_id, effective_at, available_at, source, revision, quality_status, active, is_st, listing_date, sector, asset_type, buy_lot, sell_lot, sell_delay, price_tick, commission, min_commission, limit_ratio`

历史市值池另需`float_mktcap`。缺少历史市值时，显式使用`CONFIG.research.universe_mode="pit_liquidity"`，不填今天市值。

- 股票或境内股票ETF的`asset_type`分别为`stock / equity_etf`。其他资产类别拒绝回放。
- `sell_delay`按交易日计，不能把所有ETF视为T+0。
- `commission`应涵盖已知的经纪与交易规费，最低佣金单列；A股卖出印花税由历史费率处理。
- `effective_at`与`available_at`分别约束事件何时生效、何时可知。同一证券相同两个时间出现冲突版本会报错。
- 退市证券不能从全历史文件列表删除；其退出、停牌、终止上市和结算需要完整事件数据。

## 交易日历

`calendar`文件有`session`列，须为真实交易所日期，严格升序、不重复，并至少覆盖回放结束后的下一交易日。末日没有后继日时订单保持待执行，不回填同日开盘。

## 基准

基准文件除`close`外，须有`available_at, source, revision, quality_status`；时间晚于对应交易日决策时刻的修订数据不能回填为当时可见值。实际序列逐次存档，不用价格均值替代真实指数。

## 公司行动

支持现金分红与送转，字段：

`action_id, instrument_id, record_date, ex_date, pay_date, share_date, cash_per_share, share_ratio, available_at, source, revision, quality_status`

- `record_date < ex_date`，登记日收盘股数决定权益。
- `cash_per_share`为应收现金口径，税费处理须明确；`share_ratio`为每旧股新增股数，例如送一股填1。
- `pay_date`决定应收款何时可用于买入；`share_date`决定新增股数何时可卖。
- 仅取除权前已知的最新版本，不能重复计入修订。无法解释的晚到事件会使绩效失效。
- 零碎股现金替代、配股、合并与退市结算尚不受支持，不能伪装成空事件表。

## ETF穿透与公告

ETF行业穿透：`instrument_id, effective_at, available_at, sector, weight`。无可验证穿透时归入未知ETF风险组，而非视作完全分散。

公告字段保留`available_at`和来源文档版本，但公告选股模型未启用。只有报告期的财务接口不能提供PIT输入。只有日期而没有时分秒的数据，应在导入时按真实交易日历保守推迟至下一交易日可用。

## 输出与状态

所有结果含`run_id, data_version, model_version, as_of, status, promotion_approved`。运行目录另存实际基准、代码文件哈希和依赖版本。

- `blocked_data`：未满足契约，指标为空。
- `unverified_data_gaps`：持仓估值或事件存在缺口，不发布绩效。
- `research_unvalidated`：可计算研究账本，不代表收益优势或实盘批准。
- 诊断代理数据只能显式进入诊断实验，不能经正式加载器或看板变成实盘建议。

## V3补充：覆盖证明必须匹配实测输入

正式验收要求 `coverage_evidence` 的 `security_history / raw_prices / corporate_actions / calendar / benchmark / opening_rules / industry_history` 各自引用：

```json
{"source_url":"供应商或独立核验规范URL","verification_artifact":"coverage/raw_prices.json","sha256":"文件哈希"}
```

引用文件是JSON，至少含 `kind=data_coverage`、对应`field`、`status=passed`、`input_fingerprint`、`method_version`、`source_urls`、`coverage`、`measurements`。

- `input_fingerprint`由`coverage_input_fingerprint(dataset)`计算，绑定实际表格，不循环包含manifest。
- `coverage.start/end`覆盖数据日历起点至as_of，`coverage.instruments`匹配全体证券。
- `measurements.expected_observations/actual_observations`来自独立覆盖核验且相等，不小于输入对应表计数；`unresolved_required_gaps=0`。空表公司行动必须有检查过程，不代表默认无事件。
- 校验代码会核对文件路径、哈希、输入身份、区间、证券范围及测量值。签名和授权仍须由供应商/研究者提供；文件形式合格不代替真实调查。

`source/revision`是数据来源版本；`fetched_at`是抓取时间。三者都不能替代历史available_at。

### 股息税与复杂事件

现金公司行动须显式声明 `tax_policy`：`gross_cn_individual`为税前现金与个人持有期税；`net_fixed`仅限已有可证明的固定净额口径；`exempt_etf`须有免税资格依据。不能把供应商统一税后字段冒充本账户持有期税率。

非distribution、零碎股现金替代、配股、合并/退市结算等未完整支持。遇到未支持事件，标记不可验证或阻断，不以空事件表通过。

### ETF穿透版本

每个`instrument_id/effective_at/available_at`组是一份完整权重快照。同一effective_at的新available_at整体替换旧快照。开盘检查只读取09:25已知穿透，收盘研究使用18:00已知穿透；不能用下午公告修改上午成交。

### 新产物与恢复

新run目录带 `artifacts.json` 文件摘要，原子发布；同run_id不同账本内容拒绝覆盖。pickle仅用于本地产物，不导入不可信外部文件。恢复的前向交易日标记 `recovered_session=true`，不计作当时实时产生的冻结验证材料。
