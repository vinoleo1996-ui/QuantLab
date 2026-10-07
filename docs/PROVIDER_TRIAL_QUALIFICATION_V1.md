# 外部数据供应商试用验收 V1

状态：**接口已定义；没有供应商因此获得数据、信号、训练或账户资格。**

`quantlab.data.provider_trial` 只校验已经取得的合同证据和小样本回执身份；它不发起网络请求、
不读取凭据、不写入 `ResearchSignalLedger`，也不触发模型或账户。调用方须把成功返回的
`QualifiedProviderTrial.to_record()` 另行追加到研究证据中，并在实际前向采集时重新验证回执。

## 所有来源的不可省略条件

- `provider_id`、`trial_id`、数据集和发布版本必须明确；发布版本须有 SHA-256 身份。
- 合同/条款须同时有合同编号、版本、定位位置和 SHA-256；“网页上看到过”或“待确认”不通过。
- 自动化访问与原始回执持久留存分别是独立的、`granted=true` 的许可声明，各自带范围、证据位置和
  SHA-256。任一拒绝、未知或缺失均不通过。
- 每项样本证据须引用本次试用中唯一的原始回执 SHA-256；伪造、遗漏、重复填充或跨试用引用均不通过。
- 能力声明是来源类型固定集合，必须全部显式声明且可用；少一项、重复、未知附加项或仅口头声明均不通过。

## 题材/行情来源 `theme_market`

除多对多股票—题材边、主动题材语义、涨停、炸板、日线、分页和 U0 覆盖能力外，必须交付：

- 精确 **253 个交易 session** 的历史 bar 种子身份（bar 集、日历、覆盖证券、原始回执均有 hash），
  且用途固定为 `current_u0_only`；它不能回填历史题材信号或生成历史收益。
- 一份分页快照对账：页号必须完整连续、每页有试用回执 hash、证券数完全对齐、无重复证券。
- 一份 U0 题材覆盖对账：U0 成员 hash、成员数、已映射数和未映射数都明确，且已映射数必须等于成员数、
  未映射数为零。
- 题材映射、涨停、炸板和市场 bar 的样本各自指向已登记原始回执。

## 文本来源 `text`

文本小样本至少包含两个不同原始发布者的原文，以及一篇可追溯到其中原文的转载。每条都要求：

- 当前发布者、原始发布者、规范 URL、内容 hash、原始回执 hash；
- 含时区的当前发布时间和原始发布时间，不能只给日期；
- 明确布尔 `is_reprint`。转载必须给出原文文档 ID 与 URL，并同样匹配原始发布者和发布时间；
  原文不能携带转载链接。

因此“两个看起来不同的转发页”不能凑成双源共识。

## L2 来源 `l2`

必须交付一份有 hash 的字段字典和样本矩阵身份。下列 14 个键须按冻结顺序各出现一次：

`micro_vpin_vol_ratio`、`micro_vpin_amount_ratio`、`micro_pin`、`micro_zone_distribution`、
`micro_zone_vol_ratio_T4`、`micro_zone_vol_ratio_T6`、`micro_zone_vol_ratio_T5`、
`micro_zone_vol_ratio_T3`、`micro_zone_rv_ratio_close`、`vol_price_divergence`、`micro_open_gap`、
`micro_impact_decay_half_life`、`micro_liquidity_daily_pattern`、`flow_imbalance_revert_speed`。

每个键必须有非空单位、源字段、`available_at` 字段、源版本、`source_as_of` 字段、原始回执 hash，
且缺失策略只能是 `abstain`。少一个、顺序/键重复、将公告日或供应商抓取时刻冒充
`available_at`、版本或 hash 不明，均不进入训练验收。

通过供应商试用验收后，仍须按现有 L2 契约验证真实逐字段回执、可知时点、标签和 embargo；本文件
不改变 H=5 仅信息检验、H=20 才可登记独立账户排序对照、仅 `score/rank` 可输出的边界。
