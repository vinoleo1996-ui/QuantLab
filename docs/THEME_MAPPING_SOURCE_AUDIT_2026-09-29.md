# 题材映射一手来源审计（2026-09-29）

状态：`blocked_data`。本审计只读了 easy-stock 的完整相关源码、其许可，以及少量公开
端点响应；没有把任何外部原始响应写入 `ResearchSignalLedger`，没有登记生产 source
manifest、没有启动采集/账户/回测，也没有读取、停止或重启 D2。

## 结论

**没有找到可冻结为 `theme_carrier_minimal_v1` 的一手“每个 U0 股票一个主动题材”来源。**

这不是对 easy-stock 的机制判负。它的开盘啦题材榜、涨停池和板块成分是值得保留的候选
输入；阻断的是本仓当前 V1 所要求的同时满足以下四件事的来源：

1. 对当日整个 U0 有完整且可校验的逐股覆盖；
2. 原始来源直接给出一个单一、语义为“主动题材”的标签，而不是由本仓从多概念中择一；
3. 在 18:00 前可用并能保存合法的原始回执；
4. 明确允许把原始响应持久化为研究账本证据。

因此，不冻结 `ThemeMappingPayload.source_id`、`origin_id`、URL profile 或解析器；下一次
应采集日要对该 leg 写全范围 `abstain`，而不是用本地映射、缓存、概念首项或行业名补洞。

## 已核对的 easy-stock 机制

审计对象是公开仓 `jundizhou/easy-stock` 的 commit
`f8b5e1f3014dd55b51b4a4b2f9d5b6ff0aba3836`（2026-09-29 浅克隆到临时只读目录）。以下是
源码事实，不是 README 推断。

| 输入路径 | 精确端点/字段 | 源码实际行为 | 为什么不能作为本仓 V1 直接来源 |
| --- | --- | --- | --- |
| 开盘啦题材轮动 | `POST https://duanxianxia.com/api/getPlateRotatData`，form 为 `from=kaipan&days=20&dates=`；随后 `POST /api/getLongByPlate` | 返回题材排行、强度、历史和龙头；默认只请求前 3 个题材的龙头 | 不是全股票映射。一个 `Theme` 只是题材及其少量 `Leaders`，不能覆盖 U0。 |
| 开盘啦涨停池 | `GET {datasource}/vendor/stockdata/ztpool.json`，解密/解析为涨停事件 | 每个事件可以有 `Concepts`，但对象仅是涨停池，且概念可多值 | 可成为未来“涨停事件”原始证据，不是 U0 全覆盖、单一主动题材表。 |
| easy-stock 本地交叉表 | `backend/internal/sector/kaipanla_theme_mapping.go` | 固定的开盘啦题材 → 东财名称/关键词表；注释明确“Stocks are never mapped one by one” | 这是应用自己的映射规则，且覆盖的是一次 20 日观察到的 60 个题材，不是一手逐股事实；不能搬进本仓来补洞。 |
| 东财板块成分 | `GET https://push2.eastmoney.com/api/qt/clist/get?fs=b:<BK code>` | 先本地关键词匹配板块，再请求成分；同股可以属于多个概念/节点，并有回退 | 缺少单一 primary choice；关键词匹配与回退会改变题材集合，不能伪装成直接映射。 |

相关源码锚点：

- `backend/internal/providers/duanxianxia/client.go:58-107`、`110-151`；
- `backend/internal/providers/duanxianxia/types.go:20-52`；
- `backend/internal/sector/kaipanla_theme_mapping.go:12-24`；
- `backend/docs/data-sources.md:18-35`，它也明确说只维护题材级关键词、不维护个股一一映射；
- `backend/internal/providers/eastmoney/board.go:142-223`。

easy-stock 的 `LICENSE`（SHA-256
`c33d0f2551b1f6dd06d0643109c84ef8e8f4465dae0d3d2208d7ad60ff00963f`）是 PolyForm
Noncommercial，并明确第三方数据仍受各自条款约束。这里没有复制或接入其源码。

## 唯一接近的候选：东财当前股票目录（未冻结）

easy-stock 的 `StockCatalog` 实现提供了一个值得后续按授权再验的候选，但它**不是本次
题材源的通过项**。

### 可复核的候选 profile

- HTTPS gateway：`https://data.eastmoney.com/dataapi/xuangu/list`；
- easy-stock 固定 query：
  `st=CHANGE_RATE`、`sr=-1`、`ps=3000`、`p=<page>`、
  `source=SELECT_SECURITIES`、`client=WEB`、`hyversion=v2`；
- `filter`：七类 A 股市场（上交所/深交所主板、创业板、科创板、两类风险警示板、北交所）；
- `sty`：`SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,NEW_PRICE,CHANGE_RATE,CHANGERATE_5DAYS,VOLUME,DEAL_AMOUNT,MARKET,CONCEPT,INDUSTRY`；
- 逐股候选字段：`SECUCODE` / `SECURITY_CODE`、`INDUSTRY`、`CONCEPT`；响应另有
  `MAX_TRADE_DATE`，但它只有日期而不是发布时间。

源码锚点为 easy-stock `backend/internal/providers/eastmoney/stock_catalog.go:85-105`、
`108-187`（文件 SHA-256
`6c535f861b7432e10157ca2bc481262a18e2d986b5fa21e60381c9d9b7325903`）。响应体自报的
`url` 指向 `http://datacenter.eastmoney.com/stock/selection/api/data/get`，`type` 为
`RPTA_SECURITY_STOCKSELECT`；在未取得供应商文档前，这只可记录为 gateway 所声明的上游
身份，不能把它当作受支持的 API 合约。

2026-09-29 21:00--21:08 CST（**已过 18:00，不能构成前向样本**）做了两页只读探针：

| 页 | 原始 SHA-256 | `result.count` | 实际行数 | 非空 `INDUSTRY` | `CONCEPT` 形态 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 (`ps=3000`) | `aa1d428c46f9063ebe0a2d3b2f5edd8a2324d0bff37163534114aa385d5afacc` | 5,571 | 3,000 | 3,000 | 148 string，2,852 array |
| 2 (`ps=3000`) | `43324c0d54bda5c93d7d31e5b8010309d075acf4e42420f21d84430850a86bab` | 5,571 | 2,496 | 2,496 | 138 string，2,358 array |

两页合计 5,496 个唯一 `SECUCODE`，但两页都报告总数 5,571，且第二页
`nextpage=false`；第三页返回 0 行。这是 **75 行的未解释枚举差额**。即便未来某个 U0
碰巧完全落在已返回的集合，也不能把这个响应宣称为完整全市场映射；任何拟议 adapter 至少要
将 `reported_count == unique_returned_count` 作为硬门，或取得供应商提供的、可验证的
target-scoped 映射接口。

`INDUSTRY` 在该一次样本里对每个返回行都非空，因此它可作为日后独立的
“单一行业分类”候选；但它不是原始来源声称的“主动题材”。`CONCEPT` 多为数组，择取第一项、
按强度选一项或用本地规则合并都会新增一个未经预注册的映射算法。不得把“行业”悄悄命名为
`primary_theme`，也不得把多概念压缩成单概念来填充当前 V1。

## 可得性、回执和许可阻断

1. **18:00 可得性尚未被证实。** 本次读取发生在 21:00 后；响应只有
   `MAX_TRADE_DATE=2026-09-29`，没有 `published_at` 或时分秒级供应商时间。当前
   `FORWARD_MARKET_SOURCE_MANIFEST_V1` 允许用 HTTPS 首次捕获时间作为可得性下界，
   但那只能由未来、真实不晚于目标 `decision_at=18:00` 的同日原始捕获、验证和入账证明，不能从
   今晚的结果倒推。题材映射不套用市场原始页的 15:30--17:50 捕获资格窗。
2. **无法合法持久化原始回执。** 东方财富公开法律声明
   `https://about.eastmoney.com/home/legal`（本次读取 SHA-256
   `ee229f12b7f5fb6dbb27392a9dffca332515adacabd2066630da9ba9c9e86d78`）称：未经书面
   许可不得复制网站任何内容，并特别限制行情信息的复制、传播等。它没有给这个非正式
   `dataapi/xuangu/list` 端点提供保存原始 bytes、自动化日抓或研究账本留存的授权。把响应写入
   append-only raw receipt 会正是一次持久复制，故不能以“公开可访问”替代授权。
3. **当前侧车尚未具备该来源 profile。** `capture_primary_theme_mapping()` 只接受已被调用方
   规范化成 `rows:[{code,theme_id}]` 的 bytes（`quantlab/forward_market_capture.py:967-1036`），
   并没有东财目录的 source manifest、分页/总数校验、字段语义或许可状态。为绕过这一缺口
   新建解析器反而会把未资格的来源接进账本，因此本次不改代码。

## 最小解除路径

1. 先取得**明确书面数据授权或有相同能力的、已核验许可来源**；许可必须允许自动化获取和
   为研究审计保存原始响应/哈希，并明确覆盖行情及分类数据。无此项继续 `blocked_data`。
2. 与来源方确认一个直接、一对一的逐股分类字段的语义：若只能提供 `INDUSTRY`，应另起
   `industry_carrier` 协议，不修改当前 `theme_carrier_minimal_v1` 的“主动题材”定义。
3. 在授权后，预注册专用 mapping manifest：HTTPS host/path、所有 query、页序、返回总数规则、
   `SECUCODE` 格式、字段字典、来源身份和时钟。做三次同日原始捕获；每次要求 session 匹配、
   总数/唯一行一致、U0 精确全覆盖、无重复/空分类，以及真实 `available_at`、验证和回执写入均
   不晚于目标 18:00。17:50 只适用于市场收盘原始页，不能被复用于映射。
4. 只有上述三次通过后，才实现独立 adapter 与 fail-closed tests；它仍只会让采集链路计数开始，
   不会启动 A/B、账户或交易。

在解除前，`theme_carrier_forward_collection_v1` 的市场行情 leg 可以保持独立审计，但映射 leg
不得换成新浪、腾讯、静态表、历史面板或 easy-stock 的本地交叉表。D2 的归档与本阻断相互独立，
`promotion_approved=false` 保持不变。

## `theme_carrier_graph_v1` 的后续口径修订（2026-09-29 23:38 CST）

本审计的“单一主动题材”结论只阻断 `theme_carrier_minimal_v1`，**不再阻断**新登记的
`theme_carrier_graph_v1`。图谱版本接受来源原生的多对多股票—题材边：同一股票可保留多个题材，
合格题材分数取最大值；来源明确确认但没有合格题材的股票为 0。它不允许本仓从多概念中择一，
也不允许把行业名、关键词或缓存变成题材。

这项口径放宽没有解除东方财富候选的阻断：两页目录仍有 75 行未解释的枚举差额，不能证明全 U0
覆盖；响应没有可核验的时分秒级可得性；公开法律声明也没有授予自动化日抓和持久保存原始回执的
权限。因此该候选仍不能进入图谱账本。下一条合格来源可以直接提供多对多边，但仍须同时通过
`provider_trial` 的自动访问/留存许可、完整 U0 覆盖、涨停与炸板字段、18:00 时点和原始回执验收。
