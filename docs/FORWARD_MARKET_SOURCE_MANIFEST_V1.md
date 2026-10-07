# 前向市场来源清单与传输契约 V1

状态：已实现契约与行为测试；尚未注册生产清单、尚未调用网络、尚未采集任何行情或题材样本。
本文件只约束未来 `theme_carrier_forward_collection_v1` 的原始 Eastmoney 全市场分页入口；不启动
账户实验、不修改冻结路由器、不产生订单，也不改变 `promotion_approved=false`。

## 冻结来源身份

`quantlab.forward_source_manifest` 仅支持一个 V1 profile：

- `source_id=eastmoney.push2.clist`，`source_family=eastmoney.push2`，
  `source_profile_version=eastmoney-push2-clist-eod-v2`；
- HTTPS `push2.eastmoney.com/api/qt/clist/get`；
- 固定全 A `fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81,s:2048`，按 `f6` 降序；
- 固定每页 100 条、从第 1 页起，固定请求字段
  `f2,f5,f6,f12,f14,f15,f16,f17,f18,f51`；
- 必须返回总数，并由后续 `forward_market_capture` 验证全页集合、行数和证券去重。

清单的 SHA-256 覆盖完整 URL 模板、字段、分页、时钟和失败策略。任何 host、path、参数、字段、
页大小、资格时钟或来源变更都不是“同一来源的小修复”，而是新的 profile/version；当前 V1 schema
下的市场 profile 已因此升级为 `eod-v2`，旧 profile 不可与它混用。

## 时钟、留证与决策资格

这里有两个不能混淆的时钟：**传输留证没有 18:00 停机线**；每次原始 HTTPS response 或
`fetch_error` 都必须以真实接收与写入时点追加到账本。15:30--17:50 Asia/Shanghai 只是一份
session 收盘截面进入该日 `decision_at=18:00` 的**原始页捕获资格窗**。17:55 仅是运维预警目标，
绝不是验证或入账资格截止：同一批页只要都在 17:50 前捕获，完整验证和实际写入在 18:00 前完成，
仍可进入这一次决策。它们均不是停止采集、丢弃晚到页面或停止记录失败的理由。

若供应商提供发布时间，它必须不晚于本次捕获；没有可核验供应商时钟时，账本只可把 HTTPS
首次捕获时点作为可得性的诚实下界。17:50 后或 18:00 后到达的页面仍要保存原始 bytes/错误、URL、
manifest 绑定、`received_at`、`available_at` 与实际 `written_at`；它们只标记为
`not_eligible_for_decision_at`，不得回写或改变已经冻结的 18:00 决策。它们可以按下一次明确的
资格检查进入未来决策，不能伪装成早到页面。

传输对象只接受原始 HTTPS response bytes 或一个明确的 `fetch_error`。它明确禁止：

- 新浪、腾讯或其他备用来源；
- 缓存复用、旧本地面板和解析后的 DataFrame；
- 把晚到页面伪装为本次 18:00 决策可用、非 18:00 的既定决策时钟、重复页号或不同 collection id；
- 用成功响应掩盖失败页，或用无失败原因的空页掩盖失败。

失败页和晚到页都会保留为供 raw-page sidecar 审计的显式回执；本次收盘截面随后必须弃权，不能
尝试第二个来源补洞。它们不是被丢弃的资料，也不是之后可伪造为本次快照的资料。

## 追加式登记

未来调度器须显式指定一个独立目录，以 `SourceManifestJournal` 将清单登记到
`forward-market-source-manifests.v1.jsonl`。每条记录含上一条事件哈希、清单哈希和自身事件哈希；
读取、传输开启前会重验整条链。相同清单的重复登记是幂等的；同一来源身份与 profile version 的
不同内容会拒绝覆盖。登记时点必须不晚于每一个原始页的捕获时点；事后登记不能给先前的 bytes
补来源身份。

登记清单本身不是行情回执，不提供任何 U0、题材、文本或账户资格。未来把已登记传输集合交给
`quantlab.forward_market_capture.capture_eastmoney_eod_snapshot` 时，仍须逐页写入
`ResearchSignalLedger` 原始回执，并继续满足针对**该次决策**的完整性、18:00 可得性、253 个前向
session 和题材映射门槛。晚到回执保留给未来资格检查，绝不倒灌本次截面。
