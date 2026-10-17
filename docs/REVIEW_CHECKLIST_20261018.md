# Code Review 检查清单（2026-10-18）

> 本清单用于系统性地审查对方提交的代码，聚焦核心数据错误、代码 BUG、数学逻辑错误。
> 每个检查项标注：【数据】/【代码】/【数学】/【审计】

---

## 一、数据正确性

### 1.1 停牌分类（【数据】【审计】）
- [ ] `sz.000616` 2018-01-24 ~ 2018-07-24 是否归为 `suspended`（121 条 `S`）
- [ ] 增量期间（2026-10-13~10-16）是否有新的停牌股票被正确标记
- [ ] `unexplained` 分类：缺 raw/单日不满足/覆盖不完整/跨退市日 → 必须 fail closed
- [ ] `data_quality_ok` 是否只卡 `unexplained <= 5` 且 coverage 完整
- [ ] 两源冲突时（promax suspend_d vs raw cache tradestatus）是否 fail closed

### 1.2 ETF 缺失日（【数据】）
- [ ] `sh511260` 2018-08-17 和 `sh513100` 2022-01-13 是否用 promax `fund_daily` 核对
- [ ] 调仓日缺价 → `blocked`；非调仓日 → `stale_mark`（需同时满足 4 个条件）
- [ ] ETF 去 `ffill` 后 NAV 是否改变（需逐位对比）
- [ ] `etf_missing_dates`、`etf_missing_on_rebalance`、`etf_stale_mark_days` 是否记录

### 1.3 复权因子（【数据】）
- [ ] 增量期间 adj_close 是否用 promax `adj_factor` 精确计算（不是外推）
- [ ] 分红/拆股期间复权价是否与本地历史一致
- [ ] 退市股 adj_close 是否正确处理（最后有效值 + NaN）

### 1.4 涨跌停（【数据】【执行】）
- [ ] ST 股票涨跌幅是否为 10%（2026-07-06 后变化）
- [ ] `stk_limit` 返回的 limit_up/down 是否与实际一致
- [ ] 涨停日是否 blocked（不能买入）；跌停日是否 blocked（不能卖出）

---

## 二、代码 BUG

### 2.1 边界条件（【代码】）
- [ ] `latest_panel_dir()` 是否按数据日期选，不是目录名排序
- [ ] `pool_on()` 的 `RANK_START`/`RANK_END` 边界：len(ordered) < RANK_START 时返回 []
- [ ] `decision_grid` 的 `phase::step` 是否正确处理 phase >= len(sessions)
- [ ] 空 DataFrame / NaN 传播：每个函数对空输入的处理是否一致

### 2.2 异常处理（【代码】）
- [ ] promax 429 退避是否生效（1-2s）
- [ ] API 超时/断网时是否优雅降级（不是 crash）
- [ ] `suspend_d` 返回空时是否返回 `None`（unknown），不是 `False`（trading）
- [ ] parquet 读写失败时是否原子写（tmp + os.replace）

### 2.3 竞态条件（【代码】）
- [ ] 并行会话同时写同一文件时是否有锁机制
- [ ] `p5_snapshot.py` 的状态写入是否 tmp + os.replace
- [ ] 增量采集和面板合并是否可重入（幂等）

### 2.4 类型安全（【代码】）
- [ ] `vol` 单位是否统一（promax 手 → 股 ×100）
- [ ] `amount` 单位是否统一（promax 千元 vs datahubco）
- [ ] 日期类型：string vs Timestamp vs datetime 的混用
- [ ] float32 vs float64 的精度问题（pandas parquet 默认 float32）

---

## 三、数学逻辑

### 3.1 CAGR / MDD 计算（【数学】）
- [ ] CAGR 公式：`((end / start) ** (1 / years) - 1)`，years = 天数差 / 365.25
- [ ] MDD 是否从峰值到谷值，不是从起点
- [ ] blended_cagr 的权重：股票袖 vs ETF 袖的配比是否为 65/35
- [ ] ex-2025 的日期范围是否正确（2025-01-01 ~ 2025-12-31）
- [ ] 费用是否 double-counted（成本1 和 成本2 的定义）

### 3.2 worst-path 门槛（【数学】【审计】）
- [ ] `min(blended_mdd) >= -0.25` 是否每个成本都查
- [ ] 8 条路径的 MDD 是否正确计算（不是中位 MDD）
- [ ] 800k 成本2 的 -25.57% 是否使 `passed=false`

### 3.3 统计检验（【数学】）
- [ ] NW t-stat：月度均值的 Newey-West 标准误，lag=6
- [ ] BH-FDR：Benjamini-Hochberg 多重检验校正，q≤0.10
- [ ] DSR：Deflated Sharpe Ratio 计算是否正确
- [ ] 匹配对照：同池随机 50 只、同换手、同调度

### 3.4 组合权重（【数学】）
- [ ] 50 只等权：每只权重 = gross / 50
- [ ] 行业上限：max_per_industry 是否 ≤ slots（15 或 12，需确认）
- [ ] buffer 保留：前 150 名保留，不在前 150 才卖出
- [ ] 权重和是否 ≤ 0.97（gross=0.97 时）

---

## 四、执行现实性

### 4.1 T+1（【执行】）
- [ ] 信号日收盘后生成 targets，下一交易日开盘执行
- [ ] 不使用信号日收盘后的数据（如 margin_detail 次日 09:00）

### 4.2 整手交易（【执行】）
- [ ] 100 股整手：数量是否向下取整到 100 的倍数
- [ ] 买不进一手的资金是否留在 cash
- [ ] ETF 袖是否也用 100 股整手

### 4.3 费用（【执行】）
- [ ] 成本1：佣金 + 印花税 + 过户费（双边）
- [ ] 成本2：成本1 + 冲击成本
- [ ] IPO 收益 1.5% 是否单列（不算在策略收益里）

### 4.4 退市处理（【执行】）
- [ ] 退市前最后一个可交易日是否执行清仓
- [ ] 退市后 NAV 是否按清算价计算
- [ ] `delisting_liquidations` 计数是否正确

---

## 五、审计合规

### 5.1 预注册一致性（【审计】）
- [ ] 试验参数是否与预注册文件一致（不事后改门槛）
- [ ] 新假设是否有独立预注册文件（H1-H5）
- [ ] 32 格矩阵是否标记 withdrawn

### 5.2 台账完整性（【审计】）
- [ ] 每个试验 base + outcome 是否都在 `docs/trial_ledger.csv`
- [ ] hash 是否记录（报告、summary、runs）
- [ ] `promotion_approved=false` 是否明确标注

### 5.3 密钥安全（【审计】）
- [ ] 代码中是否还有硬编码 key
- [ ] 环境变量是否正确读取
- [ ] 日志中是否不记录 key

### 5.4 冻结结果保护（【审计】）
- [ ] 是否不覆盖 `output/p5_base_lot_20261018/`
- [ ] 新目录命名是否不同（如 20261019）
- [ ] 冻结 hash 是否作为对照基线

---

## 六、Review 流程

1. **对方提交 PR / 推送分支**
2. **Trial merge**：`git worktree add` + merge + pytest
3. **逐项检查**：按本清单逐项验证
4. **数据对账**：新结果与冻结 hash 对比
5. **单路径 smoke**：`resumed=false`、gross=0.97、800k c1 13.2-13.8%
6. **撰写 review 报告**：问题分类（blocker / warning / nit）
7. **对方修复 → 重审 → 通过**

---

## 七、已知的待验证问题

| 问题 | 来源 | 验证方法 |
|---|---|---|
| INDUSTRY_SLOTS = 15 vs 预注册 12 | 代码审查 | grep call site |
| 10-22 日期：phase 0 最后网格是 08-28 不是 10-09 | 日历计算 | 运行 p4_alpha decision_grid |
| 密钥历史泄露 | 安全审计 | git log -p --grep="adapter" |
| BJ 股票排除：p4_alpha.py:86 显式排除 | 代码审查 | 确认 pool_on 调用链路 |
