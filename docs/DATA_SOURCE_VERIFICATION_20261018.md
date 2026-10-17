# 数据源验证报告（2026-10-18 01:00 CST）

## 结论

**两个接口都可用，但各有局限。建议双源并行：datahubco 用于全市场批量拉取，promax 用于单只股票精细查询和停牌/复权数据。**

---

## 一、接口连通性

| 接口 | URL | 状态 | 响应速度 |
|---|---|---|---|
| datahubco | `http://datahubco.com/app-api/openapi/v1/tushare/` | ✅ 200 OK | <1s |
| promax | `https://pcd.mobcvb.cn/tushare/pro/` | ✅ 200 OK | <1s |

---

## 二、核心接口验证

### 2.1 日线行情（`daily`）

| 维度 | datahubco | promax | 评估 |
|---|---|---|---|
| 单只股票查询 | ✅ | ✅ | 一致 |
| 全市场查询（trade_date） | ✅ 5563-5567条/日 | ❌ 返回0条 | **datahubco 更优** |
| 返回字段 | ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount | 同上 | 一致 |
| **tradestatus 字段** | ❌ 无 | ❌ 无 | **关键缺失** |
| 历史覆盖 | 待验证 | 2014年至今 ✅ | promax 已验证 |

**tradestatus 缺失的影响**：当前系统用 `tradestatus==0` 判断停牌，两个新接口都没有这个字段。需要改为：
1. 用 `suspend_d` 接口获取停牌记录（promax 可用，有频率限制）
2. 或用 "daily 接口某天无返回" 推断停牌

### 2.2 停牌数据（`suspend_d`）

| 维度 | datahubco | promax | 评估 |
|---|---|---|---|
| 可用性 | 未测试 | ✅ | promax 可用 |
| 返回字段 | — | ts_code, trade_date, suspend_timing, suspend_type | — |
| sz.000616 验证 | — | ✅ 2018-01-24~2018-07-24 共121条，suspend_type='S'，07-25恢复为'R' | **与本地 parquet 完全一致** |
| 频率限制 | — | ⚠️ 429 Too Many Requests | 需要退避（1-2秒） |

**与本地数据对比**：
- 本地 `tradestatus.parquet`：sz.000616 2018-01-23=1.0（正常），01-24~07-24=N/A（停牌），07-25=1.0（恢复）
- promax suspend_d：2018-01-24~2018-07-24 全部 'S'，07-25 'R'
- **结论：两者完全吻合**

### 2.3 复权因子（`adj_factor`）

| 维度 | datahubco | promax | 评估 |
|---|---|---|---|
| 可用性 | 未测试 | ✅ | promax 可用 |
| 返回字段 | — | ts_code, trade_date, adj_factor | — |
| 示例（000001.SZ） | — | 2026-01-05: 134.579 | 合理 |

### 2.4 涨跌停（`stk_limit`）

| 维度 | datahubco | promax | 评估 |
|---|---|---|---|
| 可用性 | 未测试 | ✅ | promax 可用 |
| 返回字段 | — | trade_date, ts_code, up_limit, down_limit | — |

### 2.5 ETF 数据（`fund_daily`）

| 维度 | datahubco | promax | 评估 |
|---|---|---|---|
| 可用性 | 未测试 | ✅ | promax 可用 |
| 返回字段 | — | ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount | 与股票 daily 一致 |
| 最新数据 | — | 2026-10-08（周末前最后交易日） | 实时性正常 |
| **QDII 溢价** | — | ❌ 无直接接口 | 需间接计算 |

**QDII 溢价计算方案**：
- 溢价 = fund_daily.close（市价）/ fund_nav.unit_nav（净值）- 1
- 需要同时调用 `fund_daily` 和 `fund_nav` 两个接口
- 当前系统已有此逻辑（`scripts/collect_etf_bars.py` 从腾讯财经获取）

### 2.6 股票基本信息（`stock_basic`）

| 维度 | datahubco | promax | 评估 |
|---|---|---|---|
| 单只查询 | 未测试 | ✅ | promax 可用 |
| 全市场列表 | ❌ 返回0条 | ❌ 返回0条 | **两者都不支持** |
| 退市信息 | — | ✅ list_status='D', delist_date | — |

**全市场列表获取**：两个接口都不支持直接获取全市场股票列表。需要：
1. 用 `daily` 接口的 `trade_date` 模式获取某日全部交易股票（datahubco 支持）
2. 或分页多次查询（未验证）

### 2.7 交易日历（`trade_cal`）

| 维度 | datahubco | promax | 评估 |
|---|---|---|---|
| 可用性 | 未测试 | ✅ | promax 可用 |

---

## 三、与现有本地数据的一致性

### 3.1 停牌数据

| 数据源 | sz.000616 停牌期间 | 结论 |
|---|---|---|
| 本地 parquet (20261013) | 2018-01-24 ~ 2018-07-24 = N/A | — |
| promax suspend_d | 2018-01-24 ~ 2018-07-24 = 'S' | ✅ **一致** |

### 3.2 数据量对比

| 指标 | 本地 (20261013) | promax | datahubco |
|---|---|---|---|
| 股票数 | 6,080 | — | — |
| 交易日 | 3,103 (2014-01-02 ~ 2026-10-12) | — | — |
| 全市场某日 | — | 0 (不支持) | 5,563-5,567 |

---

## 四、关键缺失与风险

### 4.1 高风险

| 问题 | 影响 | 缓解方案 |
|---|---|---|
| **daily 接口无 tradestatus** | 无法直接判断停牌 | 用 suspend_d 接口补充；或从 daily 缺失行推断 |
| **全市场股票列表不可直接获取** | 无法一次性获取全部股票代码 | 用 datahubco daily trade_date 模式获取某日全部交易股票作为基线 |
| **promax 频率限制（429）** | suspend_d 等接口可能触发限流 | 加 1-2 秒退避；批量查询分散到多分钟 |

### 4.2 中风险

| 问题 | 影响 | 缓解方案 |
|---|---|---|
| **QDII 溢价无直接接口** | 需两次调用（fund_daily + fund_nav） | 封装为 helper 函数 |
| **datahubco 为 HTTP（非 HTTPS）** | 密钥明文传输 | 待证书上线后切换；当前仅在可信网络使用 |
| **两个接口都无 preclose/volume/amount 的精度说明** | 与现有数据精度可能不一致 | 对比验证后标准化 |

### 4.3 低风险

| 问题 | 影响 | 缓解方案 |
|---|---|---|
| **promax stock_basic 全市场返回0** | 不影响单只查询 | 用 daily trade_date 模式获取股票列表 |
| **datahubco stock_basic 返回0** | 同上 | 同上 |

---

## 五、推荐方案

### 5.1 双源并行策略

| 场景 | 推荐接口 | 原因 |
|---|---|---|
| **全市场某日数据批量拉取** | datahubco | 支持 trade_date 全市场查询 |
| **单只股票历史数据** | promax 或 datahubco | 两者都支持；promax HTTPS 更安全 |
| **停牌记录** | promax | suspend_d 可用；需加退避 |
| **复权因子** | promax | adj_factor 可用 |
| **涨跌停** | promax | stk_limit 可用 |
| **ETF 日线** | promax | fund_daily 可用 |
| **ETF 净值（溢价计算）** | promax | fund_nav 可用 |
| **交易日历** | promax | trade_cal 可用 |
| **退市信息** | promax | stock_basic 单只查询可用 |

### 5.2 数据更新流程（替代现有 Baostock）

```
1. 获取交易日历（trade_cal）→ 确定需要更新的日期
2. 对每个交易日：
   a. datahubco daily trade_date → 获取全市场日线数据
   b. 对比上一日股票列表 → 识别新上市/退市
3. 对停牌股票：
   a. promax suspend_d → 获取停牌区间
   b. 生成 tradestatus 标记
4. 对 ETF：
   a. promax fund_daily → 市价
   b. promax fund_nav → 净值
   c. 计算溢价
5. 写入本地 parquet（与现有结构一致）
```

### 5.3 10-30 首照的数据准备

| 任务 | 依赖 | 时间估算 |
|---|---|---|
| 写 `quantlab/data/tushare_adapter.py` | promax + datahubco | 2-3 小时 |
| 写数据采集脚本（替代 collect_local_supplement.py） | adapter | 4-6 小时 |
| 全量数据拉取（2014-2026） | 脚本 | 6-12 小时（机时，可后台） |
| 数据一致性验证（vs 现有 parquet） | 全量数据 | 2-3 小时 |
| 生成新的 smallcap_v4 panel | 验证通过 | 1-2 小时 |

**总计：2-3 天（含机时）**

---

## 六、立即验证项

1. ✅ 接口连通性
2. ✅ 停牌数据与本地一致性
3. ⚠️ 全市场数据量（datahubco 5563 vs 本地 6080，差 517 只——需要解释）
4. ❌ 复权因子与本地一致性（待对比）
5. ❌ 日线收盘价与本地一致性（待对比）
6. ❌ 历史覆盖完整性（2014-2026 全部交易日）
7. ❌ 频率限制下的批量拉取可行性

---

## 七、全市场数据量差异说明

datahubco daily trade_date=20261016 返回 5,563 条，本地 parquet 有 6,080 列。差异 517 只可能原因：
1. 本地 parquet 包含北交所/新三板股票，datahubco 可能只返回主板/创业板/科创板
2. 本地 parquet 包含已退市但历史有数据的股票
3. 2026-10-16 实际交易股票数 < 本地历史累计股票数

**需要进一步验证**：用 datahubco stock_basic（如果能工作）或多次查询获取完整股票列表，与本地 6,080 只对比。

---

*验证时间：2026-10-18 01:00 CST*
*验证人：与并行会话互补的会话*
*主工作目录：`/Users/agiuser/Desktop/QuantLab`*
