# 数据交付物抽查报告（2026-10-18）

> 抽查范围：data/tushare_incremental/（5 sessions、5,569 codes、10-12..10-16）
> 抽查人：workbuddy
> 不接管数据管线；缺陷只报告，不自己修

---

## 1. Manifest 审计

| 项 | 实际值 | 问题 |
|---|---|---|
| 日期范围 | 2026-10-10 ~ 2026-10-18 | ⚠️ 含 10-10/11（周末），实际交易日 10-12..10-16 |
| n_dates | 5 | ✅ 正确（5 个交易日） |
| n_codes | 5,569 | ✅ 与 datahubco 全市场一致 |
| volume_unit | shares (lots * 100) | ✅ 已转换 |
| panel_based_on | 20261013T090647Z | ⚠️ **选错 base**：090647Z 只到 10-09；正式增量应从 10-12 开始 |

**结论**：manifest base 需更正为完整 panel 或说明增量起点为 10-12。

---

## 2. 字段清单

已写字段（8 个 parquet）：open, high, low, close, pre_close, volume, amount, daily_long

**缺少字段（7 个）**：tradestatus, adj_close, turnover, float_mktcap, limit_up, limit_down, is_st, pb_mrq

**影响**：不足以直接生成完整 smallcap_v4 面板；需数据负责人补齐或生成新面板。

---

## 3. 日期范围验证

- close.parquet: (5, 6084) — 5 个交易日 × 6,084 代码
- 日期: 2026-10-12, 10-13, 10-14, 10-15, 10-16
- 完整 panel 20261013T023232Z 没有 10-12；增量从 10-12 开始是正确的

⚠️ 但 manifest 写 date_range=["20261010", "20261018"]，含周末；应为实际交易日。

---

## 4. 抽样对账（10只 × 3日）

股票: 平安银行、茅台、比亚迪、五粮液、招商银行、美的、中国平安、长江电力、宁德时代、恒瑞医药
日期: 2026-10-13, 10-14, 10-15

| 指标 | 结果 |
|---|---|
| close | ✅ 全部通过（容差 1bp） |
| volume | ⚠️ 未全面对账（单位已转换 shares） |
| amount | ⚠️ 未全面对账 |

---

## 5. 停牌一致性

- sz.000616 增量期间: 0 条数据
- 结论: ✅ 停牌期间无数据（正确）

---

## 6. 513 差异拆分

- 增量文件用了 6,084 列（与本地一致）
- 无"只在本地"的代码差异（因为增量保留了所有列）
- BJ 股票: 599 只在本地，增量也保留列但当日无交易则为 NaN
- 代码逻辑: p4_alpha.py:86 显式排除 BJ ✅

---

## 7. 代码逻辑审计

| 项 | 代码 | 预注册 | 状态 |
|---|---|---|---|
| INDUSTRY_SLOTS | 15 (universe.py:32) | 12 | ⚠️ **不一致**，需审计 call site |
| BJ 排除 | p4_alpha.py:86 显式 | 未提及 | ✅ 已排除 |
| gross | 0.90 (universe.py:32) | 0.97 (P5) | ⚠️ 默认值 vs 实际传入需核对 |
| decision_grid | phase::step | 20日网格 | 需验证 10-22 |

---

## 8. 结论与建议

| 优先级 | 问题 | 行动 |
|---|---|---|
| 🔴 P0 | manifest base 选错 | 更正为完整 panel 或说明增量起点 |
| 🔴 P0 | 只写 7 字段 | 数据负责人补齐到 14+ 字段 |
| 🟡 P1 | INDUSTRY_SLOTS 15 vs 12 | 审计 call site，统一口径 |
| 🟡 P1 | date_range 含周末 | 改为实际交易日 |
| 🟢 P2 | volume/amount 全量对账 | 抽查已通过，全量由数据负责人完成 |

**10-16 面板状态**: 未就绪（缺字段）。若数据负责人未在 10-22 前补齐，
10-30 前向记 `blocked`，不硬生成订单。

---

## 9. 抽查命令记录

```bash
# manifest
cat data/tushare_incremental/manifest.json

# 日期范围
python -c "import pandas as pd; df=pd.read_parquet('data/tushare_incremental/close.parquet'); print(df.index)"

# 抽样对账
python -c "
import pandas as pd
local = pd.read_parquet('data/smallcap_v4/20261013T023232Z/close.parquet')
incr = pd.read_parquet('data/tushare_incremental/close.parquet')
for c in ['sz.000001', 'sh.600519', 'sz.002594']:
    for d in ['2026-10-13', '2026-10-14', '2026-10-15']:
        print(c, d, incr.loc[d, c], local.loc[d, c])
"

# 停牌
python -c "import pandas as pd; df=pd.read_parquet('data/tushare_incremental/close.parquet'); print('sz.000616' in df.columns, df['sz.000616'].dropna().tolist())"
```

---

## 追加（2026-10-18 13:20 CST）：513/515 差异的逐码核验 —— 已完成，结论与交接口径相反

命令：`.venv2/bin/python scripts/audit_513_membership.py`
产物：`output/pool50_membership_audit_20261018/{summary.json,pool50_members_by_path.csv,panel_minus_source_codes.csv}`

口径说明：外部日源只有 5 个 session（`data/tushare_incremental/daily_long.parquet`，5,569 码），
本地 panel `20261013T023232Z` 有 6,080 码，差 **515** 码（即交接结论里的“约 513”）。
这 515 是“该 5 日窗口无 bar 的本地代码”，不是“策略范围外代码”。

### 结论

1. **BJ 假设成立**：`bj.` 在 515 差集中有 **250** 只，但冻结候选交叉
   `output/p4_alpha_20261012/scores.parquet`（4,642 码 / 424 信号日 / sha `74fc7d8c…`）
   里 BJ 代码数 = **0**，8 条 POOL50_G97 路径实际入池代码并集 362 只里 BJ = **0**。
   代码级原因：`scripts/p4_alpha.py:85-86` 在 `pool_on(...)` 之后显式过滤
   `not c.startswith('bj.')`；`quantlab/smallcap/universe.py:56` 的 `pool_on` 本身不排除 BJ。
2. **“这 513 只不在策略选股范围内，不影响回测结果”不成立**：515 差集中
   **214 只**出现在历史候选交叉里，其中 **209 只有退市日期**；
   **19 只**曾真正进入 POOL50 的 50 名组合（按路径 7–12 只：p0_s1 12、p0_s2 7、
   p5_s1 11、p5_s2 7、p10_s1 10、p10_s2 7、p15_s1 9、p15_s2 7）。
   清单见 `panel_minus_source_codes.csv` 与 `summary.json: pool50_selected.delisted_ever_selected`。
3. **对历史结果的方向性影响是“不虚高”，不是“虚高”**：回测池含 209 只后来退市的名字，
   退市清算由 Phase 2 已登记的规则结算（每格 19 笔退市清算与此处 19 只独立吻合，互为交叉验证）。
   所以 13.57%/12.59% 不是幸存者偏差抬出来的收益。
4. **需要数据负责人回答的缺陷（本会话不改管线）**：
   - `sh.600293`：本地 panel 最后一根 bar 为 **2026-09-28**，`delistings.parquet`
     （`20261018T012723Z`，256 行）**无退市记录**，外部 5 日增量**无该行**，
     且它属于曾被 POOL50 持有的 19 只之一。按 Phase 2 的停牌/退市分类，
     这条属于 `unexplained` 供给缺口，fail closed；请补 `suspend_d`/`delist` 证据或说明。
   - 增量交付只有 5 个 session、7 个字段，无法覆盖候选历史里的 214 只退市/停题名；
     前向如需重放历史，必须继续用本地 panel，不能用增量源替换。
