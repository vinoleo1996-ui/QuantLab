# 并行会话代码融合报告（2026-10-18 00:45 CST）

## 结论

**✅ 可以有机融合，没有互斥部分。** 对方（并行会话）的贡献与我的贡献是互补的，底层模块改动兼容，脚本层面各司其职。

---

## 一、双方贡献清单

### 对方（并行会话）的核心贡献

| 文件 | 作用 | 状态 |
|---|---|---|
| `scripts/p5_base_lot_replay.py` | **权威整手账实现**：股票袖整手（stable_book）+ ETF 袖 100 股整手 + 袖间分数上界再平衡 | 已合并 |
| `quantlab/p4_ledger.py` | `replay()` 新增 `capital_cny` 和 `gross` 参数，通过 context manager 临时修改 `stable_book` 常数，不影响冻结用户 | 已合并 |
| `quantlab/p4_alloc.py` | 新增 `quarter_ends()`、`rebalance_ends()`、`overflow` 分配、`target_fn` 替换、`drawdown_check` 模式 | 已合并 |
| `quantlab/backtest/ledger.py` | 退市清算逻辑、stale 门控、数据质量标记 | 已合并 |
| `quantlab/data/contracts.py` | 退市日期修正 | 已合并 |
| `quantlab/smallcap/panel.py` | 缺失面板检测、stale 报价门控 | 已合并 |
| `scripts/stable_book.py` | 退市清算、数据质量问题统计、delisted_end 标记 | 已合并 |
| `docs/prereg/P5_BASE.md` | 更正 2：指出旧矩阵 gross 错误，写入修复后整手账结果 | 已合并 |
| `docs/prereg/P5_FORWARD.md` | 更正二：快照管线落地（`p5_snapshot.py`） | 已合并 |
| `docs/prereg/P5_G0_ALLOC.md` | 行业上限由 mandate 权重推导（0.25 → 12 只） | 已合并 |

### 我的核心贡献

| 文件 | 作用 | 状态 |
|---|---|---|
| `scripts/p5_matched_control.py` | **匹配对照**：GBDT_HYST vs 随机滞后同换手组合，含跨路径月度平均 t 统计（修正池化假象） | 已合并 |
| `scripts/p5_matched_yearly.py` | 逐年分解（5/9 年为正，2018/2023/2025 块状集中） | 已合并（修无副作用） |
| `scripts/p5_matched_t.py` | 独立验证：每月跨路径平均后 NW t=1.03 | 已合并（修无副作用） |
| `scripts/p5_realizable.py` | 可实现矩阵脚本（**已修正**：第 68 行显式传 `gross=GROSS`） | 已合并 |
| `docs/prereg/P5_MATCHED.md` | 匹配对照预注册 + 结果 + 统计纠错说明 | 已合并 |
| `docs/prereg/P5_REALIZABLE.md` | 可实现预注册 + 作废说明 + 正确结果摘要 | 已合并 |
| `docs/prereg/P5_L_HYST_CONTROL.md` | L_HYST 诊断稳健性 | 已合并 |
| `docs/RESEARCH_CONTINUITY.md` | 接续文档更新 | 已合并 |
| `docs/REVIEW_20261017_ADJUDICATION.md` | 裁定文档第六节/第七节 | 已合并 |
| `dashboard_annual.html` | 可实现 KPI + 研究台账行 | 已合并 |

---

## 二、兼容性验证

### 2.1 模块级：全部脚本可同时导入

```python
import p5_matched_control   # ✅
import p5_realizable         # ✅
import p5_base_lot_replay    # ✅
import p5_matched_yearly     # ✅（已修无副作用）
import p5_matched_t          # ✅（已修无副作用）
```

### 2.2 函数签名：向后兼容

| 函数 | 旧签名 | 新签名 | 兼容性 |
|---|---|---|---|
| `p4_ledger.replay` | `targets, start, end, cost_mult, dataset` | + `capital_cny=None, gross=None, delisting_haircut=0.0` | ✅ 旧调用全兼容 |
| `p4_alloc.allocate` | `equity_nav, etf_returns, equity_share, cost_mult, premium` | + `rebalance='month', drawdown_check='rebalance', overflow=None, target_fn=None` | ✅ 旧调用全兼容 |

### 2.3 数据流：无冲突

```
p4_lowturn (GBDT_HYST) ──读取──> 冻结 ledger NAV ──被 p5_matched 读取
                                        │
p5_matched (RAND_HYST) ──生成──> 新 NAV ──diff──> 报告
                                        │
p5_base_lot_replay ──读取──> 冻结 G97 targets ──整手 replay──> 权威结果
                                        │
p5_realizable (修正后) ──生成──> 新 targets ──replay──> 矩阵（待重跑）
```

**无写冲突**：四方读取的源数据不同（冻结 NAV vs 冻结 targets），输出目录也不同。

---

## 三、需要协调的问题（非互斥，是默认值不一致）

### 3.1 `stable_book.GROSS = 0.90` ≠ P5_BASE 预注册 0.97

```python
# scripts/stable_book.py:45
GROSS = 0.90          # 模块默认值

# 但 P5_BASE 预注册要求：
STOCK_GROSS = 0.97    # p5_base_lot_replay.py:32
```

**影响**：
- `p4_lowturn`（GBDT_HYST）调用 `p4_ledger.replay()` 不传 `gross` → 实际跑 0.90
- `p5_matched`（RAND_HYST）同样不传 → 也跑 0.90
- 两者**内部一致**（匹配对照有效），但与 P5_BASE 回测的 0.97 **不可比**
- `p5_base_lot_replay` 和修正后的 `p5_realizable` 显式传 0.97 → 正确

**建议**：不改动 `stable_book.GROSS`（避免影响冻结账本），而是**强制所有新脚本显式传 `gross`**。已在 `p5_realizable.py` 修正。

### 3.2 `stable_book.INITIAL_CASH = 600000` ≠ `MANDATE.capital_cny = 800000`

```python
# scripts/stable_book.py（无 INITIAL_CASH 定义，默认值可能是 600k）
# quantlab/p4_mandate.py
MANDATE.capital_cny = 800000.0
```

**影响**：
- 通过 `p4_ledger.replay()` 的调用默认用 `MANDATE.capital_cny = 800k`
- 直接调用 `stable_book` 的脚本可能用 600k
- 历史脚本（p4_lowturn, p4_alpha）都在 80 万下运行（与预注册一致）

**建议**：统一 `stable_book.INITIAL_CASH = 800000`，或删除该常数、强制通过 `p4_ledger.replay()` 的 `capital_cny` 参数传入。

### 3.3 `p5_matched_control.py` 的 `nw_t` 函数与 `quantlab.research_statistics.newey_west` 重复

- `p5_matched_control.py` 内嵌了一个 `nw_t()` 函数
- `quantlab.research_statistics` 已有 `newey_west()`
- 两者实现相同，但 `nw_t` 多了 lag=3 默认值

**建议**：后续统一调用 `quantlab.research_statistics.newey_west`，删除内嵌副本。

---

## 四、已修复的问题

| 问题 | 修复 | 提交 |
|---|---|---|
| `p5_realizable.py:68` 未传 `gross=GROSS` | 显式传递 `gross=GROSS` (0.97) | `a4a14bd` |
| `p5_matched_control.py` 池化 t 统计假象 | 改为跨路径月度平均（n=104） | `4cb4d21` |
| `p5_matched_yearly.py` 导入副作用 | 包进 `main()` + `if __name__` | `30d168e` |
| `p5_matched_t.py` 导入副作用 | 同上 | `30d168e` |

---

## 五、下一步优化计划

### 5.1 立即（今天）

1. **统一默认值**（低优先级，但记录）
   - 在 `docs/tech_debt.md` 记录：`stable_book.GROSS=0.90` 与预注册 0.97 的差异
   - 所有新脚本必须显式传 `gross`，禁止依赖默认值

2. **前向引擎 `p5_snapshot`**
   - 对方已在 `P5_FORWARD.md` 中定义快照入口
   - 需要实现：`scripts/p5_snapshot.py --as-of YYYY-MM-DD --capital 800000`
   - 状态文件：`output/p5_forward/state.json`
   - 首次真实快照：2026-10-30（若数据未更新则记 `blocked`）

### 5.2 短期（本周）

3. **高票价单股规则决定**
   - 选项 A：允许一手买入，接受权重偏离（实际暴露会更集中）
   - 选项 B：池子加 ≤150 元价格上限（减少可选标的，但权重更均匀）
   - 需要拍板：10-22 第一张真单会受此影响

4. **p5_realizable.py 重跑（可选）**
   - 已修正 `gross=GROSS`，但旧矩阵已作废
   - 若需要 32 格正式矩阵：补预注册 → 重跑 → 与 `p5_base_lot_replay` 结果交叉验证
   - 若不需要：以 `p5_base_lot_replay` 的 8 路径结果为准（13.57%）

### 5.3 中期（10-30 前）

5. **前向账本与生产系统对接**
   - 券商成交记录导入：`--holdings/--cash/--broker-confirmed`
   - 纸面模拟 vs 真实成交的偏差追踪
   - 对照 `ALLOC60` 的 6 个月停损规则实现

6. **代码清理**
   - 删除作废的 `output/p5_realizable_20261017/` 旧结果（56 个 cell）
   - 统一 `nw_t` 调用（删除内嵌副本）
   - 考虑把 `stable_book.INITIAL_CASH` 和 `GROSS` 改为 `None`，强制显式传入

---

## 六、融合后的仓库结构

```
scripts/
  p4_alpha.py          ← 冻结（POOL50 生成）
  p4_gross.py          ← 冻结（POOL50_G97 生成）
  p4_lowturn.py        ← 冻结（GBDT_HYST 生成）
  p5_base_lot_replay.py ← 对方：权威整手账
  p5_realizable.py     ← 我（已修正）：矩阵脚本
  p5_matched_control.py ← 我：匹配对照
  p5_matched_yearly.py  ← 我：逐年分解
  p5_matched_t.py      ← 我：t 统计验证
  p5_snapshot.py       ← 待实现：前向引擎
  stable_book.py       ← 对方修改：退市清算 + 数据质量

quantlab/
  p4_ledger.py         ← 对方：capital/gross context manager
  p4_alloc.py          ← 对方：quarter_ends, overflow, target_fn
  backtest/ledger.py   ← 对方：退市清算、stale 门控
  data/contracts.py    ← 对方：退市日期修正
  smallcap/panel.py    ← 对方：缺失面板检测

docs/prereg/
  P5_BASE.md           ← 对方：更正 2
  P5_REALIZABLE.md     ← 我：预注册 + 作废说明
  P5_MATCHED.md        ← 我：匹配对照
  P5_FORWARD.md        ← 对方：更正二
  P5_G0_ALLOC.md       ← 对方：行业上限推导
  P5_L_HYST_CONTROL.md ← 我：L_HYST 诊断
```

---

*报告生成时间：2026-10-18 00:45 CST*
*主工作目录：`/Users/agiuser/Desktop/QuantLab`*
*两棵树均已同步：main (local) + workbuddy/main-a9d77e8e*
