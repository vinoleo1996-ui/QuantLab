# QuantLab

A股与境内股票ETF的可追溯研究及模拟账户系统。60万元、只做多、允许现金、不加杠杆。**开发主线为原多因子策略的缺陷修复与逐项改进；支持分层的免费真实数据研究，严格数据验收和实盘晋升仍未通过。**

当前开发顺序以[策略规范](docs/STRATEGY.md)首节为准；[V3实施记录](docs/IMPLEMENTATION_V3_2026-09-24.md)保留此前状态，[数据契约](docs/DATASET_CONTRACT.md)约束研究输入。研究默认入口为`legacy_repaired`：原多因子/专家、月末评估和目标波动率配权共用现金股数账本，不需要Forecast。`opportunity`显式指定并提供预测文件。旧审查归档于docs/archive，不再维护多份现行方案。

## 原策略与全市场机会范围

24/120只固定样本仅用于历史诊断与回归测试，不代表A股机会范围。当前优先接入全历史股票目录及行情，按决策日上市资格、ST/退市状态和流动性筛选；保持原多因子/专家/月度策略，不先在小样本上继续择参。采集器和比较入口均无默认120只上限。全市场数据接入和回放尚未完成。

```bash
# MANIFEST替换为实际接入后的数据清单；证券数量不证明覆盖完整
.venv2/bin/python -m quantlab.cli backtest --manifest MANIFEST --data-mode free_research
.venv2/bin/python -m quantlab.cli compare-legacy --manifest MANIFEST
# 仅重放旧的120只诊断样本时显式指定：
.venv2/bin/python -m quantlab.cli compare-legacy --manifest data/free_research_120/dataset/manifest.json --diagnostic-sample
```

`compare-legacy`固定2020-01-02至2026-06-30，运行原策略、月度等权、波动控制等权各两组行业政策。`--industry-policy omit_for_diagnostic`只在免费研究中显式关闭行业上限，保留Mandate其余限制、同跌群组与真实成交检查；不是取消全部风控，也不是获准交易。默认`enforced`保留25%行业上限。

每组输出包含逐日合格股票数量、目标仓位、实际成交、剩余现金、限制与未成交原因、半年收益、股票贡献及退出耗时。旧120只结果仍属条件诊断；修复版不是旧14.62%的精确复刻，也不因此成为已证明盈利的策略。

已有本地大库的只读盘点工具：`scripts/audit_local_market.py --database DB --sample-raw data/free_research_120/raw`（使用装有DuckDB的Python）。它导出候选目录及覆盖/价量对账记录，不将Qlib上市日期、复权因子或当前行业直接当作历史已验证事实。

## 环境与验证

本机使用 `.venv2` 的Python3.12。旧 `.venv` 的Python3.14数值包存在二进制问题，不用于研究验收。

```bash
python3.12 -m venv .venv2
.venv2/bin/python -m pip install -r requirements-research.lock
MPLCONFIGDIR=/private/tmp/quantlab-mpl .venv2/bin/python -m unittest discover -s tests -v
.venv2/bin/python scripts/audit_parameters.py --check
```

公开数据适配器为可选依赖，见 `requirements-data.txt`；本机离线研究锁不包含所有网络后端。接口可用不等于PIT数据合格。采购和密钥授权另行决定。

## 正式研究入口

### 免费数据首轮回放

```bash
# 优先复用通过核验的本地行情，仅对缺口补采。CATALOG须包含完整历史候选范围。
# 不传--count时采集目录内所有A股代码，不按2014年或主板截断。
.venv2/bin/python scripts/collect_free_research.py --members CATALOG --output data/free_market/raw
# 交易规则与权益字段补齐后再生成回放数据；目录齐全不等于行情/规则齐全。
.venv2/bin/python scripts/build_free_dataset.py --raw data/free_market/raw --output data/free_market/dataset
```

此前24只样本及Ridge实验保留作历史诊断，不再作为当前开发主线。历史资格＋流动性池不要求历史市值全覆盖；行业、公告及执行容量分别验收。当前未核实的行业仍不得用今天的行业回填历史。

输出`conditional_metrics`是公开数据与日线开盘假设下的账面路径，`metrics=null`、`promotion_approved=false`。退市末价只用于带陈旧标记的估值，不结算成现金；缺失权益不声称损失有界。行业、公告和执行容量分别验收，不再要求首轮把所有模块的数据一次补齐。

### 扣费收益与稳定性对照

```bash
# 以下显式使用旧诊断样本，不能称为全市场策略研究：
.venv2/bin/python -m quantlab.cli compare-strategies
```

在扩大后的事前样本上，按36/6/6滚动验证选择Ridge正则强度，并列报告冻结模型、旧评分排序、简单趋势排序、入场权重缓冲与等权基准。排序对照保留相同的预测收益分布；每个测试段、成本、现金、未成交、20/60日分块区间及执行压力均保留。不会自动将测试段赢家设为默认策略。

2026-09-24已完成120只历史初始成员的七组对照及退出修复后的固定预测核验，见[本轮决策记录](output/exit_fix_replay/73b5fc3a624d2983ad82f480/DECISION.md)。旧排序仅是继续研究的候选；Ridge、5.5%入场缓冲均未证明改进，未晋级任何策略。普通调仓现在保留未完成的风险卖单，已知ST/失效持仓生成退出需求，实际成交仍受停牌等条件约束。

### 严格验收入口

```bash
.venv2/bin/python -m quantlab.cli data-audit --manifest data/research/manifest.json
.venv2/bin/python -m quantlab.cli research --manifest data/research/manifest.json
.venv2/bin/python -m quantlab.cli backtest --manifest data/research/manifest.json
```

缺清单、历史资格、原始价/公司行动、开盘状态或覆盖证据时拒绝发布绩效，不从今日TOP40缓存生成“合格历史池”。供应商真实样本及缺口见 [数据供应商核验](docs/DATA_PROVIDERS_2026-09-24.md)。

`research`默认：36/6/6月滚动、20日标签、等权校准及Ridge0.1/1/10、最多12个预登记配置、1000次随机对照、20次机械预检、9类执行压力。参数选择只用训练/验证；历史2020—2026已经被反复查看，结果只能提供回顾性证据。

## 模型与共同信号

```bash
.venv2/bin/python -m quantlab.cli backtest --manifest PATH --strategy opportunity --forecasts RUN/forecasts.json
.venv2/bin/python -m quantlab.cli signals --manifest PATH --strategy opportunity --forecasts RUN/forecasts.json
.venv2/bin/python -m quantlab.cli monitor --manifest PATH --strategy opportunity --forecasts RUN/forecasts.json
.venv2/bin/python -m quantlab.cli evidence-audit --bundle RUN/evidence/bundle.json
```

替换占位路径后使用。回测、信号和监控共用评分、预测及账本；所有输出有版本与状态。方向来自待执行订单，行情未知、跌停或现金不足仍可能不成交。信号始终标记simulation和 `executable=false`。

研究默认策略为`legacy_repaired`，原`repaired_legacy`名称不再使用。显式`opportunity`缺冻结预测就阻断，不能退回原策略；`buy_hold`、`monthly_equal`、`vol_matched_equal`共用账本。原策略采用原专家温度公式，`expert_weight_policy=uncertainty`才启用独立收缩候选；该参数及差异记录在运行配置中。

## 持久前向模拟

```bash
.venv2/bin/python -m quantlab.cli forward-step --book output/paper.sqlite3 --model-version MODEL_HASH --manifest PATH --model RUN/models/fold_0.json
.venv2/bin/python -m quantlab.cli forward-status --book output/paper.sqlite3 --model-version MODEL_HASH
.venv2/bin/python -m quantlab.cli forward-reconcile --book output/paper.sqlite3 --model-version MODEL_HASH --fills fills.json
.venv2/bin/python -m quantlab.cli forward-recover --book output/paper.sqlite3 --model-version MODEL_HASH --manifest PATH --forecasts RUN/forecasts.json
```

当前优先完成免费扩池和同账本扣费对照，前向工具待数据/研究验收后使用。模型、投资约束和代码版本固定；断档须显式恢复，恢复记录带重建标记。成交核对尚不等于券商账户资金与权益完整对账。开发日志、历史补跑和合成夹具都不计入六个月冻结验证。不创建自动实盘交易或定时作业。

## 公开行情和看板

`fetch`只维护公开缓存，`--source`仅用于此类抓取，不能改变正式研究输入。`scripts/daily_pulse.py --expected-session YYYY-MM-DD`需来自可靠交易日历的已完成日期。`quote`提供带来源/时间的公开报价；`web`展示同一版本的模拟账本，未验证数据隐藏绩效。

新结果存于 `output/runs/<run_id>/` 或研究套件目录，校验摘要后读取。旧根目录results/signals仍保留为未验证历史证据，不作为当前持仓。

## 参考项目与收益边界

[2026 GitHub排名证据与机制研究](docs/archive/GITHUB_2026_RANKING_AND_MECHANISMS.md)区分原十项目参考清单、2026新建候选表和仍未证实的全年增星TOP10。新增源码机制研究包括NautilusTrader、AKQuant、QuantDinger和OpenAlice；不以热度或代理角色数量选择策略。

最大回撤20%是研究目标。停牌、跌停和跳空可使实际损失超过目标。没有8%—12%、15%—20%或其他已证明的实盘年化预期。
