# QuantLab

A股与境内股票ETF的可追溯研究及模拟账户系统。60万元、只做多、允许现金、不加杠杆；每周评估机会、每日检查风险。**当前无合格真实数据上的新策略业绩，正式运行返回 `blocked_data`。**

本轮实现状态见 [实施记录](docs/IMPLEMENTATION_V3_2026-09-24.md)、[36项审查裁定](docs/REVIEW_R4_ADJUDICATION_2026-09-24.md)、[数据契约](docs/DATASET_CONTRACT.md)。

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

`repaired_legacy`是修复旧策略对照；`opportunity`是预期收益驱动的新模型，二者不混称同一算法。`buy_hold`、`monthly_equal`、`vol_matched_equal`共用账本。新模型允许现金，行业未知股票拒绝配权。

## 持久前向模拟

```bash
.venv2/bin/python -m quantlab.cli forward-step --book output/paper.sqlite3 --model-version MODEL_HASH --manifest PATH --model RUN/models/fold_0.json
.venv2/bin/python -m quantlab.cli forward-status --book output/paper.sqlite3 --model-version MODEL_HASH
.venv2/bin/python -m quantlab.cli forward-reconcile --book output/paper.sqlite3 --model-version MODEL_HASH --fills fills.json
.venv2/bin/python -m quantlab.cli forward-recover --book output/paper.sqlite3 --model-version MODEL_HASH --manifest PATH --forecasts RUN/forecasts.json
```

模型、投资约束和代码版本固定；断档须显式恢复，恢复记录带重建标记。成交核对尚不等于券商账户资金与权益完整对账。开发日志、历史补跑和合成夹具都不计入六个月冻结验证。不创建自动实盘交易或定时作业。

## 公开行情和看板

`fetch`只维护公开缓存，`--source`仅用于此类抓取，不能改变正式研究输入。`scripts/daily_pulse.py --expected-session YYYY-MM-DD`需来自可靠交易日历的已完成日期。`quote`提供带来源/时间的公开报价；`web`展示同一版本的模拟账本，未验证数据隐藏绩效。

新结果存于 `output/runs/<run_id>/` 或研究套件目录，校验摘要后读取。旧根目录results/signals仍保留为未验证历史证据，不作为当前持仓。

## 参考项目与收益边界

[2026 GitHub排名证据与机制研究](docs/GITHUB_2026_RANKING_AND_MECHANISMS.md)区分原十项目参考清单、2026新建候选表和仍未证实的全年增星TOP10。新增源码机制研究包括NautilusTrader、AKQuant、QuantDinger和OpenAlice；不以热度或代理角色数量选择策略。

最大回撤20%是研究目标。停牌、跌停和跳空可使实际损失超过目标。没有8%—12%、15%—20%或其他已证明的实盘年化预期。
