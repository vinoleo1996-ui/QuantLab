> **2026-09-24 实施状态**：已改为现金/股数账本、严格数据契约和版本化研究输出。现有缓存缺少PIT原始价格数据集，默认回测会返回 `blocked_data`。旧年化和专家有效性结论未经验证；新机会/公告模型尚未启用。参见 [实施记录](docs/IMPLEMENTATION_2026-09-24.md) 与 [数据契约](docs/DATASET_CONTRACT.md)。

# QuantLab

**A股 + 美股 双市场、中期持仓(~25只)、目标年化15-20%、回撤可控 的多因子量化选股系统（信号级：输出买卖建议清单，人工执行）。**

## 快速开始

```bash
cd QuantLab
# 1. 创建环境并安装依赖（已装则跳过）
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 2. 回测（真实数据优先）
./.venv/bin/python -m quantlab.cli backtest        # 生成 output/report.html
./.venv/bin/python -m quantlab.cli signals         # 输出本期买卖建议清单 + output/signals.json

# 3. 日常使用
./.venv/bin/python scripts/daily_pulse.py          # 每日收盘后：更新候选池+日线+重算专家意见
./.venv/bin/python -m quantlab.cli quote [--symbols 600519,000001]  # 实时行情(缺省=当前持仓)
./.venv/bin/python -m quantlab.cli monitor         # 随时重算专家今日意见(out: output/monitor.json)
./.venv/bin/python -m quantlab.cli web             # 打开 http://127.0.0.1:8899（四合一看板·全中文）

# 4. 指定数据源
./.venv/bin/python -m quantlab.cli backtest --source akshare   # 或 yfinance
```

真实数据为多后端 fallback 链路（全部为真实来源；所有真实后端失败时会直接报错，不会回退到任何离线人造数据）：
`AKShare(东财)` → `新浪快照(候选池)` → `腾讯前复权K线` → `东财curl_cffi直连`。
- 新浪 `Market_Center.getHQNodeData`(hs_a 分页) 提供全A实时候选池(流通市值/换手/成交额)；
- 腾讯 `fqkline` 提供按起止日期分段的前复权日线；
- 东财 WAF 可能按客户端指纹/出口风控断开 `requests`，故保留了 curl_cffi(浏览器指纹) 直连后端与失败快速退出。
- yfinance 在当前网络时常 429 限流；美股源失败时直接报错并要求人工处理网络，不做人造回退。

## 目录

```
quantlab/           核心代码
  config.py         全部参数（资金/股票池/因子/组合/成本/风控）
  data/             DataFeed 数据适配(AKShare/yfinance/多后端真实链路，无合成回退)，parquet缓存
  factors/          5大类多因子引擎(价值/质量/动量/成长/资金情绪)
  strategy/         ICIR动态加权 + 因子拥挤惩罚 + 目标波动率组合 + 分级风控
  backtest/         月度调仓回测引擎(涨跌停可交易掩码)
  report/           绩效指标 + HTML可视化报告
  webapp/           Flask 轻量看板
scripts/            fetch_data / run_backtest / generate_signals / serve_webapp
docs/
  STRATEGY.md       策略白皮书（机制与参数说明）
  RESEARCH.md       调研笔记（吸收的高星项目 + 2026论文依据）
tests/              冒烟测试
```

## 三大差异化机制（详见 docs/STRATEGY.md）

1. **涨跌停可交易掩码**：涨停开盘不可买、跌停开盘不可卖，回测不再虚高收益（arXiv:2507.07107）。
2. **拥挤度惩罚**：个股拥挤打分 + 因子IC两两相关监控（Quant Decoded FCI / 东北CNEX / 西南动态分域），防止 2026 式风格踩踏。
3. **目标波动率组合**：Ledoit-Wolf 收缩协方差 + 动态ICIR 加权，市场弱势时风控分级降仓。

## 风险提示

本系统输出**信号级建议**，仅供研究参考，不构成投资建议；实盘请结合自身资金状况、合规与数据时效性人工复核后再执行。