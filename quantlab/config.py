"""QuantLab 全局配置。

兼容参数集中定义；支持状态以 config_validation.py 和参数审计为准。
未实施或固定字段的变更会报错，不能靠修改配置静默启用新模型。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
A_STOCK_DIR = DATA_DIR / "a_stock"
US_STOCK_DIR = DATA_DIR / "us_stock"
FUND_DIR = DATA_DIR / "fundamentals"
OUTPUT_DIR = ROOT / "output"
for _d in (A_STOCK_DIR, US_STOCK_DIR, FUND_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 资金与总体约束
# ---------------------------------------------------------------------------
@dataclass
class Capital:
    init_cash: float = 600_000.0          # 初始资金(元)，A股以人民币计、美股以折算美元计
    a_share_ratio: float = 1.00           # A股资金占比
    us_ratio: float = 0.00                # 美股资金占比
    usd_cny: float = 7.20                 # 汇率(仅用于换算展示)
    target_positions: int = 23            # 总持仓只数目标
    a_share_positions: int = 20           # A股持仓只数
    us_positions: int = 5                 # 美股持仓只数
    rebalance_rule: str = "weekly"       # 调仓频率: monthly | every_n_weeks
    rebalance_n_weeks: int = 4
    min_holding_period_days: int = 20     # 调出候选的最小持有天数(抑制频繁换手)


# ---------------------------------------------------------------------------
# 股票池过滤（用于构建候选池）
# ---------------------------------------------------------------------------
@dataclass
class Universe:
    a_exclude_st: bool = True             # A股剔除 ST/*ST
    a_min_mktcap_yi: float = 60.0         # A股最小流通市值(亿)
    a_min_avg_amount_wan: float = 3000.0  # A股最小日均成交额(万元)
    a_min_listing_days: int = 60          # A股剔除次新(上市<60天)
    a_market_price_stop: bool = True      # 启用涨跌停可交易掩码
    us_min_avg_dollar: float = 20e6       # 美股最小日均成交额($)
    us_min_mktcap_my: float = 5e9         # 美股最小市值($5B)
    use_constit: bool = True              # 仅使用指数成分示意池(见data层)


# ---------------------------------------------------------------------------
# 数据源
# ---------------------------------------------------------------------------
@dataclass
class DataSource:
    cache_dir: Path = A_STOCK_DIR
    a_daily_adjust: str = ""           # 原始成交价格；公司行动和特征价格由版本数据集提供
    a_freq: str = "daily"
    max_retry: int = 4
    retry_wait: float = 1.5               # 秒，AKShare 反爬节流
    reset_cache: bool = False             # 默认走本地缓存；如需强刷用 CLI --reset-cache
    history_years: int = 6                # 回测数据起始年数
    rate_limiter_sec: float = 0.3         # 逐票请求间隔，避免被封


# ---------------------------------------------------------------------------
# 因子配置
# ---------------------------------------------------------------------------
@dataclass
class FactorConfig:
    ic_window: int = 60                   # 滚动 IC/ICIR 评估窗口(调仓期数, 月频≈60个月=5年)
    icir_floor: float = 0.30              # ICIR 下限，低于则权重压缩
    icir_cap: float = 1.60                # ICIR 上限，防单因子权重爆炸
    momentum_lookback_long: int = 250     # 12M-1M 动量窗口(交易日)
    momentum_skip: int = 21               # 跳过最近1个月(避开反转)
    momentum_short: int = 60              # 3M 动量
    reversal_window: int = 21             # 1M 反转因子窗口
    vol_window: int = 60                  # 波动率窗口
    turnover_window: int = 20             # 换手率窗口
    crowding_quantile: float = 0.95       # 拥挤剔除线(>95%分位剔除)
    crowding_quantile_mid: float = 0.60   # 拥挤降权起点
    crowding_hot_run: int = 20            # 短期涨幅窗口(与换手共振)
    crowding_hot_ret: float = 0.25        # 短期涨幅阈值
    crowding_corr_window: int = 60        # 因子收益两两相关监控窗口(调仓期)
    crowding_corr_alert: float = 0.25     # 因子相关拥挤预警阈值
    factor_decay_lookback: int = 24       # 因子失效判定窗口(IC 期数)
    expert_prior_strength: float = 12.0
    expert_temperature_floor: float = 0.05
    use_experts: bool = False              # 多专家集成打分(动量/价值/低波/流动/综合, 专家级IC加权)
    expert_ic_lookback: int = 12          # 专家质量评估的最近 IC 期数
    neutralize_by_sector: bool = False     # 因子评分前行业/市值中性化(需 meta.sector 真实行业)
    neutralize_by_cap: bool = False        # 市值分位桶去均值(小市值同源暴露剥离)


# ---------------------------------------------------------------------------
# 组合构建
# ---------------------------------------------------------------------------
@dataclass
class PortfolioConfig:
    target_vol: float = 0.14              # 目标年化波动率
    vol_floor: float = 0.09               # 波动率下限(裸仓上限)
    vol_est_window: int = 120             # 已实现波动估计窗口
    weight_max_dev: float = 0.015         # 个股相对等权最大偏离(1/N ± Δ)
    weight_floor: float = 0.0             # 单票最小权重
    max_sector_weight: float = 0.25       # 行业权重大小上限
    max_market_weight: float = 1.00       # 单市场(国家)权重上限
    turnover_rank_thresh: int = 5         # 排名变动超过该值才替换
    n_candidates_pad: int = 3             # 候选池比目标多 N 只(供风控替换)
    lw_shrinkage: bool = True             # 使用 Ledoit-Wolf 收缩协方差
    candidate_pool: int = 0              # 0=全部历史合格证券；仅诊断快照池允许截断


# ---------------------------------------------------------------------------
# 交易成本
# ---------------------------------------------------------------------------
@dataclass
class Cost:
    a_commission: float = 0.00025         # A股佣金(双边,万2.5)
    a_stamp_tax: float = 0.0005           # A股印花税(卖出,万5)
    a_slippage: float = 0.003             # A股滑点(单边30bp)
    us_commission: float = 0.0001         # 美股佣金
    us_slippage: float = 0.0015           # 美股滑点
    min_trade_value: float = 1000.0       # 最小交易金额
    cost_rounding: str = "floor"        # 手数取整: nearest | floor


# ---------------------------------------------------------------------------
# 风控
# ---------------------------------------------------------------------------
@dataclass
class RiskConfig:
    stop_loss_single: float = -0.16       # L1 单票止损线(相对成本)
    drawdown_alert: float = 0.10          # L2 组合回撤>10% -> 目标波动降档
    drawdown_reduce: float = 0.15         # 组合回撤15% -> 风险仓位上限60%
    drawdown_exit: float = 0.18            # 风险仓位上限30%
    drawdown_halt: float = 0.20           # 回撤20%：退出并人工复核
    compound_down: float = 0.30           # RESERVED：旧Regime参数，新每日风控不读取
    vol_reduction_factor: float = 0.75    # L2 目标波动缩放
    regime_window: int = 20               # 市场状态均线窗口
    regime_vol_quantile: float = 0.80     # 波动率高位分位(用于状态判断)


@dataclass
class ResearchConfig:
    model_version: str = "opportunity-ledger-v3"
    feature_profile: str = "price_volume_industry"
    horizon: int = 20
    train_months: int = 36
    validation_months: int = 6
    test_months: int = 6
    max_trials: int = 12
    ridge_alpha: float = 10.0
    min_train_rows: int = 200
    min_train_dates: int = 40
    max_stock_weight: float = 0.06
    max_etf_weight: float = 0.20
    max_etfs: int = 3
    max_cluster_weight: float = 0.30
    cash_buffer: float = 0.02
    participation: float = 0.01
    entry_percentile: float = 0.90
    exit_percentile: float = 0.70
    exit_reviews: int = 2
    downside_corr: float = 0.70
    risk_aversion: float = 5.0
    crowding_cost: float = 0.01
    liquidity_cost: float = 0.01
    recovery_buffer: float = 0.03
    recovery_sessions: int = 5
    forward_months: int = 6
    enable_announcements: bool = False
    universe_mode: str = "pit_cap"  # pit_cap | pit_liquidity | snapshot_research


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
@dataclass
class Config:
    capital: Capital = field(default_factory=Capital)
    universe: Universe = field(default_factory=Universe)
    data: DataSource = field(default_factory=DataSource)
    factors: FactorConfig = field(default_factory=FactorConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    cost: Cost = field(default_factory=Cost)
    risk: RiskConfig = field(default_factory=RiskConfig)
    research: ResearchConfig = field(default_factory=lambda: ResearchConfig())
    seed: int = 42
    start_date: str = "2020-01-01"        # 回测起点
    benchmark_a: str = "000300"           # A股基准: 沪深300
    benchmark_us: str = "SPY"             # 美股基准


CONFIG = Config()
