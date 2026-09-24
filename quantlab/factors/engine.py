"""因子引擎：从数据面板计算 5 大类因子（截面标准化，+越大越看好）。

因子类别：
- Value     价值: E/P B/P S/P FCF_Yield 股息率
- Quality   质量: ROE ROA 毛利率 资产负债率(负) 经营现金流/净利(正)
- Momentum  动量: 12M-1M 动量 3M 动量 1M 反转 A股反转 弱势偏离
- Growth    成长: 单季营收同比 单季净利同比 SUE(盈余惊喜)
- Flow      资金/情绪: 换手率(负) 北向持股变化 20日量能趋势 拥挤度(负)

计算约定：输入为标准多股票面板 {code: OHLCV DataFrame} 或
(close, volume, amount, turnover) DataFrame、以及(可选)财务宽表。
每期在截面(z)上做 winsorize + zscore 后输出"打分越高越优"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from quantlab.config import CONFIG

EPS = 1e-9


def winsorize_zscore(x):
    """截面 Winsorize(±clip σ) + z-score(中位数mad鲁棒)。支持 Series/DataFrame。"""
    if isinstance(x, pd.DataFrame):
        return x.apply(winsorize_zscore, axis=1)
    x = pd.to_numeric(x, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    # 极端情况mad=0时退化为std
    if mad < 1e-9:
        mad = x.std()
        if not np.isfinite(mad) or mad < 1e-9:
            return x * 0.0
    z = (x - med) / mad
    return z.clip(-4.0, 4.0)


# ---------------------------------------------------------------------------
# 行情类因子（纯价量，两大市场通用）
# ---------------------------------------------------------------------------
def factor_value_kline(close: pd.DataFrame, amount: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """基于行情的近似估值信号：长期便宜、累计涨幅温和(均值回归)被视为"价值"。
    有真实财务时由 factor_value_fund 替代。"""
    f = {}
    ret60 = close.pct_change(60, fill_method=None)
    ret250 = close.pct_change(250, fill_method=None)
    f["value_breadth"] = winsorize_zscore(-ret250)         # 长期涨幅大->相对贵
    f["value_medium"] = winsorize_zscore(-ret60)
    # 成交均价比：量缩价稳(低关注)近似低估
    vol_ma = amount.rolling(100).mean().replace(0, np.nan)
    vol_ratio = vol_ma / (amount.rolling(20).mean().replace(0, np.nan) + EPS)
    f["value_lowtension"] = winsorize_zscore(-vol_ratio)
    return f


def factor_momentum(close: pd.DataFrame, skip: int = 21, long: int = 250, short: int = 60, reversal: int = 21) -> dict[str, pd.DataFrame]:
    f = {}
    # 标准 12M-1M：剔除最近 1 个月反转，只用 [t-250, t-21] 区间的收益
    f["momentum_12m1m"] = winsorize_zscore(close.shift(skip) / close.shift(long + skip) - 1)
    f["momentum_3m"] = winsorize_zscore(close.shift(skip) / close.shift(short + skip) - 1)
    f["momentum_1m_reversal"] = winsorize_zscore(-(close / close.shift(reversal) - 1))  # A股短反转占优
    f["momentum_trend"] = winsorize_zscore(close / close.rolling(60).mean() - 1)
    return f


def factor_volatility(close: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    vol = close.pct_change(fill_method=None).rolling(window).std() * np.sqrt(252)
    return winsorize_zscore(-vol)  # 低波动更优(低波异象)


def factor_flow(amount: pd.DataFrame, turnover: pd.DataFrame, window: int = 20) -> dict[str, pd.DataFrame]:
    f = {}
    tov = turnover
    f["flow_turnover_low"] = winsorize_zscore(-tov.rolling(window).mean())     # 高换手=拥挤/贵
    amt_ma20 = amount.rolling(window).mean().replace(0, np.nan)
    amt_ma60 = amount.rolling(60).mean().replace(0, np.nan)
    f["flow_volume_trend"] = winsorize_zscore((amt_ma20 / amt_ma60 - 1))       # 量能温和流入
    return f


def factor_price_crowding(close: pd.DataFrame, turnover: pd.DataFrame,
                          window: int = 20,  hot_run: int = 20, hot_ret: float = 0.25) -> pd.DataFrame:
    """个股拥挤打分(负向)：高换手 + 短期暴涨共振 = 天量见天价(西南证券 IRT 思想)。"""
    tov_rank = turnover.rolling(window).mean().rank(axis=1, pct=True)
    ret_hot = close.pct_change(hot_run, fill_method=None)
    ret_rank = ret_hot.rank(axis=1, pct=True)
    bomb = (tov_rank > 0.9) & (ret_hot > hot_ret)
    crowd = 0.5 * tov_rank + 0.3 * ret_rank + 0.2 * bomb.astype(float)
    return -winsorize_zscore(crowd)  # 越拥挤打分越差


# ---------------------------------------------------------------------------
# 财务类因子（真实财务；缺数据时返回空，由主引擎自动跳过）
# ---------------------------------------------------------------------------
def factor_value_fund(close: pd.DataFrame, ep: pd.DataFrame, bp: pd.DataFrame,
                      fcf: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    f = {}
    if ep is not None and not ep.empty:
        f["value_ep"] = winsorize_zscore(ep)
    if bp is not None and not bp.empty:
        f["value_bp"] = winsorize_zscore(bp)
    if fcf is not None and not fcf.empty:
        f["value_fcf"] = winsorize_zscore(fcf)
    return f


def factor_quality(roe: pd.DataFrame, gm: pd.DataFrame, lev: pd.DataFrame,
                   ocf: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    f = {}
    if roe is not None and not roe.empty and roe.notna().any().any():
        f["quality_roe"] = winsorize_zscore(roe)
    if gm is not None and not gm.empty:
        f["quality_gm"] = winsorize_zscore(gm)
    if lev is not None and not lev.empty:
        f["quality_lever_low"] = winsorize_zscore(-lev)
    if ocf is not None and not ocf.empty:
        f["quality_ocf"] = winsorize_zscore(ocf)
    return f


def factor_growth(sales_yoy: pd.DataFrame, profit_yoy: pd.DataFrame,
                  sue: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    f = {}
    if sales_yoy is not None and not sales_yoy.empty:
        f["growth_sales_yoy"] = winsorize_zscore(sales_yoy)
    if profit_yoy is not None and not profit_yoy.empty:
        f["growth_profit_yoy"] = winsorize_zscore(profit_yoy)
    if sue is not None and not sue.empty:
        f["growth_sue"] = winsorize_zscore(sue)
    return f


# ---------------------------------------------------------------------------
# 高层: 把多个面板算成"截面因子分字典 {因子名: DataFrame(date x code)}"
# ---------------------------------------------------------------------------
class FactorEngine:
    """统一入口。输入面板切片，输出因子分(截面zscore)。"""

    def __init__(self, market: str, meta: pd.DataFrame | None = None, cfg=None):
        self.cfg = cfg or CONFIG.factors
        self.market = market
        self.meta = meta                      # 候选池元信息(code: ep/roe/...)

    def compute(self, close: pd.DataFrame, amount: pd.DataFrame,
                turnover: pd.DataFrame, fund: dict[str, pd.DataFrame] | None = None,
                north: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
        f: dict[str, pd.DataFrame] = {}
        cfg = self.cfg
        f.update(factor_momentum(close, cfg.momentum_skip, cfg.momentum_lookback_long, cfg.momentum_short, cfg.reversal_window))
        f["factor_volatility"] = factor_volatility(close, cfg.vol_window)
        f.update(factor_value_kline(close, amount))
        f.update(factor_flow(amount, turnover, cfg.turnover_window))
        f["factor_price_crowding"] = factor_price_crowding(close, turnover, cfg.turnover_window, cfg.crowding_hot_run, cfg.crowding_hot_ret)

        # 财务类（高质量时覆盖/补充价量近似）
        if self.meta is not None:
            m = self.meta
            md = m if isinstance(m, pd.DataFrame) and "code" in m.columns else m.reset_index()
            md = md.set_index("code") if "code" in md.columns else md
            if "ep" in md.columns:
                f["value_ep"] = self._meta_factor(md, "ep")
            if "bp" in md.columns:
                f["value_bp"] = self._meta_factor(md, "bp")
            if "roe" in md.columns:
                f["quality_roe"] = self._meta_factor(md, "roe")
            if "gross_margin" in md.columns:
                f["quality_gm"] = self._meta_factor(md, "gross_margin")
            if "growth" in md.columns:
                f["growth_sales_yoy"] = self._meta_factor(md, "growth")

        if fund is not None:
            f.update(self._merge_fund_factors(fund))
        if north is not None and not north.empty:
            f["flow_north"] = winsorize_zscore(north.pct_change(fill_method=None).rolling(20).mean())

        # 统一：删除全空因子
        out = {k: v for k, v in f.items() if v is not None and not v.empty and v.notna().any().any()}
        # 全程 float32 省内存
        out = {k: v.astype("float32") for k, v in out.items()}
        return out

    def _meta_factor(self, meta: pd.DataFrame, col: str) -> pd.DataFrame | None:
        s = meta[col]
        if s.isna().all():
            return None
        # 静态特征铺成面板(截面固定) —— 简化：用回测期内最新估值近似
        # 仅作为演示/兜底；真实财务由 _merge_fund_factors 提供动态面板
        return None

    def _merge_fund_factors(self, fund: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """财务宽表 -> 动态横截面因子。fund 为 {code: 长表(日期索引, 指标列)}。"""
        out: dict[str, pd.DataFrame] = {}
        # 按指标列重建面板
        first = next(iter(fund.values()), None)
        if first is None:
            return out
        for col in ["roe", "gross_margin", "eps", "bps", "net_margin", "net_profit_y", "sales_yoy", "profit_yoy"]:
            panel: dict[str, pd.Series] = {}
            for code, df in fund.items():
                if df is None or df.empty or col not in df.columns:
                    continue
                panel[code] = df[col].dropna()
            if not panel:
                continue
            p = pd.DataFrame(panel).ffill()
            if col in ("roe", "gross_margin", "net_margin"):
                out[f"quality_{col}"] = self._cross(p)
            elif col == "eps":
                out["value_ep"] = self._cross(p)
            elif col == "bps":
                out["value_bp"] = self._cross(p)
            elif col == "net_profit_y":
                out["growth_profit_yoy"] = self._cross(p.pct_change(4))
            elif col == "sales_yoy":
                out["growth_sales_yoy"] = self._cross(p)
            elif col == "profit_yoy":
                out["growth_profit_yoy"] = self._cross(p)
        return {k: v for k, v in out.items() if v is not None and not v.empty}

    def _cross(self, p: pd.DataFrame) -> pd.DataFrame:
        return p.apply(winsorize_zscore, axis=1)


if __name__ == "__main__":
    print("factor engine standalone smoke requires a data pipeline; run via tests instead.")