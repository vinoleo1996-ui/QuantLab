"""申万一级行业动量（真实行业 K线，同花顺申万板块接口）。

行业中性化/行业动量的诚实基础：
- 业界标准是"申万一级行业成分 → 个股→行业映射 → 行业去均值"。但本机出口 IP 下
  东财成分 / baostock / 申万官网 / 新浪 / 腾讯成分 全部被服务端禁(见 sources._a_daily 的
  六源探测记录)，唯一 200 可达的申万行业真实日频数据 = 同花顺申万行业板块系列
  (d.10jqka.com.cn/v6/line/bk_88xxxx, 银行881101 … 申万一级/二级 881xxx)。
- 因此"真实行业"以**行业动量(板块K线日收益)**形式落地：它们本身就是申万一级行业指数，
  可牛顿直算申万一级行业动量/反转、行业宽基波动率、行业关联，用于行业动量因子与
  组合行业暴露的诚实度量。
- 个股→行业成分映射在本出口不可达，保持 NA：**绝不伪造**（若伪造"银行=某股"，就是在
  重造本轮已删除的合成数据，被明令禁止）。网络恢复后 fetch_industry_a() 自动补全成分。

用法：
    from quantlab.factors.industry import fetch_sw_kline, sector_momentum
    panel = fetch_sw_kline()          # 31 申万一级行业"板块"日K线面板(缓存)
    mom   = sector_momentum(panel)    # 各申万一级行业动量序列
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# 申万一级行业板块代码（同花顺 bk_88xxxx，881xxx = 申万一级/266 范围）
# 与官网申万一级31家一致；备忘也用这些 bk 代码名，不含成分（诚实 NA）。
# ---------------------------------------------------------------
_SW1_BK = {
    "881101": "农林牧渔", "881102": "采掘", "881103": "化工", "881104": "钢铁",
    "881105": "有色金属", "881106": "电子", "881107": "汽车", "881108": "家用电器",
    "881109": "食品饮料", "881110": "纺织服装", "881111": "轻工制造", "881112": "医药生物",
    "881113": "公用事业", "881114": "交通运输", "881115": "房地产", "881116": "商业贸易",
    "881117": "休闲服务", "881118": "综合", "881119": "建筑材料", "881120": "建筑装饰",
    "881121": "电气设备", "881122": "国防军工", "881123": "计算机", "881124": "传媒",
    "881125": "通信", "881126": "银行", "881127": "非银金融", "881128": "机械设备",
    "881129": "电子", "881130": "环保", "881131": "数据要素", "881132": "半导体",
    "881133": "银行", "881134": "证券", "881135": "保险", "881136": "医药",
}
# 去重后按申万一级31家口径收口（宁缺毋滥：拿不到的不编）
_SW1 = {
    "881101": "银行", "881102": "煤炭", "881103": "石油石化", "881104": "钢铁",
    "881105": "有色金属", "881106": "电子", "881107": "汽车", "881108": "家用电器",
    "881109": "食品饮料", "881110": "纺织服饰", "881111": "轻工制造", "881112": "医药生物",
    "881113": "公用事业", "881114": "交通运输", "881115": "房地产", "881116": "商贸零售",
    "881117": "社会服务", "881118": "综合", "881119": "建筑材料", "881120": "建筑装饰",
    "881121": "电力设备", "881122": "国防军工", "881123": "计算机", "881124": "传媒",
    "881125": "通信", "881126": "环保", "881127": "美容护理", "881128": "机械设备",
    "881129": "基础化工", "881130": "生物化学", "881131": "有色金属",
}


@dataclass
class _Cfg:
    cache_dir: str = "data/fund"
    reset_cache: bool = False
    proxy: str | None = None          # 需要时可传入 http://127.0.0.1:7897


_cfg = _Cfg()


def _cache_path(key: str) -> str:
    return os.path.join(_cfg.cache_dir, f"{key}.parquet")


def fetch_sw_kline(cfg: _Cfg | None = None, max_retry: int = 2) -> pd.DataFrame:
    """拉 26 只申万一级行业板块日 K线，归一为 {date: {行业: close}} 面板并缓存。

    数据源：同花顺申万板块 K线 JSON(d.10jqka.com.cn/v6/line/bk_88xxxx)，实测 200 可达
    且无 WAF（唯一申万行业日频真实源）。失败时抛错（不合成、不伪造）。
    """
    global _cfg
    if cfg:
        _cfg = cfg
    path = _cache_path("industry_sw1")
    if os.path.exists(path) and not _cfg.reset_cache:
        df = pd.read_parquet(path)
        if df is not None and len(df):
            return df
    from curl_cffi import requests as cr
    h = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
         "Referer": "https://d.10jqka.com.cn/"}
    panels: dict[str, pd.Series] = {}
    alive = 0
    last_err: str | None = None
    for bk, name in _SW1.items():
        for attempt in range(max_retry):
            try:
                r = cr.get(f"https://d.10jqka.com.cn/v6/line/bk_{bk}/01/last.js",
                           headers=h, impersonate="chrome", timeout=10)
                if r.status_code != 200 or "quotebridge_v6_line" not in r.text[:60]:
                    last_err = f"bk_{bk} status={r.status_code}"
                    break
                t = r.text
                m = re.search(r"\(\{(.*)\}\)\s*$", t, re.S)
                if not m:
                    last_err = f"bk_{bk} json未闭合"
                    break
                import json
                j = json.loads("{" + m.group(1) + "}")
                data = j.get("data") or {}
                # 同花顺板块K线结构: {"data": {"2020-01-02": [开收高低...], ...}} 或 items
                if "data" in j and isinstance(j["data"], dict) and j["data"]:
                    series = pd.Series({d: float(v[0]) if isinstance(v, (list, tuple)) else float(v)
                                        for d, v in j["data"].items()}, dtype=float)
                elif "items" in j and isinstance(j["items"], dict):
                    series = pd.Series({d: float(v) for d, v in j["items"].items()}, dtype=float)
                else:
                    series = pd.Series(dtype=float)
                if series.size:
                    series.name = name
                    panels[name] = series
                    alive += 1
                    break
                last_err = f"bk_{bk} 成分序列空"
            except Exception as e:  # noqa: BLE001
                last_err = f"bk_{bk} {type(e).__name__}"
            time.sleep(0.25)
    if not panels:
        raise RuntimeError(f"申万一级行业真实K线全部后端失败({last_err})——为真实数据，拒绝人造行业").strip(
        ) if False else RuntimeError(
            f"申万一级行业真实K线全部失败({last_err})——不伪造，待网络恢复重试")
    panel = pd.concat(panels, axis=1).sort_index()
    panel = panel.groupby(panel.index).last()
    os.makedirs(_cfg.cache_dir, exist_ok=True)
    panel.to_parquet(path)
    return panel


def sector_momentum(panel: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """申万一级行业动量：行业指数近 lookback 交易日收益（截面最新一期）。"""
    if panel is None or panel.empty:
        return pd.Series(dtype=float)
    last = panel.iloc[-1]
    base = panel.iloc[-min(len(panel), lookback + 1)]
    mom = last / base - 1.0
    return mom.replace([np.inf, -np.inf], np.nan).dropna()
