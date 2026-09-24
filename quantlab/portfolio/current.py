"""Read verified versioned simulation accounts. Ledger marks and FIFO lot costs are distinct from timestamped public quotes. No broker account is inferred from a backtest."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from quantlab.config import OUTPUT_DIR
from quantlab.data.store import DataStore

YEAR_START = "2026-01-05"


def _load_result():
    import json
    from quantlab.report.runs import load_run
    latest = OUTPUT_DIR / "runs" / "latest.json"
    if latest.exists():
        run = json.loads(latest.read_text())["run_id"]
        try:
            return {"result": load_run(OUTPUT_DIR / 'runs' / run), "source": "versioned_replay"}
        except (ValueError,KeyError,OSError):
            return {'result':{'status':'unverified_artifacts'},'source':'unverified_artifacts'}
    p = OUTPUT_DIR / "results.pkl"
    if not p.exists(): return None
    return {'result':{'status':'legacy_unvalidated'},'source':'archived_legacy'}


def _entry_of(wh: pd.DataFrame, code: str) -> pd.Timestamp:
    """最近连续持有段的起点(该段最早一期)。若中间卖出(w=0)过，则为最近一次重新买入那期。"""
    s = wh[code]
    pos = s[s > 1e-6]
    if pos.empty:
        return s.index[-1]
    p = pos.index
    start = p[-1]
    for j in range(len(p) - 1, 0, -1):
        if wh.index.get_loc(p[j]) - wh.index.get_loc(p[j - 1]) > 1:
            start = p[j]
            break
        start = p[j - 1]
    return start


def snapshot(with_quotes=True):
    data = _load_result()
    if data is None or data["result"].get("status") != "research_unvalidated": return []
    result = data["result"]
    states = result.get("account", [])
    if not states: return []
    state = states[-1]
    quantities = state["quantities"]
    quotes = pd.DataFrame()
    if with_quotes and quantities:
        try:
            from quantlab.data.realtime import batch_quotes
            quotes = batch_quotes(list(quantities))
        except ImportError:
            pass  # Optional public quote adapters; dated account marks remain explicitly labelled.
    rows = []
    for code, qty in quantities.items():
        price = state.get('marks',{}).get(code)
        quote=quotes.loc[code].to_dict() if code in quotes.index else {}
        live=quote.get('price');cost=state.get('cost_basis',{}).get(code)
        rows.append(dict(code=code, name=quote.get('name',code), quantity=qty, weight=state["weights"][code],
                         account_type="simulation", as_of=str(state["date"]),run_id=result['run_id'], cost=cost, entry_date=state.get('entry_dates',{}).get(code),
                         price=price, live_quote=live,quote_time=quote.get('time'),quote_source=quote.get('source'),
                         pct_chg=quote.get('pct_chg'),market_value=qty*price if price else None,
                         return_since_entry=(price/cost-1) if cost and price else None, return_ytd=None,
                         profit_since_entry=qty*(price-cost) if cost and price else None, profit_ytd=None,
                         lot_cost=None, lot_affordable=None))
    return rows


def _r(price, base, raw: bool = False) -> float | None:
    if price is None or not base:
        return None
    return float(price / base - 1.0)


def summary() -> dict:
    """组合概况与权重口径说明(为何等权/为何 30% 仓位)。"""
    data = _load_result()
    out = {"exec_date": "", "nav": None, "drawdown": None, "daily": None}
    if data is None:
        return out
    result = data["result"]
    out["status"] = result.get("status", "legacy_unvalidated")
    out["account_type"] = "simulation"
    out['run_id']=result.get('run_id')
    if out["status"] != "research_unvalidated": return out
    nav = result.get("nav")
    if nav is not None and len(nav):
        out["nav"] = float(nav.iloc[-1])
        out["exec_date"] = str(nav.index[-1].date())
        if "drawdown" in result and len(result["drawdown"]):
            out["drawdown"] = float(result["drawdown"].iloc[-1])
    dr = result.get("daily_return")
    if dr is not None and len(dr) >= 2:
        out["daily"] = float(dr.iloc[-1])
    trades = result.get("trades") or []
    last = trades[-1] if trades else {}
    out["regime_trend"] = last.get("regime", {}).get("trend")
    out["gross_scale"] = float(last.get("risk", {}).get("gross_scale", 1.0)) if trades else 1.0
    reasons = last.get("risk", {}).get("reasons", [])
    out["risk_reasons"] = list(reasons) if reasons else []
    return out


def ytd_return() -> float | None:
    """今年以来(2026-01-05 起)组合净值收益。"""
    data = _load_result()
    if data is None:
        return None
    if data["result"].get("status") != "research_unvalidated": return None
    nav = data["result"].get("nav")
    if nav is None or len(nav) < 2:
        return None
    nav = pd.Series(nav).sort_index()
    year_start = pd.Timestamp(year=nav.index[-1].year, month=1, day=1)
    base = nav[nav.index < year_start]
    if not len(base):
        return None
    b = float(base.iloc[-1])
    return float(nav.iloc[-1] / b - 1.0) if b != 0 else None
