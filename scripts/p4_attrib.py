#!/usr/bin/env python3
"""P4 attribution from existing scores, plans and ledgers (no new account; not a trial).

  decay     rank IC of LIN/GBDT scores against the single-session open-to-open return k sessions ahead,
            k in 1..60, plus net(h) = kappa * mean IC(1..h) - 2 c tau(h) on the 20-session book
  turnover  names changed per rebalance and yearly turnover for LIN / GBDT / POOL50 plans; ledger costs
  leak      pool EW (frictionless, Beijing excluded) -> POOL50 frictionless -> x0.90 gross -> ledger cost1,
            with cost drag estimated as ledger(cost1) - ledger(cost2)
Output: output/p4_attrib_20261013/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p4_alpha as P
from quantlab.alpha.labels import adjusted_open
from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel

OUT = ROOT / 'output/p4_attrib_20261013'
ALPHA = ROOT / 'output/p4_alpha_20261012'
HORIZONS = (1, 2, 3, 5, 10, 15, 20, 30, 40, 60)


def cagr(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)


def stage_decay():
    panel = Panel(P.PANEL)
    fields = {n: panel.field(n) for n in ('open', 'close', 'adj_close')}
    adj_open = adjusted_open(fields)
    daily = adj_open.shift(-1) / adj_open - 1
    scores = pd.read_parquet(ALPHA / 'scores.parquet')
    position = pd.Series(np.arange(len(panel.sessions)), index=panel.sessions)
    rows = []
    for day, group in scores.groupby('date'):
        i = position[day]
        for k in HORIZONS:
            if i + k + 1 >= len(panel.sessions):
                continue
            target = daily.iloc[i + k].reindex(group.code).to_numpy()
            for model in ('LIN', 'GBDT'):
                mask = np.isfinite(target)
                if mask.sum() > 50:
                    rows.append(dict(date=day, k=k, model=model,
                                     ic=pd.Series(group[model].to_numpy()[mask]).corr(pd.Series(target[mask]), method='spearman')))
    frame = pd.DataFrame(rows)
    curve = frame.groupby(['model', 'k']).ic.mean().unstack(0)
    OUT.mkdir(parents=True, exist_ok=True)
    curve.to_csv(OUT / 'decay_curve.csv')
    print(json.dumps({'decay_mean_ic': curve.round(4).to_dict()}, indent=1))
    return curve


def stage_turnover():
    plans = P.plans(ALPHA, P.ANCHOR)
    rows = []
    for name, schedule in plans.items():
        days, previous, changes = sorted(schedule), set(), []
        for day in days:
            names = set(schedule[day])
            if previous:
                changes.append(len(names - previous) / max(len(names), 1))
            previous = names
        rows.append(dict(path=name, family=name.split('_')[0], mean_replaced=float(np.mean(changes)),
                         yearly_one_way=float(np.mean(changes) * 252 / 20)))
    frame = pd.DataFrame(rows)
    out = dict(plan_turnover=frame.groupby('family')[['mean_replaced', 'yearly_one_way']].median().round(3).to_dict('index'))
    for cost in (1, 2):
        path = ALPHA / f'ledger/summary_cost{cost}.csv'
        if path.exists():
            ledger = pd.read_csv(path)
            ledger['family'] = ledger.path.str.split('_').str[0]
            ledger['cost_total'] = ledger.fees + ledger.slippage
            out[f'ledger_cost{cost}'] = ledger.groupby('family')[['cagr', 'cost_total', 'fills']].median().round(4).to_dict('index')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'turnover.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


def frictionless(schedule, ret, sessions, start, end, gross=1.0):
    days = sorted(schedule)
    pieces = []
    for i, day in enumerate(days):
        stop = days[i + 1] if i + 1 < len(days) else pd.Timestamp(end)
        window = sessions[(sessions > day) & (sessions <= stop)]
        names = list(schedule[day])
        if not names or window.empty:
            continue
        block = ret.loc[window, names].fillna(0.0)
        weights = (1 + block).cumprod().shift(1).fillna(1.0)
        pieces.append(gross * (weights * block).sum(axis=1) / weights.sum(axis=1))
    series = pd.concat(pieces).sort_index()
    return (1 + series).cumprod()


def stage_leak():
    panel = Panel(P.PANEL)
    adj = panel.field('adj_close')
    ret = adj.pct_change(fill_method=None)
    sessions = panel.sessions
    features = U.rolling_features(panel, with_st=True)
    grid = U.decision_grid(sessions, P.ANCHOR, P.END, 0, 20)
    pool = {}
    for day in grid:
        pool[day] = {c: 1.0 for c in U.pool_on(day, features, 'adv', band='relative', exclude_st=True, min_history=252)
                     if not c.startswith('bj.')}
    pool_nav = frictionless(pool, ret, sessions, P.ANCHOR, P.END)
    plans = P.plans(ALPHA, P.ANCHOR)
    rows = []
    for name in sorted(plans):
        if not name.startswith('POOL50'):
            continue
        n1, n2 = ALPHA / f'ledger/nav_{name}_cost1.csv', ALPHA / f'ledger/nav_{name}_cost2.csv'
        if not (n1.exists() and n2.exists()):
            continue
        full = frictionless(plans[name], ret, sessions, P.ANCHOR, P.END)
        cash = frictionless(plans[name], ret, sessions, P.ANCHOR, P.END, gross=0.90)
        c1 = cagr(pd.read_csv(n1, index_col=0, parse_dates=True)['nav'])
        c2 = cagr(pd.read_csv(n2, index_col=0, parse_dates=True)['nav'])
        rows.append(dict(path=name, pool_ew=cagr(pool_nav), random50_full=cagr(full), random50_gross90=cagr(cash),
                         ledger_cost1=c1, ledger_cost2=c2))
    frame = pd.DataFrame(rows)
    if frame.empty:
        print(json.dumps(dict(status='no_finished_pool50_pairs')))
        return
    med = frame.drop(columns='path').median()
    table = {
        '抽样（池子等权→随机50只，不计成本、满仓）': med.pool_ew - med.random50_full,
        '现金闲置（满仓→90%仓位）': med.random50_full - med.random50_gross90,
        '交易成本（按成本1倍与2倍之差估计）': med.ledger_cost1 - med.ledger_cost2,
        '执行与其他（剩余项）': med.random50_gross90 - med.ledger_cost1 - (med.ledger_cost1 - med.ledger_cost2),
    }
    out = dict(paths=list(frame.path), medians=med.round(4).to_dict(),
               decomposition_pp={k: round(v * 100, 2) for k, v in table.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / 'leak_paths.csv', index=False)
    (OUT / 'leak.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


def stage_execution(path='POOL50_p0_s1', gross=None):
    """Replay one POOL50 path and split execution drag into named pieces (annualized share of mean NAV)."""
    from quantlab import p4_ledger
    plans = P.plans(ALPHA, P.ANCHOR)
    targets = p4_ledger.to_targets(plans[path])
    result = p4_ledger.replay(targets, start=P.ANCHOR, end=P.END, cost_mult=1.0, gross=gross)
    fills = pd.DataFrame(result['fills'])
    account = pd.DataFrame(result['account'])
    nav = result['nav']
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    scale = 1 / (float(nav.mean()) * years) * 100
    close = Panel(P.PANEL).field('close')
    fills['open'] = np.where(fills.side.eq('buy'), fills.price - fills.slippage_cost / fills.quantity,
                             fills.price + fills.slippage_cost / fills.quantity)
    signal_close = [close.at[pd.Timestamp(s), c] if pd.Timestamp(s) in close.index and c in close.columns else np.nan
                    for s, c in zip(fills.signal_date, fills.instrument_id)]
    fills['signal_close'] = signal_close
    sign = np.where(fills.side.eq('buy'), 1.0, -1.0)
    timing = float(np.nansum(sign * fills.quantity * (fills.open - fills.signal_close)))
    cash_ratio = (account['cash'] / account['nav']).astype(float) if {'cash', 'nav'} <= set(account) else pd.Series(dtype=float)
    pieces = {
        '开盘成交相对信号日收盘的价差': timing * scale,
        '滑点': float(fills.slippage_cost.sum()) * scale,
        '佣金': float(fills.commission.sum()) * scale,
        '印花税': float(fills.stamp_tax.sum()) * scale,
        '红利税': float(fills.dividend_tax.sum()) * scale,
    }
    out = dict(path=path, gross=gross, cagr=result['summary']['cagr'], years=years,
               pieces_pp_per_year={k: round(v, 3) for k, v in pieces.items()},
               mean_cash_share=float(cash_ratio.mean()) if len(cash_ratio) else None,
               fills=len(fills), checks_passed=result['summary']['account_checks_passed'])
    shortfall = [e for e in result.get('orders', []) if e.get('reject_reason')]
    reasons = pd.Series([o['reject_reason'] for o in shortfall]).value_counts().to_dict() if shortfall else {}
    out['order_reject_reasons'] = reasons
    OUT.mkdir(parents=True, exist_ok=True)
    fills.to_csv(OUT / f'execution_fills_{path}.csv', index=False)
    (OUT / f'execution_{path}.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('decay', 'turnover', 'leak', 'execution'))
    args = parser.parse_args()
    {'decay': stage_decay, 'turnover': stage_turnover, 'leak': stage_leak, 'execution': stage_execution}[args.stage]()
