"""Two preregistered changes to legacy selection, sharing its expert formula."""
import numpy as np
import pandas as pd

from quantlab.data.contracts import decision_time, timestamp, timestamps_series
from quantlab.data.industry import resolve_industries


def financial_entry_evidence(dataset, date, codes):
    """Publication-date proxies are explicit; unknown release times wait a day.

    A missing statement is never zero profit. A current value with a reported
    announcement date is conditional evidence, not a verified revision history.
    """
    date = pd.Timestamp(date)
    f = getattr(dataset, 'financials', pd.DataFrame())
    profits = {}
    if not f.empty:
        metrics = {'net_profit', 'netprofit', 'netProfit', 'net_profit_ytd'}
        rows = f[f.metric.isin(metrics) & f.instrument_id.isin(codes)].copy()
        announced = pd.to_datetime(rows.announced_on, errors='coerce')
        rows = rows.loc[announced < date]
        if not rows.empty:
            available = timestamps_series(rows.available_at)
            # Where exact publication evidence exists, it must also be known.
            rows = rows.loc[available.isna() | available.le(decision_time(date).tz_convert('UTC'))]
            rows = rows.sort_values(['report_period','announced_on','revision'])
            profits = rows.drop_duplicates('instrument_id', keep='last').set_index('instrument_id').value.to_dict()
    evidence = {}
    for code in codes:
        panel = dataset.panels.get(code)
        pb = np.nan
        if panel is not None and date in panel.index:
            row = panel.loc[date]
            known = pd.notna(row.get('available_at')) and timestamp(row.available_at)<=decision_time(date)
            if known: pb = row.get('pb_mrq', np.nan)
        profit = profits.get(code, np.nan)
        covered = bool(pd.notna(pb) and np.isfinite(pb) and pd.notna(profit) and np.isfinite(profit))
        evidence[code] = dict(pb=pb, net_profit_ytd=profit, covered=covered,
                              passes=bool(covered and profit>=0 and 0<pb<=3))
    return evidence


def industry_rotation(dataset, close, date):
    """Breadth and relative strength use members observable at each endpoint."""
    date = pd.Timestamp(date)
    dates = close.index[close.index<=date]
    if len(dates)<80: return pd.Series(dtype=float), {'reason':'insufficient_industry_history'}
    def observation(day):
        hist = close.loc[:day].tail(60)
        industry = resolve_industries(dataset,day)
        if len(hist)<60 or industry.empty: return pd.DataFrame()
        sector = industry.sector.reindex(close.columns).fillna('UNKNOWN')
        valid = hist.notna().all() & sector.ne('UNKNOWN')
        b = dataset.benchmark.loc[:day,'close'].tail(21)
        if len(b)<21 or b.isna().any() or b.iloc[0]<=0: return pd.DataFrame()
        stock20 = hist.iloc[-1]/hist.iloc[-21]-1
        stock = pd.DataFrame({'sector':sector, 'breadth':hist.iloc[-1]>hist.mean(),
                              'strength':stock20>(b.iloc[-1]/b.iloc[0]-1)})[valid]
        grouped = stock.groupby('sector')
        out = grouped[['breadth','strength']].mean()
        return out[grouped.size()>=5]
    now, previous = observation(date), observation(dates[-21])
    if now.empty or previous.empty: return pd.Series(dtype=float), {'reason':'industry_members_or_history_insufficient'}
    f = now.join(now.subtract(previous).rename(columns={'breadth':'breadth_change','strength':'strength_change'})).dropna()
    if len(f)<2: return pd.Series(dtype=float), {'reason':'insufficient_comparable_industries'}
    ranks = f.rank(pct=True).mean(axis=1)
    industry = resolve_industries(dataset,date)
    per_stock = industry.sector.map(ranks)
    return per_stock, dict(industry_mode=dataset.manifest.get('industry_mode','verified_pit'),
                           industry_features=f.to_dict('index'),industry_percentiles=ranks.to_dict())


def candidate_decision(dataset, prepared, date, scores, candidate):
    if candidate=='baseline': return scores, None, {}
    if candidate in ('quality_pb','quality_coverage'):
        evidence = financial_entry_evidence(dataset,date,list(scores.index))
        key = 'passes' if candidate=='quality_pb' else 'covered'
        allowed = {c for c,e in evidence.items() if e[key]}
        return scores, allowed, dict(financial_basis='reported_date_current_value_proxy',
                                     qualified_count=len(allowed),entries=evidence)
    if candidate=='industry_rotation':
        industry, evidence = industry_rotation(dataset,prepared['close'],date)
        base = scores.rank(pct=True)
        adjustment = .20*(industry.reindex(base.index)-.5)
        revised = base.add(adjustment.fillna(0)).sort_index().sort_values(ascending=False,kind='stable')
        return revised, None, dict(evidence, original_scores=scores.to_dict(),
                                   industry_adjustments=adjustment.dropna().to_dict())
    raise ValueError('unknown_candidate')
