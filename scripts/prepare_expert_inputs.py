"""Frozen monthly inputs for the expert/retention study; never runs an account."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/expert_selection_20260927'
MANIFEST = ROOT / 'output/account_2025_20260927/extension/dataset/manifest.json'
ARCHIVE = ROOT / 'output/score_mechanism_20260927/run2/score_rank_membership.parquet'
ARCHIVE_SHA = '48fa232c31f9fcd6d8a4a62f009188cc2d1dff4e5eaa20fc6058872b5b6e9f34'


def forward_labels(close, calendar, date, codes, gaps, horizon=20):
    """Exact mature price-path labels, matching the old attribution semantics."""
    date = pd.Timestamp(date)
    i = calendar.get_indexer([date])[0]
    if i < 0:
        raise ValueError('label_start_outside_calendar')
    y = pd.Series(np.nan, index=codes, dtype=float)
    status = pd.Series('not_matured', index=codes, dtype=object)
    if i + horizon >= len(calendar):
        return y, status, None
    end = calendar[i+horizon]
    path = close.reindex(index=calendar[i:i+horizon+1], columns=codes)
    valid = np.isfinite(path).all() & path.gt(0).all()
    status[:] = 'missing_price_path'
    missing_endpoint = ~np.isfinite(path.iloc[0]) | ~np.isfinite(path.iloc[-1])
    status.loc[missing_endpoint] = 'missing_endpoint'
    values = path.iloc[-1] / path.iloc[0] - 1
    y.loc[valid] = values[valid]
    status.loc[valid] = 'observed'
    bad = {g['instrument_id'] for g in gaps if date < pd.Timestamp(g['date']) <= end}
    bad = sorted(bad.intersection(codes))
    y.loc[bad] = np.nan
    status.loc[bad] = 'accounting_gap'
    return y, status, end


def build():
    import continuity_cadence as shared
    # Resolve symlink-backed read-only dependencies before frozen-module checks.
    shared.FROZEN = shared.FROZEN.resolve()
    shared.load_frozen()
    from quantlab.config import CONFIG
    from quantlab.policy_comparison import arm_config
    from quantlab.data.contracts import decision_time, timestamps_series
    from quantlab.data.market import universe_at
    from quantlab.strategy.decision import FactorDecisionService, review_dates
    from quantlab.strategy import experts, scoring

    def progress(stage, **kwargs):
        record = dict(stage=stage, at=datetime.now(timezone.utc).isoformat(), **kwargs)
        shared.freeze_json(OUT/'features_status.json', record, immutable=False)
        print(json.dumps(record), flush=True)

    shared.verify_file(ARCHIVE, ARCHIVE_SHA)
    m = json.loads(MANIFEST.read_text())
    root = MANIFEST.parent
    def read(entry, columns=None):
        p = root/entry['path']
        shared.verify_file(p, entry['sha256'])
        return pd.read_parquet(p, columns=columns)
    calendar = pd.DatetimeIndex(read(m['calendar']).session)
    calendar = calendar[calendar <= '2026-09-22']
    cfg = arm_config(dict(minimum_adv=5e6, risk_policy='P2', candidate='quality_pb', allocation_policy='legacy'))
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=shared.sha256(MANIFEST), archive_sha256=ARCHIVE_SHA,
        source_sha256=shared.sha256(__file__), frozen_source=shared.source_identity(),
        parameters=dict(cfg.factors.__dict__), cutoff='2026-09-22',
        prehistory='all complete monthly dates from2015 using2014 warmup; eligibility same listing+ADV5m policy',
        original_cohorts='preserve exact74 archive dates, features and predictions; compare regenerated values before use',
        added_router_cohorts='only complete monthly dates after2026-06-30',
        labels='20-session feature-price path; no filling; actual mature endpoint18:00; preexisting observed labels must match',
        limitation='historical adjustment snapshot and original conditional eligibility; not PIT certification')
    shared.freeze_json(OUT/'features_protocol.json', protocol)
    cutoffs = pd.Series([decision_time(d) for d in calendar], index=calendar)
    close, amount, turnover, panels = {}, {}, {}, {}
    cols = ['close','feature_close','amount','turnover','is_st_at_open','available_at','quality_status']
    progress('reading_feature_projection', panels=len(m['panels']))
    for j, (code, entry) in enumerate(sorted(m['panels'].items())):
        p = read(entry, cols).reindex(calendar)
        available = timestamps_series(p.available_at)
        known = available.notna() & available.le(cutoffs) & p.quality_status.eq('ok')
        x = p.feature_close.where(known)
        first = x.first_valid_index()
        close[code] = x / x.loc[first] * 100 if first is not None else x
        amount[code] = p.amount.where(known)
        turnover[code] = p.turnover.where(known)
        panels[code] = p[['close','is_st_at_open']]
        if j % 500 == 0: progress('reading_feature_projection', loaded=j)
    close, amount, turnover = pd.DataFrame(close), pd.DataFrame(amount), pd.DataFrame(turnover)
    ds = SimpleNamespace(panels=panels, calendar=calendar, manifest=m,
        instruments=read(m['instruments']), industry_records=read(m['industry_records']))
    progress('computing_frozen_factors', rows=len(calendar), securities=len(close.columns))
    service = FactorDecisionService(close, amount, turnover, calendar, cfg, m.get('data_gaps',[]))
    archived = pd.read_parquet(ARCHIVE)
    old = {pd.Timestamp(day): f.set_index('instrument_id').sort_index() for day,f in archived.groupby('date')}
    months = [d for d in review_dates(calendar) if d >= pd.Timestamp('2015-01-01')]
    score_columns = ['score_'+e for e in experts.EXPERT_SPECS] + ['score_mixture']
    feature_columns = ['factor_'+f for f in service.factors]
    checks, files, routed = [], {}, []
    for n, day in enumerate(months):
        eligible, records = universe_at(ds, day, amount, cfg)
        eligible = [c for c in eligible if records.at[c,'asset_type']=='stock']
        decision = service.score(day, eligible)
        trace = decision.trace
        if not trace or 'standardized_features' not in trace:
            continue
        full = {k:pd.DataFrame([v], index=[day]) for k,v in trace['standardized_features'].items()}
        hist = service.ic.loc[:day,list(full)]
        pen,_ = scoring.factor_crowding_penalty(hist,cfg.factors.crowding_corr_window,cfg.factors.crowding_corr_alert)
        if not pen.empty: pen=pen.reindex(pen.index.union([day])).sort_index().ffill().loc[[day]]
        weights = scoring.weight_row(pd.Series(trace['factor_weights']), day)
        mixture,_ = experts.experts_score(full,weights,pen,hist,cfg=cfg.factors)
        available = mixture.loc[day].reindex(eligible).dropna()
        available = available.drop(index=list(trace.get('unavailable_score_history',{})),errors='ignore')
        crowd = full['factor_price_crowding'].loc[day].reindex(available.index)
        available = available[crowd.rank(pct=True) > 1-cfg.factors.crowding_quantile]
        frame = pd.DataFrame(index=available.index)
        frame.index.name='instrument_id'
        for f, values in full.items(): frame['factor_'+f]=values.loc[day].reindex(frame.index)
        for e,names in experts.EXPERT_SPECS.items():
            frame['score_'+e]=experts._group_score(full,weights,pen,names).loc[day].rank(pct=True).reindex(frame.index)
        frame['score_mixture']=available
        y,status,end=forward_labels(close,calendar,day,list(frame.index),m.get('data_gaps',[]),20)
        frame['return_20'],frame['coverage_20']=y,status
        frame['label_end']=end
        frame['label_available_at']=str(decision_time(end)) if end is not None else None
        frame['date']=str(day.date())
        if day in old:
            prior=old[day]
            if set(prior.index)!=set(frame.index):
                raise ValueError('historical_member_mismatch:'+str(day)+':'+str(len(set(prior.index)^set(frame.index))))
            error=float(np.nanmax(np.abs(frame.reindex(prior.index)[feature_columns+score_columns].to_numpy()-prior[feature_columns+score_columns].to_numpy())))
            if error>1e-10: raise ValueError('historical_feature_score_mismatch:'+str(day)+':'+str(error))
            observed=prior.coverage_20.eq('observed')
            err=float((frame.return_20.reindex(prior.index)[observed]-prior.return_20[observed]).abs().max()) if observed.any() else 0.
            if not np.isfinite(frame.return_20.reindex(prior.index)[observed]).all() or err>1e-10:
                raise ValueError('historical_observed_label_mismatch:'+str(day))
            frame.loc[prior.index,feature_columns+score_columns]=prior[feature_columns+score_columns]
            checks.append(dict(date=str(day.date()),members=len(prior),max_feature_score_error=error,max_label_error=err))
        path=OUT/'features'/f'{day.date()}.parquet'
        shared.publish(path,lambda h,f=frame:f.reset_index().to_parquet(h,index=False))
        files[str(day.date())]=dict(path=str(path.resolve()),sha256=shared.sha256(path),rows=len(frame))
        if day in old or day>pd.Timestamp('2026-06-30'):
            routed.append(frame.reset_index())
        if n%5==0: progress('monthly_features',completed=n+1,total=len(months),date=str(day.date()),rows=len(frame))
    if len(checks)!=len(old): raise ValueError('not_all_archived_months_reconstructed')
    route_frame=pd.concat(routed,ignore_index=True)
    target=OUT/'router_scores.parquet'
    shared.publish(target,lambda h:route_frame.to_parquet(h,index=False))
    shared.freeze_json(OUT/'features_complete.json',dict(files=files,overlap=checks,
        router_scores=dict(path=str(target.resolve()),sha256=shared.sha256(target)),
        protocol_sha256=shared.sha256(OUT/'features_protocol.json'),
        new_router_dates=sorted(set(route_frame.date)-set(archived.date.astype(str)))))
    progress('features_complete',months=len(files),overlap=len(checks),router_rows=len(route_frame))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();build()
