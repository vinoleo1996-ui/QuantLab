#!/usr/bin/env python3
"""Offline review evidence; never rewrites production outputs or fetches data.

Run: .venv2/bin/python scripts/review_audit.py [--replay]
Fixtures expose execution contracts, not investment performance. Replay uses the
saved processed factors and regime decisions, with cached OHLC; it is a controlled
sensitivity experiment, not a PIT or out-of-sample validation.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import quantlab.backtest.engine as em
from quantlab.config import Config
from quantlab.data.sources import DataFeed
from quantlab.data.store import DataStore
from quantlab.strategy.experts import experts_score, expert_top_union
from quantlab.strategy.scoring import composite_score
from quantlab.strategy.portfolio import target_vol_weights
from quantlab.report.metrics import performance_table

OUT = ROOT / 'output/review_20260924'
OUT.mkdir(parents=True, exist_ok=True)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def stats(r):
    p = performance_table(r['nav'], r['daily_return'])
    p.update(start=str(r['nav'].index[0].date()), end=str(r['nav'].index[-1].date()),
             days=len(r['nav']), trades=len(r['trades']))
    return p


def fixture(mode):
    idx = pd.bdate_range('2022-01-03', '2023-04-14')
    codes = [f'60000{i}' for i in range(6)]
    panels = {c: pd.DataFrame({'open':10., 'close':10., 'amount':1e8, 'turnover':1.},index=idx) for c in codes}
    if mode in ('rotate', 'stop', 'overweight'):
        panels[codes[0]].loc['2023-03-01':, ['open','close']] = 9.
    if mode == 'stop':
        # Jan entry at 10; February losses already exceed L1, Mar 1 is limit down.
        panels[codes[0]].loc['2023-02-02':'2023-02-28', ['open','close']] = 8.
        panels[codes[0]].loc['2023-03-01':, ['open','close']] = 7.2
    if mode == 'drift':
        panels[codes[0]].loc['2023-02-02', ['open','close']] = 11.
    if mode == 'missing':
        panels[codes[0]].loc['2023-02-02', ['open','close']] = np.nan
        panels[codes[0]].loc['2023-02-02', 'amount'] = 0.
        panels[codes[0]].loc['2023-02-03':, ['open','close']] = 8.
    cfg = Config()
    cfg.start_date = '2023-01-01'
    cfg.factors.use_experts = False
    cfg.factors.neutralize_by_sector = False
    cfg.capital.a_share_positions = 3
    cfg.capital.min_holding_period_days = 0
    cfg.portfolio.n_candidates_pad = 0
    cfg.portfolio.turnover_rank_thresh = 0
    if mode not in ('rotate',):
        cfg.cost.a_commission = cfg.cost.a_stamp_tax = cfg.cost.a_slippage = 0.
    scores = pd.DataFrame(np.tile([6.,5.,4.,3.,2.,1.],(len(idx),1)),index=idx,columns=codes)
    if mode == 'rotate':
        scores.loc['2023-02-28':] = [1.,2.,3.,6.,5.,4.]
    if mode == 'skip':
        scores.loc['2023-02-28'] = np.nan
        for c in codes: panels[c].loc['2023-03-02':,['open','close']] = 11.
    def weights(ret, sector, cfg, vol_mult=1.):
        w = np.ones(len(ret.columns))/len(ret.columns)
        if mode == 'overweight' and ret.index[-1] >= pd.Timestamp('2023-02-28'):
            w = np.array([0.1,0.45,0.45])
        return pd.DataFrame({'weight':w},index=ret.columns)
    with ExitStack() as st:
        st.enter_context(patch.object(em.FactorEngine,'compute',return_value={'fixture':scores}))
        # Engine slices historical factors; expose the requested last signal only.
        st.enter_context(patch.object(em,'composite_score',side_effect=lambda full,*a: scores.loc[full['fixture'].index]))
        st.enter_context(patch.object(em,'target_vol_weights',side_effect=weights))
        st.enter_context(patch.object(em.Regime,'state',return_value={'trend':1,'vol_high':False,'multiplier':1.}))
        st.enter_context(patch.object(em.RiskManager,'evaluate',return_value={'gross_scale':1.,'vol_scale':1.,'drawdown':0.,'reasons':[]}))
        result = em.Engine(cfg).run(panels)
    return result, panels


def diagnostics(saved):
    results = {}
    t = saved['ic_matrix'].index[-1]
    full = {k:v.loc[:t] for k,v in saved['factor_panels'].items()}
    wf = pd.Series(saved['trades'][-1]['factor_weights'])
    bad = wf.to_frame('w').T.reindex([t])
    good = pd.DataFrame([wf.to_dict()],index=[t])
    for label,w in [('current',bad),('correct_index_only',good)]:
        score,exp = experts_score(full,w,None,saved['ic_matrix'].fillna(0))
        tops=expert_top_union(full,w,None,full['factor_price_crowding'].loc[t],28)
        results[label+'_scoring']={'missing_weights':int(w.isna().sum().sum()),'unique_stock_scores':int(score.iloc[-1].nunique()),'distinct_expert_lists':len({tuple(v) for v in tops.values()}),'score_min':float(score.iloc[-1].min()),'score_max':float(score.iloc[-1].max())}
    assert results['current_scoring']['unique_stock_scores'] == 1
    assert results['correct_index_only_scoring']['unique_stock_scores'] > 1

    r,pan = fixture('rotate')
    trade = next(t for t in r['trades'] if t['exec_date'].startswith('2023-03-01'))
    initial = r['weight_history'][0]
    current = next(w for w in r['weight_history'] if w.name == pd.Timestamp('2023-03-01'))
    a,b = current.align(initial,fill_value=0)
    expected_turnover=float((a-b).abs().sum()/2)
    results['outgoing_positions']={'actual_turnover':trade['turnover'],'turnover_if_all_recorded_changes_filled':expected_turnover,'actual_fee':trade['fees'],'fee_if_all_recorded_changes_filled':0.007,'limit_down_old_position_after':float(current.get('600000',0)),'old_position_before':float(initial['600000'])}
    assert trade['turnover'] == 0 and np.isclose(expected_turnover, 1)
    assert current.get('600000',0) == 0
    r,_=fixture('stop'); trade=next(t for t in r['trades'] if t['exec_date'].startswith('2023-03-01'))
    results['stop_overrides_limit_down']={'sold_weight':trade['weights']['600000'],'reasons':trade['risk']['reasons']}
    assert trade['weights']['600000'] == 0
    r,_=fixture('overweight');trade=next(t for t in r['trades'] if t['exec_date'].startswith('2023-03-01'))
    results['unfunded_buys']={'gross':sum(trade['weights'].values()),'cash_implied':1-sum(trade['weights'].values())}
    assert results['unfunded_buys']['gross']>1
    r,_=fixture('drift')
    actual=float((1+r['daily_return'].loc['2023-02-01':'2023-02-03']).prod()-1)
    results['weight_drift']={'path':'one of three equal positions: 10 -> 11 -> 10; others flat; no trades/costs','engine_return':actual,'buy_and_hold_return':0.}
    assert actual>0.002
    r,_=fixture('missing')
    results['missing_quote_resumption']={'engine_return_on_resumption':float(r['daily_return'].loc['2023-02-03']),
        'true_return_for_one_third_position':(8/10-1)/3,'path':'10 -> absent bar -> 8, other positions flat'}
    assert results['missing_quote_resumption']['engine_return_on_resumption']==0
    r,_=fixture('skip')
    results['skipped_signal']={'market_return_on_2023_03_02':0.1,'portfolio_return_on_2023_03_02':float(r['daily_return'].loc['2023-03-02'])}
    assert results['skipped_signal']['portfolio_return_on_2023_03_02']==0
    results['terminal_execution']={'signal_date':r['trades'][-1]['signal_date'],'exec_date':r['trades'][-1]['exec_date']}
    assert results['terminal_execution']['signal_date']==results['terminal_execution']['exec_date']
    rm=em.RiskManager()
    rm.evaluate(1.,1.)
    monthly=rm.evaluate(1.1,1.)
    results['risk_peak_sampling']={'intramonth_peak':1.5,'month_end_nav':1.1,
        'true_drawdown':1.1/1.5-1,'engine_drawdown_with_monthly_calls':monthly['drawdown'],
        'engine_gross_scale':monthly['gross_scale']}
    assert monthly['drawdown']==0 and monthly['gross_scale']==1

    # The broken weight alignment also makes factor sign flips invisible.
    day=pd.Timestamp('2023-04-14');cols=list('ABCDEF')
    x=pd.DataFrame([[6,5,4,3,2,1]],index=[day],columns=cols)
    wb=pd.Series(1.,index=['fixture']).to_frame('w').T.reindex([day])
    low=composite_score({'fixture':x},wb,None)
    high=composite_score({'fixture':-x},wb,None)
    results['factor_perturbation_current_path']={'max_score_change_after_sign_flip':float((low-high).abs().to_numpy().max())}
    assert results['factor_perturbation_current_path']['max_score_change_after_sign_flip']==0

    # IPO rule uses whole-sample row count; no 60-day as-of eligibility at execution.
    # Crowding series is a NEGATIVE score: high rank means least crowded.
    crowd=pd.Series([-3.,-2.,-1.,0.,1.,2.,3.],index=list('ABCDEFG'))
    results['crowding_direction']={'least_crowded_removed_in_single_expert':crowd.index[(crowd.rank(pct=True)>=0.95)].tolist(),'crowding_filter_runs_in_expert_union':bool(t in crowd.index)}

    rr=pd.DataFrame(np.random.default_rng(10).normal(0,.002,(120,10)),columns=list('ABCDEFGHIJ'))
    pp=Config().portfolio; pp.target_vol=1.
    ww=target_vol_weights(rr,pd.Series(['bank']*10,index=rr.columns),pp)
    results['infeasible_sector_constraint']={'sector_cap':pp.max_sector_weight,'actual_sector_weight':float(ww.weight.sum())}
    assert ww.weight.sum()>pp.max_sector_weight

    class EmptyStore:
        def load(self,*a,**k): return None
        def save(self,*a,**k): raise AssertionError('must not save')
    feed=DataFeed(store=EmptyStore())
    with patch.object(feed,'_yf',return_value=SimpleNamespace(Ticker=lambda s:None)):
        try: feed.fetch_candidates('us')
        except RuntimeError as e: results['us_candidates']={'error':str(e)}
    assert 'error' in results['us_candidates']
    fake = pd.DataFrame({'close':[1.]},index=pd.to_datetime(['2020-01-02']))
    with patch.object(feed,'_a_daily',return_value=fake) as m:
        feed.fetch_benchmark('a')
        results['benchmark_route']={'called_stock_daily_with':m.call_args.args[0]}
    # Cached history is returned regardless of age, with no incremental fetch.
    with patch.object(feed.store,'load',return_value=fake),patch.object(feed,'_ak',side_effect=AssertionError('network')):
        old=feed.fetch_daily('a','600000')
        results['stale_cache']={'returned_last_date':str(old.index[-1].date()),'network_refresh_attempted':False}

    results['evidence_note']='Assertions confirm observed defects in current code; not a passing correctness suite.'
    dump('diagnostics.json',results)
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)
    return results


def transformed_engine(changes):
    src=inspect.getsource(em.Engine)
    for old,new in changes:
        assert src.count(old)==1,(old,src.count(old))
        src=src.replace(old,new)
    namespace=dict(em.__dict__)
    exec(compile(src,'<review-only Engine variant>','exec'),namespace)
    return namespace['Engine']


def replay(saved):
    store=DataStore()
    panels={c:store.load('a',f'daily_{c}') for c in saved['codes']}
    close=pd.DataFrame({c:p['close'] for c,p in panels.items()}).sort_index()
    op=pd.DataFrame({c:p['open'] for c,p in panels.items()}).reindex(close.index)
    wh=pd.DataFrame(saved['weight_history']).fillna(0)
    ret=close.pct_change(fill_method=None).fillna(0)
    # Recover saved daily return from saved target weights and the documented formula.
    reconstructed=pd.Series(0.,index=saved['daily_return'].index)
    previous=pd.Series(0.,index=close.columns)
    for i,(date,w) in enumerate(wh.iterrows()):
        dates=reconstructed.index[reconstructed.index>=date]
        if i+1<len(wh): dates=dates[dates<wh.index[i+1]]
        if len(dates)==0: continue
        gap=(op.loc[date]/close.shift().loc[date]-1).fillna(0)
        intraday=(close.loc[date]/op.loc[date]-1).fillna(0)
        ww=w.reindex(close.columns,fill_value=0)
        reconstructed.loc[dates]=(ret.loc[dates]*ww).sum(axis=1)
        reconstructed.loc[date]=float((gap*previous).sum()+(intraday*ww).sum()-saved['trades'][i]['fees'])
        previous=ww
    error=float((reconstructed-saved['daily_return']).abs().max())
    # Freeze processed factors and external regime states to make variants comparable.
    regimes={pd.Timestamp(t['signal_date']):t['regime'] for t in saved['trades']}
    def state(_self,t): return regimes.get(t,{'trend':1,'vol_high':False,'multiplier':1.})
    cfg=Config();cfg.factors.neutralize_by_sector=False
    factor_fix=('w_row = w_fac.to_frame("w").T.reindex([t])','w_row = pd.DataFrame([w_fac.to_dict()], index=[t])')
    terminal_fix=('for i, t in enumerate(rebal):\n            t_exec = exec_dates[i]','for i, t in enumerate(rebal):\n            if t >= all_dates[-1]:\n                continue\n            t_exec = exec_dates[i]')
    fee_fix=('turnover_w = (w_exec - w_now).abs().sum() / 2.0\n                fee = turnover_w * (cost.a_commission * 2 + cost.a_stamp_tax + cost.a_slippage * 2)',
             'wa, wb = w_exec.align(w_now, fill_value=0.0)\n                dw = wa - wb\n                turnover_w = dw.abs().sum() / 2.0\n                fee = dw.clip(lower=0).sum() * (cost.a_commission + cost.a_slippage) + (-dw.clip(upper=0)).sum() * (cost.a_commission + cost.a_slippage + cost.a_stamp_tax)')
    variants={'current_replay':[], 'factor_index_only':[factor_fix], 'terminal_only':[terminal_fix], 'fees_only':[fee_fix], 'factor_terminal_fees':[factor_fix,terminal_fix,fee_fix]}
    results={'saved':stats(saved),'saved_return_reconstruction_max_abs_error':error,'variants':{}}
    navs={'saved':saved['nav']}
    with patch.object(em.FactorEngine,'compute',return_value=saved['factor_panels']),patch.object(em.Regime,'state',state):
        for label,changes in variants.items():
            print('REPLAY',label,flush=True)
            er=transformed_engine(changes)(cfg).run(panels)
            results['variants'][label]=stats(er)
            results['variants'][label]['same_day_executions']=sum(t['signal_date']==t['exec_date'] for t in er['trades'])
            navs[label]=er['nav']
            print(label,results['variants'][label],flush=True)
    # Self-financing buy-and-hold between the SAME archived orders (diagnostic only).
    # Start with 1 cash. At monthly opens, trade to archived target weights and pay costs.
    # Cash/asset units are then unchanged until the next order. No intramonth stop added.
    units=pd.Series(0.,index=close.columns);cash=1.;nav=pd.Series(index=saved['nav'].index,dtype=float)
    frozen_fees=0.;max_gross=0.
    events={d:w for d,w in wh.iterrows() if d < close.index[-1]}
    marked=close.ffill()
    for d in nav.index:
        px=marked.loc[d]
        if d in events:
            price=op.loc[d].fillna(px)
            v=float((units*price).sum()+cash)
            target=events[d].reindex(close.columns,fill_value=0)
            before=units*price
            notional=target*v
            delta=notional-before
            fee=float(delta.clip(lower=0).sum()*.00325-delta.clip(upper=0).sum()*.00375)
            # Scale target if necessary so fees never create negative cash.
            if notional.sum()+fee>v:
                notional*=max(0,(v-fee)/max(notional.sum(),1e-12))
                delta=notional-before
                fee=float(delta.clip(lower=0).sum()*.00325-delta.clip(upper=0).sum()*.00375)
            units=notional/price
            cash=v-float(notional.sum())-fee
            frozen_fees+=fee
        nav.loc[d]=float((units*px).sum()+cash)
        max_gross=max(max_gross,float((units*px).sum())/nav.loc[d])
    dd=nav.pct_change(fill_method=None);dd.iloc[0]=nav.iloc[0]-1
    results['frozen_orders_self_financing']={**performance_table(nav,dd),'max_gross':max_gross,'note':'Same saved targets, no terminal trade; drift and explicit cash/units/fees, no selection or risk feedback. Still inherits original invalid orders, snapshot universe and data defects.'}
    navs['frozen_orders_self_financing']=nav
    pd.DataFrame(navs).to_csv(OUT/'sensitivity_nav.csv')
    dump('replay.json',results)
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


def inventory(saved):
    store=DataStore(); cand=store.load('a','candidates_a')
    records=[]
    for code in saved['codes']:
        d=store.load('a',f'daily_{code}')
        records.append({'code':code,'rows':len(d),'first':str(d.index[0].date()),'last':str(d.index[-1].date()),'source':d.attrs.get('source'),'amount_present':int((d['amount'].fillna(0)>0).sum()) if 'amount' in d else 0,'turnover_present':int((d['turnover'].fillna(0)>0).sum()) if 'turnover' in d else 0})
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'quantlab').rglob('*.py'))}
    hashes['output/results.pkl']=hashlib.sha256((ROOT/'output/results.pkl').read_bytes()).hexdigest()
    for p in sorted((ROOT/'data').rglob('*.parquet')): hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump('manifest.json',{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'hashes':hashes})
    dump('inventory.json',{'saved_stats':stats(saved),'calendar_cagr':float((saved['nav'].iloc[-1]/saved['nav'].iloc[0])**(365.2425/(saved['nav'].index[-1]-saved['nav'].index[0]).days)-1),'candidate_rows':len(cand),'sector_present':'sector' in cand,'benchmark_cache_keys':store.list_keys('bench'),'daily':records})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    saved=pickle.load((ROOT/'output/results.pkl').open('rb'))['result']
    inventory(saved)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        diagnostics(saved)
        if args.replay: replay(saved)

if __name__=='__main__': main()
