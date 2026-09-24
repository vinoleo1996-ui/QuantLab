"""Daily ledger P&L attribution, with removal diagnostics explicitly not tradable replays."""
from collections import defaultdict
import math

import pandas as pd
from quantlab.data.contracts import asof_records,decision_time
from quantlab.data.market import exposure_matrix


def instrument_returns(run, *, actions=None):
    """Stock contributions to daily net returns, with mandatory reconciliation.

    Compact runs may supply their frozen action table explicitly. Recorded cash
    entitlements are income on the ex-date, including amounts still receivable;
    later payment only transfers receivables to cash. Bonus shares enter marked
    position values. Fill fees already include dividend tax, and slippage is
    already embedded in fill prices, so neither is charged a second time.
    Reconciliation does not resolve the run's outstanding source-data gaps.
    """
    returns=run['daily_return'];states=run.get('account',[])
    dates=pd.DatetimeIndex([pd.Timestamp(s['date']) for s in states])
    if dates.has_duplicates or not dates.is_monotonic_increasing or not dates.equals(returns.index):
        raise ValueError('attribution_calendar_mismatch')
    actions=run.get('input_actions',pd.DataFrame()) if actions is None else actions
    action_codes={}
    if not actions.empty:
        if not {'action_id','instrument_id'}.issubset(actions):
            raise ValueError('attribution_missing_action_mapping: action_id and instrument_id required')
        if actions.groupby('action_id',observed=True).instrument_id.nunique(dropna=False).gt(1).any():
            raise ValueError('attribution_ambiguous_action_mapping')
        action_codes=actions.drop_duplicates('action_id').set_index('action_id').instrument_id.to_dict()
    fills,accruals=defaultdict(list),defaultdict(list)
    for fill in run.get('fills',[]):
        if fill['side'] not in ('buy','sell'):raise ValueError('attribution_invalid_fill_side')
        fills[pd.Timestamp(fill['date'])].append(fill)
    for event in run.get('corporate_action_events',[]):
        if event.get('type')!='corporate_action' or not (event.get('cash',0) or event.get('bonus_shares',0)):
            continue
        code=action_codes.get(event['action_id'])
        if code is None or pd.isna(code):
            raise ValueError('attribution_missing_action_mapping: pass actions= for '+str(event['action_id']))
        accruals[pd.Timestamp(event['date'])].append((code,float(event.get('cash',0))))
    if (set(fills)|set(accruals))-set(dates):raise ValueError('attribution_event_outside_calendar')
    prior_nav=float(run['config']['capital']['init_cash']);previous={};rows={}
    max_pnl_error=max_return_error=0.
    for state in states:
        date=pd.Timestamp(state['date']);nav=float(state['nav'])
        if not math.isfinite(prior_nav) or prior_nav<=0:raise ValueError('attribution_invalid_prior_nav')
        if 'quantities' in state and 'marks' in state:
            values={c:q*state['marks'][c] for c,q in state['quantities'].items()}
        else:
            values={c:w*nav for c,w in state['weights'].items()}
        pnl={c:values.get(c,0)-previous.get(c,0) for c in sorted(set(values)|set(previous))}
        for fill in fills.get(date,[]):
            c=fill['instrument_id'];pnl[c]=pnl.get(c,0)+fill['notional']*(1 if fill['side']=='sell' else -1)-fill['fees']
        for code,cash in accruals.get(date,[]):pnl[code]=pnl.get(code,0)+cash
        total=sum(pnl.values());pnl_error=total-(nav-prior_nav)
        return_error=total/prior_nav-float(returns.loc[date])
        if (not all(math.isfinite(v) for v in (*pnl.values(),pnl_error,return_error)) or
                abs(pnl_error)>1e-6 or abs(return_error)>1e-12):
            raise ValueError(f'attribution_reconciliation_failed:{date.date()}:pnl_cny={pnl_error}:return={return_error}')
        max_pnl_error=max(max_pnl_error,abs(pnl_error));max_return_error=max(max_return_error,abs(return_error))
        rows[date]={c:value/prior_nav for c,value in pnl.items()}
        previous=values;prior_nav=nav
    result=pd.DataFrame.from_dict(rows,orient='index').reindex(returns.index).fillna(0.)
    result.attrs.update(reconciliation_max_error=max_return_error,reconciliation_max_pnl_error=max_pnl_error,
                        unit='daily_net_return_contribution',dividend_income_basis='recorded_ex_date_accrual')
    return result


def concentration_diagnostics(strategy,control,ds):
    a=instrument_returns(strategy,actions=strategy.get('input_actions',getattr(ds,'actions',None)))
    b=instrument_returns(control,actions=control.get('input_actions',getattr(ds,'actions',None)))
    if not a.index.equals(b.index):raise ValueError('attribution_comparison_calendar_mismatch')
    excess=a.subtract(b,fill_value=0).fillna(0.)
    if excess.empty:return dict(status='insufficient_data',checks={})
    total=excess.sum(axis=1);stock=excess.sum().idxmax();sectors={}
    for d,row in excess.iterrows():
        records=asof_records(ds.instruments,decision_time(d))
        matrix=exposure_matrix(ds,d,records,list(row.index))
        sectors[d]=(matrix@row).to_dict()
    sector_frame=pd.DataFrame.from_dict(sectors,orient='index').fillna(0.)
    industry=sector_frame.sum().idxmax();best_days=total.nlargest(min(5,len(total))).index
    measures=dict(without_best_stock=float((total-excess[stock]).mean()*252),
        without_best_industry=float((total-sector_frame[industry]).mean()*252),
        without_best_five_days=float(total.drop(best_days).mean()*252) if len(total)>5 else None)
    return dict(status='diagnostic_only',estimator='additive_daily_excess_contribution_removal_not_reoptimized_account',
        strongest_stock=stock,strongest_industry=industry,removed_days=[str(x) for x in best_days],annualized_excess=measures,
        checks={k:v is not None and v>0 for k,v in measures.items()},
        reconciliation_max_error=float((a.sum(axis=1)-strategy['daily_return']).abs().max()))
