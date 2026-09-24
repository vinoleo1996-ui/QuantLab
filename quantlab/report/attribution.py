"""Daily ledger P&L attribution, with removal diagnostics explicitly not tradable replays."""
import pandas as pd
from quantlab.data.contracts import asof_records,decision_time
from quantlab.data.market import exposure_matrix


def instrument_returns(run):
    capital=run['config']['capital']['init_cash'];prior_nav=capital;previous={};rows={}
    actions=run.get('input_actions',pd.DataFrame())
    action_codes=actions.drop_duplicates('action_id').set_index('action_id').instrument_id.to_dict() if not actions.empty else {}
    for state in run.get('account',[]):
        date=pd.Timestamp(state['date']);values={c:w*state['nav'] for c,w in state['weights'].items()}
        pnl={c:values.get(c,0)-previous.get(c,0) for c in set(values)|set(previous)}
        for fill in run.get('fills',[]):
            if pd.Timestamp(fill['date'])!=date:continue
            c=fill['instrument_id'];pnl[c]=pnl.get(c,0)+fill['notional']*(1 if fill['side']=='sell' else -1)-fill['fees']
        for event in run.get('corporate_action_events',[]):
            if event.get('type')!='corporate_action':continue
            if pd.Timestamp(event['date'])!=date:continue
            c=action_codes.get(event['action_id'])
            if c is not None:pnl[c]=pnl.get(c,0)+event['cash']
        rows[date]={c:value/prior_nav for c,value in pnl.items()}
        previous=values;prior_nav=state['nav']
    return pd.DataFrame.from_dict(rows,orient='index').reindex(run['daily_return'].index).fillna(0.)


def concentration_diagnostics(strategy,control,ds):
    a,b=instrument_returns(strategy),instrument_returns(control)
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
