"""Decision-to-fill capital use; interacting limits are not additive attributions."""
from collections import Counter, defaultdict
import pandas as pd


def capital_report(run):
    orders = {o['order_id']:o for o in run.get('orders',[])}
    fills = defaultdict(list)
    for f in run.get('fills',[]):fills[f['order_id']].append(f)
    states = {str(pd.Timestamp(a['date']).date()):a for a in run.get('account',[])}
    rows=[]
    for t in run.get('signal_trace',[]):
        capital=t.get('capital',{})
        ids=capital.get('order_ids',[])
        actual=sorted([f for oid in ids for f in fills[oid]],key=lambda f:(f['date'],f['order_id']))
        row=dict(date=t['date'],status=t['status'],**capital)
        row.update(industry_policy=run.get('industry_policy'),
            filled_buy_notional=sum(f['notional'] for f in actual if f['side']=='buy'),
            filled_sell_notional=sum(f['notional'] for f in actual if f['side']=='sell'),
            actual_fees=sum(f['fees'] for f in actual),actual_slippage=sum(f['slippage_cost'] for f in actual),
            unfilled=[dict(order_id=oid,remaining=orders[oid]['remaining'],status=orders[oid]['status'],reason=orders[oid]['reject_reason'])
                      for oid in ids if orders[oid]['remaining']],
            cash_after_last_fill=states[actual[-1]['date']]['cash'] if actual else capital.get('cash_before_orders'),
            order_not_submitted=[e for e in run.get('corporate_action_events',[])
                                 if e.get('type')=='order_not_submitted' and e.get('date')==str(pd.Timestamp(t['date']).date())])
        rows.append(row)
    bound=Counter(x for t in rows for x in t.get('binding_constraints',[]))
    reasons=Counter(o['reject_reason'] for o in orders.values() if o['reject_reason'])
    exits=[]
    calendar=pd.DatetimeIndex(run.get('calendar',[]))
    for o in orders.values():
        if o['side']!='sell' or o['reason']!='risk_exit':continue
        fs=fills[o['order_id']]
        end=pd.Timestamp(fs[-1]['date']) if o['remaining']==0 and fs else pd.Timestamp(run['as_of'])
        duration=int(((calendar>pd.Timestamp(o['signal_date']))&(calendar<=end)).sum())
        exits.append(dict(order_id=o['order_id'],instrument_id=o['instrument_id'],signal_date=o['signal_date'],
                          remaining=o['remaining'],completed=o['remaining']==0,elapsed_sessions=duration,
                          reason=o['reject_reason']))
    return dict(definition='Sequential decision and actual-fill accounting; interacting constraint effects are not additive.',
                decisions=rows,binding_constraint_counts=dict(bound),unfilled_reason_counts=dict(reasons),risk_exits=exits)
