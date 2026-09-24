"""Human-readable rotation evidence from the actual shared replay decisions."""
def rotation_advice(result):
    accounts=result.get('account',[])
    if not accounts: return dict(status='no_account',holdings=[],orders=[])
    state=accounts[-1]
    traces=result.get('signal_trace',[])
    scored=[t for t in result.get('weekly_observations',traces) if t.get('scores')]
    current=scored[-1] if scored else {}
    previous=scored[-2] if len(scored)>1 else {}
    def rankings(trace):
        return {c:i+1 for i,c in enumerate(sorted(trace.get('scores',{}),key=lambda c:(-trace['scores'][c],c)))}
    rank=rankings(current);oldrank=rankings(previous)
    pending=[o for o in result.get('orders',[]) if o['status'] in ('pending','partial')]
    orders=[]
    for o in pending:
        c=o['instrument_id'];price=state.get('marks',{}).get(c)
        orders.append(dict(o,remaining_notional=(o.get('estimated_notional',0.)*o['remaining']/max(o['quantity'],1)) if price is None else price*o['remaining'],
                           funding='settled_cash_only' if o['side']=='buy' else 'existing_shares',
                           execution='next_available_open_subject_to_permission',
                           cancellation='new_monthly_decision_or_risk_tightening' if o['side']=='buy' else 'risk_exit_persists'))
    holdings=[]
    for c,quantity in state.get('quantities',{}).items():
        sells=[o for o in pending if o['instrument_id']==c and o['side']=='sell']
        support=[e for e,names in current.get('expert_candidates',{}).items() if c in names]
        holdings.append(dict(instrument_id=c,quantity=quantity,weight=state['weights'].get(c),
            action='reduce' if sells else 'hold',reason=sells[0]['reason'] if sells else 'no_pending_exit',
            current_rank=rank.get(c),previous_rank=oldrank.get(c),supporting_experts=support,
            expert_weights=current.get('expert_weights',{})))
    risk=result.get('risk_history',[])
    return dict(as_of=result.get('as_of'),signal_as_of=current.get('date'),
                risk_budget=risk[-1].get('risk_budget') if risk else None,
                holdings=holdings,orders=orders,cash=state['cash'],
                candidate_evidence=current.get('candidate_evidence',{}),
                executable=False,account_type='simulation',
                observation_policy='weekly_observation_monthly_rotation_daily_risk',
                data_issues=result.get('data_issues',[]))
