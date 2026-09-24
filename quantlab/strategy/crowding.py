"""Exit-risk proxies; these do not estimate other investors' actual positions."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class RiskSnapshot:
    as_of: str
    groups: dict
    overheating: pd.Series
    illiquidity: pd.Series
    reasons: list

    def matrix(self,codes):
        return pd.DataFrame({c:{self.groups.get(c,'UNKNOWN'):1.} for c in codes}).fillna(0.)


def exit_risk(close,turnover,amount,market,date,codes,threshold=.70,nav=600000.,weight=.06):
    ret=close.loc[:date,codes].pct_change(fill_method=None).tail(120)
    m=market.loc[:date].pct_change(fill_method=None).reindex(ret.index)
    down=ret.loc[m<0]
    groups={};parent={c:c for c in codes};unknown=set()
    def root(c):
        while parent[c]!=c: c=parent[c]
        return c
    for c in codes:
        if down[c].notna().sum()<30: unknown.add(c)
    for i,a in enumerate(codes):
        for b in codes[i+1:]:
            pair=down[[a,b]].dropna()
            if len(pair)<30: unknown.update((a,b));continue
            corr=pair[a].corr(pair[b])
            if not np.isfinite(corr): unknown.update((a,b));continue
            if corr>=threshold: parent[root(b)]=root(a)
    # If any member has unknown correlation, its connected component is conservative unknown.
    unknown_roots={root(c) for c in unknown}
    for c in codes: groups[c]='UNKNOWN' if root(c) in unknown_roots else 'DOWNSIDE:'+root(c)
    heat={};illiquid={};reasons=[]
    for c in codes:
        t=turnover.loc[:date,c].tail(252).dropna()
        px=close.loc[:date,c].dropna();amt=amount.loc[:date,c].tail(20)
        if len(t)<252 or len(px)<21:
            heat[c]=.5;reasons.append(c+':overheating_history_incomplete')
        else:
            q=float((t<=t.iloc[-1]).mean())
            acceleration=(px.iloc[-1]/px.iloc[-6]-1)-(px.iloc[-6]/px.iloc[-21]-1)/3
            adv=amt.mean(); divergence=max(0.,(px.iloc[-1]/px.iloc[-21]-1))*max(0.,1-float(amt.tail(5).mean()/adv))
            heat[c]=float(np.clip(((q-.6)/.4+max(0.,acceleration)/.1+divergence/.1)/3,0,1))
        stress_adv=amt.mean()*.3
        illiquid[c]=float(np.clip(nav*weight/max(stress_adv*.01,1.),0,1)) if amt.notna().sum()==20 else 1.
    return RiskSnapshot(str(date),groups,pd.Series(heat),pd.Series(illiquid),reasons)
