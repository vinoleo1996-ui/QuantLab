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
    uncertain_pairs: tuple = ()

    def matrix(self,codes):
        """Exposure constraints, including uncertainty, shared by all risk stages.

        UNKNOWN is a budget of its own and is also added to every observed group:
        it may co-fall with any of them. An unknown pair of otherwise observed
        groups receives a joint cap without erasing the groups' other evidence.
        Rows may overlap and must not be added as portfolio attribution totals.
        """
        labels={c:self.groups.get(c,'UNKNOWN') for c in codes}
        names=sorted(set(labels.values()))
        rows={name:{c:float(group==name or group=='UNKNOWN' and name!='UNKNOWN')
                    for c,group in labels.items()} for name in names}
        pairs=set()
        for a,b in self.uncertain_pairs:
            groups=tuple(sorted({self.groups.get(a,'UNKNOWN'),self.groups.get(b,'UNKNOWN')}))
            if len(groups)==2 and 'UNKNOWN' not in groups:pairs.add(groups)
        for groups in sorted(pairs):
            rows['UNCERTAIN_PAIR:'+'|'.join(groups)]={
                c:float(group in groups or group=='UNKNOWN') for c,group in labels.items()}
        return pd.DataFrame.from_dict(rows,orient='index').reindex(columns=codes,fill_value=0.)


def exit_risk(close,turnover,amount,market,date,codes,threshold=.70,nav=600000.,weight=.06,*,include_penalties=True):
    ret=close.loc[:date,codes].pct_change(fill_method=None).tail(120)
    m=market.loc[:date].pct_change(fill_method=None).reindex(ret.index)
    down=ret.loc[m<0]
    groups={};parent={c:c for c in codes};unknown=set();uncertain=[];reasons=[]
    def root(c):
        while parent[c]!=c: c=parent[c]
        return c
    for c in codes:
        count=int(down[c].notna().sum())
        if count<30:
            unknown.add(c);reasons.append(f'downside_history_incomplete:{c}:{count}')
        elif down[c].dropna().nunique()<2:
            unknown.add(c);reasons.append(f'downside_variance_undefined:{c}')
    present=down.notna().astype(np.int64)
    observations=(present.T@present).to_numpy()
    correlations=down.corr(min_periods=30).to_numpy()
    for i,a in enumerate(codes):
        for j in range(i+1,len(codes)):
            b=codes[j]
            if a in unknown or b in unknown:continue
            if observations[i,j]<30:
                uncertain.append((a,b))
                reasons.append(f'pair_history_incomplete:{a}:{b}:{observations[i,j]}')
                continue
            corr=correlations[i,j]
            if not np.isfinite(corr):
                uncertain.append((a,b));reasons.append(f'pair_correlation_undefined:{a}:{b}');continue
            # Keep the previous decision exactly at the correlation threshold;
            # otherwise matrix and pairwise Pearson differ only in roundoff.
            if abs(corr-threshold)<1e-12:
                pair=down[[a,b]].dropna();corr=pair[a].corr(pair[b])
            if corr>=threshold: parent[root(b)]=root(a)
    # Only deficient securities are UNKNOWN. Their potential correlation with
    # observed groups is represented explicitly by overlapping matrix rows.
    for c in codes: groups[c]='UNKNOWN' if c in unknown else 'DOWNSIDE:'+root(c)
    if not include_penalties:
        return RiskSnapshot(str(date),groups,pd.Series(0.,index=codes),pd.Series(0.,index=codes),reasons,tuple(uncertain))
    heat={};illiquid={}
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
    return RiskSnapshot(str(date),groups,pd.Series(heat),pd.Series(illiquid),reasons,tuple(uncertain))
