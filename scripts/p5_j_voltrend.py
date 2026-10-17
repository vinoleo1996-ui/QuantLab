import sys; from pathlib import Path
ROOT=Path('/Users/agiuser/Desktop/QuantLab'); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np, pandas as pd
import p4_alloc as R
from quantlab import p4_alloc as A
import p4_alpha as P
from quantlab.smallcap import universe as U
from quantlab.smallcap.panel import Panel

panel=Panel(P.PANEL); f=U.rolling_features(panel, with_st=True)
ret=panel.field('adj_close').pct_change(fill_method=None); sessions=panel.sessions
# pool EW daily index (ADV bottom 40%, ex-ST, ex-BJ, ADV>=5m, hist>=252)
idx_rows={}
for d in sessions[::5]:
    a=f['adv'].loc[d]; h=f['history'].loc[d]; st=f['st'].loc[d]
    keep=(h>=252)&a.notna()&(a>=5e6)&~st.eq(1)
    codes=[c for c in a[keep].index if not c.startswith('bj.')]
    idx_rows[d]=codes
pool_ret=[]
dates=sorted(idx_rows)
for i,d in enumerate(dates):
    nxt=dates[i+1] if i+1<len(dates) else None
    if nxt is None: break
    win=sessions[(sessions>d)&(sessions<=nxt)]
    r=ret.loc[win, idx_rows[d]].mean(axis=1)
    pool_ret.append((r.index, r))
pool=pd.concat([r for _,r in pool_ret]).groupby(level=0).first().sort_index()
pool_idx=(1+pool).cumprod()

def qends(s):
    x=pd.Series(s,index=s); return set(x.groupby([s.years,0]).max()) if False else set(x.groupby([s.year,(s.month-1)//3]).max())
def etf(sess):
    b=pd.read_parquet(ROOT/'data/smallcap_assets/etf_bars.parquet')
    w=b.pivot(index='date',columns='code',values='close_hfq').sort_index(); w.index=pd.to_datetime(w.index)
    w=w[['sh518880','sh511260','sh513100']].rename(columns={'sh518880':'gold','sh511260':'t10','sh513100':'nasdaq'})
    return w.reindex(w.index.union(sess)).ffill().reindex(sess).pct_change(fill_method=None).fillna(0.0)

def run(cost):
    out=[]
    for p in sorted((ROOT/'output/p4_gross_20261013/ledger').glob('nav_POOL50_G97_p*_cost%d.csv'%cost)):
        eq=pd.read_csv(p,index_col=0,parse_dates=True)['nav'].loc['2018-01-02':'2026-09-22']
        eqr=eq.pct_change(fill_method=None).fillna(0.0)
        E=etf(eq.index)
        S=pool_idx.reindex(eq.index).ffill()
        ma120=S.rolling(120,min_periods=60).mean()
        vol20=(pool.reindex(eq.index).fillna(0).rolling(20).std()*np.sqrt(252))
        month_ends=set(pd.Series(eq.index).groupby([eq.index.year,eq.index.month]).max().values)
        q_ends=qends(eq.index)
        w=np.array([0.65,0.13125,0.13125,0.0875]); nav=1.0; ser=[1.0]
        for t,day in enumerate(eq.index):
            if t>0:
                g=w*(1+np.array([eqr.iloc[t],E.gold.iloc[t],E.t10.iloc[t],E.nasdaq.iloc[t]]))
                nav*=g.sum(); w=g/g.sum()
            ser.append(nav)
            if day in month_ends:
                trend=S.loc[day]<ma120.loc[day] if pd.notna(ma120.loc[day]) else False
                sig=vol20.loc[day] if pd.notna(vol20.loc[day]) else 0
                if trend and sig>0.25:
                    new_eq=max(0.35,0.65*0.25/sig)
                else:
                    new_eq=0.65
                freed=0.65-new_eq
                if day in q_ends or abs(new_eq-0.65)>1e-9:
                    new=np.array([new_eq,0.13125+freed/2,0.13125+freed/2,0.0875])
                else:
                    new=w.copy(); new[0]=new_eq; rest=new[1:4].sum(); new[1:4]=new[1:4]/(rest+1e-9)*0.35 if rest>0 else np.array([0.13125+freed/2,0.13125+freed/2,0.0875])
                c=np.array([0.004,0.0005,0.0005,0.0005])*cost
                nav*=1-float((np.abs(new-w)*c).sum()); w=new/new.sum() if new.sum()>0 else w
        navs=pd.Series([np.nan]*0)
        yrs=(eq.index[-1]-eq.index[0]).days/365.25
        ns=pd.Series(ser)
        out.append((nav**(1/yrs)-1,float((ns/ns.cummax()-1).min())))
    return out
for cost in (1,2):
    res=run(cost)
    print('cost',cost,'cagr med %.2f'%(np.median([r[0] for r in res])*100),'mdd med %.2f'%(np.median([r[1] for r in res])*100))
