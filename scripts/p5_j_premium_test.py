import sys; from pathlib import Path
ROOT=Path('/Users/agiuser/Desktop/QuantLab'); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np, pandas as pd
import p4_alloc as R
sessions=pd.date_range('2018-01-02','2026-09-22',freq='B')
prem=R.qdii_premium_frame(sessions); p=prem['nasdaq'].dropna()
etf=pd.read_parquet(ROOT/'data/smallcap_assets/etf_bars.parquet')
w=etf.pivot(index='date',columns='code',values='close_hfq').sort_index(); w.index=pd.to_datetime(w.index)
r=w['sh513100'].pct_change()
pr=p.reindex(r.index).ffill()
for lab,m in (('>8%',pr>0.08),('<=8%',pr<=0.08)):
    rr=r[m].dropna()
    yrs=(rr.index[-1]-rr.index[0]).days/365.25
    print(lab,'n',len(rr),'total %.1f%%'%(((1+rr).prod()-1)*100),'ann %.1f%%'%((((1+rr).prod())**(1/yrs)-1)*100))
hi=pr[pr>0.08]
print('years with >8% days:', sorted(set(hi.index.year)), 'share', {y: round((pr[pr.index.year==y]>0.08).mean(),2) for y in range(2018,2027)})
