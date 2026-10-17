"""P5_J premium re-test on the share-split-cleaned premium series.

The earlier run used the raw close/unit_nav premium, which contains -82% artifacts
around the 2022-01 share split of 513100 (and smaller jumps). This re-test (a) prints
the masked-day counts, (b) re-runs the same high-premium spell return split on the
cleaned series, (c) reports the 3% pause trigger counts that the alloc layer sees.
"""
import sys
from pathlib import Path

ROOT = Path('/Users/agiuser/Desktop/QuantLab')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import p4_alloc as R

sessions = pd.date_range('2018-01-02', '2026-09-22', freq='B')
prem = R.qdii_premium_frame(sessions, basis='legacy')
print('masked data-error days:', prem.attrs.get('masked'))

etf = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
w = etf.pivot(index='date', columns='code', values='close_hfq').sort_index()
w.index = pd.to_datetime(w.index)
r = w['sh513100'].pct_change()

for sleeve in ('nasdaq', 'sp500'):
    code = {'nasdaq': 'sh513100', 'sp500': 'sh513500'}[sleeve]
    rr = w[code].pct_change()
    pr = prem[sleeve].reindex(rr.index).ffill()
    rows = {}
    for lab, m in (('prem>8%', pr > 0.08), ('prem<=8%', pr <= 0.08)):
        s = rr[m].dropna()
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        rows[lab] = dict(n=len(s), ann=round((((1 + s).prod()) ** (1 / yrs) - 1) * 100, 2))
    pause = (pr > 0.03).mean()
    print(sleeve, rows, 'pause-day share %.1f%%' % (pause * 100),
          'years with >8%:', sorted(set(pr[pr > 0.08].index.year)))
