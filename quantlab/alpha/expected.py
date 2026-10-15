"""Map a within-domain score percentile to the mean completed forward return of that decile.

The decile is assigned inside one date and one domain, so score scales from different refits are not
pooled. A decision may use a row only after ``label_known_pos`` is at or before the decision session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 10


def assign_decile(frame, n_bins=N_BINS):
    """``frame`` needs ``date``, ``domain`` and ``GBDT``. Adds ``decile`` in ``0 .. n_bins-1``."""
    out = frame.copy()
    pct = out.groupby(['date', 'domain'], sort=False)['GBDT'].rank(pct=True, method='average')
    out['decile'] = np.minimum((pct.to_numpy() * n_bins).astype(int), n_bins - 1)
    return out


def daily_decile_totals(frame):
    """Sum completed ``label_raw`` by domain, decile and the session at which the label is known."""
    done = frame.dropna(subset=['label_raw']).copy()
    done['label_known_pos'] = done['label_known_pos'].astype(int)
    return (done.groupby(['domain', 'decile', 'label_known_pos'], as_index=False)
            .agg(label_sum=('label_raw', 'sum'), label_n=('label_raw', 'size')))


def lagged_decile_means(daily, asof_pos):
    """Mean completed return of each ``(domain, decile)`` whose label session is ``<= asof_pos``."""
    done = daily[daily.label_known_pos <= asof_pos]
    if done.empty:
        return pd.Series(dtype=float)
    grouped = done.groupby(['domain', 'decile'])[['label_sum', 'label_n']].sum()
    return grouped.label_sum / grouped.label_n
