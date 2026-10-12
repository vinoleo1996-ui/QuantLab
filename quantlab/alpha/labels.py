"""H=20 labels: adjusted open of session t+1 to adjusted open of session t+21.

A sample is unlabeled when the entry open cannot be bought: suspended at t+1
or opening at the limit-up price. Labels are returned with the index of the
session at which they are fully known (t+21), for the training embargo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 20


def adjusted_open(fields):
    factor = fields['adj_close'] / fields['close']
    return fields['open'] * factor


def forward_open_returns(fields, horizon=HORIZON):
    adj_open = adjusted_open(fields)
    entry, exit_ = adj_open.shift(-1), adj_open.shift(-(1 + horizon))
    label = exit_ / entry - 1
    tradable = fields['tradestatus'].shift(-1).eq(1)
    if 'limit_up' in fields:
        locked = fields['open'].shift(-1) >= fields['limit_up'].shift(-1) - 1e-6
        tradable &= ~locked.fillna(False)
    label = label.where(tradable & np.isfinite(label))
    return label


def label_end_position(sessions, day, horizon=HORIZON):
    """Position in ``sessions`` of the session at which the label of ``day`` is known."""
    return int(pd.DatetimeIndex(sessions).get_loc(pd.Timestamp(day))) + 1 + horizon
