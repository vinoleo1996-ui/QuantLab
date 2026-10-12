"""Point-in-time feature library for the P4 alpha engine (own implementation).

A feature on day ``t`` uses only data known at ``t`` 18:00 Asia/Shanghai:
- price/volume fields of sessions ``<= t`` from the v4/v5 panel;
- financial rows whose ``available_at`` is ``<= t`` 18:00 (cloud ``financials_pit``);
- Shenwan L1 membership with ``in_date <= t < out_date``.

Families (fixed in docs/prereg/P4_ALPHA.md before any account run):
  price:  ret_5, ret_20, mom_120_20, vol_20, vol_60, idio_vol_60, max_ret_20, close_to_high_250
  liquidity: turn_20, abn_turn_20_250, amihud_20, log_float_cap, hl_range_20
  industry: ind_ret_20, rel_ind_ret_20
  fundamental: ep_ttm, bp, roe, roe_yoy, gross_margin, debt_to_assets, ocf_to_or, profit_yoy, revenue_yoy
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FEATURES = ('ret_5', 'ret_20', 'mom_120_20', 'vol_20', 'vol_60', 'idio_vol_60', 'max_ret_20',
                  'close_to_high_250')
LIQUIDITY_FEATURES = ('turn_20', 'abn_turn_20_250', 'amihud_20', 'log_float_cap', 'hl_range_20')
INDUSTRY_FEATURES = ('ind_ret_20', 'rel_ind_ret_20')
FUNDAMENTAL_FEATURES = ('ep_ttm', 'bp', 'roe', 'roe_yoy', 'gross_margin', 'debt_to_assets', 'ocf_to_or',
                        'profit_yoy', 'revenue_yoy')
ALL_FEATURES = PRICE_FEATURES + LIQUIDITY_FEATURES + INDUSTRY_FEATURES + FUNDAMENTAL_FEATURES
DECISION_HOUR = 18


class PriceVolume:
    """Wide date x code matrices computed once; every rolling window ends at the row's own session."""

    def __init__(self, fields):
        close, adj = fields['close'], fields['adj_close']
        ret = adj.pct_change(fill_method=None)
        amount = fields['amount'].where(fields['amount'] > 0)
        market = ret.mean(axis=1)
        self.ret = ret
        self.m = {
            'ret_5': adj / adj.shift(5) - 1,
            'ret_20': adj / adj.shift(20) - 1,
            'mom_120_20': adj.shift(20) / adj.shift(120) - 1,
            'vol_20': ret.rolling(20, min_periods=15).std(),
            'vol_60': ret.rolling(60, min_periods=45).std(),
            'idio_vol_60': ret.sub(market, axis=0).rolling(60, min_periods=45).std(),
            'max_ret_20': ret.rolling(20, min_periods=15).max(),
            'close_to_high_250': adj / adj.rolling(250, min_periods=120).max(),
            'turn_20': fields['turnover'].rolling(20, min_periods=15).mean(),
            'abn_turn_20_250': fields['turnover'].rolling(20, min_periods=15).mean()
                               / fields['turnover'].rolling(250, min_periods=120).mean(),
            'amihud_20': (ret.abs() / amount * 1e8).rolling(20, min_periods=15).mean(),
            'log_float_cap': np.log(fields['float_mktcap'].rolling(20, min_periods=10).mean()),
            'hl_range_20': ((fields['high'] - fields['low']) / close).rolling(20, min_periods=15).mean(),
        }

    def on(self, day, codes):
        return pd.DataFrame({name: frame.loc[day].reindex(codes) for name, frame in self.m.items()})


def decision_cutoff(day):
    return pd.Timestamp(day).tz_localize('Asia/Shanghai') + pd.Timedelta(hours=DECISION_HOUR)


def _ts(values):
    out = pd.to_datetime(values, utc=True, errors='coerce')
    return out.dt.tz_convert('Asia/Shanghai')


def code_of(ts_code):
    number, market = str(ts_code).split('.')
    return f'{market.lower()}.{number}'


class Fundamentals:
    """As-of reads of cloud ``financials_pit`` (cumulative year-to-date values per ``end_date``)."""

    COLUMNS = ('n_income_attr_p', 'revenue', 'basic_eps', 'bps', 'roe', 'grossprofit_margin', 'debt_to_assets',
               'ocf_to_or')

    def __init__(self, table):
        frame = table.copy()
        frame['code'] = frame['ts_code'].map(code_of) if 'code' not in frame else frame['code']
        frame['available'] = _ts(frame['available_at'])
        frame['end'] = pd.to_datetime(frame['end_date'].astype(str)).dt.normalize()
        frame = frame.dropna(subset=['available', 'end'])
        for column in self.COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
        self.table = frame.sort_values(['available', 'code', 'end'])

    def snapshot(self, day):
        """Latest version of every (code, end_date) known by ``day`` 18:00."""
        rows = self.table[self.table['available'] <= decision_cutoff(day)]
        return rows.drop_duplicates(['code', 'end'], keep='last').set_index(['code', 'end']).sort_index()

    def on(self, day, codes, close):
        known = self.snapshot(day)
        index = pd.Index(codes, name='code')
        out = pd.DataFrame(np.nan, index=index, columns=list(FUNDAMENTAL_FEATURES))
        if known.empty:
            return out
        latest_end = known.reset_index().groupby('code')['end'].max().reindex(index)
        have = latest_end.notna()
        ends = pd.DatetimeIndex(latest_end[have])
        names = latest_end.index[have]
        row = known.reindex(pd.MultiIndex.from_arrays([names, ends]))
        prior = known.reindex(pd.MultiIndex.from_arrays([names, ends - pd.DateOffset(years=1)]))
        annual = known.reindex(pd.MultiIndex.from_arrays(
            [names, pd.to_datetime([f'{y - 1}-12-31' for y in ends.year])]))
        price = pd.to_numeric(close.reindex(names), errors='coerce').to_numpy()
        price = np.where(price > 0, price, np.nan)
        eps = row.basic_eps.to_numpy()
        eps_ttm = np.where(ends.month == 12, eps, eps + annual.basic_eps.to_numpy() - prior.basic_eps.to_numpy())

        def growth(now, before):
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(np.isfinite(before) & (before != 0), (now - before) / np.abs(before), np.nan)

        values = dict(ep_ttm=eps_ttm / price, bp=row.bps.to_numpy() / price, roe=row.roe.to_numpy(),
                      roe_yoy=row.roe.to_numpy() - prior.roe.to_numpy(),
                      gross_margin=row.grossprofit_margin.to_numpy(), debt_to_assets=row.debt_to_assets.to_numpy(),
                      ocf_to_or=row.ocf_to_or.to_numpy(),
                      profit_yoy=growth(row.n_income_attr_p.to_numpy(), prior.n_income_attr_p.to_numpy()),
                      revenue_yoy=growth(row.revenue.to_numpy(), prior.revenue.to_numpy()))
        out.loc[names, list(values)] = np.column_stack([values[k] for k in values])
        return out


class Industries:
    """Shenwan L1 membership from cloud ``sw_l1_members`` (``in_date``/``out_date`` and ``available_at``)."""

    def __init__(self, table):
        frame = table.copy()
        frame['code'] = frame['instrument_id'] if 'instrument_id' in frame else frame['ts_code'].map(code_of)
        frame['in'] = pd.to_datetime(frame['in_date'].astype(str), errors='coerce')
        frame['out'] = pd.to_datetime(frame['out_date'].astype(str), errors='coerce')
        frame['available'] = _ts(frame['available_at']) if 'available_at' in frame else pd.NaT
        self.table = frame.dropna(subset=['in'])

    def on(self, day):
        day = pd.Timestamp(day)
        rows = self.table[(self.table['in'] <= day) & (self.table['out'].isna() | (self.table['out'] > day))]
        if rows['available'].notna().any():
            rows = rows[rows['available'].isna() | (rows['available'] <= decision_cutoff(day))]
        return rows.sort_values('in').drop_duplicates('code', keep='last').set_index('code')['sw_code']


def industry_features(day, codes, ret_20_all, industry_map):
    """Industry mean 20-day return over all classified names that day, and the stock's excess over it."""
    frame = pd.DataFrame({'ret_20': ret_20_all, 'industry': industry_map.reindex(ret_20_all.index)}).dropna()
    means = frame.groupby('industry')['ret_20'].mean()
    industry = industry_map.reindex(codes)
    ind_ret = industry.map(means)
    return pd.DataFrame({'ind_ret_20': ind_ret.to_numpy(),
                         'rel_ind_ret_20': ret_20_all.reindex(codes).to_numpy() - ind_ret.to_numpy()},
                        index=pd.Index(codes, name='code'))


def cross_section(day, codes, price, fundamentals=None, industries=None, close=None):
    parts = [price.on(day, codes)]
    if industries is not None:
        parts.append(industry_features(day, codes, price.m['ret_20'].loc[day], industries.on(day)))
    if fundamentals is not None:
        parts.append(fundamentals.on(day, codes, close.loc[day] if close is not None else pd.Series(dtype=float)))
    frame = pd.concat([p.set_axis(pd.Index(codes, name='code')) for p in parts], axis=1)
    frame.insert(0, 'date', pd.Timestamp(day))
    return frame.reset_index()
