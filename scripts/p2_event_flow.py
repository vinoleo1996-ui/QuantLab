#!/usr/bin/env python3
"""P2 event/flow family on the large/mid-cap pool (docs/prereg/P2_EVENT_FLOW_BROAD.md).

  python scripts/p2_event_flow.py [--window is|ref2017]

Writes output/p2_event_flow_20261008/<window>/{periods.parquet, daily.parquet, report.json}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import pandas as pd

from neutral_alpha import neutralize, rank_ic, sector_letters, stamp, zscore
from quantlab import research_statistics as RS
from quantlab import research_trials as T
from quantlab.smallcap.panel import Panel
from quantlab.smallcap.universe import Industries

VERSION = '20261006T152412Z'
PREREG = 'docs/prereg/P2_EVENT_FLOW_BROAD.md'
FAMILY = 'P2_event_flow_broad'
INFO = ROOT / 'data/smallcap_info'
OUT = ROOT / 'output/p2_event_flow_20261008'
WINDOWS = {'is': ('2018-01-02', '2026-09-22'), 'ref2017': ('2017-03-01', '2017-12-29')}
SECTION = {'MF1': -1, 'MF2': -1, 'MF3': -1, 'K1': 1, 'H1': -1, 'G1': -1, 'E2': 1}
EVENTS = {'B1': -1, 'B2': 1, 'K2': -1, 'E1': 1, 'S1': 1, 'S2': 1}
STEP, HORIZON, HOLD, EXIT_SEARCH = 20, 20, 20, 20
MIN_ADV, TOP, SMALL = 5_000_000.0, 0.20, 0.30
SLIP, COMMISSION = 0.003, 0.00025
MF3_END = pd.Timestamp('2023-10-13')
MARGIN_SEASONING = 60
FORECAST_SIGN = {'预增': 1, '扭亏': 1, '略增': 1, '续盈': 1, '预减': -1, '首亏': -1, '续亏': -1, '略减': -1}


def panel_code(code):
    code = str(code).zfill(6)
    if code.startswith('6'):
        return f'sh.{code}'
    if code[0] in '03':
        return f'sz.{code}'
    return None


class Data:
    def __init__(self):
        t0 = time.time()
        panel = Panel(VERSION)
        f = lambda name: panel.field(name).astype('float64')
        close, adj = f('close'), f('adj_close')
        self.dates, self.codes = close.index, close.columns
        self.col = {c: j for j, c in enumerate(self.codes)}
        factor = (adj / close).ffill()
        self.adj = adj.to_numpy()
        self.adj_open = (f('open') * factor).to_numpy()
        self.open = f('open').to_numpy()
        self.up = f('limit_up').to_numpy()
        self.down = f('limit_down').to_numpy()
        self.status = f('tradestatus').to_numpy()
        self.st = f('is_st').to_numpy()
        amount = f('amount')
        self.amount = amount.to_numpy()
        self.adv20 = amount.rolling(20, min_periods=15).mean().to_numpy()
        self.cap = f('float_mktcap').to_numpy()
        self.pb = f('pb_mrq').to_numpy()
        self.ret = adj.pct_change(fill_method=None).to_numpy()
        self.industries = Industries(panel.table('industry'))
        self.pool, self.month_of = self.load_pool()
        self.buckets = {}
        self.load_events()
        print(json.dumps({'stage': 'loaded', 'seconds': round(time.time() - t0)}), flush=True)

    def idx(self, days, side='left'):
        return self.dates.searchsorted(pd.DatetimeIndex(pd.to_datetime(days)), side=side)

    def load_pool(self):
        pool = pd.read_parquet(ROOT / 'data/broad/top1500.parquet')
        months = sorted(pool.date.unique())
        member = {m: np.zeros(len(self.codes), bool) for m in months}
        for m, group in pool.groupby('date'):
            member[m][[self.col[c] for c in group.code]] = True
        month_of = np.full(len(self.dates), -1)
        mi = self.idx(months)
        for k, i in enumerate(mi):
            month_of[i:] = k
        self.months = [pd.Timestamp(m) for m in months]
        self.month_idx = mi
        return [member[m] for m in months], month_of

    def in_pool(self, i):
        k = self.month_of[i]
        return self.pool[k] if k >= 0 else np.zeros(len(self.codes), bool)

    def eligible(self, i):
        return (self.in_pool(i) & (self.status[i] == 1) & (self.st[i] != 1) & (self.adv20[i] >= MIN_ADV)
                & np.isfinite(self.cap[i]))

    def bucket(self, i):
        """Pool cap quintile x industry letter as of the latest month end (-1 outside the pool)."""
        k = self.month_of[i]
        if k not in self.buckets:
            m = self.month_idx[k]
            members = self.pool[k]
            out = np.full(len(self.codes), -1)
            caps = self.cap[m, members]
            quint = np.minimum((pd.Series(caps).rank(pct=True).to_numpy() * 5).astype(int), 4)
            letters = sector_letters(self.industries, self.dates[m], self.codes[members])
            keys = pd.Series([f'{q}{l}' for q, l in zip(quint, letters)])
            out[members] = pd.factorize(keys)[0]
            self.buckets[k] = out
        return self.buckets[k]

    # -------- event tables
    def table(self, name, code_col='SECURITY_CODE'):
        frame = pd.read_parquet(INFO / f'{name}.parquet')
        raw = frame[code_col].astype(str)
        frame['code'] = raw.map({c: panel_code(c) for c in raw.unique()})
        return frame[frame.code.isin(self.col)].copy()

    def load_events(self):
        n = len(self.dates)
        margin = self.table('margin', 'SCODE')
        margin['DATE'] = pd.to_datetime(margin.DATE)
        margin = margin[margin.DATE < pd.Timestamp('2026-09-30')]
        sessions = self.dates[(self.dates >= margin.DATE.min()) & (self.dates <= margin.DATE.max())]
        missing = sessions.difference(pd.DatetimeIndex(margin.DATE.unique()))
        self.margin_missing = [str(d.date()) for d in missing]
        pivot = lambda field: margin.pivot_table(index='DATE', columns='code', values=field, aggfunc='last') \
            .reindex(index=self.dates, columns=self.codes)
        rzye, rqye = pivot('RZYE'), pivot('RQYE')
        for day in missing:
            i = self.dates.get_loc(day)
            rzye.iloc[i] = rzye.iloc[i - 1]
            rqye.iloc[i] = rqye.iloc[i - 1]
        self.rzye, self.rqye = rzye.to_numpy(float), rqye.to_numpy(float)
        self.rzmre = pivot('RZMRE').to_numpy(float)
        first = pd.Series(self.idx(margin.groupby('code').DATE.min()), index=margin.groupby('code').DATE.min().index)
        self.margin_first = np.full(len(self.codes), 10 ** 9)
        self.margin_first[[self.col[c] for c in first.index]] = first.to_numpy()

        block = self.table('block_trade')
        block['i'] = self.idx(block.TRADE_DATE)
        block = block[block.i < n]
        block['wp'] = block.DEAL_AMT * block.PREMIUM_RATIO
        agg = block.groupby(['i', 'code'])[['DEAL_AMT', 'wp']].sum().reset_index()
        self.block_amt = np.zeros((n, len(self.codes)))
        self.block_wp = np.zeros((n, len(self.codes)))
        cols = agg.code.map(self.col).to_numpy()
        self.block_amt[agg.i.to_numpy(), cols] = agg.DEAL_AMT.to_numpy()
        self.block_wp[agg.i.to_numpy(), cols] = agg.wp.to_numpy()
        self.block_amt_cum, self.block_wp_cum = self.block_amt.cumsum(0), self.block_wp.cumsum(0)

        holder = self.table('holder_num')
        holder['avail'] = self.idx(holder.HOLD_NOTICE_DATE, side='right')
        self.holder = holder.sort_values(['END_DATE', 'HOLD_NOTICE_DATE'])[['code', 'avail', 'HOLDER_NUM_RATIO']]
        pledge = self.table('pledge')
        pledge['avail'] = self.idx(pledge.TRADE_DATE) + 5
        self.pledge = pledge.sort_values('TRADE_DATE')[['code', 'avail', 'PLEDGE_RATIO']]
        express = self.table('express')
        express['avail'] = self.idx(express.NOTICE_DATE, side='right')
        self.express = express.sort_values('NOTICE_DATE')[['code', 'avail', 'JLRTBZCL']]

        self.events = {name: self.event_frame(name) for name in EVENTS}

    def event_frame(self, name):
        """Columns: code, avail (session index whose close makes the event usable), value (signed)."""
        if name in ('B1', 'B2', 'K2'):
            source = {'B1': 'billboard', 'B2': 'billboard_inst', 'K2': 'block_trade'}[name]
            frame = self.table(source)
            frame['i'] = self.idx(frame.TRADE_DATE)
            frame = frame[frame.i < len(self.dates)]
            if name == 'B1':
                out = frame.groupby(['i', 'code']).size().rename('value').reset_index()
            elif name == 'B2':
                out = frame.groupby(['i', 'code']).NET_BUY_AMT.sum().rename('value').reset_index()
            else:
                out = frame.groupby(['i', 'code']).DEAL_AMT.sum().rename('value').reset_index()
                cap = self.cap[out.i.to_numpy(), out.code.map(self.col).to_numpy()]
                out['value'] = np.where(out.value.to_numpy() / cap >= 0.005, 1.0, 0.0)
            out['avail'] = out.i + 1
        elif name == 'E1':
            frame = self.table('forecast')
            frame['rank'] = frame.PREDICT_FINANCE_CODE.map({'004': 0, '002': 1})
            frame = frame.dropna(subset=['rank']).sort_values('rank')
            frame = frame.groupby(['code', 'NOTICE_DATE', 'REPORT_DATE']).first().reset_index()
            frame['value'] = frame.PREDICT_TYPE.map(FORECAST_SIGN).fillna(0.0)
            frame['avail'] = self.idx(frame.NOTICE_DATE, side='right')
            out = frame.groupby(['avail', 'code']).value.sum().reset_index()
        elif name == 'S1':
            frame = self.table('holder_change')
            frame['value'] = np.where(frame.DIRECTION == '增持', 1.0, -1.0) * frame.CHANGE_FREE_RATIO.fillna(0.0)
            frame['avail'] = self.idx(frame.NOTICE_DATE, side='right')
            out = frame.groupby(['avail', 'code']).value.sum().reset_index()
        else:
            frame = self.table('exec_trade')
            frame['avail'] = self.idx(frame.CHANGE_DATE) + 3
            out = frame.groupby(['avail', 'code']).CHANGE_SHARES.sum().rename('value').reset_index()
        out = out[(out.value != 0) & (out.avail < len(self.dates))]
        return out[['code', 'avail', 'value']].reset_index(drop=True)

    # -------- cross-sectional signals at decision i (information through the close of i)
    def latest(self, frame, column, i, mask, window=None):
        part = frame[frame.avail <= i]
        if window is not None:
            part = part[part.avail > i - window]
        values = part.groupby('code')[column].last()
        series = pd.Series(np.nan, index=self.codes)
        series.loc[values.index] = values.to_numpy()
        return series.to_numpy()[mask]

    def section(self, i, mask):
        j = i - 1
        seasoned = (j - self.margin_first) >= MARGIN_SEASONING
        with np.errstate(divide='ignore', invalid='ignore'):
            mf1 = (self.rzye[j] - self.rzye[j - 20]) / self.cap[j]
            flows = self.rzmre[j - 19:j + 1]
            avail = np.isfinite(flows).sum(0)
            buy = np.nansum(flows, 0) * 20 / np.maximum(avail, 1)
            mf2 = np.where(avail >= 15, buy / np.nansum(self.amount[j - 19:j + 1], 0), np.nan)
            mf3 = self.rqye[j] / self.cap[j]
            amt = self.block_amt_cum[j] - self.block_amt_cum[j - 60]
            k1 = np.where(amt > 0, (self.block_wp_cum[j] - self.block_wp_cum[j - 60]) / amt, np.nan)
        out = {'MF1': np.where(seasoned, mf1, np.nan)[mask], 'MF2': np.where(seasoned, mf2, np.nan)[mask],
               'MF3': np.where(seasoned, mf3, np.nan)[mask], 'K1': k1[mask],
               'H1': self.latest(self.holder, 'HOLDER_NUM_RATIO', i, mask),
               'G1': self.latest(self.pledge, 'PLEDGE_RATIO', i, mask),
               'E2': np.clip(self.latest(self.express, 'JLRTBZCL', i, mask, window=120), -200, 200)}
        return out

    def forward(self, i, mask):
        n, e = len(self.dates), i + 1
        cols = np.where(mask)[0]
        if e >= n:
            return np.full(len(cols), np.nan)
        entry = self.adj_open[e, cols]
        blocked = (~np.isfinite(entry) | (self.status[e, cols] != 1)
                   | (np.isfinite(self.up[e, cols]) & (self.open[e, cols] >= self.up[e, cols] - 0.005)))
        exit_ = np.full(len(cols), np.nan)
        todo = np.ones(len(cols), bool)
        for x in range(e + HORIZON, min(e + HORIZON + EXIT_SEARCH + 1, n)):
            price = self.adj_open[x, cols]
            ok = todo & np.isfinite(price) & (self.status[x, cols] == 1) & ~(
                np.isfinite(self.down[x, cols]) & (self.open[x, cols] <= self.down[x, cols] + 0.005))
            exit_[ok] = price[ok]
            todo &= ~ok
        if e + HORIZON >= n:
            return np.full(len(cols), np.nan)
        if todo.any():
            last = min(e + HORIZON + EXIT_SEARCH, n - 1)
            exit_[todo] = pd.DataFrame(self.adj[e:last + 1, cols[todo]]).ffill().iloc[-1].to_numpy() * 0.5
        return np.where(blocked, np.nan, exit_ / entry - 1)


def round_trip(day):
    return 2 * (SLIP + COMMISSION) + stamp(day)


def cross_sectional(data, first, last):
    rows, previous = [], {name: None for name in SECTION}
    for i in range(first, last + 1, STEP):
        mask = data.eligible(i)
        codes = data.codes[mask]
        fwd = data.forward(i, mask)
        valid = np.isfinite(fwd)
        logcap = np.log(data.cap[i, mask])
        industry = sector_letters(data.industries, data.dates[i], codes)
        small = logcap <= np.quantile(logcap, SMALL)
        raw = data.section(i, mask)
        day = data.dates[i]
        universe = float(np.nanmean(fwd))
        lo, hi = np.quantile(logcap, [0.3, 0.7])
        pb = data.pb[i, mask]
        pos = np.isfinite(pb) & (pb > 0)
        plo, phi = np.quantile(pb[pos], [0.3, 0.7])
        row = {'date': day, 'names': int(valid.sum()), 'MKT': universe,
               'SMB': float(np.nanmean(fwd[logcap <= lo]) - np.nanmean(fwd[logcap >= hi])),
               'HML': float(np.nanmean(fwd[pos & (pb <= plo)]) - np.nanmean(fwd[pos & (pb >= phi)]))}
        cap_pct = pd.Series(logcap).rank(pct=True).to_numpy()
        for name, direction in SECTION.items():
            s = direction * neutralize(raw[name], logcap, industry, small)
            robust = np.full(len(s), np.nan)
            robust[~small] = direction * neutralize(raw[name][~small], logcap[~small], industry[~small])
            row[f'{name}_ic'] = rank_ic(s, fwd)
            row[f'{name}_ic_robust'] = rank_ic(robust, fwd)
            ok = np.isfinite(s) & valid
            row[f'{name}_coverage'] = float(ok.sum() / max(valid.sum(), 1))
            if ok.sum() < 50:
                for key in ('gross', 'net', 'net2x', 'raw_net', 'ls'):
                    row[f'{name}_{key}'] = np.nan
                continue
            order = np.argsort(-np.where(ok, s, -np.inf))
            n_cut = int(math.floor(TOP * ok.sum()))
            best, worst = order[:n_cut], order[ok.sum() - n_cut:ok.sum()]
            row[f'{name}_ls'] = float(np.mean(fwd[best]) - np.mean(fwd[worst]))
            if direction > 0:
                chosen = set(codes[best])
                gross = float(np.mean(fwd[best])) - universe
                tau = 1.0 if previous[name] is None else 1 - len(chosen & previous[name]) / max(len(chosen), 1)
                cost = tau * round_trip(day)
                row[f'{name}_size_pct'] = float(np.median(cap_pct[best]))
                row[f'{name}_adv_1pct'] = float(np.median(data.adv20[i, mask][best]) * 0.01)
            else:
                chosen = set(codes[worst])
                keep = valid.copy()
                keep[worst] = False
                gross = float(np.mean(fwd[keep])) - universe
                tau = 1.0 if previous[name] is None else 1 - len(chosen & previous[name]) / max(len(chosen), 1)
                cost = tau * len(chosen) / max(keep.sum(), 1) * round_trip(day)
                row[f'{name}_size_pct'] = float(np.median(cap_pct[keep]))
                row[f'{name}_adv_1pct'] = float(np.median(data.adv20[i, mask][keep]) * 0.01)
            previous[name] = chosen
            row[f'{name}_gross'] = gross
            row[f'{name}_net'] = gross - cost
            row[f'{name}_net2x'] = gross - 2 * cost
            row[f'{name}_raw_net'] = gross - cost + universe
        rows.append(row)
    return pd.DataFrame(rows).set_index('date')


def daily_factors(data, t):
    mask = data.eligible(t - 1) & np.isfinite(data.ret[t])
    r, c, pb = data.ret[t, mask], data.cap[t - 1, mask], data.pb[t - 1, mask]
    if mask.sum() < 100:
        return np.nan, np.nan, np.nan, mask
    lo, hi = np.quantile(c, [0.3, 0.7])
    pos = np.isfinite(pb) & (pb > 0)
    plo, phi = np.quantile(pb[pos], [0.3, 0.7])
    return (float(r.mean()), float(r[c <= lo].mean() - r[c >= hi].mean()),
            float(r[pos & (pb <= plo)].mean() - r[pos & (pb >= phi)].mean()), mask)


def event_daily(data, first, last):
    """Calendar-time portfolios for every event signal, one pass over the sessions."""
    by_entry = {}
    for name, frame in data.events.items():
        direction = EVENTS[name]
        good = frame[frame.value * direction > 0] if direction > 0 else frame
        entries = {}
        for e, group in good.groupby(good.avail + 1):
            entries[int(e)] = [data.col[c] for c in group.code]
        by_entry[name] = entries
    held = {name: {} for name in EVENTS}
    excluded = {name: {} for name in EVENTS if EVENTS[name] < 0}
    rows = []
    for t in range(first, last + 1):
        day = data.dates[t]
        mkt, smb, hml, live = daily_factors(data, t)
        bucket = data.bucket(t)
        ok_ret = live & (bucket >= 0)
        sums = np.bincount(bucket[ok_ret], weights=data.ret[t, ok_ret], minlength=bucket.max() + 1)
        counts = np.bincount(bucket[ok_ret], minlength=bucket.max() + 1)
        bmean = np.divide(sums, counts, out=np.full(len(sums), np.nan), where=counts > 0)
        row = {'date': day, 'MKT': mkt, 'SMB': smb, 'HML': hml}
        eligible = data.eligible(t)
        for name, direction in EVENTS.items():
            new = by_entry[name].get(t, [])
            if direction > 0:
                book = held[name]
                entered = 0
                for j in new:
                    if j in book or not eligible[j] or not np.isfinite(data.adj_open[t, j]):
                        continue
                    if np.isfinite(data.up[t, j]) and data.open[t, j] >= data.up[t, j] - 0.005:
                        continue
                    book[j] = t + HOLD - 1
                    entered += 1
                returns, excess = [], []
                for j, exit_t in book.items():
                    r = data.adj[t, j] / data.adj_open[t, j] - 1 if exit_t == t + HOLD - 1 else data.ret[t, j]
                    if not np.isfinite(r):
                        r = 0.0
                    b = bmean[bucket[j]] if bucket[j] >= 0 and bucket[j] < len(bmean) else np.nan
                    returns.append(r)
                    excess.append(r - (b if np.isfinite(b) else mkt))
                exits = [j for j, exit_t in book.items() if exit_t == t]
                n = len(book)
                cost = (entered * (SLIP + COMMISSION) + len(exits) * (SLIP + COMMISSION + stamp(day))) / n if n else 0.0
                row[f'{name}_n'] = n
                row[f'{name}_excess'] = float(np.mean(excess)) - cost if n else 0.0
                row[f'{name}_excess2x'] = float(np.mean(excess)) - 2 * cost if n else 0.0
                row[f'{name}_raw'] = float(np.mean(returns)) - cost if n else 0.0
                row[f'{name}_size_pct'] = float(np.median(
                    pd.Series(data.cap[t - 1]).where(live).rank(pct=True).to_numpy()[list(book)])) if n else np.nan
                for j in exits:
                    del book[j]
            else:
                book = excluded[name]
                fresh = 0
                for j in new:
                    if j not in book:
                        fresh += 1
                    book[j] = t + HOLD - 1
                universe = live
                drop = np.zeros(len(data.codes), bool)
                drop[list(book)] = True
                keep = universe & ~drop
                leaving = [j for j, exit_t in book.items() if exit_t == t]
                n_keep = keep.sum()
                if n_keep and universe.sum():
                    gross = float(data.ret[t, keep].mean() - data.ret[t, universe].mean())
                    cost = (fresh * (SLIP + COMMISSION + stamp(day)) + len(leaving) * (SLIP + COMMISSION)) / n_keep
                    row[f'{name}_excess'] = gross - cost
                    row[f'{name}_excess2x'] = gross - 2 * cost
                    row[f'{name}_raw'] = float(data.ret[t, keep].mean()) - cost
                else:
                    row[f'{name}_excess'] = row[f'{name}_excess2x'] = row[f'{name}_raw'] = 0.0
                row[f'{name}_n'] = int((drop & universe).sum())
                for j in leaving:
                    del book[j]
        rows.append(row)
    return pd.DataFrame(rows).set_index('date')


def nw(series, x=None, lags=3):
    frame = pd.DataFrame({'y': series}) if x is None else pd.concat([series.rename('y'), x], axis=1)
    frame = frame.dropna()
    if len(frame) < 12:
        return [np.nan], [np.nan]
    beta, t = RS.newey_west(frame.y.to_numpy(), None if x is None else frame.drop(columns='y').to_numpy(), lags=lags)
    return [float(b) for b in beta], [float(v) for v in t]


def years_positive(series, end=None):
    s = series if end is None else series[series.index <= end]
    years = s.groupby(s.index.year).sum()
    return {int(k): float(v) for k, v in years.items()}


def evaluate(periods, daily, m2):
    trials = max(T.count(FAMILY, include_diagnostics=False), 2)
    annual_sharpes, info = {}, {}
    for name, direction in SECTION.items():
        net = periods[f'{name}_net']
        annual_sharpes[name] = float(net.mean() / net.std() * math.sqrt(252 / STEP)) if net.std() > 0 else np.nan
    for name in EVENTS:
        ex = daily[f'{name}_excess']
        annual_sharpes[name] = float(ex.mean() / ex.std() * math.sqrt(252)) if ex.std() > 0 else np.nan
    spread = float(pd.Series(annual_sharpes).std())
    for name, direction in SECTION.items():
        net = periods[f'{name}_net']
        end = MF3_END if name == 'MF3' else None
        gate_net = net if end is None else net[net.index <= end]
        _, ic_t = nw(periods[f'{name}_ic'] if end is None else periods[f'{name}_ic'][periods.index <= end])
        _, net_t = nw(gate_net)
        beta, tstat = nw(periods[f'{name}_raw_net'] if end is None else periods[f'{name}_raw_net'][periods.index <= end],
                         periods[['MKT', 'SMB', 'HML']])
        years = years_positive(gate_net)
        need = 4 if name == 'MF3' else 6
        gate = {'ic_t>=3': bool(ic_t[0] >= 3), 'long_only_t>=2': bool(net_t[0] >= 2),
                'net_annual>0': bool(gate_net.mean() > 0),
                f'years_positive>={need}': bool(sum(v > 0 for v in years.values()) >= need),
                'ff3_alpha_t>=2': bool(tstat[0] >= 2)}
        gate['pass'] = all(gate.values())
        m2_corr = float(pd.concat([net, m2.reindex(net.index)], axis=1).dropna().corr().iloc[0, 1]) if m2 is not None else None
        dsr = RS.deflated_sharpe(net.fillna(0.0), trials=trials, sharpe_std=spread / math.sqrt(252 / STEP))
        info[name] = {'framework': 'cross_sectional', 'direction': direction, 'gate': gate,
                      'ic_mean': float(periods[f'{name}_ic'].mean()), 'ic_nw_t': ic_t[0],
                      'robust_ex_small30_ic_t': nw(periods[f'{name}_ic_robust'])[1][0],
                      'long_only_net_annual': float(gate_net.mean() * 252 / STEP), 'long_only_nw_t': net_t[0],
                      'net2x_annual': float(periods[f'{name}_net2x'].mean() * 252 / STEP),
                      'long_short_gross_annual': float(periods[f'{name}_ls'].mean() * 252 / STEP),
                      'ff3_alpha_annual': beta[0] * 252 / STEP, 'ff3_alpha_t': tstat[0],
                      'betas': dict(zip(('MKT', 'SMB', 'HML'), beta[1:])), 'years': years,
                      'years_full_8': {k: v for k, v in years.items() if k < 2026},
                      'coverage_median': float(periods[f'{name}_coverage'].median()),
                      'size_pct_median': float(periods[f'{name}_size_pct'].median()),
                      'capacity_cny_median': float(periods[f'{name}_adv_1pct'].median()),
                      'corr_with_m2': m2_corr, 'dsr': dsr.get('value'),
                      'report_only_after_2023_10_13': (float(net[net.index > MF3_END].mean() * 252 / STEP)
                                                       if name == 'MF3' else None)}
    for name, direction in EVENTS.items():
        ex = daily[f'{name}_excess']
        _, t = nw(ex, lags=20)
        beta, tstat = nw(daily[f'{name}_raw'], daily[['MKT', 'SMB', 'HML']], lags=20)
        years = years_positive(ex)
        active = daily[f'{name}_n'] > 0
        gate = {'nw_t>=3': bool(t[0] >= 3), 'net_annual>0': bool(ex.mean() > 0),
                'years_positive>=6': bool(sum(v > 0 for v in years.values()) >= 6),
                'ff3_alpha_t>=2': bool(tstat[0] >= 2)}
        gate['pass'] = all(gate.values())
        dsr = RS.deflated_sharpe(ex.fillna(0.0), trials=trials, sharpe_std=spread / math.sqrt(252))
        info[name] = {'framework': 'event', 'direction': direction, 'gate': gate,
                      'excess_annual': float(ex.mean() * 252), 'nw_t': t[0],
                      'excess2x_annual': float(daily[f'{name}_excess2x'].mean() * 252),
                      'excess_annual_active_days_only': float(ex[active].mean() * 252) if active.any() else None,
                      'active_share_by_year': {int(k): float(v) for k, v in active.groupby(active.index.year).mean().items()},
                      'mean_names': float(daily[f'{name}_n'][active].mean()) if active.any() else 0.0,
                      'ff3_alpha_annual': beta[0] * 252, 'ff3_alpha_t': tstat[0],
                      'betas': dict(zip(('MKT', 'SMB', 'HML'), beta[1:])), 'years': years,
                      'years_full_8': {k: v for k, v in years.items() if k < 2026},
                      'size_pct_median': float(daily[f'{name}_size_pct'].median()) if f'{name}_size_pct' in daily else None,
                      'corr_with_m2': None, 'dsr': dsr.get('value')}
    return {'family_trials': trials, 'global_trials': T.count(include_diagnostics=False),
            'annual_sharpe_spread': spread, 'signals': info,
            'passed': [n for n, v in info.items() if v['gate']['pass']]}


def m2_returns(window):
    path = ROOT / 'output/p1_m2_20261008/is/ledger_nav_0_1_cost1.csv'
    if window != 'is' or not path.exists():
        return None, None
    nav = pd.read_csv(path, index_col=0, parse_dates=True)['nav']
    return nav.pct_change().dropna(), nav


def main(window):
    data = Data()
    start, end = WINDOWS[window]
    first = data.dates.searchsorted(pd.Timestamp(start))
    last = data.dates.searchsorted(pd.Timestamp(end), side='right') - 1
    t0 = time.time()
    periods = cross_sectional(data, first, last)
    print(json.dumps({'stage': 'cross_sectional', 'periods': len(periods), 'seconds': round(time.time() - t0)}), flush=True)
    daily = event_daily(data, first, last)
    print(json.dumps({'stage': 'events', 'days': len(daily), 'seconds': round(time.time() - t0)}), flush=True)
    m2_daily, nav = m2_returns(window)
    m2_period = None
    if nav is not None:
        m2_period = nav.reindex(periods.index, method='ffill').pct_change().shift(-1)
    folder = OUT / window
    folder.mkdir(parents=True, exist_ok=True)
    periods.to_parquet(folder / 'periods.parquet')
    daily.to_parquet(folder / 'daily.parquet')
    report = evaluate(periods, daily, m2_period)
    for name in EVENTS:
        if m2_daily is not None:
            ex = daily[f'{name}_excess']
            report['signals'][name]['corr_with_m2'] = float(pd.concat([ex, m2_daily.reindex(ex.index)], axis=1)
                                                            .dropna().corr().iloc[0, 1])
    report.update(window=window, prereg=PREREG, prereg_sha256=T.sha256_file(ROOT / PREREG), panel=VERSION,
                  margin_missing_days=data.margin_missing, promotion_approved=False)
    (folder / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({'passed': report['passed'],
                      'summary': {n: {k: v.get(k) for k in ('long_only_net_annual', 'excess_annual', 'ff3_alpha_t')}
                                  for n, v in report['signals'].items()}}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default='is', choices=tuple(WINDOWS))
    main(parser.parse_args().window)
