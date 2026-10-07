#!/usr/bin/env python3
"""P1-B price-volume family, whole market, size- and industry-neutral (docs/prereg/P1_B_PRICE_VOLUME_NEUTRAL.md).

  python scripts/neutral_alpha.py [--window is|confirm]

Writes output/p1_b_20261008/<window>/{daily.parquet, weekly_ic.parquet, report.json}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

from quantlab import research_statistics as RS
from quantlab import research_trials as T
from quantlab.smallcap.panel import Panel
from quantlab.smallcap.universe import Industries

VERSION = '20261006T152412Z'
OUT = ROOT / 'output/p1_b_20261008'
WINDOWS = {'is': ('2018-01-02', '2026-09-22'), 'confirm': ('2015-07-01', '2017-12-29')}
SIGNALS = {'REV': -1, 'TURN': -1, 'ATURN': -1, 'IVOL': -1, 'MAX': -1, 'TREND': 1}
TREND_LAGS = (3, 5, 10, 20, 50, 100, 200)
STEP, HORIZON, WEEK = 20, 20, 5
MIN_HISTORY, MIN_ADV = 252, 5_000_000.0
TOP, SMALL = 0.20, 0.30
SLIP, COMMISSION = 0.003, 0.00025
STAMP_CUT = pd.Timestamp('2023-08-28')
EXIT_SEARCH = 20
FAMILY = 'B_price_volume'


class Data:
    def __init__(self):
        t0 = time.time()
        panel = Panel(VERSION)
        f = lambda name: panel.field(name).astype('float64')
        close, adj = f('close'), f('adj_close')
        self.dates, self.codes = close.index, close.columns
        factor = (adj / close).ffill()
        self.adj = adj.to_numpy()
        self.adj_open = (f('open') * factor).to_numpy()
        self.open = f('open').to_numpy()
        self.up = f('limit_up').to_numpy()
        self.down = f('limit_down').to_numpy()
        self.status = f('tradestatus').to_numpy()
        self.st = f('is_st').to_numpy()
        amount = f('amount')
        self.history = amount.where(amount > 0).notna().cumsum().to_numpy()
        self.adv20 = amount.rolling(20, min_periods=15).mean().to_numpy()
        turnover = f('turnover')
        self.turn20 = turnover.rolling(20, min_periods=15).mean().to_numpy()
        self.turn250 = turnover.rolling(250, min_periods=200).mean().to_numpy()
        self.cap = f('float_mktcap').rolling(20, min_periods=10).mean().to_numpy()
        self.pb = f('pb_mrq').to_numpy()
        ret = adj.pct_change(fill_method=None)
        self.ret = ret.to_numpy()
        self.max20 = ret.rolling(20, min_periods=15).max().to_numpy()
        self.ma = {lag: (adj / adj.rolling(lag, min_periods=lag).mean() - 1).to_numpy() for lag in TREND_LAGS}
        self.bj = np.array([c.startswith('bj.') for c in self.codes])
        self.industries = Industries(panel.table('industry'))
        self.mkt, self.smb = self.daily_factors()
        print(json.dumps({'stage': 'loaded', 'seconds': round(time.time() - t0)}), flush=True)

    def daily_factors(self):
        """Equal-weight market and size factor returns over tradable non-ST names (cap known the day before)."""
        live = (self.status == 1) & (self.st != 1) & ~self.bj[None, :] & np.isfinite(self.ret)
        mkt = np.full(len(self.dates), np.nan)
        smb = np.full(len(self.dates), np.nan)
        for t in range(1, len(self.dates)):
            ok = live[t] & np.isfinite(self.cap[t - 1])
            if ok.sum() < 100:
                continue
            r, c = self.ret[t, ok], self.cap[t - 1, ok]
            lo, hi = np.quantile(c, [0.3, 0.7])
            mkt[t] = r.mean()
            smb[t] = r[c <= lo].mean() - r[c >= hi].mean()
        return mkt, smb

    def eligible(self, i):
        return ((self.st[i] != 1) & ~self.bj & (self.history[i] >= MIN_HISTORY) & (self.status[i] == 1)
                & (self.adv20[i] >= MIN_ADV) & np.isfinite(self.cap[i]))

    def ivol(self, i, mask):
        rows = slice(i - 19, i + 1)
        X = np.column_stack([np.ones(20), self.mkt[rows], self.smb[rows]])
        Y = self.ret[rows][:, mask]
        out = np.full(mask.sum(), np.nan)
        if not np.isfinite(X).all():
            return out
        full = np.isfinite(Y).all(axis=0)
        if full.any():
            beta, *_ = np.linalg.lstsq(X, Y[:, full], rcond=None)
            out[full] = (Y[:, full] - X @ beta).std(axis=0, ddof=3)
        return out

    def raw_signals(self, i, mask):
        with np.errstate(divide='ignore', invalid='ignore'):
            out = {'REV': self.adj[i - 1, mask] / self.adj[i - 21, mask] - 1,
                   'TURN': self.turn20[i, mask],
                   'ATURN': self.turn20[i, mask] / self.turn250[i, mask],
                   'IVOL': self.ivol(i, mask),
                   'MAX': self.max20[i, mask]}
        out['TREND'] = np.nanmean(np.column_stack([zscore(self.ma[lag][i, mask]) for lag in TREND_LAGS]), axis=1)
        return out

    def forward(self, i, mask, horizon):
        """Entry at the next open, exit at open i+1+horizon (searching forward if not sellable)."""
        n = len(self.dates)
        e = i + 1
        cols = np.where(mask)[0]
        entry = self.adj_open[e, cols] if e < n else np.full(len(cols), np.nan)
        blocked = (~np.isfinite(entry) | (self.status[e, cols] != 1)
                   | (np.isfinite(self.up[e, cols]) & (self.open[e, cols] >= self.up[e, cols] - 0.005))) if e < n \
            else np.ones(len(cols), bool)
        exit_ = np.full(len(cols), np.nan)
        todo = np.ones(len(cols), bool)
        for x in range(e + horizon, min(e + horizon + EXIT_SEARCH + 1, n)):
            price = self.adj_open[x, cols]
            ok = todo & np.isfinite(price) & (self.status[x, cols] == 1) & ~(
                np.isfinite(self.down[x, cols]) & (self.open[x, cols] <= self.down[x, cols] + 0.005))
            exit_[ok] = price[ok]
            todo &= ~ok
        if todo.any():
            last = min(e + horizon + EXIT_SEARCH, n - 1)
            if last < e + horizon:
                return np.full(len(cols), np.nan), blocked
            history = pd.DataFrame(self.adj[e:last + 1, cols[todo]]).ffill().iloc[-1].to_numpy()
            exit_[todo] = history * 0.5
        return exit_ / entry - 1, blocked


def zscore(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    out = np.full(len(x), np.nan)
    if ok.sum() < 30:
        return out
    lo, hi = np.quantile(x[ok], [0.01, 0.99])
    y = np.clip(x[ok], lo, hi)
    sd = y.std()
    out[ok] = (y - y.mean()) / sd if sd > 0 else 0.0
    return out


def neutralize(signal, logcap, industry, small_dummy=None):
    z = zscore(signal)
    ok = np.isfinite(z) & np.isfinite(logcap)
    out = np.full(len(z), np.nan)
    if ok.sum() < 100:
        return out
    dummies = pd.get_dummies(pd.Series(industry[ok])).to_numpy(dtype=float)
    columns = [logcap[ok][:, None], dummies]
    if small_dummy is not None:
        columns.append(small_dummy[ok][:, None].astype(float))
    X = np.hstack(columns)
    beta, *_ = np.linalg.lstsq(X, z[ok], rcond=None)
    out[ok] = z[ok] - X @ beta
    return out


def rank_ic(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return np.nan
    return float(pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank()))


def sector_letters(industries, day, codes):
    mapping = industries.on(day)
    return np.array([str(mapping.get(c, 'unclassified'))[:1] if mapping.get(c, 'unclassified') != 'unclassified'
                     else 'U' for c in codes])


def neutral_set(data, i, mask, robust=False):
    codes = data.codes[mask]
    logcap = np.log(data.cap[i, mask])
    industry = sector_letters(data.industries, data.dates[i], codes)
    raw = data.raw_signals(i, mask)
    small = logcap <= np.quantile(logcap, SMALL)
    out = {}
    for name, direction in SIGNALS.items():
        if robust:
            values = np.full(len(codes), np.nan)
            keep = ~small
            values[keep] = neutralize(raw[name][keep], logcap[keep], industry[keep])
        else:
            values = neutralize(raw[name], logcap, industry, small)
        out[name] = direction * values
    return out, logcap, small


def stamp(day):
    return 0.001 if day < STAMP_CUT else 0.0005


def run(window):
    data = Data()
    start, end = WINDOWS[window]
    dates = data.dates
    first = dates.searchsorted(pd.Timestamp(start))
    last = dates.searchsorted(pd.Timestamp(end), side='right') - 1
    monthly = list(range(first, last + 1, STEP))
    rows, previous_top = [], {name: None for name in SIGNALS}
    t0 = time.time()
    for k, i in enumerate(monthly):
        mask = data.eligible(i)
        fwd, blocked = data.forward(i, mask, HORIZON)
        fwd = np.where(blocked, np.nan, fwd)
        signals, logcap, small = neutral_set(data, i, mask)
        robust, _, _ = neutral_set(data, i, mask, robust=True)
        codes = data.codes[mask]
        valid = np.isfinite(fwd)
        universe_mean = float(np.nanmean(fwd)) if valid.any() else np.nan
        day = dates[i]
        row = {'date': day, 'names': int(valid.sum()), 'universe_fwd': universe_mean}
        capr = pd.Series(-logcap).rank(method='first').to_numpy()
        pb = data.pb[i, mask]
        lo_cap, hi_cap = np.quantile(logcap, [0.3, 0.7])
        row['SMB_fwd'] = float(np.nanmean(fwd[logcap <= lo_cap]) - np.nanmean(fwd[logcap >= hi_cap]))
        pos = np.isfinite(pb) & (pb > 0)
        lo_pb, hi_pb = np.quantile(pb[pos], [0.3, 0.7])
        row['HML_fwd'] = float(np.nanmean(fwd[pos & (pb <= lo_pb)]) - np.nanmean(fwd[pos & (pb >= hi_pb)]))
        complete = np.all([np.isfinite(signals[s]) for s in SIGNALS], axis=0)
        matrix = np.column_stack([signals[s] for s in SIGNALS])
        if complete.sum() > 200:
            corr = np.corrcoef(matrix[complete].T)
            for a, sa in enumerate(SIGNALS):
                for b, sb in enumerate(SIGNALS):
                    if a < b:
                        row[f'corr_{sa}_{sb}'] = float(corr[a, b])
        for a, name in enumerate(SIGNALS):
            s = signals[name]
            row[f'{name}_ic'] = rank_ic(s, fwd)
            row[f'{name}_ic_robust'] = rank_ic(robust[name], fwd)
            for tier, (lo, hi) in {'t1_300': (1, 300), 't301_800': (301, 800), 't801_1800': (801, 1800),
                                   't1801p': (1801, 10 ** 9)}.items():
                part = (capr >= lo) & (capr <= hi)
                row[f'{name}_ic_{tier}'] = rank_ic(np.where(part, s, np.nan), fwd)
            if complete.sum() > 200:
                others = np.delete(matrix[complete], a, axis=1)
                X = np.column_stack([np.ones(complete.sum()), others])
                beta, *_ = np.linalg.lstsq(X, matrix[complete, a], rcond=None)
                resid = np.full(len(s), np.nan)
                resid[complete] = matrix[complete, a] - X @ beta
                row[f'{name}_ic_incr'] = rank_ic(resid, fwd)
            ok = np.isfinite(s) & valid
            n_top = int(math.floor(TOP * ok.sum()))
            order = np.argsort(-np.where(ok, s, -np.inf))[:n_top]
            top = set(codes[order])
            gross = float(np.mean(fwd[order])) - universe_mean
            prev = previous_top[name]
            tau = 1.0 if prev is None else 1 - len(top & prev) / max(len(top), 1)
            cost = tau * (2 * (SLIP + COMMISSION) + stamp(day))
            previous_top[name] = top
            row[f'{name}_gross'] = gross
            row[f'{name}_turnover'] = tau
            row[f'{name}_net'] = gross - cost
            row[f'{name}_net2x'] = gross - 2 * cost
            row[f'{name}_top_adv_1pct'] = float(np.median(data.adv20[i, mask][order]) * 0.01)
        rows.append(row)
        if k % 10 == 0:
            print(json.dumps({'stage': 'monthly', 'done': k, 'total': len(monthly), 'seconds': round(time.time() - t0)}),
                  flush=True)
    daily = pd.DataFrame(rows).set_index('date')
    weekly = []
    for i in range(first, last + 1, WEEK):
        mask = data.eligible(i)
        fwd, blocked = data.forward(i, mask, WEEK)
        fwd = np.where(blocked, np.nan, fwd)
        signals, _, _ = neutral_set(data, i, mask)
        weekly.append({'date': dates[i], **{name: rank_ic(signals[name], fwd) for name in SIGNALS}})
    weekly = pd.DataFrame(weekly).set_index('date')
    folder = OUT / window
    folder.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(folder / 'daily.parquet')
    weekly.to_parquet(folder / 'weekly_ic.parquet')
    report = evaluate(daily, weekly)
    report.update(window=window, prereg='docs/prereg/P1_B_PRICE_VOLUME_NEUTRAL.md',
                  prereg_sha256=T.sha256_file(ROOT / 'docs/prereg/P1_B_PRICE_VOLUME_NEUTRAL.md'),
                  panel=VERSION, promotion_approved=False)
    (folder / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: report[k] for k in ('gates', 'effective_independent_signals')}, indent=1, default=float))


def nw_t(series, lags=3):
    s = pd.Series(series).dropna()
    if len(s) < 12:
        return float('nan'), float('nan')
    beta, t = RS.newey_west(s.to_numpy(), lags=lags)
    return float(beta[0]), float(t[0])


def evaluate(daily, weekly):
    periods = 252 / STEP
    out = {'periods': int(len(daily)), 'family_trials': T.count(FAMILY), 'global_trials': T.count(), 'signals': {}}
    nets = pd.DataFrame({name: daily[f'{name}_net'] for name in SIGNALS})
    per_obs_sharpe = nets.mean() / nets.std(ddof=1)
    family_std = float(per_obs_sharpe.std(ddof=1))
    gates = {}
    for name in SIGNALS:
        ic_mean, ic_t = nw_t(daily[f'{name}_ic'])
        net = daily[f'{name}_net']
        years = net.groupby(net.index.year).sum()
        positive_years = int((years > 0).sum())
        gate = {'ic_t>=3': bool(ic_t >= 3), 'net_mean>0': bool(net.mean() > 0),
                'years_positive>=2/3': bool(positive_years >= math.ceil(2 * len(years) / 3))}
        gate['pass'] = all(gate.values())
        rob_mean, rob_t = nw_t(daily[f'{name}_ic_robust'])
        incr_mean, incr_t = nw_t(daily[f'{name}_ic_incr']) if f'{name}_ic_incr' in daily else (np.nan, np.nan)
        wk_mean, wk_t = nw_t(weekly[name], lags=4)
        reg = pd.DataFrame({'y': net, 'smb': daily.SMB_fwd, 'hml': daily.HML_fwd}).dropna()
        beta, tstat = RS.newey_west(reg.y.to_numpy(), reg[['smb', 'hml']].to_numpy(), lags=3)
        dsr = RS.deflated_sharpe(net.fillna(0.0), trials=max(T.count(FAMILY), 2), sharpe_std=family_std)
        info = {
            'ic_mean': ic_mean, 'ic_nw_t': ic_t,
            'net_excess_per_period': float(net.mean()), 'net_excess_annual': float(net.mean() * periods),
            'gross_excess_annual': float(daily[f'{name}_gross'].mean() * periods),
            'net2x_excess_annual': float(daily[f'{name}_net2x'].mean() * periods),
            'cost_sensitive': bool(daily[f'{name}_net2x'].mean() <= 0),
            'one_way_turnover_annual': float(daily[f'{name}_turnover'].iloc[1:].mean() * periods),
            'years': {int(k): float(v) for k, v in years.items()}, 'positive_years': positive_years,
            'robust_ex_small30_ic_mean': rob_mean, 'robust_ex_small30_ic_t': rob_t,
            'shell_or_size': bool(not (rob_t >= 2)),
            'incremental_ic_mean': incr_mean, 'incremental_ic_t': incr_t,
            'weekly_ic_mean': wk_mean, 'weekly_ic_t': wk_t,
            'tiers': {tier: nw_t(daily[f'{name}_ic_{tier}']) for tier in ('t1_300', 't301_800', 't801_1800', 't1801p')},
            'smb_hml_alpha_per_period': float(beta[0]), 'smb_hml_alpha_t': float(tstat[0]),
            'smb_beta': float(beta[1]), 'hml_beta': float(beta[2]),
            'dsr': dsr.get('value'), 'median_top_adv_1pct_cny': float(daily[f'{name}_top_adv_1pct'].median()),
            'gate': gate,
        }
        info['effective_independent'] = bool(gate['pass'] and not info['shell_or_size'] and incr_t >= 3)
        out['signals'][name] = info
        gates[name] = gate['pass']
    out['gates'] = gates
    out['effective_independent_signals'] = [n for n, v in out['signals'].items() if v['effective_independent']]
    out['pbo'] = __import__('smallcap_audit').pbo_cscv(nets.dropna(), blocks=16)
    corr_cols = [c for c in daily.columns if c.startswith('corr_')]
    out['mean_corr'] = {c[5:]: float(daily[c].mean()) for c in corr_cols}
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default='is', choices=tuple(WINDOWS))
    run(parser.parse_args().window)
