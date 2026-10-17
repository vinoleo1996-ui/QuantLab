#!/usr/bin/env python3
"""P5_G0: restore the fixed 3:3:2 sleeve on the existing P4 stock books. Diagnostic, no stock replay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import os
ROOT = Path(os.environ.get('QUANTLAB_ROOT', '/Users/agiuser/Desktop/QuantLab'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

import p4_alloc as R
from quantlab import p4_alloc as A
from quantlab import research_trials as T
from quantlab.research_statistics import deflated_sharpe

START, END = '2018-01-02', '2026-09-22'
ALPHA = ROOT / 'output/p4_alpha_20261012/ledger'
GROSS = ROOT / 'output/p4_gross_20261013/ledger'
PREREG = 'docs/prereg/P5_G0_ALLOC.md'
OUT = ROOT / 'output/p5_g0_20261017'


def etf_g0(sessions):
    bars = pd.read_parquet(ROOT / 'data/smallcap_assets/etf_bars.parquet')
    wide = bars.pivot(index='date', columns='code', values='close_hfq').sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide[['sh518880', 'sh511260', 'sh513100']].rename(
        columns={'sh518880': 'gold', 'sh511260': 'treasury', 'sh513100': 'nasdaq'})
    wide = wide.reindex(wide.index.union(sessions)).ffill().reindex(sessions)
    return wide.pct_change(fill_method=None).fillna(0.0)


ORIG_TARGET = A.target_weights


def g0_target(etf_returns_upto, equity_share):
    sleeve = 1.0 - equity_share
    return np.array([equity_share, 0.375 * sleeve, 0.375 * sleeve, 0.25 * sleeve])


def run(folder, prefix, share, line, cost, sleeve, rebalance='quarter'):
    files = sorted(folder.glob(f'nav_{prefix}p*_cost{cost}.csv'))
    out, daily = [], []
    target_fn = g0_target if sleeve == 'g0' else ORIG_TARGET
    overflow = {'gold': 0.5, 'treasury': 0.5} if sleeve == 'g0' else None
    for path in files:
        equity = pd.read_csv(path, index_col=0, parse_dates=True)['nav'].loc[START:END]
        if sleeve == 'g0':
            etf = etf_g0(equity.index)
        else:
            etf = R.etf_returns(equity.index)
        premium = R.qdii_premium_frame(equity.index, basis='legacy')
        A.DISASTER_DD, A.RECOVER_DD, A.DISASTER_EQUITY = line, 0.08, 0.50
        nav, trace = A.allocate(equity, etf, share, cost_mult=cost, premium=premium,
                                rebalance=rebalance, overflow=overflow, target_fn=target_fn)
        s = A.stats(nav)
        daily.append(nav.pct_change(fill_method=None).fillna(0.0))
        out.append(dict(path=path.stem, cagr=s['cagr'], mdd=s['mdd'],
                        triggers=trace['disaster_triggers']))
    frame = pd.DataFrame(out)
    return dict(n=len(frame), cagr=float(frame.cagr.median()), mdd=float(frame.mdd.median()),
                triggers=frame.triggers.tolist(), _daily=daily)


def cell_name(tag, share, line, cost, rebalance):
    line_tag = 'noline' if line >= 1.0 else str(int(round(line * 100)))
    suffix = 'q' if rebalance == 'quarter' else 'm'
    return f'{tag}_{int(share * 100)}_{line_tag}_{suffix}_c{cost:g}'


def decide(cells, rebalance):
    """Apply the P5_G0 prereg decision rule; never choose a cell from another frequency."""
    suffix = 'q' if rebalance == 'quarter' else 'm'
    base = cells.get(f'G97_65_23_{suffix}_c1')
    challenger = cells.get(f'G97_70_noline_{suffix}_c1')
    if base is None:
        return dict(selected=None, reason='base_cell_missing')
    if challenger is not None and challenger['mdd'] >= -0.25 and challenger['cagr'] > base['cagr']:
        return dict(selected=f'G97_70_noline_{suffix}',
                    reason='prereg: 70/no-line has MDD >= -25% and higher CAGR')
    return dict(selected=f'G97_65_23_{suffix}',
                reason='prereg: 70/no-line does not beat 65/23 on both CAGR and MDD')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebalance', choices=('month', 'quarter'), default='quarter')
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--monthly-diagnostics', action='store_true',
                        help='also run the G97_70 no-line monthly cell as a diagnostic')
    args = parser.parse_args()
    cells = {}
    # Validation: reproduce ALLOC60 with the existing inverse-vol rule and 18% line (monthly P4_ALLOC).
    baseline = run(ALPHA, 'POOL50_', 0.60, 0.18, 1, 'g1', rebalance='month')
    cells['ALLOC60_invvol_60_18_m_c1'] = dict(baseline, share=0.60, line=0.18, cost=1,
                                              sleeve='g1', rebalance='month')
    # G0 sleeve.
    for share in (0.60, 0.70):
        for line in (0.23, 1.0):
            for cost in (1, 2):
                name = cell_name('G0', share, line, cost, args.rebalance)
                cells[name] = dict(run(ALPHA, 'POOL50_', share, line, cost, 'g0', args.rebalance),
                                   share=share, line=line, cost=cost, sleeve='g0', rebalance=args.rebalance)
    # G97 with the G0 sleeve.
    for share in (0.65, 0.70):
        for line in (0.23, 1.0):
            for cost in (1, 2):
                name = cell_name('G97', share, line, cost, args.rebalance)
                cells[name] = dict(run(GROSS, 'POOL50_G97_', share, line, cost, 'g0', args.rebalance),
                                   share=share, line=line, cost=cost, sleeve='g0', rebalance=args.rebalance)
    if args.monthly_diagnostics:
        cells['G97_70_noline_m_c1'] = dict(run(GROSS, 'POOL50_G97_', 0.70, 1.0, 1, 'g0', 'month'),
                                           share=0.70, line=1.0, cost=1, sleeve='g0', rebalance='month')
    for name, row in sorted(cells.items()):
        if row.get('rebalance') != args.rebalance:
            continue
        print(f'{name:28s} n={row["n"]:2d} cagr={row["cagr"]*100:6.2f}% +ipo={row["cagr"]*100+1.5:6.2f}% '
              f'mdd={row["mdd"]*100:6.2f}% triggers={row["triggers"]}')
    decision = decide(cells, args.rebalance)
    selected = cells.get(decision['selected'] + '_c1') if decision['selected'] else None
    selected_c2 = cells.get(decision['selected'] + '_c2') if decision['selected'] else None
    grid = [row for row in cells.values() if row.get('rebalance') == args.rebalance]
    dsr = {'n_grid_cells': len(grid),
           'n_family_trials': T.count('P5_g0', include_diagnostics=False),
           'n_global_trials': T.count(include_diagnostics=False),
           'n_families': len(T.families())}
    if selected is not None and selected.get('_daily'):
        paths = selected['_daily']
        matrix = np.vstack([series.reindex(paths[0].index).to_numpy() for series in paths])
        median_return = np.median(matrix, axis=0)
        sharpes = [float(series.mean() / series.std(ddof=1)) for series in paths
                   if series.std(ddof=1) > 0]
        dispersion = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else None
        dsr['dsr_family'] = deflated_sharpe(median_return, trials=max(len(grid), 2),
                                           sharpe_std=dispersion)
        dsr['dsr_global'] = deflated_sharpe(median_return,
                                            trials=max(T.count(include_diagnostics=False), 2))
    for row in cells.values():
        row.pop('_daily', None)
    gates = {}
    if selected is not None:
        gates = {'cost1_cagr_plus_ipo_ge_12_5': selected['cagr'] + 0.015 >= 0.125,
                 'mdd_ge_-25': selected['mdd'] >= -0.25,
                 'cost2_cagr_plus_ipo_ge_10_5': selected_c2 is not None and selected_c2['cagr'] + 0.015 >= 0.105}
    report = {'trial': 'P5_g0', 'prereg': PREREG, 'prereg_sha256': T.sha256_file(ROOT / PREREG),
              'window': [START, END], 'rebalance': args.rebalance, 'premium_basis': 'legacy',
              'n_trials': T.count(), 'n_formal_trials': T.count(include_diagnostics=False),
              'baseline_reproduced': cells['ALLOC60_invvol_60_18_m_c1'],
              'cells': cells, 'decision': decision, 'dsr': dsr, 'gates': gates,
              'passed': bool(gates) and all(gates.values()), 'promotion_approved': False}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'report_v2_{args.rebalance}.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({'report': str(path), 'decision': decision, 'gates': gates, 'passed': report['passed']},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
