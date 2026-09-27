"""Selection-value audit: does the expert score add anything at portfolio level?

Read-only. Consumes the already-published Q2-a archive
  output/continuity_review_20260927/attribution/scored_outcomes.parquet
(74 archived decision dates, 292532 scored-eligible observations, frozen
ledger 6157c64, data 5403b090). Does not run the account engine, does not
touch frozen source, does not read or write any account artifact.

Question this is allowed to answer
---------------------------------
Q2-a showed the top score quintile does not beat the middle quintile
(Q5-Q3 < 0 for all six models). It never compared, at portfolio level,
score-based selection against score-blind selection of the same size from
the same pool. If a score-blind equal-weight draw of N names matches or beats
the N names the strategy actually selected, the scoring overlay is destroying
value and "add more experts" is the wrong direction.

Pre-registered decision rule
----------------------------
If score-blind equal-weight N=20 draws from the same eligible pool beat the
actually-selected 20 on the date-weighted mean in a majority of the three
complete phases at both 20d and 60d, the direction changes: stop adding
complementary experts and stop tuning the router, and re-open the question of
whether the scoring/risk overlay should exist at all.
If the selected set is at or above the random-draw median, the score is
carrying real selection value and expert work stays the direction.

Not claimed
-----------
Forward feature-price means are not account returns. Windows overlap. The
74 dates are the strategy's own historical decision dates, already inspected
many times, not an independent out-of-sample sample. No costs, no cash drag,
no P2 sizing, no execution. This ranks selection value only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'output/continuity_review_20260927/attribution/scored_outcomes.parquet'
OUT = ROOT / 'output/selection_value_20260927'

N_SELECT = 20
DRAWS = 200
SEED = 20260927

PHASES = (('2020-21', '2020-01-01', '2021-12-31'),
          ('2022-23', '2022-01-01', '2023-12-31'),
          ('2024-25', '2024-01-01', '2025-12-31'),
          ('2026H1', '2026-01-01', '2026-12-31'))


def phase_of(dates: pd.Series) -> pd.Series:
    out = pd.Series(index=dates.index, dtype=object)
    for name, lo, hi in PHASES:
        m = (dates >= lo) & (dates <= hi)
        out[m] = name
    return out


def draw_stats(values: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Equal-weight means of N random draws, without replacement."""
    n = len(values)
    if n < N_SELECT:
        return np.array([])
    idx = rng.integers(0, n, size=(n_draws, N_SELECT))
    # reject duplicates within a draw by resampling collisions
    for _ in range(8):
        s = np.sort(idx, axis=1)
        dup = (np.diff(s, axis=1) == 0).any(axis=1)
        if not dup.any():
            break
        idx[dup] = rng.integers(0, n, size=(dup.sum(), N_SELECT))
    return values[idx].mean(axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(SRC)
    frame['date'] = frame['date'].astype(str)
    frame['phase'] = phase_of(frame['date'])
    rng = np.random.default_rng(SEED)

    rows: list[dict] = []
    for date, block in frame.groupby('date', sort=True):
        phase = block['phase'].iloc[0]
        for horizon in (20, 60):
            col, cov = f'return_{horizon}', f'coverage_{horizon}'
            ok = block[block[cov] == 'observed']
            if len(ok) < N_SELECT:
                continue
            ret = ok[col].to_numpy(dtype=float)
            # variant A: uniform over the whole scored-eligible pool
            a = draw_stats(ret, DRAWS, rng)
            # variant B: uniform inside the recorded union + financial pass set,
            # so only the score ranking differs, not the candidate filter
            same = ok[ok['in_expert_union'] & (ok['quality_pass'] == True)]
            b = (draw_stats(same[col].to_numpy(dtype=float), DRAWS, rng)
                 if len(same) >= N_SELECT else np.array([]))
            sel = ok[ok['selected']]
            top = ok.nlargest(N_SELECT, 'score')
            rows.append(dict(
                date=date, phase=phase, horizon=horizon,
                pool_n=len(ok), same_set_n=len(same), selected_n=len(sel),
                selected_mean=float(sel[col].mean()) if len(sel) else np.nan,
                top20_by_score=float(top[col].mean()),
                pool_mean=float(ret.mean()),
                drawA_mean=float(a.mean()), drawA_p50=float(np.median(a)),
                drawA_p10=float(np.percentile(a, 10)), drawA_p90=float(np.percentile(a, 90)),
                selected_pctile=float((a < sel[col].mean()).mean() * 100) if len(sel) else np.nan,
                top20_pctile=float((a < top[col].mean()).mean() * 100),
                drawB_mean=float(b.mean()) if len(b) else np.nan,
                selected_minus_drawA=float(sel[col].mean() - a.mean()) if len(sel) else np.nan,
                top20_minus_drawA=float(top[col].mean() - a.mean()),
                selected_minus_drawB=float(sel[col].mean() - b.mean()) if len(sel) and len(b) else np.nan,
                drawB_n=int(len(b)),
            ))

    daily = pd.DataFrame(rows)
    daily.to_parquet(OUT / 'selection_value_daily.parquet', index=False)

    def agg(sub: pd.DataFrame) -> dict:
        return dict(
            dates=int(sub['date'].nunique()),
            selected_vs_drawA_pp=float((sub['selected_minus_drawA']).mean() * 100),
            top20_vs_drawA_pp=float((sub['top20_minus_drawA']).mean() * 100),
            selected_vs_drawB_pp=float(sub['selected_minus_drawB'].mean() * 100),
            selected_pctile_median=float(sub['selected_pctile'].median()),
            top20_pctile_median=float(sub['top20_pctile'].median()),
            pool_n_median=float(sub['pool_n'].median()),
            same_set_n_median=float(sub['same_set_n'].median()),
        )

    by_phase = []
    for horizon in (20, 60):
        for phase, g in daily[daily['horizon'] == horizon].groupby('phase', sort=False):
            by_phase.append(dict(horizon=horizon, phase=phase, **agg(g)))
    by_phase = pd.DataFrame(by_phase)

    overall = []
    for horizon in (20, 60):
        g = daily[(daily['horizon'] == horizon) & (daily['phase'] != '2026H1')]
        overall.append(dict(horizon=horizon, phase='complete_3', **agg(g)))
    overall = pd.DataFrame(overall)

    # breadth: does widening the same scored pool fix the ranking? scale-invariant?
    breadth = []
    for horizon in (20, 60):
        col, cov = f'return_{horizon}', f'coverage_{horizon}'
        for n in (10, 20, 40, 60, 100, 150, 200):
            per = []
            for date, g in frame[frame['phase'] != '2026H1'].groupby('date', sort=True):
                ok = g[g[cov] == 'observed']
                if len(ok) < n:
                    continue
                per.append((date, float(ok.nlargest(n, 'score')[col].mean() - ok[col].mean())))
            sub = pd.DataFrame(per, columns=['date', 'x'])
            sub['phase'] = phase_of(sub['date'])
            breadth.append(dict(
                horizon=horizon, n=n, dates=int(len(sub)),
                all_complete_phases_pp=float(sub['x'].mean() * 100),
                **{p: float(sub[sub['phase'] == p]['x'].mean() * 100)
                   for p in ('2020-21', '2022-23', '2024-25')}))

    # direction gate: complete phases, both horizons, score-blind beats selected.
    # by_phase already holds one row per phase x horizon, so cells == len(complete).
    complete = by_phase[by_phase['phase'] != '2026H1']
    cells = int(len(complete))
    lose = int((complete['selected_vs_drawA_pp'] < 0).sum())
    below_median = int((complete['selected_pctile_median'] < 50).sum())

    # paired inference on the date level, so the verdict is not eyeballed cells
    paired = []
    for horizon in (20, 60):
        g = daily[(daily['horizon'] == horizon) & (daily['phase'] != '2026H1')]
        d = g['selected_minus_drawA'].to_numpy(dtype=float)
        boot = np.random.default_rng(SEED + horizon)
        idx = boot.integers(0, len(d), size=(4000, len(d)))
        means = d[idx].mean(axis=1)
        paired.append(dict(
            horizon=horizon, dates=int(len(d)),
            mean_diff_pp=float(d.mean() * 100),
            boot_ci95_pp=[float(np.percentile(means, 2.5) * 100),
                          float(np.percentile(means, 97.5) * 100)],
            share_of_dates_selected_worse=float((d < 0).mean()),
            share_of_dates_above_random_median=float((g['selected_pctile'] > 50).mean()),
        ))
    summary = dict(
        created_at=pd.Timestamp.now('UTC').isoformat(),
        status='conditional_offline_diagnostic',
        promotion_approved=False,
        account_replayed=False,
        source=dict(path=str(SRC.relative_to(ROOT)), rows=int(len(frame)),
                    dates=int(frame['date'].nunique())),
        method=dict(n_select=N_SELECT, draws_per_date=DRAWS, seed=SEED,
                    variant_A='uniform over scored-eligible pool',
                    variant_B='uniform inside recorded union AND financial pass',
                    weighting='equal weight, forward feature-price mean',
                    costs_included=False, cash_drag_included=False,
                    p2_sizing_included=False, execution_included=False),
        overall=overall.to_dict('records'),
        by_phase=by_phase.to_dict('records'),
        paired_date_level=paired,
        direction_gate=dict(
            rule='score-blind draw beats the selected set in a majority of '
                 'complete-phase x horizon cells (cells == len of complete rows)',
            cells=cells, cells_where_score_blind_wins=lose,
            cells_where_selected_below_random_median=below_median,
            triggered=bool(lose > cells / 2),
            interpretation='triggered means the scoring overlay is not earning '
                           'its keep at portfolio level, so adding more experts '
                           'is the wrong direction; the effect size still has to '
                           'be weighed against the 12pp gap to target'),
        breadth_test=breadth,
        limits=[
            'forward feature-price means are not account returns',
            '20/60d windows overlap across the 74 dates',
            'the 74 dates are the strategy own inspected decision dates, not independent OOS',
            'no costs, no cash drag, no P2 sizing, no execution, no dividends',
            '2026H1 has only 5/3 mature dates and is supplementary',
            'one of 74 dates lacks enough 20d-mature names to draw and is dropped, so 73 dates appear in the daily file and 68 in the complete-phase paired test',
            'random draws are a same-distribution reference, not a tradeable counterfactual',
        ],
    )
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    pd.set_option('display.width', 240)
    print('=== complete phases, date-weighted mean forward return difference (pp) ===')
    print(by_phase[by_phase['phase'] != '2026H1'].to_string(index=False))
    print('\n=== 2026H1 (supplementary, too few mature dates) ===')
    print(by_phase[by_phase['phase'] == '2026H1'].to_string(index=False))
    print('\n=== overall complete_3 ===')
    print(overall.to_string(index=False))
    print('\n=== paired date-level (complete 3 phases) ===')
    for p in summary['paired_date_level']:
        print(f"  {p['horizon']:>2}d  n={p['dates']}  mean {p['mean_diff_pp']:+.3f}pp  "
              f"CI95 [{p['boot_ci95_pp'][0]:+.3f}, {p['boot_ci95_pp'][1]:+.3f}]pp  "
              f"selected-worse on {p['share_of_dates_selected_worse']:.1%} of dates")
    print('\n=== direction gate ===')
    print(json.dumps(summary['direction_gate'], ensure_ascii=False, indent=1))
    print(f"\nwrote {OUT/'summary.json'}")


if __name__ == '__main__':
    main()
