#!/usr/bin/env python3
"""Code-level check of the 513-name panel/external-source difference against POOL50 membership.

Answers the question the spot-check note left open: are the names that the external daily
source omits (delisted, Beijing exchange, never-trading) ever present in the frozen POOL50
candidate cross or in an actually selected 50-name portfolio? Reads the frozen score book,
replays the frozen scheduling code path, and compares against the local panel code list and
the incremental daily source file. Read-only; writes an audit folder only.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from quantlab.alpha.portfolio import random_order, schedule

SCORES = ROOT / 'output/p4_alpha_20261012/scores.parquet'
INCREMENTAL = ROOT / 'data/tushare_incremental/daily_long.parquet'
PANEL_FULL = ROOT / 'data/smallcap_v4/20261013T023232Z/close.parquet'
ANCHOR, END = '2018-01-02', '2026-09-22'
STEP, REBALANCE, PHASES, SEEDS = 5, 20, (0, 5, 10, 15), (1, 2)
GROSS = 0.97
OUT = ROOT / 'output/pool50_membership_audit_20261018'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel_codes(path: Path) -> list[str]:
    schema = pq.ParquetFile(path).schema_arrow
    return [name for name in schema.names if re.match(r'^(sh|sz|bj)\.', str(name))]


def members(scores: pd.DataFrame) -> dict:
    """Rebuild the frozen POOL50_G97 plans and return per-path selected codes."""
    scores = scores[scores.date >= pd.Timestamp(ANCHOR)]
    by_day = {day: group for day, group in scores.groupby('date')}
    days = sorted(by_day)
    out = {}
    for phase in PHASES:
        rebalance = days[phase // STEP::REBALANCE // STEP]
        industries = lambda day: dict(zip(by_day[day].code, by_day[day].industry))
        for seed in SEEDS:
            order = lambda day, s=seed: random_order(list(by_day[day].code), s)
            plan = schedule(rebalance, order, industries, gross=GROSS)
            codes = sorted({code for day_plan in plan.values() for code in day_plan})
            out[f'POOL50_G97_p{phase}_s{seed}'] = codes
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(SCORES, columns=['date', 'code', 'industry'])
    candidate_codes = pd.Index(sorted(scores.code.astype(str).unique()))
    selected = members(scores)
    selected_union = sorted({code for codes in selected.values() for code in codes})

    panel = pd.Index(panel_codes(PANEL_FULL))
    daily = pd.read_parquet(INCREMENTAL, columns=['code', 'date'])
    source_codes = pd.Index(sorted(daily.code.astype(str).unique()))

    prefixes = pd.Index(candidate_codes.map(lambda c: c.split('.')[0])).value_counts().to_dict()
    candidate_bj = [c for c in candidate_codes if str(c).startswith('bj.')]
    selected_bj = [c for c in selected_union if str(c).startswith('bj.')]

    delist_path = sorted((ROOT / 'data/smallcap_v4').glob('*/delistings.parquet'))[-1]
    delistings = pd.read_parquet(delist_path)
    key = [c for c in delistings.columns if c.lower() in ('code', 'instrument_id', 'ts_code')][0]
    dcol = [c for c in delistings.columns if 'delist' in c.lower()][0]
    delist_map = {str(r[key]): str(r[dcol]) for r in delistings.to_dict('records')
                  if pd.notna(r[dcol]) and str(r[dcol]) not in ('', 'nan', 'None')}
    candidate_delisted = sorted(set(candidate_codes) & set(delist_map))
    selected_delisted = {c: delist_map[c] for c in selected_union if c in delist_map}

    diff = sorted(set(panel) - set(source_codes))
    diff_in_candidates = sorted(set(diff) & set(candidate_codes))
    diff_in_selected = sorted(set(diff) & set(selected_union))

    summary = {
        'inputs': {
            'scores': {'path': str(SCORES.relative_to(ROOT)), 'sha256': sha256(SCORES),
                       'rows': int(len(scores)), 'signal_days': int(scores.date.nunique())},
            'incremental_daily': {'path': str(INCREMENTAL.relative_to(ROOT)),
                                  'codes': int(len(source_codes)),
                                  'sessions': int(daily.date.nunique())},
            'panel_full': {'path': str(PANEL_FULL.relative_to(ROOT)), 'codes': int(len(panel))},
            'delistings': {'path': str(delist_path.relative_to(ROOT)), 'rows': int(len(delistings))},
        },
        'candidate_cross': {'codes': int(len(candidate_codes)), 'exchange_prefixes': prefixes,
                            'bj_codes': len(candidate_bj), 'bj_examples': candidate_bj[:10]},
        'pool50_selected': {'paths': len(selected), 'union_codes': int(len(selected_union)),
                            'bj_codes': len(selected_bj),
                            'delisted_ever_selected': selected_delisted},
        'difference_set': {'panel_minus_source': len(diff),
                           'in_candidate_cross': len(diff_in_candidates),
                           'in_pool50_selected': len(diff_in_selected),
                           'examples_selected': diff_in_selected[:20]},
        'delisted_in_candidate_cross': candidate_delisted,
        'code_path_note': ('scripts/p4_alpha.py:85-86 filters pool_on(...) with '
                           "explicit `not c.startswith('bj.')`; quantlab/smallcap/universe.py:56 "
                           'pool_on itself has no prefix exclusion.'),
        'promotion_approved': False,
    }
    pd.DataFrame([{'path': name, 'codes': ';'.join(codes)} for name, codes in selected.items()]
                 ).to_csv(OUT / 'pool50_members_by_path.csv', index=False)
    pd.DataFrame({'code': diff}).to_csv(OUT / 'panel_minus_source_codes.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
