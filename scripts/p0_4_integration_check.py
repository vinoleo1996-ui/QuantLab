#!/usr/bin/env python3
"""P0-4 integration check: delisting settlement, data quality and end holdings.

Runs one frozen G97 path over 2019 only (the year sz.002070 delisted) at the real
65% sub-account scale and writes the evidence under output/audit_20261017/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import p4_gross as G
from quantlab import p4_ledger


def main():
    plan = G.plans()['POOL50_G97_p0_s1']
    targets = p4_ledger.to_targets(plan)
    codes = sorted({c for row in targets for c in list(row['weights']) + row['sell']})
    dataset = p4_ledger.load_dataset(codes)
    result = p4_ledger.replay(targets, start='2019-01-02', end='2019-12-31',
                              dataset=dataset, capital_cny=520_000.0, gross=0.97)
    summary = result['summary']
    end_holdings = result['account'][-1]['quantities']
    report = {
        'window': ['2019-01-02', '2019-12-31'],
        'capital': 520_000.0,
        'delisted_code': 'sz.002070',
        'end_holdings_count': len(end_holdings),
        'delisted_still_held': 'sz.002070' in end_holdings,
        'delisting_liquidations': summary['delisting_liquidations'],
        'data_quality_ok': summary['data_quality_ok'],
        'stale_held_max_days': summary['stale_held_max_days'],
        'missing_panel_count': summary['missing_panel_count'],
        'data_issues_by_reason': summary['data_issues_by_reason'],
        'account_checks_passed': summary['account_checks_passed'],
        'final_equity': summary.get('final_equity'),
    }
    out = ROOT / 'output/audit_20261017/p0_4_integration.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
