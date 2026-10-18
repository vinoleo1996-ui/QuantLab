import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import p5_realizable as R


def fake_scores(days=4):
    dates = pd.bdate_range('2020-01-02', periods=days)
    rows = []
    for day in dates:
        for code, industry in [('a', 'x'), ('b', 'y')]:
            rows.append(dict(date=day, code=code, industry=industry))
    return pd.DataFrame(rows)


def test_plans_pass_gross_097(monkeypatch):
    seen = []

    def fake_schedule(rebalance, order, industries, gross=None, **kwargs):
        seen.append(gross)
        return {}

    monkeypatch.setattr(R.pd, 'read_parquet', lambda *a, **k: fake_scores())
    monkeypatch.setattr(R, 'schedule', fake_schedule)
    R.plans()
    assert seen and set(seen) == {0.97}


def test_existing_output_without_resume_exits(monkeypatch, tmp_path):
    (tmp_path / 'nav_cap520000_cost1_p0_s1.csv').write_text('nav\n1\n')
    monkeypatch.setattr(sys, 'argv', ['p5_realizable.py', '--out', str(tmp_path), '--quick'])
    with pytest.raises(SystemExit) as exc:
        R.main()
    assert exc.value.code == 1


def test_report_records_gross_quality_and_resumed_flag(monkeypatch, tmp_path):
    nav = pd.Series(np.linspace(1.0, 1.1, 5),
                    index=pd.bdate_range('2020-01-02', periods=5))
    account = [dict(date=day, nav=float(value), cash=0.0, marks={'a': 1.0},
                    quantities={'a': 100}) for day, value in nav.items()]
    summary = dict(data_issues=2, data_quality_ok=True, stale_held_max_days=3,
                   stale_held_codes={'a': 3}, delisting_liquidations=0,
                   missing_panel_count=0)

    monkeypatch.setattr(R, 'plans', lambda: {(0, 1): [
        dict(signal_date='2020-01-02', weights={'a': 0.97}, active=['a'], sell=[], sleeve=0)]})
    monkeypatch.setattr(R.p4_ledger, 'load_dataset', lambda codes: object())
    monkeypatch.setattr(R.p4_ledger, 'replay', lambda *a, **k: dict(
        account=account, summary=summary, orders=[{}, {}], data_issues=[]))
    monkeypatch.setattr(R, 'equity_stats', lambda result: pd.DataFrame(
        dict(exposure=[0.9], cash_ratio=[0.1], n=[1])))
    monkeypatch.setattr(R, 'combo', lambda series, cost: dict(
        cagr=0.13, mdd=-0.22, nav=series, premium_basis='adjusted'))
    monkeypatch.setattr(R, 'DELISTINGS', tmp_path / 'missing_delistings.parquet')
    monkeypatch.setattr(sys, 'argv', ['p5_realizable.py', '--out', str(tmp_path),
                                      '--capitals', '520000', '--costs', '1', '--quick'])
    R.main()

    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['gross'] == 0.97
    assert report['share'] == 0.65
    assert report['resumed'] is False
    assert report['promotion_approved'] is False
    cell = report['cells']['cap520000_cost1_p0_s1']
    assert cell['resumed'] is False
    assert cell['data_quality_ok'] is True
    assert cell['stale_held_max_days'] == 3
    assert cell['missing_panel_count'] == 0
    assert cell['premium_basis'] == 'adjusted'
    summary_cell = report['cells']['summary_cap520000_cost1']
    assert summary_cell['data_quality_ok'] is True
    assert summary_cell['data_quality_ok'] is not None
