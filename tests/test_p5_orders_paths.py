"""A residual sweep into 511010 is a diagnostic, so it may not share a file with P5_BASE orders."""
import json

import pandas as pd

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
orders_module = importlib.import_module('p5_base_orders')

FORMAL = [{'code': 'sz.000001', 'leg': 'stock', 'side': 'buy', 'shares': 100},
          {'code': 'sh518880', 'leg': 'etf', 'side': 'buy', 'shares': 200}]
SWEEP = {'code': 'sh511010', 'leg': 'sweep', 'side': 'buy', 'shares': 300}
SUMMARY = {'signal_date': '2026-09-28', 'execution_date': '2026-09-29', 'cash_sweep': None}


def test_formal_run_writes_only_the_preregistered_files(tmp_path):
    orders_module.write_outputs(tmp_path, FORMAL, dict(SUMMARY), sweep=False)
    assert (tmp_path / 'orders.csv').exists()
    assert json.loads((tmp_path / 'summary.json').read_text())['signal_date'] == '2026-09-28'
    assert not (tmp_path / 'orders_sweep.csv').exists()


def test_sweep_is_written_to_separate_files_and_is_labelled(tmp_path):
    (tmp_path / 'orders.csv').write_bytes(b'frozen\n')
    (tmp_path / 'summary.json').write_text('{"signal_date": "2026-09-28"}')
    before = (tmp_path / 'orders.csv').read_bytes()
    orders_module.write_outputs(tmp_path, FORMAL + [SWEEP], dict(SUMMARY, cash_sweep=SWEEP), sweep=True)
    assert (tmp_path / 'orders.csv').read_bytes() == before
    assert json.loads((tmp_path / 'summary.json').read_text()) == {'signal_date': '2026-09-28'}
    frame = pd.read_csv(tmp_path / 'orders_sweep.csv')
    assert 'sh511010' in set(frame.code)
    report = json.loads((tmp_path / 'summary_sweep.json').read_text())
    assert report['diagnostic'] is True
    assert report['cash_sweep']['code'] == 'sh511010'
