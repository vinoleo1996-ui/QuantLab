"""The incremental-state helpers must stay out of the production snapshot module.

They price a book from a forward-filled open and settle it from a state file, which contradicts the
replay engine (P5_FORWARD amendment 3) and Phase 2's rule that a filled bar may not be ffilled into
an execution price. Keeping them importable from ``p5_snapshot`` risks one being reached by accident.
"""
import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

LEGACY = ['etf_books', 'load_scores', 'mark_snapshot', 'settle_orders', 'apply_pending',
          'stock_targets', 'etf_targets', 'build_rebalance_orders', 'settlement_fee']


def defined(path):
    tree = ast.parse(Path(path).read_text())
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_production_module_no_longer_defines_the_incremental_helpers():
    assert not set(LEGACY) & defined(ROOT / 'scripts/p5_snapshot.py')


def test_legacy_module_defines_them_and_they_are_not_reachable_from_production():
    assert set(LEGACY) <= defined(ROOT / 'scripts/_legacy_p5_helpers.py')
    production = importlib.import_module('p5_snapshot')
    for name in LEGACY:
        assert not hasattr(production, name), name


def test_production_surface_keeps_the_replay_engine():
    names = defined(ROOT / 'scripts/p5_snapshot.py')
    assert {'replay_book', 'frozen_schedule', 'pending_orders', 'grid_report',
            'alloc60_series', 'six_month_check', 'run_parity', 'run_prestart'} <= names
