"""Restricted factor formulas over date x code panels.

Only past-looking operators exist: every time-series window ends at the row's own
session and ``delay``/``delta`` take a positive lag, so a formula cannot read the
future by construction. Formulas are parsed with Python's ``ast`` against a
whitelist; anything else is rejected before evaluation.

Fields: open high low close volume amount turnover vwap ret float_cap
Time series (window 2..250): ts_mean ts_std ts_sum ts_max ts_min ts_rank delay delta ts_corr(x, y, d)
Cross section: rank zscore
Elementwise: log abs sign neg, + - * /, numeric constants
"""
from __future__ import annotations

import ast
import hashlib

import numpy as np
import pandas as pd

FIELDS = ('open', 'high', 'low', 'close', 'volume', 'amount', 'turnover', 'vwap', 'ret', 'float_cap')
TS_UNARY = ('ts_mean', 'ts_std', 'ts_sum', 'ts_max', 'ts_min', 'ts_rank', 'delay', 'delta')
TS_BINARY = ('ts_corr',)
CS = ('rank', 'zscore')
ELEMENTWISE = ('log', 'abs', 'sign', 'neg')
MAX_WINDOW, MAX_NODES, MAX_DEPTH = 250, 30, 6


class FormulaError(ValueError):
    pass


def _window(node):
    if not isinstance(node, ast.Constant) or not isinstance(node.value, int) or isinstance(node.value, bool):
        raise FormulaError('window_must_be_integer_literal')
    if not 1 <= node.value <= MAX_WINDOW:
        raise FormulaError('window_out_of_range')
    return node.value


def validate(text):
    """Parse and check a formula; returns the AST expression node."""
    try:
        tree = ast.parse(text.strip(), mode='eval')
    except SyntaxError as exc:
        raise FormulaError(f'syntax:{exc.msg}') from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES * 2:
        raise FormulaError('too_many_nodes')

    def depth(node):
        children = list(ast.iter_child_nodes(node))
        return 1 + max((depth(c) for c in children), default=0)

    if depth(tree.body) > MAX_DEPTH * 2:
        raise FormulaError('too_deep')
    for node in nodes:
        if isinstance(node, (ast.Expression, ast.Load, ast.operator, ast.unaryop)):
            continue
        if isinstance(node, ast.Name):
            if node.id not in FIELDS and node.id not in TS_UNARY + TS_BINARY + CS + ELEMENTWISE:
                raise FormulaError(f'unknown_name:{node.id}')
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise FormulaError('bad_call')
            name, args = node.func.id, node.args
            if name in TS_UNARY:
                if len(args) != 2:
                    raise FormulaError(f'{name}_needs_2_args')
                lag = _window(args[1])
                if name in ('delay', 'delta') and lag < 1:
                    raise FormulaError('non_positive_lag')
                if name not in ('delay', 'delta') and lag < 2:
                    raise FormulaError('window_too_short')
            elif name in TS_BINARY:
                if len(args) != 3:
                    raise FormulaError('ts_corr_needs_3_args')
                if _window(args[2]) < 5:
                    raise FormulaError('window_too_short')
            elif name in CS + ELEMENTWISE:
                if len(args) != 1:
                    raise FormulaError(f'{name}_needs_1_arg')
            else:
                raise FormulaError(f'unknown_function:{name}')
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                raise FormulaError('operator_not_allowed')
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ast.USub):
                raise FormulaError('operator_not_allowed')
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise FormulaError('constant_not_numeric')
        else:
            raise FormulaError(f'node_not_allowed:{type(node).__name__}')
    if isinstance(tree.body, ast.Constant):
        raise FormulaError('constant_formula')
    return tree.body


def canonical(text):
    return ast.unparse(validate(text))


def formula_id(text):
    return hashlib.sha256(canonical(text).encode()).hexdigest()[:16]


def panels_from_fields(fields):
    adj = fields['adj_close']
    volume = fields['volume'].where(fields['volume'] > 0)
    return dict(open=fields['open'] * adj / fields['close'], high=fields['high'] * adj / fields['close'],
                low=fields['low'] * adj / fields['close'], close=adj, volume=volume, amount=fields['amount'],
                turnover=fields['turnover'], vwap=(fields['amount'] / volume) * adj / fields['close'],
                ret=adj.pct_change(fill_method=None), float_cap=fields['float_mktcap'])


def _ts_rank(frame, window):
    return frame.rolling(window, min_periods=max(2, window // 2)).rank(pct=True)


def evaluate(text, panels):
    node = validate(text)

    def ev(n):
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Name):
            return panels[n.id]
        if isinstance(n, ast.UnaryOp):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            out = a / b
            return out.replace([np.inf, -np.inf], np.nan) if isinstance(out, pd.DataFrame) else out
        name, args = n.func.id, n.args
        if name in TS_UNARY:
            x, w = ev(args[0]), args[1].value
            minimum = max(2, w // 2)
            return {'ts_mean': lambda: x.rolling(w, min_periods=minimum).mean(),
                    'ts_std': lambda: x.rolling(w, min_periods=minimum).std(),
                    'ts_sum': lambda: x.rolling(w, min_periods=minimum).sum(),
                    'ts_max': lambda: x.rolling(w, min_periods=minimum).max(),
                    'ts_min': lambda: x.rolling(w, min_periods=minimum).min(),
                    'ts_rank': lambda: _ts_rank(x, w),
                    'delay': lambda: x.shift(w),
                    'delta': lambda: x - x.shift(w)}[name]()
        if name == 'ts_corr':
            x, y, w = ev(args[0]), ev(args[1]), args[2].value
            return x.rolling(w, min_periods=w // 2).corr(y).replace([np.inf, -np.inf], np.nan)
        x = ev(args[0])
        if name == 'rank':
            return x.rank(axis=1, pct=True)
        if name == 'zscore':
            return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
        if name == 'log':
            return np.log(x.where(x > 0))
        if name == 'abs':
            return x.abs()
        if name == 'sign':
            return np.sign(x)
        return -x

    out = ev(node)
    if not isinstance(out, pd.DataFrame):
        raise FormulaError('formula_has_no_field')
    return out
