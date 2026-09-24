"""Round a sell-only risk intent at portfolio level, then recheck its budgets.

Rounding every target down sells one lot of every security for arbitrarily
small risk excesses. Start with sales rounded down instead, and add only the
lots needed to satisfy the declared portfolio constraints. This is a bounded
greedy rounding repair, not a claim of a globally optimal integer portfolio.
"""
import numpy as np
import pandas as pd


def round_risk_sales(quantities, marks, nav, targets, rules, *, linear_limits=(),
                     covariance=None, volatility_limit=None, reserved=None):
    codes = sorted(quantities)
    if any(c not in marks for c in codes):raise ValueError('unpriced_holding_in_risk_rounding')
    price = np.array([marks[c] for c in codes], dtype=float)
    tradable = np.array([c in rules for c in codes],dtype=bool)
    held = np.array([max(0, quantities[c] - ((reserved or {}).get(c, 0) if c in rules else 0)) for c in codes], dtype=int)
    lots = np.array([int(rules[c].sell_lot) if c in rules else 1 for c in codes])
    desired = np.array([max(0., nav * targets.get(c, 0.) / marks[c]) for c in codes])
    sold = np.array([rules[c].order_quantity(max(0., held[i]-desired[i]), 'sell', holding=held[i])
                     if c in rules else 0 for i,c in enumerate(codes)])
    result = held-sold
    result[desired <= 1e-10] = 0
    result[~tradable] = held[~tradable]
    rows, caps = [], []
    for matrix, limits in linear_limits:
        aligned = matrix.reindex(columns=codes, fill_value=0.)
        rows.extend(aligned.to_numpy(dtype=float))
        caps.extend(pd.Series(limits).reindex(aligned.index).to_numpy(dtype=float))
    matrix = np.asarray(rows).reshape((-1, len(codes))) if codes else np.empty((0, 0))
    cap = np.asarray(caps)
    cov = covariance.reindex(index=codes, columns=codes).to_numpy() if covariance is not None else None
    if nav <= 0 or not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError('invalid_risk_rounding_account')
    if matrix.size and (not np.isfinite(matrix).all() or not np.isfinite(cap).all() or (cap < 0).any()):
        raise ValueError('invalid_risk_rounding_constraints')
    if volatility_limit is not None and (not np.isfinite(volatility_limit) or volatility_limit<0):
        raise ValueError('invalid_risk_rounding_volatility_limit')
    if cov is not None and (not np.isfinite(cov).all() or not np.allclose(cov,cov.T,rtol=1e-8,atol=1e-12)
                            or len(cov) and np.linalg.eigvalsh(cov).min() < -1e-10):
        raise ValueError('invalid_risk_rounding_covariance')

    def evaluate(q):
        w = q*price/nav
        excess = np.maximum(matrix@w-cap, 0.) if len(cap) else np.array([])
        vol = float(np.sqrt(max(0., w@cov@w))) if cov is not None else None
        ve = max(0., vol-volatility_limit) if vol is not None and volatility_limit is not None else 0.
        valid = (not len(excess) or excess.max() <= 1e-10) and ve <= 1e-10
        score = float((excess/np.maximum(cap, 1e-4)).sum()) + ve/max(volatility_limit or 1., 1e-4)
        return valid, score, vol, float(excess.max()) if len(excess) else 0.

    rounds = 0
    unresolved_reason = None
    while not evaluate(result)[0]:
        before = evaluate(result)[1]
        choices = []
        for i in np.flatnonzero((result>0)&tradable):
            trial = result.copy()
            minimum = rules[codes[i]].minimum_quantity('sell') if held[i]==result[i] else lots[i]
            delta = min(max(lots[i], minimum), result[i])
            trial[i] -= delta
            valid, score, _, _ = evaluate(trial)
            improvement = before-score
            # A fixed holding may make a budget impossible to satisfy. Selling
            # unrelated positions must not continue when it cannot reduce the
            # remaining violation (or would make that violation worse).
            if not valid and improvement <= 1e-12*max(1., abs(before)):
                continue
            value = float(delta*price[i])
            # Prefer the cheapest one-lot solution if it completes the repair;
            # otherwise prefer the largest budget improvement per yuan sold.
            key = (0, value, codes[i]) if valid else (1, -improvement/value, codes[i])
            choices.append((key, trial))
        if not choices:
            unresolved_reason = 'no_improving_sell'
            break  # Unknown-rule holdings remain real, fixed account exposure.
        result = min(choices, key=lambda item: item[0])[1]
        rounds += 1
    valid, _, vol, linear_excess = evaluate(result)
    shares = dict(zip(codes, map(int, result)))
    return shares, dict(method='portfolio_lot_rounding_repair', additional_lots=rounds,
        projected_gross=float(result@price/nav), projected_annualized_vol=vol,
        maximum_linear_excess=linear_excess,
        projected_constraints_passed=bool(valid and (cov is not None or volatility_limit is None)),
        unresolved_reason=unresolved_reason,
        unresolved_fixed_holdings=[c for c in codes if c not in rules],
        volatility_checked=cov is not None,
        expected_new_sell_value=float((held-result)@price),
        reserved_sell_shares=dict(reserved or {}),
        assumption='signal_marks_before_future_fees; pending_sales_are_intents_not_cash_or_completed_risk_reduction')
