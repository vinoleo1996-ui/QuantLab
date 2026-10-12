"""Score -> 50-name equal-weight targets with a retention buffer and an industry cap."""
from __future__ import annotations

from collections import Counter

from quantlab.smallcap.universe import priority

HOLDINGS, BUFFER, INDUSTRY_CAP, GROSS = 50, 100, 12, 0.90


def select(ordered, industries, current, holdings=HOLDINGS, buffer=BUFFER, cap=INDUSTRY_CAP):
    """``ordered``: codes best first. Held names inside the top ``buffer`` are kept first."""
    held = set(current)
    retained = [code for code in ordered[:buffer] if code in held]
    chosen, counts = [], Counter()
    for code in retained + list(ordered):
        industry = industries.get(code, 'unclassified')
        if code in chosen or counts[industry] >= cap:
            continue
        chosen.append(code)
        counts[industry] += 1
        if len(chosen) == holdings:
            break
    return chosen


def schedule(rebalance_days, order_on, industries_on, gross=GROSS, holdings=HOLDINGS):
    """``order_on(day)`` -> codes best first; ``industries_on(day)`` -> {code: industry}."""
    current, out = [], {}
    for day in rebalance_days:
        ordered = order_on(day)
        current = select(ordered, industries_on(day), current, holdings=holdings) if ordered else []
        out[day] = {code: gross / len(current) for code in current} if current else {}
    return out


def random_order(codes, seed):
    return sorted(codes, key=lambda code: priority(seed, code))
