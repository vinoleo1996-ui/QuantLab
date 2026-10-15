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


def select_hysteresis(ordered, industries, current, holdings=HOLDINGS, entry_frac=0.10, exit_frac=0.50,
                      cap=INDUSTRY_CAP, entry_count=None, exit_count=None):
    """Keep held names ranked inside the top ``exit_frac`` of the pool; fill vacancies only from the top
    ``entry_frac``, best first, under the industry cap (docs/prereg/P4_LOWTURN.md).

    ``entry_count`` and ``exit_count`` replace the fractions with absolute depths. Zero is an empty
    zone. The fractional path is unchanged when both counts are omitted.
    """
    n = len(ordered)
    keep_n = max(1, int(n * exit_frac)) if exit_count is None else max(0, int(exit_count))
    entry_n = max(1, int(n * entry_frac)) if entry_count is None else max(0, int(entry_count))
    keep_zone = set(ordered[:keep_n])
    entry_zone = ordered[:entry_n]
    chosen, counts = [], Counter()
    for code in ordered:
        if code in current and code in keep_zone and counts[industries.get(code, 'unclassified')] < cap:
            chosen.append(code)
            counts[industries.get(code, 'unclassified')] += 1
    chosen = chosen[:holdings]
    for code in entry_zone:
        if len(chosen) == holdings:
            break
        industry = industries.get(code, 'unclassified')
        if code in chosen or counts[industry] >= cap:
            continue
        chosen.append(code)
        counts[industry] += 1
    return chosen


def select_domain_book(orders, industries, current, slots, buffer_mult=2, cap=INDUSTRY_CAP):
    """Fill each domain's slot from that domain's own ranking. Names are not borrowed across domains.

    A held name is kept only while it sits inside the first ``buffer_mult * slot`` of its current domain.
    The industry cap counts across the whole book. A slot left short stays empty.
    """
    held = set(current)
    chosen, counts = [], Counter()
    for domain, slot in slots:
        ordered = list(orders.get(domain) or [])
        if slot <= 0 or not ordered:
            continue
        buffer = set(ordered[:max(slot, slot * buffer_mult)])
        retained = [code for code in ordered if code in held and code in buffer]
        picked = []
        for code in retained + ordered:
            industry = industries.get(code, 'unclassified')
            if code in chosen or code in picked or counts[industry] >= cap:
                continue
            picked.append(code)
            counts[industry] += 1
            if len(picked) == slot:
                break
        chosen.extend(picked)
    return chosen


def schedule(rebalance_days, order_on, industries_on, gross=GROSS, holdings=HOLDINGS, selector=None):
    """``order_on(day)`` -> codes best first; ``industries_on(day)`` -> {code: industry}."""
    current, out = [], {}
    selector = selector or select
    for day in rebalance_days:
        ordered = order_on(day)
        current = selector(ordered, industries_on(day), current, holdings=holdings) if ordered else []
        out[day] = {code: gross / len(current) for code in current} if current else {}
    return out


def random_order(codes, seed):
    return sorted(codes, key=lambda code: priority(seed, code))
