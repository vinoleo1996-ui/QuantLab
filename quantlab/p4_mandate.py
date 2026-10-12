"""P4 account mandate (user decision 2026-10-12 23:20 CST).

Replaces the 600k capital of earlier research for every P4 component. Earlier
frozen results keep their own capital and are not restated. Values here are the
only source for P4 scripts; a change is a new mandate version, not an edit.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class P4Mandate:
    version: str = 'p4_v1_20261012'
    capital_cny: float = 800_000.0
    long_only: bool = True
    leverage: float = 1.0
    max_stocks: int = 50
    max_stock_weight: float = 0.04
    max_industry_weight_in_equity: float = 0.25
    allowed_assets: tuple = ('a_share_stock', 'exchange_etf', 'ipo_subscription')
    etf_universe: dict = field(default_factory=lambda: {
        'gold': '518880', 'nasdaq_qdii': '513100', 'sp500_qdii': '513500',
        'treasury_10y': '511260', 'treasury_5y': '511010'})
    excluded_assets: tuple = ('convertible_bond', 'margin', 'short', 'futures', 'options')
    equity_weight_bounds: tuple = (0.50, 0.70)
    qdii_premium_pause: float = 0.03
    min_net_cagr: float = 0.12
    target_net_cagr: float = 0.15
    max_drawdown: float = 0.25
    min_net_cagr_cost2: float = 0.10
    ipo_credit_cap: float = 0.015
    promotion_approved: bool = False

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(payload).hexdigest()


MANDATE = P4Mandate()
