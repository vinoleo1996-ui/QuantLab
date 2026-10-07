#!/usr/bin/env python3
"""One-time backfill of the small-cap family into docs/trial_ledger.csv (refuses to run twice)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from quantlab import research_trials as T

AUDIT = 'output/smallcap_audit_20261008/PREREGISTRATION.md'
PANEL = 'smallcap_v4/20261006T152412Z'


def main():
    import smallcap_audit as AU
    family = 'smallcap'
    if T.count(family):
        raise SystemExit('smallcap family already backfilled')
    for name in AU.TRIALS:
        T.register(family, name, prereg=AUDIT, source='0efe326', data=PANEL, segment='2018-01..2026-09',
                   outcome='no_improvement', note='equity leg counted in audit stats (17 legs)')
    groups = (('E4_variant', 6, 'A/D x equity-only, cross-asset, gold-as-cash'),
              ('season_window', 12, 'exploration season windows'),
              ('asset_set', 3, 'G0/G1/G2'),
              ('theme_combo', 5, 'T1, T2, T3, C1, mainline'))
    for prefix, n, note in groups:
        for i in range(1, n + 1):
            T.register(family, f'{prefix}_{i}', prereg=AUDIT, source='0efe326', data=PANEL,
                       segment='2018-01..2026-09', outcome='no_improvement',
                       note=f'counted from RESULTS files (lower bound): {note}')
    for name in ('R_A_adv', 'R_D_pead_adv'):
        T.register(family, name, prereg=AUDIT, source='0efe326', data=PANEL, segment='2018-01..2026-09 + 2015-07..2017-12',
                   confirm_uses=1, outcome='no_improvement',
                   note='relative band R (addendum); out-of-sample viewed once; DSR fails under the family')
    print(T.count(family), 'trials;', T.confirm_uses(family), 'confirmation uses')


if __name__ == '__main__':
    main()
