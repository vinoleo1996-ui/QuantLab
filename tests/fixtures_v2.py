"""Deterministic synthetic fixtures, never loaded by production data adapters."""
import numpy as np
import pandas as pd
from quantlab.data.contracts import MarketDataset


def dataset(days=70, stocks=6):
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2023-01-02", periods=days+5)
    panels, records = {}, []
    for i in range(stocks):
        c = f"S{i:03}"
        close = 20*np.exp(np.cumsum(rng.normal(.0003, .008, days)))
        prev = np.r_[close[0], close[:-1]]
        panels[c] = pd.DataFrame(dict(open=prev, close=close, amount=1e8, turnover=1.+rng.uniform(size=days),
                       source="TEST_FIXTURE", revision="1", quality_status="ok", open_status="trading",
                       reference_price=prev, limit_up=prev*1.1, limit_down=prev*.9,
                       available_at=[str(d.date())+" 16:00:00" for d in dates[:days]],
                       open_status_available_at=[str(d.date())+" 09:25:00" for d in dates[:days]]), index=dates[:days])
        records.append(dict(instrument_id=c, available_at="2022-01-01", effective_at="2022-01-01",
                            source="TEST_FIXTURE", revision="1", quality_status="ok", active=True, is_st=False,
                            listing_date="2020-01-01", float_mktcap=1e10, sector=f"sector{i%4}",
                            asset_type="stock", buy_lot=100, sell_lot=100, sell_delay=1, price_tick=.01,
                            commission=.00025, min_commission=5., limit_ratio=.1))
    bench = pd.DataFrame({"close": 100*np.exp(np.cumsum(rng.normal(.0002,.005,days)))}, index=dates[:days])
    bench["source"], bench["revision"], bench["quality_status"] = "TEST_FIXTURE", "1", "ok"
    bench["available_at"] = [str(d.date())+" 16:00:00" for d in bench.index]
    return MarketDataset(panels, pd.DataFrame(records), dates, benchmark=bench,
                         manifest=dict(source="TEST_FIXTURE", revision="1", universe_coverage="historical",
                                       actions_coverage="complete", price_basis="raw", benchmark_type="index",
                                       currency="CNY",amount_unit="CNY",turnover_unit="percent",
                                       as_of=str(dates[days-1].date())))
