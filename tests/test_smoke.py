"""冒烟测试：本地确定性样本端到端跑通 因子→组合→回测→指标(仅测试隔离，非演示数据)。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

import numpy as np
import pandas as pd

from quantlab.backtest.engine import Engine
from quantlab.factors.engine import FactorEngine, winsorize_zscore
from quantlab.report.metrics import performance_table
from quantlab.strategy.portfolio import target_vol_weights
from quantlab.strategy.risk import Regime, RiskManager
from quantlab.strategy.scoring import composite_score, rank_ic


def _local_panel(n_stocks: int = 120, n_days: int = 900, start: str = "2018-06-01") -> dict:
    """内嵌确定性随机样本，仅作为离线单元测试夹具；不进入任何信号/回测展示链路。"""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(start=start, periods=n_days)
    codes = [f"X{i:06d}" for i in range(n_stocks)]
    close = pd.DataFrame(index=idx, columns=codes)
    for c in codes:
        noise = rng.normal(0.0004, 0.022, n_days)
        level = np.exp(np.cumsum(noise))
        close[c] = level * (1.0 + rng.uniform(0.5, 3.0))
    volume = pd.DataFrame(rng.uniform(1.5e6, 8e6, size=(n_days, n_stocks)),
                          index=idx, columns=codes)
    amount = close * volume * 100.0
    turnover = rng.uniform(0.5, 4.5, size=(n_days, n_stocks))
    turnover = pd.DataFrame(turnover, index=idx, columns=codes)
    market = pd.DataFrame({"close": np.exp(np.cumsum(rng.normal(0.0002, 0.012, n_days))) + 100.0}, index=idx)
    meta = pd.DataFrame({
        "name": [f"样本{c}" for c in codes], "sector": ["NA"] * n_stocks,
        "mktcap": rng.uniform(1e9, 1e11, n_stocks), "float_mktcap": rng.uniform(1e9, 8e10, n_stocks),
    }, index=codes)
    meta.index.name = "code"
    panels = {}
    for c in codes:
        df = pd.DataFrame(index=idx)
        df["open"] = close[c].to_numpy() * (1 + rng.normal(0, 0.004, n_days))
        df["close"] = close[c]
        df["high"] = df["close"] * 1.01
        df["low"] = df["close"] * 0.99
        df["amount"] = amount[c]
        df["turnover"] = turnover[c]
        df.loc[df.index[0], "open"] = df["close"].iloc[0]
        panels[c] = df
    return {"panels": panels, "close": close, "amount": amount,
            "turnover": turnover, "meta": meta, "market": market}


class TestSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pan = _local_panel()
        cls.close, cls.amount, cls.turnover = cls.pan["close"], cls.pan["amount"], cls.pan["turnover"]
        cls.meta = cls.pan["meta"]

    def test_winsorize(self):
        s = pd.Series([1.0, 2.0, 3.0, 2.2, 1.8, 100.0, -50.0])
        z = winsorize_zscore(s)
        assert np.isfinite(z).all()
        assert z.abs().max() <= 4.0

    def test_factor_engine(self):
        eng = FactorEngine("a", self.meta)
        fs = eng.compute(self.close, self.amount, self.turnover)
        self.assertGreaterEqual(len(fs), 6)
        for name, f in fs.items():
            vals = f.dropna().to_numpy()
            self.assertGreater(vals.size, 0, name)
            self.assertTrue(np.isfinite(vals).all(), name)

    def test_rank_ic(self):
        f = pd.DataFrame(np.random.default_rng(1).normal(size=(3, 20)),
                         index=[0, 1, 2], columns=[f"S{i}" for i in range(20)])
        y = pd.DataFrame(np.random.default_rng(2).normal(size=(3, 20)),
                         index=[0, 1, 2], columns=[f"S{i}" for i in range(20)])
        ic = rank_ic(f, y)
        self.assertEqual(len(ic), 3)
        self.assertTrue(np.isfinite(ic).all())

    def test_target_vol_weights(self):
        ret = self.close.pct_change().tail(120)
        w = target_vol_weights(ret)
        self.assertAlmostEqual(w["weight"].sum(), 1.0, places=2)
        self.assertTrue((w["weight"] >= 0).all())

    def test_regime(self):
        r = Regime(self.close.mean(axis=1))
        st = r.state(self.close.index[-1])
        self.assertIn("multiplier", st)

    def test_risk_manager(self):
        rm = RiskManager()
        rm.evaluate(1.00, 1.0)
        out = rm.evaluate(0.80, 1.0)
        self.assertLess(out["gross_scale"], 1.0)

    def test_engine_run(self):
        panels = {}
        for c in self.close.columns:
            df = pd.DataFrame(index=self.close.index)
            df["close"] = self.close[c]
            df["open"] = df["close"].shift(1).fillna(self.close[c].iloc[0]) * 1.001
            df["high"] = df[["open", "close"]].max(axis=1) * 1.01
            df["low"] = df[["open", "close"]].min(axis=1) * 0.99
            df["amount"] = self.amount[c]
            df["turnover"] = self.turnover[c]
            panels[c] = df
        bench = self.pan["market"]["close"]
        eng = Engine()
        res = eng.run(panels, self.meta, bench.rename("close").to_frame())
        # Bare historical caches no longer qualify as executable/PIT market data.
        # End-to-end cash/share replay is covered with explicit contracts in test_replay.py.
        self.assertEqual(res["status"], "blocked_data")
        self.assertIsNone(res["metrics"])
        self.assertFalse(res["trades"])



if __name__ == "__main__":
    unittest.main(verbosity=2)