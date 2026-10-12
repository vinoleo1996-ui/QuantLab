"""The two P4 rankers. Inputs are per-day percentile features (0..1, missing = 0.5) and a percentile label."""
from __future__ import annotations

import numpy as np
import pandas as pd

GBDT_PARAMS = dict(objective='regression', num_leaves=31, learning_rate=0.03, min_data_in_leaf=500,
                   feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbosity=-1)
GBDT_MAX_ROUNDS, GBDT_PATIENCE, GBDT_VALID_SHARE = 1000, 50, 0.10
SEEDS = (1, 2, 3, 4, 5)


def daily_rank_ic(frame, features, label='label'):
    """Per-day Pearson correlation of percentile features with the percentile label (a rank IC)."""
    def one(group):
        y = group[label]
        return pd.Series({f: np.corrcoef(group[f], y)[0, 1] if group[f].std() > 0 and y.std() > 0 else np.nan
                          for f in features})
    return frame.groupby('date')[list(features) + [label]].apply(one)


class LinearICIR:
    name = 'LIN'

    def fit(self, train, features):
        ic = daily_rank_ic(train, features)
        self.features = list(features)
        self.weights = (ic.mean() / ic.std()).fillna(0.0).reindex(self.features)
        self.ic = ic
        return self

    def predict(self, frame):
        return (frame[self.features] - 0.5).to_numpy() @ self.weights.to_numpy()


class GBDTEnsemble:
    name = 'GBDT'

    def fit(self, train, features, seeds=SEEDS):
        import lightgbm as lgb
        self.features = list(features)
        dates = np.sort(train['date'].unique())
        cut = dates[int(len(dates) * (1 - GBDT_VALID_SHARE))]
        fit, valid = train[train['date'] < cut], train[train['date'] >= cut]
        self.boosters = []
        for seed in seeds:
            params = dict(GBDT_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
                          num_threads=8, deterministic=True)
            booster = lgb.train(params, lgb.Dataset(fit[self.features], fit['label']), GBDT_MAX_ROUNDS,
                                valid_sets=[lgb.Dataset(valid[self.features], valid['label'])],
                                callbacks=[lgb.early_stopping(GBDT_PATIENCE, verbose=False)])
            self.boosters.append(booster)
        return self

    def predict(self, frame):
        x = frame[self.features]
        return np.mean([b.predict(x, num_iteration=b.best_iteration) for b in self.boosters], axis=0)

    def importance(self):
        gains = [pd.Series(b.feature_importance('gain'), index=self.features) for b in self.boosters]
        return pd.concat(gains, axis=1).mean(axis=1).sort_values(ascending=False)


def percentile_frame(frame, features):
    """Per-day percentile of each feature; missing values become 0.5 after ranking."""
    out = frame.copy()
    out[list(features)] = frame.groupby('date')[list(features)].rank(pct=True).fillna(0.5)
    return out
