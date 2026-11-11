"""
modeling.py  (Steps 3-5)
------------------------
Cross-validated comparison of:
  - baseline model: city + time only        ("where and when")
  - full model:     baseline + weather
The R² gain (delta_R2) is weather's added explanatory power.

Folds are grouped by local calendar day so neighboring hours never
end up on both sides of a split (which would inflate the scores).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

import config


def cv_score(data, features, target):
    """Mean R², std R², mean MAE, and n over grouped K-fold CV."""
    d = data.dropna(subset=[target])          # only rows missing the TARGET are dropped
    X, y, groups = d[features], d[target], d[config.GROUP_COL]

    r2s, maes = [], []
    for tr, te in GroupKFold(n_splits=config.N_SPLITS).split(X, y, groups):
        model = lgb.LGBMRegressor(**config.LGBM_PARAMS)    # fresh model every fold
        model.fit(X.iloc[tr], y.iloc[tr])                  # .iloc: split returns positions
        pred = model.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], pred))
        maes.append(mean_absolute_error(y.iloc[te], pred))

    return np.mean(r2s), np.std(r2s), np.mean(maes), len(d)


def evaluate_all(df, targets=config.TARGETS, verbose=True):
    rows = []
    for t in targets:
        if verbose:
            print(f"  cross-validating {t} ...")
        b_mean, b_sd, b_mae, n = cv_score(df, config.BASELINE, t)
        w_mean, w_sd, w_mae, _ = cv_score(df, config.WEATHER, t)
        f_mean, f_sd, f_mae, _ = cv_score(df, config.BASELINE + config.WEATHER, t)
        rows.append({
            "target": t,
            "n": n,
            "baseline_R2": b_mean, "baseline_R2_sd": b_sd,
            "weather_only_R2": w_mean, "weather_only_R2_sd": w_sd,
            "full_R2": f_mean, "full_R2_sd": f_sd,
            "full_MAE": f_mae,
            "delta_R2": f_mean - b_mean,
        })
    return pd.DataFrame(rows).round(3)


def fit_full_model(df, target):
    """Train one model on all rows that have the target (used for SHAP)."""
    d = df.dropna(subset=[target])
    X = d[config.BASELINE + config.WEATHER]
    model = lgb.LGBMRegressor(**config.LGBM_PARAMS).fit(X, d[target])
    return model, X, d[target]
