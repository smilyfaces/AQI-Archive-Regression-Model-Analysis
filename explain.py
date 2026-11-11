"""
explain.py  (Step 6)
--------------------
SHAP analysis for one target:
  6a. feature importance + weather's share of total attribution
  6b. dependence plots for the top weather drivers
  6c. strongest feature-pair interactions
Plots are saved as PNGs in outputs/<target>/.
"""

import matplotlib
matplotlib.use("Agg")                 # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

import config
from modeling import fit_full_model


def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def explain_target(df, target, n_dependence=3, verbose=True):
    outdir = config.OUTPUT_DIR / target
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- 6a. SHAP values and importance ----------
    model, X, y = fit_full_model(df, target)
    X_s = X.sample(min(config.SHAP_SAMPLE, len(X)), random_state=config.RANDOM_STATE)
    sv = shap.TreeExplainer(model)(X_s)

    # SHAP's plots can't color by text categories, so give the plots a copy
    # of the data with categories as numeric codes (missing -> NaN).
    # The SHAP values themselves are unchanged.
    X_plot = X_s.copy()
    for col in config.CATEGORICAL_COLS:
        X_plot[col] = X_plot[col].cat.codes.replace(-1, np.nan).astype(float)
    sv = shap.Explanation(values=sv.values, base_values=sv.base_values,
                          data=X_plot.to_numpy(dtype=float),
                          feature_names=list(X_s.columns))

    shap.plots.bar(sv, max_display=15, show=False)
    _save(outdir / "importance_bar.png")
    shap.plots.beeswarm(sv, max_display=15, show=False)
    _save(outdir / "importance_beeswarm.png")

    imp = (pd.Series(np.abs(sv.values).mean(axis=0), index=X_s.columns)
             .sort_values(ascending=False))
    weather_share = imp[config.WEATHER].sum() / imp.sum()

    # ---------- 6b. Dependence plots for the top weather features ----------
    top_weather = [f for f in imp.index if f in config.WEATHER][:n_dependence]
    for feat in top_weather:
        shap.plots.scatter(sv[:, feat], color=sv, show=False)
        _save(outdir / f"dependence_{feat}.png")

    # ---------- 6c. Interactions ----------
    # TreeSHAP interactions fail on native categorical columns, so this
    # helper model uses integer codes instead. Main results above still
    # come from the categorical model.
    Xc = X.copy()
    for col in config.CATEGORICAL_COLS:
        Xc[col] = Xc[col].cat.codes
    model_int = lgb.LGBMRegressor(**config.LGBM_PARAMS).fit(Xc, y)
    X_int = Xc.loc[X_s.index[:config.INTERACTION_SAMPLE]]
    inter = shap.TreeExplainer(model_int).shap_interaction_values(X_int)

    M = np.abs(inter).mean(axis=0)
    np.fill_diagonal(M, 0)                       # remove main effects
    i, j = np.triu_indices_from(M, k=1)          # each pair once
    pairs = (pd.DataFrame({"feature_a": Xc.columns[i],
                           "feature_b": Xc.columns[j],
                           "strength": M[i, j]})
               .sort_values("strength", ascending=False)
               .reset_index(drop=True))

    is_w = lambda s: s.isin(config.WEATHER)
    weather_pairs = pairs[is_w(pairs["feature_a"]) & is_w(pairs["feature_b"])]

    imp.rename("mean_abs_shap").to_csv(outdir / "importance.csv")
    pairs.head(config.TOP_PAIRS).to_csv(outdir / "top_interactions_all.csv", index=False)
    weather_pairs.head(config.TOP_PAIRS).to_csv(outdir / "top_interactions_weather.csv",
                                                index=False)

    if verbose:
        print(f"\n=== {target} (n = {len(X):,}) ===")
        print(f"Weather share of SHAP attribution: {weather_share:.0%}")
        print("Top features:")
        print(imp.head(8).round(3).to_string())
        print("Top weather x weather interactions:")
        print(weather_pairs.head(5).round(3).to_string(index=False))

    return {"target": target, "n": len(X), "weather_share": round(weather_share, 3),
            "top_features": ", ".join(imp.index[:5]),
            "top_weather_pair": (" x ".join(weather_pairs.iloc[0, :2])
                                 if len(weather_pairs) else "")}
