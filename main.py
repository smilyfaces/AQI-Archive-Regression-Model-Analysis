"""
main.py
-------
Runs the whole pipeline:
  1. clean          (clean_data.py)
  2. features       (features.py)
  3-5. CV models    (modeling.py)   -> outputs/model_comparison.csv
  6. SHAP           (explain.py)    -> outputs/<target>/*.png, *.csv
                                     -> outputs/shap_summary.csv
Run:  python main.py
"""

import warnings

import pandas as pd

import config
from clean_data import load_and_clean
from explain import explain_target
from features import build_features
from modeling import evaluate_all

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


def main():
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    print("Step 1: cleaning")
    df = load_and_clean()

    print("\nStep 2: building features")
    df = build_features(df)

    print("\nSteps 3-5: cross-validated baseline vs. full models")
    results = evaluate_all(df)
    results.to_csv(config.OUTPUT_DIR / "model_comparison.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(results)

    print("\nStep 6: SHAP explanations")
    summary = pd.DataFrame([explain_target(df, t) for t in config.TARGETS])
    summary.to_csv(config.OUTPUT_DIR / "shap_summary.csv", index=False)

    print(f"\nDone. Results saved in {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
