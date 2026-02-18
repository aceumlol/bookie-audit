import os
import pandas as pd
from src.load_data import load_all
from src.features import build_features
from src.analysis.calibration import brier_by_season, calibration_data, favorite_accuracy
from src.analysis.line_movement import movement_win_rates, steamed_vs_implied, movement_by_season
from src.analysis.value_gap import gap_summary, gap_by_outcome, gap_distribution, gap_by_season
from src.ml.train import build_ml_features, split, train_models
from src.ml.evaluate import evaluate_all, roc_data, pr_data, model_calibration_data, feature_importance
from src.viz.plots import (
    plot_calibration_curves, plot_brier_by_season, plot_favorite_accuracy,
    plot_movement_win_rates, plot_steamed_vs_implied,
    plot_value_gap_summary, plot_gap_by_outcome,
    plot_roc_curves, plot_pr_curves, plot_feature_importance, plot_model_calibration,
)
from walk_forward import walk_forward_validate
from backtest import threshold_sweep, plot_cumulative_pnl, plot_walk_forward_auc

os.makedirs("output/plots", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

def main():
    # load data
    df = load_all("data/raw")
    df = build_features(df)

    # calibration
    print("\n--- calibration ---")
    brier = brier_by_season(df)
    brier.to_csv("output/results/brier_by_season.csv", index=False)

    fav = favorite_accuracy(df, threshold=0.7)
    fav.to_csv("output/results/favorite_accuracy.csv", index=False)

    plot_calibration_curves(df, calibration_data)
    plot_brier_by_season(brier)
    plot_favorite_accuracy(fav)

    # line movement
    print("\n--- line movement ---")
    mov = movement_win_rates(df)
    mov.to_csv("output/results/movement_win_rates.csv", index=False)

    steamed = steamed_vs_implied(df)
    steamed.to_csv("output/results/steamed_vs_implied.csv", index=False)

    tmp = movement_by_season(df)
    tmp.to_csv("output/results/movement_by_season.csv", index=False)

    plot_movement_win_rates(mov)
    plot_steamed_vs_implied(steamed)

    # value gap
    print("\n--- value gap ---")
    gap = gap_summary(df)
    gap.to_csv("output/results/gap_summary.csv", index=False)

    gap_out = gap_by_outcome(df)
    gap_out.to_csv("output/results/gap_by_outcome.csv", index=False)

    gap_by_season(df).to_csv("output/results/gap_by_season.csv", index=False)
    gap_distribution(df).to_csv("output/results/gap_distribution.csv", index=False)

    plot_value_gap_summary(gap)
    plot_gap_by_outcome(gap_out)

    # ml single split
    print("\n--- ml (single split) ---")
    ml_df, feat_cols = build_ml_features(df)
    X_train, X_test, y_train, y_test = split(ml_df)

    models = train_models(X_train, y_train)

    res, probas = evaluate_all(models, X_test, y_test)
    res.to_csv("output/results/model_metrics.csv", index=False)
    print(res.to_string(index=False))

    imp = feature_importance(models, feat_cols)
    imp.to_csv("output/results/feature_importance.csv", index=False)

    roc = roc_data(y_test, probas)
    pr = pr_data(y_test, probas)
    cal = model_calibration_data(y_test, probas)

    plot_roc_curves(roc, res)
    plot_pr_curves(pr, res)
    plot_feature_importance(imp)
    plot_model_calibration(cal)

    # walk forward
    print("\n--- walk-forward validation ---")
    preds_df, fold_stats = walk_forward_validate(df)

    fold_stats.to_csv("output/results/walk_forward_metrics.csv", index=False)
    preds_df.to_csv("output/results/walk_forward_preds.csv")

    plot_walk_forward_auc(fold_stats)

    # backtest
    print("\n--- backtest ---")
    sweep = threshold_sweep(preds_df, df, model="logreg")
    sweep.to_csv("output/results/backtest_sweep.csv", index=False)

    #TBC 0.3 seems decent, might try other thresholds later
    plot_cumulative_pnl(preds_df, df, threshold=0.3)

    print("\ndone. outputs in output/")

if __name__ == "__main__":
    main()
