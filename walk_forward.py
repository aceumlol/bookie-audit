import pandas as pd
import numpy as np
from src.ml.train import build_ml_features, train_models
from src.ml.evaluate import evaluate_all

seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

def walk_forward_validate(df):
    ml_df, feat_cols = build_ml_features(df.copy())

    all_preds = []
    stats = []

    for i in range(2, len(seasons)):
        target = seasons[i]
        train_seasons = seasons[:i]

        train = ml_df[ml_df["season"].isin(train_seasons)].drop(columns="season")
        test = ml_df[ml_df["season"] == target].drop(columns="season")

        if len(test) == 0:
            print(f"  skip {target} — no rows")
            continue
        if test["high_gap"].sum() == 0:
            print(f"  skip {target} — zero positives")
            continue

        X_tr = train.drop(columns="high_gap")
        y_tr = train["high_gap"]
        X_te = test.drop(columns="high_gap")
        y_te = test["high_gap"]

        print(f"fold → train: {train_seasons}  |  test: {target}  "
              f"|  train_pos: {y_tr.mean():.2%}  "
              f"|  test_pos: {y_te.mean():.2%}  "
              f"|  n_test: {len(test)}")

        fitted = train_models(X_tr, y_tr)
        results, probas = evaluate_all(fitted, X_te, y_te)

        # collect metrics per model
        for _, row in results.iterrows():
            stats.append({
                "season": target,
                "model": row["model"],
                "roc_auc": row["roc_auc"],
                "avg_precision": row["avg_precision"],
                "n_test": len(test),
                "n_pos": int(y_te.sum()),
            })

        tmp = pd.DataFrame(index=y_te.index)
        tmp["season"] = target
        tmp["y_true"] = y_te.values
        for m, probs in probas.items():
            tmp[f"{m}_prob"] = probs
        all_preds.append(tmp)

    all_preds_df = pd.concat(all_preds)
    stats_df = pd.DataFrame(stats)

    print("\n--- walk-forward: per-fold results ---")
    print(stats_df.to_string(index=False))

    print("\n--- walk-forward: averaged across folds ---")
    avg = stats_df.groupby("model")[["roc_auc", "avg_precision"]].mean().round(4)
    print(avg)

    return all_preds_df, stats_df