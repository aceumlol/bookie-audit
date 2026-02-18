import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

def brier_score(probs, outcomes):
    return np.mean((probs - outcomes) ** 2)

def brier_by_season(df):
    records = []
    for season, grp in df.groupby("season"):
        grp = grp.dropna(subset=["b365_ph", "b365_pd", "b365_pa"])
        if len(grp) < 10:
            continue

        h_actual = (grp["FTR"] == "H").astype(int)
        d_actual = (grp["FTR"] == "D").astype(int)
        a_actual = (grp["FTR"] == "A").astype(int)

        records.append({
            "season": season,
            "brier_h": brier_score(grp["b365_ph"], h_actual),
            "brier_d": brier_score(grp["b365_pd"], d_actual),
            "brier_a": brier_score(grp["b365_pa"], a_actual),
            "brier_ps_h": brier_score(grp["ps_ph"].dropna(), (grp.loc[grp["ps_ph"].notna(), "FTR"] == "H").astype(int)),
            "n": len(grp),
        })

    return pd.DataFrame(records)

def calibration_data(df, outcome, prob_col, n_bins=10):
    mask = df[prob_col].notna()
    y = (df.loc[mask, "FTR"] == outcome).astype(int)
    p = df.loc[mask, prob_col]
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos

def favorite_accuracy(df, threshold=0.7):
    # matches where b365 implied prob on any outcome exceeds threshold
    heavy = df[
        (df[["b365_ph", "b365_pd", "b365_pa"]].max(axis=1) >= threshold) &
        df[["b365_ph", "b365_pd", "b365_pa"]].notna().all(axis=1)
    ].copy()

    heavy["fav_outcome"] = heavy[["b365_ph", "b365_pd", "b365_pa"]].idxmax(axis=1).map({
        "b365_ph": "H", "b365_pd": "D", "b365_pa": "A"
    })
    heavy["fav_implied"] = heavy[["b365_ph", "b365_pd", "b365_pa"]].max(axis=1)
    heavy["fav_won"] = heavy["fav_outcome"] == heavy["FTR"]

    bins = np.arange(threshold, 1.01, 0.05)
    heavy["prob_bucket"] = pd.cut(heavy["fav_implied"], bins=bins)

    summary = heavy.groupby("prob_bucket", observed=True).agg(
        n=("fav_won", "count"),
        win_rate=("fav_won", "mean"),
        avg_implied=("fav_implied", "mean"),
    ).reset_index()

    return summary