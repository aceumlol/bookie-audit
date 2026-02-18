import pandas as pd

def gap_summary(df):
    mask = df["max_gap"].notna()
    grp = df[mask].copy()

    grp["fav_outcome"] = grp[["b365_ph", "b365_pd", "b365_pa"]].idxmax(axis=1).map({
        "b365_ph": "H", "b365_pd": "D", "b365_pa": "A"
    })
    grp["fav_implied"] = grp[["b365_ph", "b365_pd", "b365_pa"]].max(axis=1)
    grp["fav_won"] = grp["fav_outcome"] == grp["FTR"]

    summary = grp.groupby("high_gap", observed=True).agg(
        n=("fav_won", "count"),
        win_rate=("fav_won", "mean"),
        avg_b365_implied=("fav_implied", "mean"),
        avg_max_gap=("max_gap", "mean"),
    ).reset_index()

    return summary

def gap_by_outcome(df):
    # break down where the gap is coming from — h, d, or a
    rows = []
    for outcome, gap_col, prob_col in [("H", "gap_h", "b365_ph"), ("D", "gap_d", "b365_pd"), ("A", "gap_a", "b365_pa")]:
        sub = df[df[gap_col].notna()].copy()
        sub["actual"] = (sub["FTR"] == outcome).astype(int)
        sub["high_gap"] = sub[gap_col] > 0.03

        agg = sub.groupby("high_gap", observed=True).agg(
            n=("actual", "count"),
            win_rate=("actual", "mean"),
            avg_gap=(gap_col, "mean"),
            avg_implied=(prob_col, "mean"),
        ).reset_index()
        agg["outcome"] = outcome
        rows.append(agg)

    return pd.concat(rows, ignore_index=True)

def gap_distribution(df):
    mask = df[["gap_h", "gap_d", "gap_a"]].notna().all(axis=1)
    melted = df[mask][["gap_h", "gap_d", "gap_a"]].melt(var_name="outcome", value_name="gap")
    melted["outcome"] = melted["outcome"].map({"gap_h": "H", "gap_d": "D", "gap_a": "A"})
    return melted

def gap_by_season(df):
    mask = df["high_gap"].notna()
    return (
        df[mask]
        .groupby(["season", "high_gap"], observed=True)
        .agg(n=("high_gap", "count"), avg_gap=("max_gap", "mean"))
        .reset_index()
    )