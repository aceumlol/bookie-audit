import pandas as pd

def movement_win_rates(df):
    mask = df["movement_cat"].notna()
    grp = df[mask].copy()

    # who is the fav at close?
    grp["fav_outcome"] = grp[["b365_ph", "b365_pd", "b365_pa"]].idxmax(axis=1).map({
        "b365_ph": "H", "b365_pd": "D", "b365_pa": "A"
    })
    grp["fav_implied_close"] = grp[["b365_ph", "b365_pd", "b365_pa"]].max(axis=1)
    grp["fav_won"] = grp["fav_outcome"] == grp["FTR"]

    summary = grp.groupby("movement_cat", observed=True).agg(
        n=("fav_won", "count"),
        win_rate=("fav_won", "mean"),
        avg_implied_close=("fav_implied_close", "mean"),
    ).reset_index()

    return summary

def steamed_vs_implied(df):
    # for steamed favs specifically: how does actual win rate compare
    # to what the closing implied prob predicted?
    steamed = df[df["movement_cat"] == "steamed_fav"].copy()
    steamed = steamed.dropna(subset=["b365_ph", "b365_pa"])

    steamed["fav_outcome"] = steamed[["b365_ph", "b365_pd", "b365_pa"]].idxmax(axis=1).map({
        "b365_ph": "H", "b365_pd": "D", "b365_pa": "A"
    })
    steamed["fav_implied_close"] = steamed[["b365_ph", "b365_pd", "b365_pa"]].max(axis=1)
    steamed["fav_won"] = steamed["fav_outcome"] == steamed["FTR"]

    bins = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    steamed["implied_bucket"] = pd.cut(steamed["fav_implied_close"], bins=bins)

    summary = steamed.groupby("implied_bucket", observed=True).agg(
        n=("fav_won", "count"),
        win_rate=("fav_won", "mean"),
        avg_implied=("fav_implied_close", "mean"),
    ).reset_index()

    return summary

def movement_by_season(df):
    mask = df["movement_cat"].notna()
    return (
        df[mask]
        .groupby(["season", "movement_cat"], observed=True)
        .size()
        .reset_index(name="n")
    )