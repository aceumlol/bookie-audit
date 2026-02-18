import numpy as np
import pandas as pd

def remove_vig(h, d, a):
    total = 1/h + 1/d + 1/a
    return 1/(h * total), 1/(d * total), 1/(a * total)

def add_implied_probs(df):
    for prefix, cols in [
        ("b365", ("B365H", "B365D", "B365A")),
        ("ps",   ("PSH",   "PSD",   "PSA")),
        ("max",  ("MaxH",  "MaxD",  "MaxA")),
    ]:
        mask = df[list(cols)].notna().all(axis=1)
        ph, pd_, pa = remove_vig(df.loc[mask, cols[0]], df.loc[mask, cols[1]], df.loc[mask, cols[2]])
        df.loc[mask, f"{prefix}_ph"] = ph
        df.loc[mask, f"{prefix}_pd"] = pd_
        df.loc[mask, f"{prefix}_pa"] = pa

    # opening b365 implied probs (used for line movement)
    mask = df[["B365CH", "B365CD", "B365CA"]].notna().all(axis=1)
    ph, pd_, pa = remove_vig(df.loc[mask, "B365CH"], df.loc[mask, "B365CD"], df.loc[mask, "B365CA"])
    df.loc[mask, "b365_open_ph"] = ph
    df.loc[mask, "b365_open_pd"] = pd_
    df.loc[mask, "b365_open_pa"] = pa

    return df

def add_overround(df):
    df["b365_overround"] = 1/df["B365H"] + 1/df["B365D"] + 1/df["B365A"]
    df["ps_overround"]   = 1/df["PSH"]   + 1/df["PSD"]   + 1/df["PSA"]
    return df

def categorise_movement(row):
    # compare b365 closing vs opening implied prob for the favourite
    # favourite defined by closing b365 implied prob
    if any(np.isnan(v) for v in [row["b365_ph"], row["b365_pa"], row["b365_open_ph"], row["b365_open_pa"]]):
        return np.nan

    home_fav = row["b365_ph"] >= row["b365_pa"]
    if home_fav:
        close_p, open_p = row["b365_ph"], row["b365_open_ph"]
    else:
        close_p, open_p = row["b365_pa"], row["b365_open_pa"]

    delta = close_p - open_p  # positive = steamed (shortened), negative = drifted

    if abs(delta) < 0.02:
        return "stable_fav" if close_p >= 0.5 else "stable_dog"
    elif delta > 0.02:
        return "steamed_fav"
    elif delta < -0.02:
        return "drifted_fav"
    else:
        # implied favourite flipped between open and close
        return "flip"

def add_line_movement(df):
    df["movement_cat"] = df.apply(categorise_movement, axis=1)
    df["b365_close_open_delta_h"] = df["b365_ph"] - df["b365_open_ph"]
    df["b365_close_open_delta_a"] = df["b365_pa"] - df["b365_open_pa"]
    return df

def add_value_gap(df):
    # value gap: how much lower is b365 implied prob vs market max
    # positive gap = b365 is less generous than the market max
    mask = df[["max_ph", "b365_ph", "max_pa", "b365_pa", "max_pd", "b365_pd"]].notna().all(axis=1)
    df.loc[mask, "gap_h"] = df.loc[mask, "max_ph"] - df.loc[mask, "b365_ph"]
    df.loc[mask, "gap_d"] = df.loc[mask, "max_pd"] - df.loc[mask, "b365_pd"]
    df.loc[mask, "gap_a"] = df.loc[mask, "max_pa"] - df.loc[mask, "b365_pa"]

    # max gap across any outcome for this match
    df.loc[mask, "max_gap"] = df.loc[mask, ["gap_h", "gap_d", "gap_a"]].max(axis=1)
    df["high_gap"] = df["max_gap"] > 0.03

    return df

def add_season_phase(df):
    # early = first 10 GWs, late = last 10 GWs, mid = everything else
    # rough approximation: sort by date within season, assign decile
    df["season_phase"] = "mid"
    for season, grp in df.groupby("season"):
        n = len(grp)
        early_cut = grp["Date"].quantile(0.26)
        late_cut  = grp["Date"].quantile(0.74)
        df.loc[grp[grp["Date"] <= early_cut].index, "season_phase"] = "early"
        df.loc[grp[grp["Date"] >= late_cut].index,  "season_phase"] = "late"

    return df

def build_features(df):
    df = add_implied_probs(df)
    df = add_overround(df)
    df = add_line_movement(df)
    df = add_value_gap(df)
    df = add_season_phase(df)
    return df