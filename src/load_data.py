import os
import glob
import pandas as pd

KEEP_COLS = [
    "Div", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A",
    "B365CH", "B365CD", "B365CA",
    "PSH", "PSD", "PSA",
    "PSCH", "PSCD", "PSCA",
    "MaxH", "MaxD", "MaxA",
    "MaxCH", "MaxCD", "MaxCA",
    "AvgH", "AvgD", "AvgA",
    "B365>2.5", "B365<2.5", "P>2.5", "P<2.5", "Max>2.5",
    "AHh", "B365AHH", "B365AHA", "PAHH", "PAHA",
]

LEAGUE_NAMES = {
    "E0":  "Premier League", #added
    "SP1": "La Liga", #added
    "D1":  "Bundesliga", #added
    "I1":  "Serie A", #added
    "F1":  "Ligue 1", #added
    "N1":  "Eredivisie", #added
    "P1":  "Primeira Liga", #added
    "B1":  "Belgian Pro League", #added
}

def load_all(data_dir="data/raw"):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSVs found in {data_dir}")

    print(f"loading {len(files)} files...")
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin-1")

        # SP1_2122.csv -> league="SP1", season="2021-22"
        stem = os.path.splitext(os.path.basename(f))[0]
        parts = stem.split("_")
        league_code = parts[0]
        tag = parts[-1]

        df["season"] = f"20{tag[:2]}-{tag[2:]}" if len(tag) == 4 else tag
        df["league"] = LEAGUE_NAMES.get(league_code, league_code)

        keep = [c for c in KEEP_COLS if c in df.columns] + ["season", "league"]
        frames.append(df[keep])

    df = pd.concat(frames, ignore_index=True)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FTR", "Date"])
    df = df[df["FTR"].str.strip().str.upper().isin(["H", "D", "A"])]
    df["FTR"] = df["FTR"].str.strip().str.upper()

    non_numeric = ["Div", "Date", "HomeTeam", "AwayTeam", "FTR", "season", "league"]
    odds_cols = [c for c in df.columns if c not in non_numeric]
    df[odds_cols] = df[odds_cols].apply(pd.to_numeric, errors="coerce")

    # quick sanity check
    # print(df.isna().sum()) 
    # print(df['league'].value_counts())

    df = df.sort_values(["league", "Date"]).reset_index(drop=True)
    print("Loaded", len(df), "rows across", df['league'].nunique(), "leagues")
    return df