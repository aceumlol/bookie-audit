import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

TRAIN_CUTOFF = "2022-23"

def build_ml_features(df):
    feat_cols = [
        "b365_ph", "b365_pd", "b365_pa",
        "b365_close_open_delta_h", "b365_close_open_delta_a",
        "b365_overround",
        "ps_overround",
    ]

    df["fav_implied"] = df[["b365_ph", "b365_pd", "b365_pa"]].max(axis=1)
    df["fav_implied_bucket"] = pd.cut(
        df["fav_implied"],
        bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    df["b365_spread"] = df[["b365_ph", "b365_pd", "b365_pa"]].std(axis=1)

    phase_map = {"early": 0, "mid": 1, "late": 2}
    df["season_phase_enc"] = df["season_phase"].map(phase_map)

    # league matters — B365 shading patterns differ by market
    le = LabelEncoder()
    df["league_enc"] = le.fit_transform(df["league"].astype(str))

    feat_cols += ["fav_implied_bucket", "b365_spread", "season_phase_enc", "fav_implied", "league_enc"]

    ml_df = df[feat_cols + ["high_gap", "season"]].dropna()
    ml_df["high_gap"] = ml_df["high_gap"].astype(int)

    pos_rate = ml_df["high_gap"].mean()
    pos_count = ml_df['high_gap'].sum()
    total_rows = len(ml_df)
    pos_pct = round(pos_rate * 100, 2)

    print(f"High gap rate: {pos_pct}%")
    print(f"Total positive: {pos_count} out of {total_rows}")

    return ml_df, feat_cols

def split(ml_df):
    train = ml_df[ml_df["season"] <= TRAIN_CUTOFF].drop(columns="season")
    test  = ml_df[ml_df["season"] >  TRAIN_CUTOFF].drop(columns="season")

    X_train, y_train = train.drop(columns="high_gap"), train["high_gap"]
    X_test,  y_test  = test.drop(columns="high_gap"),  test["high_gap"]

    print(f"Train size: {len(train)}, Test size: {len(test)}")
    # print(f"Train pos rate: {round(y_train.mean()*100, 1)}%")
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_model.fit(X_train, y_train)
    print("logreg done")

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42)
    rf_model.fit(X_train, y_train)
    print("rf done")

    xgb_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              scale_pos_weight=neg/pos, eval_metric="logloss", random_state=42)
    xgb_model.fit(X_train, y_train)
    print("xgb done")
    
    fitted = {"logreg": lr_model, "rf": rf_model, "xgb": xgb_model}

    return fitted