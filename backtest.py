import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # just in case
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def simulate_returns(preds, df_full, model="logreg", threshold=0.3, stake=1.0):
    prob_col = f"{model}_prob"
    flagged = preds[preds[prob_col] >= threshold].copy()

    if len(flagged) == 0:
        print(f"  no bets at threshold={threshold} for {model}")
        return pd.DataFrame(), {}

    need = ["B365H", "B365D", "B365A", "FTR", "b365_ph", "b365_pd", "b365_pa", "Date", "league"]
    need = [c for c in need if c in df_full.columns]
    flagged = flagged.join(df_full[need], how="left")

    flagged["fav_col"] = flagged[["b365_ph", "b365_pd", "b365_pa"]].idxmax(axis=1)
    flagged["fav_outcome"] = flagged["fav_col"].map(
        {"b365_ph": "H", "b365_pd": "D", "b365_pa": "A"}
    )

    odds_map = {"b365_ph": "B365H", "b365_pd": "B365D", "b365_pa": "B365A"}
    flagged["fav_odds"] = flagged.apply(lambda r: r[odds_map[r["fav_col"]]], axis=1)
    flagged["won"] = flagged["fav_outcome"] == flagged["FTR"]

    flagged["pnl"] = np.where(flagged["won"], stake * (flagged["fav_odds"] - 1), -stake)
    flagged["cumulative_pnl"] = flagged["pnl"].cumsum()

    n = len(flagged)
    total = flagged["pnl"].sum()

    summary = {
        "model": model,
        "threshold": threshold,
        "n_bets": n,
        "n_won": int(flagged["won"].sum()),
        "strike_rate": f"{flagged['won'].mean():.1%}",
        "avg_odds": f"{flagged['fav_odds'].mean():.2f}",
        "total_pnl": f"£{total:.2f}",
        "roi": f"{(total / (n * stake)) * 100:.1f}%",
    }

    return flagged, summary

def threshold_sweep(preds, df_full, model="logreg", stake=1.0):
    rows = []
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for t in thresholds:
        _, s = simulate_returns(preds, df_full, model=model, threshold=t, stake=stake)
        if s:
            rows.append(s)

    out = pd.DataFrame(rows)
    print(f"\n--- threshold sweep ({model}) ---")
    print(out.to_string(index=False))
    return out

def plot_cumulative_pnl(preds, df_full, threshold=0.3, stake=1.0,
                        save_path="output/plots/backtest_pnl.png"):
    colors = {"logreg": "steelblue", "rf": "tomato", "xgb": "seagreen"}

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_facecolor("#f5f5f5")
    fig.patch.set_facecolor("white")

    for m in ["logreg", "rf", "xgb"]:
        bets, info = simulate_returns(preds, df_full, model=m,
                                      threshold=threshold, stake=stake)
        if len(bets) == 0:
            continue

        bets = bets.sort_index()
        lbl = f"{m}  (n={info['n_bets']}, roi={info['roi']}, pnl={info['total_pnl']})"
        ax.plot(range(len(bets)), bets["cumulative_pnl"],
                label=lbl, color=colors[m], lw=2)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("bet number (chronological)", fontsize=11)
    ax.set_ylabel(f"cumulative p&l (£{stake}/bet)", fontsize=11)
    ax.set_title(f"backtest — cumulative p&l  |  threshold = {threshold}", fontsize=13)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("£%.0f"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved {save_path}")

def plot_walk_forward_auc(fold_stats, save_path="output/plots/walk_forward_auc.png"):
    colors = {"logreg": "steelblue", "rf": "tomato", "xgb": "seagreen"}

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_facecolor("#f5f5f5")
    fig.patch.set_facecolor("white")

    for name, grp in fold_stats.groupby("model"):
        ax.plot(grp["season"], grp["roc_auc"], marker="o",
                label=name, color=colors.get(name, "grey"), lw=2)

    ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="random baseline")
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("roc-auc", fontsize=11)
    ax.set_title("walk-forward auc by season", fontsize=13)
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved {save_path}")
