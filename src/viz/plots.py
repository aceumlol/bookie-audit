import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

style = {
    "axes.facecolor":   "#f9f9f9",
    "figure.facecolor": "#ffffff",
    "axes.grid":        True,
    "grid.color":       "#e0e0e0",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "sans-serif",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
}
colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]
out = "output/plots"

def save_fig(fig, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.png")

def plot_calibration_curves(df, calibration_data_fn):
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

        outcomes = [
            ("H", "b365_ph", "Home win"),
            ("D", "b365_pd", "Draw"),
            ("A", "b365_pa", "Away win"),
        ]

        for i, (outcome, col, label) in enumerate(outcomes):
            ax = axes[i]
            mean_pred, frac_pos = calibration_data_fn(df, outcome, col)
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
            ax.plot(mean_pred, frac_pos, "o-", color=colors[i], lw=2, ms=5, label="B365")
            ax.set_title(label)
            ax.set_xlabel("Implied probability")
            ax.legend(fontsize=8)

        axes[0].set_ylabel("Actual win rate")
        fig.suptitle("B365 calibration — implied prob vs actual win rate", y=1.01)
    save_fig(fig, "calibration_curves")

def plot_brier_by_season(brier_df):
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(brier_df))
        w = 0.25

        ax.bar(x - w, brier_df["brier_h"], w, label="Home", color=colors[0], alpha=0.85)
        ax.bar(x,      brier_df["brier_d"], w, label="Draw", color=colors[1], alpha=0.85)
        ax.bar(x + w, brier_df["brier_a"], w, label="Away",  color=colors[2], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(brier_df["season"], rotation=30, ha="right")
        ax.set_ylabel("Brier score (lower = better)")
        ax.set_title("B365 Brier score by season and outcome")
        ax.legend()
    save_fig(fig, "brier_by_season")

def plot_favorite_accuracy(fav_df):
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(fav_df))
        ax.bar(x, fav_df["win_rate"], color=colors[0], alpha=0.85, label="Actual win rate")
        ax.plot(x, fav_df["avg_implied"], "o--", color=colors[1], lw=1.5, ms=5,
                label="Avg implied prob")

        ax.set_xticks(x)
        lbls = fav_df["prob_bucket"].astype(str)
        ax.set_xticklabels(lbls, rotation=30, ha="right")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylabel("Win rate")
        ax.set_xlabel("Implied probability bucket")
        ax.set_title("Heavy favourite accuracy — actual vs implied")
        ax.legend()
    save_fig(fig, "favorite_accuracy")

def plot_movement_win_rates(mov_df):
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        cats = mov_df["movement_cat"]
        x = np.arange(len(cats))

        bars = ax.bar(x, mov_df["win_rate"], color=colors[0], alpha=0.85)
        ax.plot(x, mov_df["avg_implied_close"], "o--", color=colors[1],
                lw=1.5, ms=5, label="Avg implied (close)")

        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylabel("Favourite win rate")
        ax.set_title("Win rate by line movement category")
        ax.legend()

        for b, n in zip(bars, mov_df["n"]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"n={n}", ha="center", va="bottom", fontsize=8)

    save_fig(fig, "movement_win_rates")

def plot_steamed_vs_implied(steamed_df):
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(steamed_df))
        ax.bar(x, steamed_df["win_rate"], color=colors[2], alpha=0.85,
               label="Actual win rate")
        ax.plot(x, steamed_df["avg_implied"], "o--", color=colors[1], lw=1.5, ms=5,
                label="Avg implied (close)")

        ax.set_xticks(x)
        ax.set_xticklabels(steamed_df["implied_bucket"].astype(str), rotation=30, ha="right")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylabel("Win rate")
        ax.set_xlabel("Closing implied prob bucket")
        ax.set_title("Steamed favourites — actual win rate vs closing implied")
        ax.legend()
    save_fig(fig, "steamed_vs_implied")

def plot_value_gap_summary(gap_df):
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        lbls = gap_df["high_gap"].map({True: "High gap (>3%)", False: "Normal"})
        x = np.arange(len(gap_df))

        bars = ax.bar(x, gap_df["win_rate"], color=[colors[1], colors[0]], alpha=0.85)
        ax.plot(x, gap_df["avg_b365_implied"], "o--", color="black", lw=1.5, ms=5,
                label="B365 implied")
        ax.set_xticks(x)
        ax.set_xticklabels(lbls)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylabel("Favourite win rate")
        ax.set_title("Value gap — does B365 shade affect outcomes?")
        ax.legend()
    save_fig(fig, "value_gap_summary")

def plot_gap_by_outcome(gap_outcome_df):
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

        for ax, outcome, c in zip(axes, ["H", "D", "A"], colors[:3]):
            sub = gap_outcome_df[gap_outcome_df["outcome"] == outcome]
            x = np.arange(len(sub))
            ax.bar(x, sub["win_rate"], color=c, alpha=0.85)
            ax.plot(x, sub["avg_implied"], "o--", color="black", lw=1.5, ms=5)
            ax.set_xticks(x)
            lbls = sub["high_gap"].map({True: "High gap", False: "Normal"})
            ax.set_xticklabels(lbls)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

            titles = {"H": "Home win", "D": "Draw", "A": "Away win"}
            ax.set_title(titles[outcome])

        axes[0].set_ylabel("Actual win rate")
        fig.suptitle("Value gap breakdown by outcome", y=1.01)
    save_fig(fig, "gap_by_outcome")

def plot_roc_curves(roc_curves, auc_df):
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

        for (name, (fpr, tpr)), c in zip(roc_curves.items(), colors):
            auc_val = auc_df.loc[auc_df["model"] == name, "roc_auc"].values[0]
            ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={auc_val:.3f})")

        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves — value gap classifier")
        ax.legend()
    save_fig(fig, "roc_curves")

def plot_pr_curves(pr_curves, ap_df):
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7, 6))

        for (name, (prec, rec)), c in zip(pr_curves.items(), colors):
            ap = ap_df.loc[ap_df["model"] == name, "avg_precision"].values[0]
            ax.plot(rec, prec, color=c, lw=2, label=f"{name} (AP={ap:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-recall curves — value gap classifier")
        ax.legend()
    save_fig(fig, "pr_curves")

def plot_feature_importance(imp_df):
    with plt.rc_context(style):
        models = imp_df["model"].unique()
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
        if len(models) == 1:
            axes = [axes]

        for ax, model, c in zip(axes, models, colors):
            sub = imp_df[imp_df["model"] == model].sort_values("importance")
            ax.barh(sub["feature"], sub["importance"], color=c, alpha=0.85)
            ax.set_title(model)
            ax.set_xlabel("Importance")

        fig.suptitle("Feature importance (RF + XGB)", y=1.01)
    save_fig(fig, "feature_importance")

def plot_model_calibration(cal_data):
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")

        for (name, (mean_pred, frac_pos)), c in zip(cal_data.items(), colors):
            ax.plot(mean_pred, frac_pos, "o-", color=c, lw=2, ms=5, label=name)

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Model calibration — value gap predictions")
        ax.legend()
    save_fig(fig, "model_calibration")
