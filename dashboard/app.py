import os
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Bookie Audit", layout="wide")

plots_dir = "output/plots"
results_dir = "output/results"

def load_img(name):
    p = os.path.join(plots_dir, f"{name}.png")
    if os.path.exists(p):
        return Image.open(p)
    return None

def load_csv(name):
    p = os.path.join(results_dir, f"{name}.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


st.title("Premier League betting market audit")
st.caption("B365 vs Pinnacle vs Market Max — seasons 2017-18 → 2024-25")

tab1, tab2, tab3, tab4 = st.tabs(["Calibration", "Line movement", "Value gap", "ML model"])

# ---- calibration tab ----
with tab1:
    st.subheader("Implied probability calibration")
    pic = load_img("calibration_curves")
    if pic:
        st.image(pic, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Brier score by season")
        pic = load_img("brier_by_season")
        if pic:
            st.image(pic, use_container_width=True)
        tmp = load_csv("brier_by_season")
        if tmp is not None:
            # TODO: maybe add more columns here later
            st.dataframe(tmp.style.format({"brier_h": "{:.4f}", "brier_d": "{:.4f}",
                                           "brier_a": "{:.4f}", "brier_ps_h": "{:.4f}"}),
                         use_container_width=True)

    with col2:
        st.subheader("Heavy favourite accuracy")
        pic = load_img("favorite_accuracy")
        if pic:
            st.image(pic, use_container_width=True)
        tmp = load_csv("favorite_accuracy")
        if tmp is not None:
            st.dataframe(tmp, use_container_width=True)

# ---- line movement ----
with tab2:
    st.subheader("Line movement categories")
    col1, col2 = st.columns(2)
    with col1:
        pic = load_img("movement_win_rates")
        if pic: st.image(pic, use_container_width=True)
        tmp = load_csv("movement_win_rates")
        if tmp is not None:
            st.dataframe(tmp, use_container_width=True)

    with col2:
        st.subheader("Steamed favourites — actual vs implied")
        pic = load_img("steamed_vs_implied")
        if pic: st.image(pic, use_container_width=True)
        tmp = load_csv("steamed_vs_implied")
        if tmp is not None:
            st.dataframe(tmp, use_container_width=True)

    st.subheader("Movement category counts by season")
    mov_df = load_csv("movement_by_season")
    if mov_df is not None:
        pivot = mov_df.pivot(index="season", columns="movement_cat", values="n")
        pivot = pivot.fillna(0).astype(int)
        st.dataframe(pivot, use_container_width=True)

# ---- value gap stuff ----
with tab3:
    st.subheader("B365 vs market max — value gap")
    col1, col2 = st.columns(2)
    with col1:
        pic = load_img("value_gap_summary")
        if pic: st.image(pic, use_container_width=True)
        tmp = load_csv("gap_summary")
        if tmp is not None:
            st.dataframe(tmp, use_container_width=True)

    with col2:
        pic = load_img("gap_by_outcome")
        if pic: st.image(pic, use_container_width=True)
        tmp = load_csv("gap_by_outcome")
        if tmp is not None:
            st.dataframe(tmp, use_container_width=True)

    st.subheader("Gap prevalence by season")
    tmp = load_csv("gap_by_season")
    if tmp is not None:
        st.dataframe(tmp, use_container_width=True)

    st.subheader("Gap distribution (raw)")
    gap_df = load_csv("gap_distribution")
    if gap_df is not None:
        desc = gap_df.groupby("outcome")["gap"].describe()
        st.dataframe(desc.round(4), use_container_width=True)

# ---- ml model results ----
with tab4:
    st.subheader("Value gap classifier — model comparison")
    metrics = load_csv("model_metrics")
    if metrics is not None:
        st.dataframe(metrics.style.format({
            "roc_auc": "{:.4f}",
            "avg_precision": "{:.4f}"
        }), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        pic = load_img("roc_curves")
        if pic: st.image(pic, use_container_width=True)
    with col2:
        pic = load_img("pr_curves")
        if pic: st.image(pic, use_container_width=True)

    col1, col2 = st.columns(2)  # reusing names lol
    with col1:
        pic = load_img("feature_importance")
        if pic: st.image(pic, use_container_width=True)
        tmp = load_csv("feature_importance")
        if tmp is not None:
            st.dataframe(tmp, use_container_width=True)
    with col2:
        pic = load_img("model_calibration")
        if pic: st.image(pic, use_container_width=True)
