import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Residual Aviation Stress Dashboard", layout="wide")

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

PANEL_PATH = os.path.join(BASE_DIR, "hub_daily_panel.csv")
FUTURES_PATH = os.path.join(BASE_DIR, "futures_returns.csv")
BASELINE_GLOB = os.path.join(PROJECT_DIR, "flight_csv", "OnTime_2018_*.csv")

st.write("BASE_DIR:", BASE_DIR)
st.write("PROJECT_DIR:", PROJECT_DIR)
st.write("BASELINE_GLOB:", BASELINE_GLOB)
st.write("Matched baseline files:", glob.glob(BASELINE_GLOB)[:12])

HUBS = {"ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "CLT", "LAS", "PHX", "IAH"}

# =========================================================
# HELPERS
# =========================================================
def make_stress_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stress_index"] = (
        df["avg_dep_delay"] * 0.5
        + df["cancel_rate"] * 100 * 0.3
        + df["frac_delay_30"] * 50 * 0.2
    )
    return df

def get_holiday_windows(years):
    # Major US travel-heavy holidays
    holiday_dates = [
        # New Year
        "2018-01-01", "2021-01-01", "2022-01-01", "2023-01-01",
        # Memorial Day
        "2018-05-28", "2021-05-31", "2022-05-30", "2023-05-29",
        # July 4th
        "2018-07-04", "2021-07-04", "2022-07-04", "2023-07-04",
        # Thanksgiving
        "2018-11-22", "2021-11-25", "2022-11-24", "2023-11-23",
        # Christmas
        "2018-12-25", "2021-12-25", "2022-12-25", "2023-12-25",
    ]
    holiday_dates = pd.to_datetime(holiday_dates)
    holiday_dates = holiday_dates[holiday_dates.year.isin(years)]
    return holiday_dates

def add_holiday_flags(df: pd.DataFrame, window_days: int = 3) -> pd.DataFrame:
    df = df.copy()
    holiday_dates = get_holiday_windows(df["date"].dt.year.unique())
    df["is_holiday_window"] = df["date"].apply(
        lambda x: any(abs((x - h).days) <= window_days for h in holiday_dates)
    )
    return df

@st.cache_data
def load_main_data():
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    futures = pd.read_csv(FUTURES_PATH, parse_dates=["date"])
    return panel, futures

@st.cache_data
def load_2018_baseline():
    files = sorted(glob.glob(BASELINE_GLOB))
    if not files:
        return None, "No 2018 baseline files found."

    chunks = []
    usecols = ["FL_DATE", "ORIGIN", "DEP_DELAY", "CANCELLED"]

    for f in files:
        try:
            d = pd.read_csv(f, usecols=usecols, low_memory=False)
            d = d[d["ORIGIN"].isin(HUBS)].copy()
            d["FL_DATE"] = pd.to_datetime(d["FL_DATE"], errors="coerce")
            d = d.dropna(subset=["FL_DATE"])
            d["DEP_DELAY"] = pd.to_numeric(d["DEP_DELAY"], errors="coerce")
            d["CANCELLED"] = pd.to_numeric(d["CANCELLED"], errors="coerce").fillna(0)

            # Non-cancelled flights missing delay -> fill as 0
            non_cancelled = d["CANCELLED"] == 0
            d.loc[non_cancelled, "DEP_DELAY"] = d.loc[non_cancelled, "DEP_DELAY"].fillna(0)

            d["delay30"] = (d["DEP_DELAY"] >= 30).astype(int)
            chunks.append(d)
        except Exception:
            continue

    if not chunks:
        return None, "2018 files were found but could not be parsed."

    raw = pd.concat(chunks, ignore_index=True)

    daily = (
        raw.groupby(["FL_DATE", "ORIGIN"], as_index=False)
        .agg(
            n_flights=("DEP_DELAY", "size"),
            n_cancelled=("CANCELLED", "sum"),
            avg_dep_delay=("DEP_DELAY", "mean"),
            frac_delay_30=("delay30", "mean"),
        )
        .rename(columns={"FL_DATE": "date", "ORIGIN": "hub"})
    )

    daily["cancel_rate"] = daily["n_cancelled"] / daily["n_flights"]

    daily = make_stress_index(daily)

    # Build expected stress by day-of-year and hub
    daily["dayofyear"] = daily["date"].dt.dayofyear

    baseline_by_hub = (
        daily.groupby(["hub", "dayofyear"], as_index=False)["stress_index"]
        .mean()
        .rename(columns={"stress_index": "expected_stress_hub"})
    )

    baseline_global = (
        daily.groupby("dayofyear", as_index=False)["stress_index"]
        .mean()
        .rename(columns={"stress_index": "expected_stress_global"})
    )

    return {
        "daily_2018": daily,
        "baseline_by_hub": baseline_by_hub,
        "baseline_global": baseline_global,
    }, None

def merge_expected_stress(panel: pd.DataFrame, baseline_dict):
    panel = panel.copy()
    panel["dayofyear"] = panel["date"].dt.dayofyear

    if baseline_dict is None:
        panel["expected_stress"] = np.nan
        panel["unexplained_stress"] = np.nan
        return panel

    baseline_by_hub = baseline_dict["baseline_by_hub"]
    baseline_global = baseline_dict["baseline_global"]

    panel = panel.merge(baseline_by_hub, on=["hub", "dayofyear"], how="left")
    panel = panel.merge(baseline_global, on="dayofyear", how="left")

    # Prefer hub-specific baseline, fall back to global day-of-year baseline
    panel["expected_stress"] = panel["expected_stress_hub"].fillna(panel["expected_stress_global"])
    panel["unexplained_stress"] = panel["stress_index"] - panel["expected_stress"]

    return panel

def compute_lag_corr(series_x, series_y, max_lag=14):
    lags = list(range(0, max_lag + 1))
    corrs = []
    for lag in lags:
        tmp = pd.DataFrame({
            "x": series_x.shift(lag),
            "y": series_y
        }).dropna()
        if len(tmp) > 2:
            corrs.append(tmp["x"].corr(tmp["y"]))
        else:
            corrs.append(np.nan)
    return lags, corrs

# =========================================================
# LOAD
# =========================================================
panel, futures = load_main_data()
baseline_dict, baseline_error = load_2018_baseline()

# =========================================================
# FEATURE ENGINEERING
# =========================================================
panel = make_stress_index(panel)
panel = add_holiday_flags(panel, window_days=3)
panel = merge_expected_stress(panel, baseline_dict)

# Aggregate national daily values
daily = (
    panel.groupby("date", as_index=False)
    .agg(
        stress_index=("stress_index", "mean"),
        expected_stress=("expected_stress", "mean"),
        unexplained_stress=("unexplained_stress", "mean"),
        total_flights=("n_flights", "sum"),
        avg_cancel_rate=("cancel_rate", "mean"),
        avg_dep_delay=("avg_dep_delay", "mean"),
        holiday_share=("is_holiday_window", "mean"),
    )
    .sort_values("date")
)

daily["is_holiday_window"] = daily["holiday_share"] > 0

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Controls")

market_col = st.sidebar.selectbox(
    "Choose market series",
    ["ES_return", "ZN_return", "CL_return"]
)

signal_col = st.sidebar.selectbox(
    "Choose aviation signal",
    ["unexplained_stress", "stress_index", "expected_stress"]
)

lag_days = st.sidebar.slider("Lag (days)", 0, 14, 3)
vol_window = st.sidebar.slider("Volatility window", 3, 20, 5)

futures = futures.copy().sort_values("date")
futures["volatility"] = futures[market_col].rolling(vol_window).std()

df = daily.merge(
    futures[["date", market_col, "volatility"]],
    on="date",
    how="inner"
).sort_values("date")

df["signal_lag"] = df[signal_col].shift(lag_days)

# =========================================================
# HEADER
# =========================================================
st.title("✈️ Residual Aviation Stress vs Market Behaviour")
st.caption(
    "Real BTS hub data + real futures returns. "
    "Expected stress is learned from 2018 seasonality. "
    "Unexplained stress = observed stress − expected stress."
)

if baseline_error:
    st.warning(baseline_error)

# =========================================================
# METRICS
# =========================================================
valid = df[["signal_lag", "volatility"]].dropna()
corr_val = valid["signal_lag"].corr(valid["volatility"]) if len(valid) > 2 else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Merged rows", len(df))
c2.metric("Market", market_col)
c3.metric("Signal", signal_col)
c4.metric("Lagged corr", f"{corr_val:.3f}" if pd.notna(corr_val) else "N/A")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Expected vs Unexplained",
    "Lag Analysis",
    "Hubs",
    "Holiday Effect",
])

# =========================================================
# TAB 1
# =========================================================
with tab1:
    st.subheader("Signal vs Market Volatility")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df[signal_col], label=signal_col)
    ax.plot(df["date"], df["volatility"], label=f"{market_col} volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.write(
        f"This compares **{signal_col}** with **{market_col} volatility**. "
        f"The goal is to see whether aviation-system disruption aligns with later market instability."
    )

# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.subheader("Observed Stress vs Expected Stress vs Unexplained Stress")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["date"], daily["stress_index"], label="Observed stress")
    ax.plot(daily["date"], daily["expected_stress"], label="Expected stress (2018 baseline)")
    ax.plot(daily["date"], daily["unexplained_stress"], label="Unexplained stress")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.write(
        "**Unexplained stress** is the part of aviation disruption that exceeds normal seasonal patterns. "
        "This is your main novelty signal."
    )

# =========================================================
# TAB 3
# =========================================================
with tab3:
    st.subheader("Lag Correlation")

    lags, corrs = compute_lag_corr(df[signal_col], df["volatility"], max_lag=14)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags, corrs, marker="o")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Correlation with market volatility")
    ax.set_title(f"{signal_col} leading {market_col} volatility")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    if not np.all(np.isnan(corrs)):
        best_idx = int(np.nanargmax(np.abs(corrs)))
        best_lag = lags[best_idx]
        best_corr = corrs[best_idx]
        st.write(f"**Best lag:** {best_lag} days")
        st.write(f"**Correlation at best lag:** {best_corr:.3f}")

# =========================================================
# TAB 4
# =========================================================
with tab4:
    st.subheader("Hub-Level Unexplained Stress")

    hub_signal = (
        panel.groupby("hub", as_index=False)["unexplained_stress"]
        .mean()
        .sort_values("unexplained_stress")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(hub_signal["hub"], hub_signal["unexplained_stress"])
    ax.set_xlabel("Average Unexplained Stress")
    ax.set_ylabel("Hub")
    ax.grid(True, axis="x", alpha=0.3)
    st.pyplot(fig)

    st.subheader("Average Daily Flights by Hub")

    hub_flights = (
        panel.groupby("hub", as_index=False)["n_flights"]
        .mean()
        .sort_values("n_flights")
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(hub_flights["hub"], hub_flights["n_flights"])
    ax2.set_xlabel("Average Daily Flights")
    ax2.set_ylabel("Hub")
    ax2.grid(True, axis="x", alpha=0.3)
    st.pyplot(fig2)

# =========================================================
# TAB 5
# =========================================================
with tab5:
    st.subheader("Holiday vs Normal Stress")

    holiday_df = daily[["date", "stress_index", "unexplained_stress", "is_holiday_window"]].copy()

    holiday_mean = holiday_df[holiday_df["is_holiday_window"]]["stress_index"].mean()
    normal_mean = holiday_df[~holiday_df["is_holiday_window"]]["stress_index"].mean()

    col1, col2 = st.columns(2)
    col1.metric("Holiday-window avg stress", f"{holiday_mean:.2f}" if pd.notna(holiday_mean) else "N/A")
    col2.metric("Normal-day avg stress", f"{normal_mean:.2f}" if pd.notna(normal_mean) else "N/A")

    fig, ax = plt.subplots(figsize=(8, 4))
    holiday_vals = holiday_df[holiday_df["is_holiday_window"]]["stress_index"].dropna()
    normal_vals = holiday_df[~holiday_df["is_holiday_window"]]["stress_index"].dropna()

    ax.boxplot([normal_vals, holiday_vals], labels=["Normal", "Holiday window"])
    ax.set_ylabel("Observed Stress")
    ax.set_title("Holiday stress deviation")
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig)

    st.write(
        "This helps separate **predictable seasonal congestion** from **abnormal disruption**. "
        "If markets respond more to unexplained stress than holiday stress, that strengthens your hypothesis."
    )

# =========================================================
# DEBUG / PREVIEW
# =========================================================
with st.expander("Show merged daily preview"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Show futures columns"):
    st.write(futures.columns.tolist())

with st.expander("Show baseline status"):
    if baseline_dict is None:
        st.write("No baseline loaded.")
    else:
        st.write("2018 baseline loaded successfully.")
        st.write("Files used:", len(glob.glob(BASELINE_GLOB)))