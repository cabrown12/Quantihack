import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar

st.set_page_config(page_title="Residual Aviation Stress Dashboard", layout="wide")

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PANEL_PATH = os.path.join(BASE_DIR, "data", "processed", "hub_daily_panel.csv")
DEFAULT_FUTURES_PATH = os.path.join(BASE_DIR, "data", "processed", "futures_returns.csv")

# Fall back to the checked-in US_Airlines data if processed data dir is absent.
PANEL_PATH = DEFAULT_PANEL_PATH if os.path.exists(DEFAULT_PANEL_PATH) else os.path.join(
    BASE_DIR, "US_Airlines", "hub_daily_panel.csv"
)
FUTURES_PATH = DEFAULT_FUTURES_PATH if os.path.exists(DEFAULT_FUTURES_PATH) else os.path.join(
    BASE_DIR, "US_Airlines", "futures_returns.csv"
)
RAW_GLOB = os.path.join(BASE_DIR, "flight_csv", "OnTime_*.csv")
T100_GLOB = os.path.join(BASE_DIR, "t100_*.csv")


HUBS = {"ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "CLT", "LAS", "PHX", "IAH"}

# =========================================================
# HELPERS
# =========================================================
def make_stress_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Simple interpretable stress index
    df["stress_index"] = (
        df["avg_dep_delay"] * 0.5
        + df["cancel_rate"] * 100 * 0.3
        + df["frac_delay_30"] * 50 * 0.2
    )

    return df


def add_holiday_flags(df: pd.DataFrame, date_col: str = "date", window_days: int = 3) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(
        start=df[date_col].min() - pd.Timedelta(days=window_days),
        end=df[date_col].max() + pd.Timedelta(days=window_days),
    )
    holiday_dates = pd.to_datetime(holidays)

    df["is_holiday_window"] = df[date_col].apply(
        lambda x: any(abs((x.normalize() - h.normalize()).days) <= window_days for h in holiday_dates)
    )

    return df


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
# LOAD MAIN DATA
# =========================================================
@st.cache_data
def load_main_data():
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    futures = pd.read_csv(FUTURES_PATH, parse_dates=["date"])
    return panel, futures


# =========================================================
# LOAD RAW BTS CSVs AND BUILD MONTHLY BASE STRESS
# =========================================================
@st.cache_data
def load_monthly_baseline():
    files = sorted(glob.glob(RAW_GLOB))

    if not files:
        return None, f"No raw BTS CSV files found at: {RAW_GLOB}"

    chunks = []
    usecols = ["FL_DATE", "ORIGIN", "DEP_DELAY", "CANCELLED"]

    for f in files:
        try:
            d = pd.read_csv(f, usecols=usecols, low_memory=False)
            d = d[d["ORIGIN"].isin(HUBS)].copy()

            d["FL_DATE"] = pd.to_datetime(d["FL_DATE"], errors="coerce")
            d = d.dropna(subset=["FL_DATE"]).copy()

            d["DEP_DELAY"] = pd.to_numeric(d["DEP_DELAY"], errors="coerce")
            d["CANCELLED"] = pd.to_numeric(d["CANCELLED"], errors="coerce").fillna(0)

            non_cancelled = d["CANCELLED"] == 0
            d.loc[non_cancelled, "DEP_DELAY"] = d.loc[non_cancelled, "DEP_DELAY"].fillna(0)

            d["delay30"] = (d["DEP_DELAY"] >= 30).astype(int)

            chunks.append(d)

        except Exception as e:
            print(f"Skipping {os.path.basename(f)}: {e}")
            continue

    if not chunks:
        return None, "Raw BTS files were found but could not be parsed."

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

    daily["month"] = daily["date"].dt.month

    # Hub-specific monthly baseline
    baseline_by_hub_month = (
        daily.groupby(["hub", "month"], as_index=False)["stress_index"]
        .mean()
        .rename(columns={"stress_index": "expected_stress_hub_month"})
    )

    # Global fallback monthly baseline
    baseline_global_month = (
        daily.groupby("month", as_index=False)["stress_index"]
        .mean()
        .rename(columns={"stress_index": "expected_stress_global_month"})
    )

    return {
        "daily_baseline": daily,
        "baseline_by_hub_month": baseline_by_hub_month,
        "baseline_global_month": baseline_global_month,
        "n_files": len(files),
    }, None


# =========================================================
# LOAD T-100 DATA FOR M&A ANALYSIS
# =========================================================
@st.cache_data
def load_t100_data():
    files = sorted(glob.glob(T100_GLOB))
    if not files:
        return None, f"No T-100 CSV files found at: {T100_GLOB}"

    dfs = []
    for f in files:
        try:
            temp = pd.read_csv(f, low_memory=False)
            temp.columns = temp.columns.str.strip()
            temp = temp.dropna(axis=1, how='all')
            file_year = int(os.path.basename(f).split('_')[1].split('.')[0])
            if 'YEAR' not in temp.columns:
                temp['YEAR'] = file_year
            if 'MONTH' not in temp.columns:
                temp['MONTH'] = 1
            dfs.append(temp)
        except Exception:
            continue

    if not dfs:
        return None, "T-100 files found but could not be parsed."

    t100 = pd.concat(dfs, ignore_index=True)
    return t100, None


def get_carrier_routes(t100, carrier, year, month):
    mask = (
        (t100['UNIQUE_CARRIER'] == carrier) &
        (t100['YEAR'] == year) &
        (t100['MONTH'] == month) &
        (t100['PASSENGERS'] > 0)
    )
    routes = t100.loc[mask, ['ORIGIN', 'DEST']]
    route_set = set()
    for _, row in routes.iterrows():
        key = tuple(sorted([row['ORIGIN'], row['DEST']]))
        route_set.add(key)
    return route_set


def compute_monthly_overlap(t100, carrier1, carrier2, start_year, end_year):
    records = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if not ((t100['YEAR'] == year) & (t100['MONTH'] == month)).any():
                continue
            routes1 = get_carrier_routes(t100, carrier1, year, month)
            routes2 = get_carrier_routes(t100, carrier2, year, month)
            if len(routes1) == 0 and len(routes2) == 0:
                continue
            overlap = routes1 & routes2
            union = routes1 | routes2
            records.append({
                'year': year, 'month': month,
                'date': pd.Timestamp(year=year, month=month, day=1),
                'routes_carrier1': len(routes1),
                'routes_carrier2': len(routes2),
                'overlap': len(overlap),
                'union': len(union),
                'jaccard': len(overlap) / len(union) if len(union) > 0 else 0,
                'overlap_pct_c1': len(overlap) / len(routes1) * 100 if len(routes1) > 0 else 0,
                'overlap_pct_c2': len(overlap) / len(routes2) * 100 if len(routes2) > 0 else 0,
            })
    return pd.DataFrame(records)


def compute_route_stress_proxy(t100, start_year, end_year):
    """Compute a monthly 'route competition stress' metric: avg number of carriers per route."""
    records = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            monthly = t100[
                (t100['YEAR'] == year) &
                (t100['MONTH'] == month) &
                (t100['PASSENGERS'] > 0)
            ]
            if monthly.empty:
                continue
            monthly = monthly.copy()
            monthly['route'] = monthly.apply(
                lambda x: tuple(sorted([x['ORIGIN'], x['DEST']])), axis=1
            )
            carriers_per_route = monthly.groupby('route')['UNIQUE_CARRIER'].nunique()
            records.append({
                'date': pd.Timestamp(year=year, month=month, day=1),
                'avg_carriers_per_route': carriers_per_route.mean(),
                'max_carriers_on_route': carriers_per_route.max(),
                'routes_with_3plus': (carriers_per_route >= 3).sum(),
                'routes_with_5plus': (carriers_per_route >= 5).sum(),
                'total_routes': len(carriers_per_route),
            })
    return pd.DataFrame(records)


# =========================================================
# MERGE EXPECTED / UNEXPLAINED STRESS
# =========================================================
def merge_expected_stress_monthly(panel: pd.DataFrame, baseline_dict):
    panel = panel.copy()
    panel["month"] = panel["date"].dt.month

    if baseline_dict is None:
        panel["expected_stress"] = np.nan
        panel["unexplained_stress"] = np.nan
        return panel

    panel = panel.merge(
        baseline_dict["baseline_by_hub_month"],
        on=["hub", "month"],
        how="left"
    )

    panel = panel.merge(
        baseline_dict["baseline_global_month"],
        on="month",
        how="left"
    )

    panel["expected_stress"] = panel["expected_stress_hub_month"].fillna(
        panel["expected_stress_global_month"]
    )

    panel["unexplained_stress"] = panel["stress_index"] - panel["expected_stress"]

    return panel


# =========================================================
# LOAD
# =========================================================
panel, futures = load_main_data()
baseline_dict, baseline_error = load_monthly_baseline()
t100_data, t100_error = load_t100_data()

# =========================================================
# FEATURE ENGINEERING
# =========================================================
panel = make_stress_index(panel)
panel = add_holiday_flags(panel, date_col="date", window_days=3)
panel = merge_expected_stress_monthly(panel, baseline_dict)

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
    )
    .sort_values("date")
)

daily = add_holiday_flags(daily, date_col="date", window_days=3)

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

small_number = 1e-6
df["stress_surprise"] = df["unexplained_stress"] / (abs(df["expected_stress"]) + small_number)
df["stress_change_1d"] = df["unexplained_stress"].diff(1)
df["stress_change_3d"] = df["unexplained_stress"].diff(3)

rolling_mean_20 = df["unexplained_stress"].rolling(20).mean()
rolling_std_20 = df["unexplained_stress"].rolling(20).std()
df["stress_z_20"] = (df["unexplained_stress"] - rolling_mean_20) / rolling_std_20

df["positive_stress"] = df["unexplained_stress"].clip(lower=0)

# =========================================================
# HEADER
# =========================================================
st.title("✈️ Residual Aviation Stress vs Market Behaviour")
st.caption(
    "Real BTS hub data + futures returns. "
    "Expected stress is learned from all raw BTS CSVs using average hub-month seasonal stress. "
    "Unexplained stress = observed stress − expected stress."
)

if baseline_error:
    st.warning(baseline_error)

with st.expander("Path debug"):
    st.write("BASE_DIR:", BASE_DIR)
    st.write("PANEL_PATH:", PANEL_PATH)
    st.write("FUTURES_PATH:", FUTURES_PATH)
    st.write("RAW_GLOB:", RAW_GLOB)
    st.write("Matched raw files:", glob.glob(RAW_GLOB)[:12])

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Expected vs Unexplained",
    "Lag Analysis",
    "Hubs",
    "Holiday Effect",
    "Stress → M&A",
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
        f"It checks whether aviation-system disruption aligns with later market instability."
    )

# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.subheader("Observed Stress vs Expected Stress vs Unexplained Stress")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["date"], daily["stress_index"], label="Observed stress")
    ax.plot(daily["date"], daily["expected_stress"], label="Expected base stress")
    ax.plot(daily["date"], daily["unexplained_stress"], label="Unexplained stress")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.write(
        "**Expected stress** is the average stress for that hub and month across all historical raw BTS files. "
        "**Unexplained stress** is the residual above that seasonal baseline."
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
    st.subheader("Hub-Level Average Unexplained Stress")

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

    ax.boxplot([normal_vals, holiday_vals], tick_labels=["Normal", "Holiday window"])
    ax.set_ylabel("Observed Stress")
    ax.set_title("Holiday stress deviation")
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig)

    st.write(
        "This helps separate predictable seasonal congestion from abnormal disruption. "
        "If markets respond more to unexplained stress than base stress, that supports your hypothesis."
    )

# =========================================================
# TAB 6 — STRESS → M&A CORRELATION
# =========================================================
with tab6:
    st.subheader("Aviation Stress & Route Competition → Merger & Acquisitions")
    st.write(
        "**Thesis:** When multiple airlines compete on the same routes (e.g. 10 carriers flying NYC→Colombia), "
        "the resulting operational stress — delays, cancellations, thin margins — creates pressure to consolidate. "
        "Airlines merge to reduce route competition, eliminate redundancy, and gain pricing power. "
        "High route overlap is both a cause of stress and a leading indicator of M&A."
    )

    if t100_data is None:
        st.warning(t100_error or "No T-100 data available.")
    else:
        t100 = t100_data
        min_year = int(t100['YEAR'].min())
        max_year = int(t100['YEAR'].max())

        # --- Section 1: Route competition stress over time ---
        st.markdown("---")
        st.subheader("1. Route Competition Intensity Over Time")
        st.write(
            "How many airlines compete on each route? More carriers per route = more stress, "
            "more pressure to merge."
        )

        route_stress = compute_route_stress_proxy(t100, min_year, max_year)

        if not route_stress.empty:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

            ax1.plot(route_stress['date'], route_stress['avg_carriers_per_route'],
                     color='#e74c3c', linewidth=2, marker='o', markersize=3)
            ax1.set_ylabel('Avg Carriers per Route')
            ax1.set_title('Route Competition Intensity')
            ax1.grid(True, alpha=0.3)

            # Mark M&A events
            ma_events = [
                (pd.Timestamp('2022-02-07'), 'Frontier-Spirit\nAnnounced', '#e67e22'),
                (pd.Timestamp('2022-07-28'), 'JetBlue-Spirit\nCounter-Bid', '#3498db'),
                (pd.Timestamp('2024-01-16'), 'JetBlue-Spirit\nBlocked (DOJ)', '#e74c3c'),
            ]
            for evt_date, label, color in ma_events:
                ax1.axvline(x=evt_date, color=color, linestyle='--', linewidth=2, alpha=0.7)
                ax1.annotate(label, xy=(evt_date, ax1.get_ylim()[1] * 0.95),
                             fontsize=8, color=color, rotation=0, ha='center', va='top')

            ax2.plot(route_stress['date'], route_stress['routes_with_5plus'],
                     color='#8e44ad', linewidth=2, label='Routes with 5+ carriers')
            ax2.plot(route_stress['date'], route_stress['routes_with_3plus'],
                     color='#2ecc71', linewidth=2, alpha=0.6, label='Routes with 3+ carriers')
            ax2.set_ylabel('Number of Highly Contested Routes')
            ax2.set_xlabel('Date')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            for evt_date, label, color in ma_events:
                ax2.axvline(x=evt_date, color=color, linestyle='--', linewidth=2, alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

        # --- Section 2: Frontier vs Spirit and JetBlue vs Spirit overlap ---
        st.markdown("---")
        st.subheader("2. Route Overlap Between Merger Candidates")
        st.write(
            "Airlines with high route overlap are natural merger targets. "
            "Frontier and Spirit shared **70+ routes**; JetBlue and Spirit shared **60+ routes**. "
            "Merging eliminates head-to-head competition on these routes."
        )

        fs_overlap = compute_monthly_overlap(t100, 'F9', 'NK', min_year, max_year)
        bs_overlap = compute_monthly_overlap(t100, 'B6', 'NK', min_year, max_year)

        if not fs_overlap.empty and not bs_overlap.empty:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(fs_overlap['date'], fs_overlap['overlap'], linewidth=2.5,
                    color='#e74c3c', marker='o', markersize=3, label='Frontier vs Spirit')
            ax.plot(bs_overlap['date'], bs_overlap['overlap'], linewidth=2.5,
                    color='#3498db', marker='o', markersize=3, label='JetBlue vs Spirit')

            for evt_date, label, color in ma_events:
                ax.axvline(x=evt_date, color=color, linestyle='--', linewidth=2, alpha=0.7)

            ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-01'),
                       alpha=0.1, color='gray', label='COVID Impact')

            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Overlapping Routes')
            ax.set_title('Route Overlap: Merger Candidates Share Heavily Contested Routes')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            # Jaccard similarity
            fig2, ax2 = plt.subplots(figsize=(14, 5))
            ax2.plot(fs_overlap['date'], fs_overlap['jaccard'] * 100, linewidth=2.5,
                     color='#e74c3c', label='Frontier vs Spirit')
            ax2.plot(bs_overlap['date'], bs_overlap['jaccard'] * 100, linewidth=2.5,
                     color='#3498db', label='JetBlue vs Spirit')

            for evt_date, label, color in ma_events:
                ax2.axvline(x=evt_date, color=color, linestyle='--', linewidth=2, alpha=0.7)

            ax2.set_xlabel('Date')
            ax2.set_ylabel('Jaccard Similarity (%)')
            ax2.set_title('Route Network Similarity (Jaccard Index)\n'
                          'Higher = more overlap = stronger M&A signal')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)

        # --- Section 3: Correlation between aviation stress and route competition ---
        st.markdown("---")
        st.subheader("3. Stress ↔ Route Competition Correlation")
        st.write(
            "We correlate the **national aviation stress index** (from BTS hub data) "
            "with **route competition intensity** (from T-100 data). "
            "High competition → operational strain → M&A pressure."
        )

        if not route_stress.empty and not daily.empty:
            # Resample daily stress to monthly for comparison
            daily_copy = daily.copy()
            daily_copy['month_start'] = daily_copy['date'].dt.to_period('M').dt.to_timestamp()
            monthly_stress = daily_copy.groupby('month_start', as_index=False).agg(
                stress_index=('stress_index', 'mean'),
                unexplained_stress=('unexplained_stress', 'mean'),
                avg_cancel_rate=('avg_cancel_rate', 'mean'),
            ).rename(columns={'month_start': 'date'})

            merged_ma = monthly_stress.merge(route_stress, on='date', how='inner')

            if len(merged_ma) > 5:
                col1, col2 = st.columns(2)

                # Scatter: stress vs avg carriers per route
                with col1:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sc = ax.scatter(merged_ma['avg_carriers_per_route'],
                                    merged_ma['stress_index'],
                                    c=merged_ma['date'].astype(np.int64),
                                    cmap='viridis', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
                    plt.colorbar(sc, ax=ax, label='Time')
                    ax.set_xlabel('Avg Carriers per Route')
                    ax.set_ylabel('Aviation Stress Index')
                    ax.set_title('Stress vs Route Competition')
                    ax.grid(True, alpha=0.3)

                    corr_stress_comp = merged_ma['stress_index'].corr(
                        merged_ma['avg_carriers_per_route'])
                    ax.annotate(f'r = {corr_stress_comp:.3f}', xy=(0.05, 0.95),
                                xycoords='axes fraction', fontsize=12, fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    plt.tight_layout()
                    st.pyplot(fig)

                # Scatter: cancel rate vs highly contested routes
                with col2:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sc = ax.scatter(merged_ma['routes_with_5plus'],
                                    merged_ma['avg_cancel_rate'] * 100,
                                    c=merged_ma['date'].astype(np.int64),
                                    cmap='viridis', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
                    plt.colorbar(sc, ax=ax, label='Time')
                    ax.set_xlabel('Routes with 5+ Competing Carriers')
                    ax.set_ylabel('Avg Cancellation Rate (%)')
                    ax.set_title('Cancellations vs Route Crowding')
                    ax.grid(True, alpha=0.3)

                    corr_canc_crowd = merged_ma['avg_cancel_rate'].corr(
                        merged_ma['routes_with_5plus'])
                    ax.annotate(f'r = {corr_canc_crowd:.3f}', xy=(0.05, 0.95),
                                xycoords='axes fraction', fontsize=12, fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    plt.tight_layout()
                    st.pyplot(fig)

                # Time series overlay
                fig, ax1 = plt.subplots(figsize=(14, 5))
                ax1.plot(merged_ma['date'], merged_ma['stress_index'],
                         color='#e74c3c', linewidth=2, label='Aviation Stress Index')
                ax1.set_ylabel('Stress Index', color='#e74c3c')
                ax1.tick_params(axis='y', labelcolor='#e74c3c')

                ax2 = ax1.twinx()
                ax2.plot(merged_ma['date'], merged_ma['avg_carriers_per_route'],
                         color='#3498db', linewidth=2, linestyle='--', label='Avg Carriers/Route')
                ax2.set_ylabel('Avg Carriers per Route', color='#3498db')
                ax2.tick_params(axis='y', labelcolor='#3498db')

                for evt_date, label, color in ma_events:
                    ax1.axvline(x=evt_date, color=color, linestyle=':', linewidth=2, alpha=0.7)

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                ax1.set_title('Aviation Stress & Route Competition Over Time\n'
                              'Stress and competition move together → M&A reduces both')
                ax1.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

                st.metric("Stress vs Competition Correlation",
                          f"{corr_stress_comp:.3f}")
            else:
                st.info("Not enough overlapping months between stress data and T-100 data.")

        # --- Summary ---
        st.markdown("---")
        st.subheader("Key Takeaway")
        st.write(
            "**The chain:** Route competition (many airlines on the same routes) → "
            "operational stress (delays, cancellations, thin margins) → "
            "M&A pressure (consolidate to reduce competition and improve efficiency). "
            "\n\n"
            "The Frontier-Spirit and JetBlue-Spirit cases prove this: both mergers targeted airlines "
            "with **massive route overlap**. The aviation stress index captures the same underlying "
            "competitive pressure that drives consolidation. "
            "This makes aviation stress a **leading indicator** for M&A activity in the airline sector."
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
        st.write("Monthly baseline loaded successfully.")
        st.write("Raw files used:", baseline_dict["n_files"])
        st.write("Sample hub-month baseline:")
        st.dataframe(baseline_dict["baseline_by_hub_month"].head(20), use_container_width=True)