import os
import glob
import numpy as np
import pandas as pd

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = BASE_DIR

PANEL_CANDIDATES = [
    os.path.join(BASE_DIR, "hub_daily_panel.csv"),
    os.path.join(PROJECT_DIR, "data", "processed", "hub_daily_panel.csv"),
    os.path.join(PROJECT_DIR, "hub_daily_panel.csv"),
]

FUTURES_CANDIDATES = [
    os.path.join(BASE_DIR, "futures_returns.csv"),
    os.path.join(PROJECT_DIR, "data", "processed", "futures_returns.csv"),
    os.path.join(PROJECT_DIR, "futures_returns.csv"),
]

RAW_PATTERNS = [
    os.path.join(PROJECT_DIR, "flight_csv", "OnTime_2018_*.csv"),
    os.path.join(PROJECT_DIR, "flight_csv", "OnTime_2019_*.csv"),
    os.path.join(PROJECT_DIR, "flight_csv", "OnTime_2020_*.csv"),
]


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


PANEL_PATH = find_first_existing(PANEL_CANDIDATES)
FUTURES_PATH = find_first_existing(FUTURES_CANDIDATES)

RAW_FILES = []
for pattern in RAW_PATTERNS:
    RAW_FILES.extend(glob.glob(pattern))
RAW_FILES = sorted(set(RAW_FILES))

HUBS = {"ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "CLT", "LAS", "PHX", "IAH"}

# =========================================================
# CORE HELPERS
# =========================================================
def make_stress_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stress_index"] = (
        df["avg_dep_delay"] * 0.5
        + df["cancel_rate"] * 100 * 0.3
        + df["frac_delay_30"] * 50 * 0.2
    )
    return df


def load_main_data():
    if PANEL_PATH is None:
        raise FileNotFoundError("Could not find hub_daily_panel.csv")
    if FUTURES_PATH is None:
        raise FileNotFoundError("Could not find futures_returns.csv")

    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    futures = pd.read_csv(FUTURES_PATH, parse_dates=["date"])
    return panel, futures


def load_monthly_baseline():
    if not RAW_FILES:
        return None

    chunks = []
    usecols = ["FL_DATE", "ORIGIN", "DEP_DELAY", "CANCELLED"]

    for f in RAW_FILES:
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

    if not chunks:
        return None

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

    baseline_by_hub_month = (
        daily.groupby(["hub", "month"], as_index=False)["stress_index"]
        .mean()
        .rename(columns={"stress_index": "expected_stress_hub_month"})
    )

    baseline_global_month = (
        daily.groupby("month", as_index=False)["stress_index"]
        .mean()
        .rename(columns={"stress_index": "expected_stress_global_month"})
    )

    return {
        "baseline_by_hub_month": baseline_by_hub_month,
        "baseline_global_month": baseline_global_month,
        "n_files": len(RAW_FILES),
    }


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


def build_daily_signal():
    panel, futures = load_main_data()

    panel = make_stress_index(panel)
    baseline_dict = load_monthly_baseline()
    panel = merge_expected_stress_monthly(panel, baseline_dict)

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

    return daily, futures, baseline_dict


# =========================================================
# BACKTEST HELPERS
# =========================================================
def make_zscore_signal(series: pd.Series, lookback: int = 20):
    roll_mean = series.rolling(lookback).mean()
    roll_std = series.rolling(lookback).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    return z


def max_drawdown(cum_returns: pd.Series):
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1.0
    return drawdown.min()


def run_strategy(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str,
    lag: int = 1,
    lookback: int = 20,
    threshold: float = 1.0,
    mode: str = "long_short"
):
    out = df.copy().sort_values("date").reset_index(drop=True)

    # signal known at t, trade next day after lag
    out["signal_raw"] = out[signal_col].shift(lag)
    out["signal_z"] = make_zscore_signal(out["signal_raw"], lookback=lookback)

    if mode == "long_short":
        out["position"] = np.where(
            out["signal_z"] > threshold, -1,
            np.where(out["signal_z"] < -threshold, 1, 0)
        )
    elif mode == "long_only":
        out["position"] = np.where(out["signal_z"] < -threshold, 1, 0)
    elif mode == "short_only":
        out["position"] = np.where(out["signal_z"] > threshold, -1, 0)
    else:
        raise ValueError("mode must be: long_short, long_only, short_only")

    out["strategy_return"] = out["position"] * out[return_col]
    out["cum_strategy"] = (1 + out["strategy_return"].fillna(0)).cumprod()
    out["cum_buyhold"] = (1 + out[return_col].fillna(0)).cumprod()

    valid = out["strategy_return"].dropna()
    mean_ret = valid.mean()
    std_ret = valid.std()

    sharpe = np.nan
    if std_ret and not np.isnan(std_ret) and std_ret != 0:
        sharpe = np.sqrt(252) * mean_ret / std_ret

    hit_rate = np.nan
    active = out[out["position"] != 0]["strategy_return"]
    if len(active) > 0:
        hit_rate = (active > 0).mean()

    results = {
        "rows": len(out),
        "trades": int((out["position"].diff().fillna(0) != 0).sum()),
        "active_days": int((out["position"] != 0).sum()),
        "avg_daily_return": mean_ret,
        "vol_daily": std_ret,
        "sharpe": sharpe,
        "cum_return": out["cum_strategy"].iloc[-1] - 1,
        "buy_hold_return": out["cum_buyhold"].iloc[-1] - 1,
        "max_drawdown": max_drawdown(out["cum_strategy"]),
        "hit_rate": hit_rate,
    }

    return out, results


def grid_search(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str,
    lags=(1, 2, 3, 5, 7),
    lookbacks=(10, 20, 30),
    thresholds=(0.5, 1.0, 1.5),
    modes=("long_short", "long_only", "short_only"),
):
    rows = []

    for lag in lags:
        for lookback in lookbacks:
            for threshold in thresholds:
                for mode in modes:
                    try:
                        _, res = run_strategy(
                            df=df,
                            signal_col=signal_col,
                            return_col=return_col,
                            lag=lag,
                            lookback=lookback,
                            threshold=threshold,
                            mode=mode,
                        )
                        rows.append({
                            "signal": signal_col,
                            "market": return_col,
                            "lag": lag,
                            "lookback": lookback,
                            "threshold": threshold,
                            "mode": mode,
                            **res
                        })
                    except Exception:
                        continue

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["sharpe", "cum_return"], ascending=False)
    return results


# =========================================================
# MAIN
# =========================================================
def main():
    daily, futures, baseline_dict = build_daily_signal()

    print("=" * 80)
    print("BACKTEST: RESIDUAL AVIATION STRESS")
    print("=" * 80)
    print(f"PANEL_PATH   : {PANEL_PATH}")
    print(f"FUTURES_PATH : {FUTURES_PATH}")
    print(f"RAW FILES    : {len(RAW_FILES)}")
    if baseline_dict is not None:
        print(f"BASELINE FILES USED: {baseline_dict['n_files']}")
    else:
        print("BASELINE FILES USED: 0 (expected/unexplained stress may be NaN)")

    merged = daily.merge(futures, on="date", how="inner").sort_values("date")

    signal_cols = ["stress_index", "expected_stress", "unexplained_stress"]
    market_cols = [c for c in ["ES_return", "ZN_return", "CL_return"] if c in merged.columns]

    print("\nAvailable signals:", signal_cols)
    print("Available markets:", market_cols)

    output_dir = os.path.join(BASE_DIR, "backtest_outputs")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for signal_col in signal_cols:
        for market_col in market_cols:
            print(f"\nRunning grid search: {signal_col} vs {market_col}")

            results = grid_search(
                df=merged,
                signal_col=signal_col,
                return_col=market_col,
                lags=(1, 2, 3, 5, 7, 10),
                lookbacks=(10, 20, 40),
                thresholds=(0.5, 1.0, 1.5, 2.0),
                modes=("long_short", "long_only", "short_only"),
            )

            if results.empty:
                print("  No results.")
                continue

            all_results.append(results)

            save_name = f"grid_{signal_col}_{market_col}.csv"
            save_path = os.path.join(output_dir, save_name)
            results.to_csv(save_path, index=False)

            best = results.iloc[0]
            print("  BEST RESULT")
            print(f"    mode         : {best['mode']}")
            print(f"    lag          : {best['lag']}")
            print(f"    lookback     : {best['lookback']}")
            print(f"    threshold    : {best['threshold']}")
            print(f"    sharpe       : {best['sharpe']:.3f}")
            print(f"    cum_return   : {best['cum_return']:.3%}")
            print(f"    max_drawdown : {best['max_drawdown']:.3%}")
            print(f"    hit_rate     : {best['hit_rate']:.3%}" if pd.notna(best["hit_rate"]) else "    hit_rate     : N/A")

            best_path = os.path.join(output_dir, f"equity_{signal_col}_{market_col}.csv")
            equity_df, _ = run_strategy(
                df=merged,
                signal_col=signal_col,
                return_col=market_col,
                lag=int(best["lag"]),
                lookback=int(best["lookback"]),
                threshold=float(best["threshold"]),
                mode=best["mode"],
            )
            equity_df.to_csv(best_path, index=False)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values(["sharpe", "cum_return"], ascending=False)
        combined.to_csv(os.path.join(output_dir, "all_backtests_ranked.csv"), index=False)

        print("\n" + "=" * 80)
        print("TOP 15 BACKTESTS")
        print("=" * 80)
        print(
            combined[
                ["signal", "market", "mode", "lag", "lookback", "threshold", "sharpe", "cum_return", "max_drawdown", "hit_rate"]
            ].head(15).to_string(index=False)
        )

        print(f"\nSaved outputs to: {output_dir}")
    else:
        print("\nNo backtest results were produced.")


if __name__ == "__main__":
    main()