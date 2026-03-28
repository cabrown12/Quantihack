import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ====================================
# CONFIG
# ====================================

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed")

HUB_AIRPORTS = [
    'ATL', 'ORD', 'DFW', 'DEN', 'LAX',
    'JFK', 'CLT', 'LAS', 'PHX', 'IAH'
]

KEEP_COLUMNS = [
    'FL_DATE', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST',
    'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'ARR_DELAY',
    'CANCELLED', 'DIVERTED',
    'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY',
    'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'DISTANCE',
]

COLUMN_NAME_MAP = {
    'FlightDate': 'FL_DATE',
    'FL_DATE': 'FL_DATE',
    'Reporting_Airline': 'OP_UNIQUE_CARRIER',
    'REPORTING_AIRLINE': 'OP_UNIQUE_CARRIER',
    'OP_UNIQUE_CARRIER': 'OP_UNIQUE_CARRIER',
    'UNIQUE_CARRIER': 'OP_UNIQUE_CARRIER',
    'Origin': 'ORIGIN',
    'ORIGIN': 'ORIGIN',
    'Dest': 'DEST',
    'DEST': 'DEST',
    'CRSDepTime': 'CRS_DEP_TIME',
    'CRS_DEP_TIME': 'CRS_DEP_TIME',
    'DepTime': 'DEP_TIME',
    'DEP_TIME': 'DEP_TIME',
    'DepDelay': 'DEP_DELAY',
    'DEP_DELAY': 'DEP_DELAY',
    'ArrDelay': 'ARR_DELAY',
    'ARR_DELAY': 'ARR_DELAY',
    'Cancelled': 'CANCELLED',
    'CANCELLED': 'CANCELLED',
    'Diverted': 'DIVERTED',
    'DIVERTED': 'DIVERTED',
    'CarrierDelay': 'CARRIER_DELAY',
    'CARRIER_DELAY': 'CARRIER_DELAY',
    'WeatherDelay': 'WEATHER_DELAY',
    'WEATHER_DELAY': 'WEATHER_DELAY',
    'NASDelay': 'NAS_DELAY',
    'NAS_DELAY': 'NAS_DELAY',
    'SecurityDelay': 'SECURITY_DELAY',
    'SECURITY_DELAY': 'SECURITY_DELAY',
    'LateAircraftDelay': 'LATE_AIRCRAFT_DELAY',
    'LATE_AIRCRAFT_DELAY': 'LATE_AIRCRAFT_DELAY',
    'Distance': 'DISTANCE',
    'DISTANCE': 'DISTANCE',
}


# ====================================
# HELPERS
# ====================================

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


# ====================================
# LOAD ALL MONTHLY CSVS
# ====================================

def load_flights_from_csvs(folderpath=DATA_DIR):
    folder = Path(folderpath)
    csv_files = sorted(folder.glob("OnTime_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No OnTime_*.csv files found in {folder.resolve()}")

    frames = []
    for fp in csv_files:
        print(f"Loading {fp.name}...")
        df = pd.read_csv(fp, low_memory=False)

        # Drop junk columns
        df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")
        df.columns = df.columns.str.strip().str.strip('"')

        rename = {c: COLUMN_NAME_MAP[c] for c in df.columns if c in COLUMN_NAME_MAP}
        if rename:
            df = df.rename(columns=rename)

        available = [c for c in KEEP_COLUMNS if c in df.columns]
        if len(available) < 3:
            print(f"  Skipping {fp.name}: too few matched columns")
            continue

        df = df[available].copy()

        # Date cleanup
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
        df = df.dropna(subset=["FL_DATE"]).copy()

        # Filter to relevant hubs early
        if "DEST" in df.columns:
            mask = df["ORIGIN"].isin(HUB_AIRPORTS) | df["DEST"].isin(HUB_AIRPORTS)
        else:
            mask = df["ORIGIN"].isin(HUB_AIRPORTS)
        df = df[mask].copy()

        frames.append(df)

    if not frames:
        raise ValueError("No usable flight data loaded from CSVs.")

    flights = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(csv_files)} files, total rows: {len(flights):,}")
    return flights


# ====================================
# CLEAN FLIGHTS
# ====================================

def clean_flights(df):
    df = df.copy()

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df = df.dropna(subset=["FL_DATE"]).copy()

    for col in [
        "DEP_DELAY", "ARR_DELAY", "CARRIER_DELAY", "WEATHER_DELAY",
        "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["CANCELLED", "DIVERTED"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    non_cancelled = df["CANCELLED"] == 0
    for col in ["DEP_DELAY", "ARR_DELAY"]:
        if col in df.columns:
            df.loc[non_cancelled, col] = df.loc[non_cancelled, col].fillna(0)

    if "DEP_DELAY" in df.columns:
        df["DEP_DELAY_30"] = (df["DEP_DELAY"] >= 30).astype(int)
        df["DEP_DELAY_60"] = (df["DEP_DELAY"] >= 60).astype(int)

    return df


# ====================================
# AGGREGATE HUB x DATE
# ====================================

def aggregate_hub_daily(df, hub_list=HUB_AIRPORTS):
    hub_df = df[df["ORIGIN"].isin(set(hub_list))].copy()
    if hub_df.empty:
        return pd.DataFrame()

    records = []

    for (hub, dt), grp in hub_df.groupby(["ORIGIN", "FL_DATE"]):
        n = len(grp)
        n_canc = int(grp["CANCELLED"].sum())
        n_div = int(grp["DIVERTED"].sum()) if "DIVERTED" in grp.columns else 0
        ops = grp[grp["CANCELLED"] == 0]
        n_ops = len(ops)

        rec = {
            "date": dt,
            "hub": hub,
            "n_flights": n,
            "n_operating": n_ops,
            "n_cancelled": n_canc,
            "n_diverted": n_div,
            "cancel_rate": n_canc / n if n > 0 else np.nan,
            "divert_rate": n_div / n if n > 0 else np.nan,
        }

        if n_ops > 0:
            for col in ["DEP_DELAY", "ARR_DELAY"]:
                if col in ops.columns:
                    rec[f"avg_{col.lower()}"] = ops[col].mean()

            if "DEP_DELAY_30" in ops.columns:
                rec["frac_delay_30"] = ops["DEP_DELAY_30"].mean()
                rec["frac_delay_60"] = ops["DEP_DELAY_60"].mean()

            for cause in ["WEATHER_DELAY", "NAS_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY"]:
                if cause in ops.columns:
                    vals = ops[cause].fillna(0)
                    rec[f"avg_{cause.lower()}"] = vals.mean()
                    rec[f"pct_{cause.lower()}"] = (vals > 0).mean()
        else:
            rec.update({
                "avg_dep_delay": np.nan,
                "avg_arr_delay": np.nan,
                "frac_delay_30": np.nan,
                "frac_delay_60": np.nan,
                "avg_weather_delay": np.nan,
                "avg_nas_delay": np.nan,
                "avg_carrier_delay": np.nan,
                "avg_late_aircraft_delay": np.nan,
                "pct_weather_delay": np.nan,
                "pct_nas_delay": np.nan,
                "pct_carrier_delay": np.nan,
                "pct_late_aircraft_delay": np.nan,
            })

        records.append(rec)

    panel = pd.DataFrame(records)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["hub", "date"]).reset_index(drop=True)


# ====================================
# BUILD TOTAL STRESS
# ====================================

def build_total_stress(panel):
    panel = panel.copy()

    needed = [
        "avg_dep_delay",
        "avg_arr_delay",
        "cancel_rate",
        "divert_rate",
        "frac_delay_30",
        "frac_delay_60",
        "pct_nas_delay",
        "pct_late_aircraft_delay",
        "pct_weather_delay",
    ]

    for col in needed:
        if col not in panel.columns:
            panel[col] = np.nan

    # Global z-scores
    panel["z_avg_dep_delay"] = zscore(panel["avg_dep_delay"]).fillna(0)
    panel["z_avg_arr_delay"] = zscore(panel["avg_arr_delay"]).fillna(0)
    panel["z_cancel_rate"] = zscore(panel["cancel_rate"]).fillna(0)
    panel["z_divert_rate"] = zscore(panel["divert_rate"]).fillna(0)
    panel["z_frac_delay_30"] = zscore(panel["frac_delay_30"]).fillna(0)
    panel["z_frac_delay_60"] = zscore(panel["frac_delay_60"]).fillna(0)
    panel["z_pct_nas_delay"] = zscore(panel["pct_nas_delay"]).fillna(0)
    panel["z_pct_late_aircraft_delay"] = zscore(panel["pct_late_aircraft_delay"]).fillna(0)
    panel["z_pct_weather_delay"] = zscore(panel["pct_weather_delay"]).fillna(0)

    # Main total stress, includes weather
    panel["total_stress"] = (
        0.25 * panel["z_avg_dep_delay"] +
        0.10 * panel["z_avg_arr_delay"] +
        0.20 * panel["z_cancel_rate"] +
        0.05 * panel["z_divert_rate"] +
        0.15 * panel["z_frac_delay_30"] +
        0.10 * panel["z_frac_delay_60"] +
        0.10 * panel["z_pct_nas_delay"] +
        0.03 * panel["z_pct_late_aircraft_delay"] +
        0.02 * panel["z_pct_weather_delay"]
    )

    # Alternative version excluding weather
    panel["total_stress_ex_weather"] = (
        0.26 * panel["z_avg_dep_delay"] +
        0.11 * panel["z_avg_arr_delay"] +
        0.21 * panel["z_cancel_rate"] +
        0.05 * panel["z_divert_rate"] +
        0.16 * panel["z_frac_delay_30"] +
        0.11 * panel["z_frac_delay_60"] +
        0.10 * panel["z_pct_nas_delay"] +
        0.05 * panel["z_pct_late_aircraft_delay"]
    )

    return panel


# ====================================
# WRITE SUMMARY
# ====================================

def write_summary(panel, output_dir=OUTPUT_DIR):
    lines = []
    lines.append("=" * 70)
    lines.append("TOTAL STRESS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Generated: {pd.Timestamp.now()}")
    lines.append(f"Rows: {len(panel):,}")
    lines.append(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
    lines.append(f"Hubs: {sorted(panel['hub'].unique())}")
    lines.append("")
    lines.append("Total stress distribution:")
    lines.append(panel["total_stress"].describe().to_string())
    lines.append("")
    lines.append("Top 20 total stress hub-days:")
    lines.append(
        panel[["date", "hub", "total_stress", "total_stress_ex_weather",
               "avg_dep_delay", "cancel_rate", "frac_delay_30", "pct_nas_delay"]]
        .sort_values("total_stress", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    (output_dir / "total_stress_summary.txt").write_text("\n".join(lines))


# ====================================
# RUN
# ====================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flights = load_flights_from_csvs(DATA_DIR)
    flights = clean_flights(flights)

    panel = aggregate_hub_daily(flights, HUB_AIRPORTS)
    panel = build_total_stress(panel)

    panel.to_csv(OUTPUT_DIR / "hub_total_stress.csv", index=False)
    write_summary(panel, OUTPUT_DIR)

    print("\nSaved:")
    print(OUTPUT_DIR / "hub_total_stress.csv")
    print(OUTPUT_DIR / "total_stress_summary.txt")

    print("\nSample:")
    print(
        panel[[
            "date", "hub", "total_stress", "total_stress_ex_weather",
            "avg_dep_delay", "cancel_rate", "frac_delay_30", "pct_nas_delay"
        ]].head(20)
    )