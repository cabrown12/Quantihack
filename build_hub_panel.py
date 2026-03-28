import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("flight_csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HUB_AIRPORTS = {"ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "CLT", "LAS", "PHX", "IAH"}

KEEP_COLUMNS = [
    "FL_DATE", "ORIGIN", "DEST", "OP_UNIQUE_CARRIER",
    "DEP_DELAY", "ARR_DELAY", "CANCELLED", "DIVERTED",
    "WEATHER_DELAY", "NAS_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY"
]

COLUMN_NAME_MAP = {
    "FlightDate": "FL_DATE",
    "FL_DATE": "FL_DATE",
    "Origin": "ORIGIN",
    "ORIGIN": "ORIGIN",
    "Dest": "DEST",
    "DEST": "DEST",
    "Reporting_Airline": "OP_UNIQUE_CARRIER",
    "OP_UNIQUE_CARRIER": "OP_UNIQUE_CARRIER",
    "UNIQUE_CARRIER": "OP_UNIQUE_CARRIER",
    "DepDelay": "DEP_DELAY",
    "DEP_DELAY": "DEP_DELAY",
    "ArrDelay": "ARR_DELAY",
    "ARR_DELAY": "ARR_DELAY",
    "Cancelled": "CANCELLED",
    "CANCELLED": "CANCELLED",
    "Diverted": "DIVERTED",
    "DIVERTED": "DIVERTED",
    "WeatherDelay": "WEATHER_DELAY",
    "WEATHER_DELAY": "WEATHER_DELAY",
    "NASDelay": "NAS_DELAY",
    "NAS_DELAY": "NAS_DELAY",
    "CarrierDelay": "CARRIER_DELAY",
    "CARRIER_DELAY": "CARRIER_DELAY",
    "LateAircraftDelay": "LATE_AIRCRAFT_DELAY",
    "LATE_AIRCRAFT_DELAY": "LATE_AIRCRAFT_DELAY",
}


def load_all_raw_csvs():
    files = sorted(RAW_DIR.glob("OnTime_*.csv"))
    if not files:
        raise FileNotFoundError(f"No OnTime_*.csv files found in {RAW_DIR.resolve()}")

    frames = []

    for fp in files:
        print(f"Loading {fp.name}...")
        df = pd.read_csv(fp, low_memory=False)

        df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")
        df.columns = df.columns.str.strip().str.strip('"')

        rename = {c: COLUMN_NAME_MAP[c] for c in df.columns if c in COLUMN_NAME_MAP}
        if rename:
            df = df.rename(columns=rename)

        available = [c for c in KEEP_COLUMNS if c in df.columns]
        if len(available) < 4:
            print(f"Skipping {fp.name}: not enough matching columns")
            continue

        df = df[available].copy()

        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
        df = df.dropna(subset=["FL_DATE"]).copy()

        if "DEST" in df.columns:
            mask = df["ORIGIN"].isin(HUB_AIRPORTS) | df["DEST"].isin(HUB_AIRPORTS)
        else:
            mask = df["ORIGIN"].isin(HUB_AIRPORTS)
        df = df[mask].copy()

        frames.append(df)

    if not frames:
        raise ValueError("No usable raw CSVs were loaded.")

    return pd.concat(frames, ignore_index=True)


def clean_flights(df):
    df = df.copy()

    for col in [
        "DEP_DELAY", "ARR_DELAY", "WEATHER_DELAY",
        "NAS_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["CANCELLED", "DIVERTED"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    non_cancelled = df["CANCELLED"] == 0
    if "DEP_DELAY" in df.columns:
        df.loc[non_cancelled, "DEP_DELAY"] = df.loc[non_cancelled, "DEP_DELAY"].fillna(0)
        df["DEP_DELAY_30"] = (df["DEP_DELAY"] >= 30).astype(int)

    return df


def aggregate_hub_daily(df):
    hub_df = df[df["ORIGIN"].isin(HUB_AIRPORTS)].copy()

    records = []
    for (hub, dt), grp in hub_df.groupby(["ORIGIN", "FL_DATE"]):
        n = len(grp)
        n_cancelled = int(grp["CANCELLED"].sum())
        n_diverted = int(grp["DIVERTED"].sum()) if "DIVERTED" in grp.columns else 0

        ops = grp[grp["CANCELLED"] == 0]
        n_ops = len(ops)

        rec = {
            "date": dt,
            "hub": hub,
            "n_flights": n,
            "n_operating": n_ops,
            "n_cancelled": n_cancelled,
            "n_diverted": n_diverted,
            "cancel_rate": n_cancelled / n if n > 0 else np.nan,
            "divert_rate": n_diverted / n if n > 0 else np.nan,
        }

        if n_ops > 0:
            rec["avg_dep_delay"] = ops["DEP_DELAY"].mean() if "DEP_DELAY" in ops.columns else np.nan
            rec["avg_arr_delay"] = ops["ARR_DELAY"].mean() if "ARR_DELAY" in ops.columns else np.nan
            rec["frac_delay_30"] = ops["DEP_DELAY_30"].mean() if "DEP_DELAY_30" in ops.columns else np.nan

            for cause in ["WEATHER_DELAY", "NAS_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY"]:
                if cause in ops.columns:
                    vals = ops[cause].fillna(0)
                    rec[f"avg_{cause.lower()}"] = vals.mean()
                    rec[f"pct_{cause.lower()}"] = (vals > 0).mean()
        else:
            rec["avg_dep_delay"] = np.nan
            rec["avg_arr_delay"] = np.nan
            rec["frac_delay_30"] = np.nan

        records.append(rec)

    panel = pd.DataFrame(records).sort_values(["hub", "date"]).reset_index(drop=True)
    return panel


if __name__ == "__main__":
    flights = load_all_raw_csvs()
    flights = clean_flights(flights)
    panel = aggregate_hub_daily(flights)

    out_path = OUT_DIR / "hub_daily_panel.csv"
    panel.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(panel.head())