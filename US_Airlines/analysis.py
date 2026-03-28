#!/usr/bin/env python3
"""
analyze_bts_data.py — Process real BTS ZIPs + comprehensive data analysis
==========================================================================
Run this from your Quantihack directory where data/raw/ has the 36 ZIPs.

  python3 analyze_bts_data.py

It will:
  1. Delete any old synthetic panel
  2. Extract real flight data from every ZIP
  3. Build the hub x daily panel
  4. Run a full data suitability & quality analysis
  5. Save everything to data/processed/
"""

import io
import sys
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

HUB_AIRPORTS = ['ATL','ORD','DFW','DEN','LAX','JFK','CLT','LAS','PHX','IAH']

KEEP_COLUMNS = [
    'FL_DATE', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST',
    'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'ARR_DELAY',
    'CANCELLED', 'DIVERTED',
    'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY',
    'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'DISTANCE',
]

# BTS has used different column names across years and download methods.
# Map every known variant to our canonical names above.
COLUMN_NAME_MAP = {
    # Date
    'FlightDate':                   'FL_DATE',
    'FL_DATE':                      'FL_DATE',
    # Carrier
    'Reporting_Airline':            'OP_UNIQUE_CARRIER',
    'REPORTING_AIRLINE':            'OP_UNIQUE_CARRIER',
    'IATA_CODE_Reporting_Airline':  'IATA_CARRIER',
    'OP_UNIQUE_CARRIER':            'OP_UNIQUE_CARRIER',
    'UNIQUE_CARRIER':               'OP_UNIQUE_CARRIER',
    # Origin / Dest
    'Origin':                       'ORIGIN',
    'ORIGIN':                       'ORIGIN',
    'Dest':                         'DEST',
    'DEST':                         'DEST',
    # Times
    'CRSDepTime':                   'CRS_DEP_TIME',
    'CRS_DEP_TIME':                 'CRS_DEP_TIME',
    'DepTime':                      'DEP_TIME',
    'DEP_TIME':                     'DEP_TIME',
    # Delays
    'DepDelay':                     'DEP_DELAY',
    'DEP_DELAY':                    'DEP_DELAY',
    'DepDelayMinutes':              'DEP_DELAY_MINS',
    'ArrDelay':                     'ARR_DELAY',
    'ARR_DELAY':                    'ARR_DELAY',
    'ArrDelayMinutes':              'ARR_DELAY_MINS',
    # Status
    'Cancelled':                    'CANCELLED',
    'CANCELLED':                    'CANCELLED',
    'Diverted':                     'DIVERTED',
    'DIVERTED':                     'DIVERTED',
    # Delay causes
    'CarrierDelay':                 'CARRIER_DELAY',
    'CARRIER_DELAY':                'CARRIER_DELAY',
    'WeatherDelay':                 'WEATHER_DELAY',
    'WEATHER_DELAY':                'WEATHER_DELAY',
    'NASDelay':                     'NAS_DELAY',
    'NAS_DELAY':                    'NAS_DELAY',
    'SecurityDelay':                'SECURITY_DELAY',
    'SECURITY_DELAY':               'SECURITY_DELAY',
    'LateAircraftDelay':            'LATE_AIRCRAFT_DELAY',
    'LATE_AIRCRAFT_DELAY':          'LATE_AIRCRAFT_DELAY',
    # Distance
    'Distance':                     'DISTANCE',
    'DISTANCE':                     'DISTANCE',
}

RAW_DIR = Path('./data/raw')
PROC_DIR = Path('./data/processed')

# ─────────────────────────────────────────────────────────────────────
# ZIP EXTRACTION
# ─────────────────────────────────────────────────────────────────────

def extract_csv_from_zip(zip_path, hub_filter=None):
    """Open a BTS ZIP, find the CSV, parse it, filter to hubs."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            csv_files = [n for n in names if n.lower().endswith('.csv')]

            if not csv_files:
                # Try nested ZIP
                nested = [n for n in names if n.lower().endswith('.zip')]
                if nested:
                    with zf.open(nested[0]) as inner:
                        inner_zf = zipfile.ZipFile(io.BytesIO(inner.read()))
                    csv_files = [n for n in inner_zf.namelist() if n.lower().endswith('.csv')]
                    if csv_files:
                        with inner_zf.open(csv_files[0]) as f:
                            return _parse_csv(f, hub_filter)

                # Check for non-.csv data files
                for name in names:
                    if '__MACOSX' in name or name.startswith('.'):
                        continue
                    with zf.open(name) as f:
                        header = f.readline().decode('latin-1', errors='replace')
                    if 'FL_DATE' in header or 'ORIGIN' in header or 'FlightDate' in header or 'Origin' in header:
                        with zf.open(name) as f:
                            return _parse_csv(f, hub_filter)

                return None

            with zf.open(csv_files[0]) as f:
                return _parse_csv(f, hub_filter)

    except Exception as e:
        print(f"  ERROR reading {zip_path.name}: {e}")
        return None


def _parse_csv(file_obj, hub_filter=None):
    """Parse raw BTS CSV into DataFrame, mapping column names to canonical form."""
    try:
        df = pd.read_csv(file_obj, encoding='latin-1', low_memory=False, on_bad_lines='skip')
    except UnicodeDecodeError:
        file_obj.seek(0)
        df = pd.read_csv(file_obj, encoding='utf-8', low_memory=False, on_bad_lines='skip')

    if df.empty:
        return None

    # Drop trailing-comma columns
    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in str(c)], errors='ignore')

    # Strip whitespace and quotes from column names
    df.columns = df.columns.str.strip().str.strip('"')

    # Map all known column name variants to our canonical names
    rename = {}
    for orig_col in df.columns:
        if orig_col in COLUMN_NAME_MAP:
            rename[orig_col] = COLUMN_NAME_MAP[orig_col]
    if rename:
        df = df.rename(columns=rename)

    # Keep only the columns we need
    available = [c for c in KEEP_COLUMNS if c in df.columns]
    if len(available) < 3:
        print(f"[only {len(available)} cols matched: {available}] ", end='')
        print(f"[raw cols: {list(df.columns)[:10]}] ", end='')
        return None
    df = df[available].copy()

    # Filter to hub airports to save memory
    if hub_filter and 'ORIGIN' in df.columns:
        hub_set = set(hub_filter)
        if 'DEST' in df.columns:
            mask = df['ORIGIN'].isin(hub_set) | df['DEST'].isin(hub_set)
        else:
            mask = df['ORIGIN'].isin(hub_set)
        df = df[mask]

    return df


# ─────────────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────────────

def clean_flights(df):
    """Parse dates, coerce numerics, add delay flags."""
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    df = df.dropna(subset=['FL_DATE'])

    for col in ['DEP_DELAY','ARR_DELAY','CARRIER_DELAY','WEATHER_DELAY',
                'NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['CANCELLED','DIVERTED']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    non_canc = df['CANCELLED'] == 0
    for col in ['DEP_DELAY','ARR_DELAY']:
        if col in df.columns:
            df.loc[non_canc, col] = df.loc[non_canc, col].fillna(0)

    if 'DEP_DELAY' in df.columns:
        df['DEP_DELAY_30'] = (df['DEP_DELAY'] >= 30).astype(int)
        df['DEP_DELAY_60'] = (df['DEP_DELAY'] >= 60).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────────────

def aggregate_hub_daily(df, hub_list):
    """Build hub x date panel from flight-level data."""
    hub_df = df[df['ORIGIN'].isin(set(hub_list))].copy()
    if hub_df.empty:
        return pd.DataFrame()

    records = []
    for (hub, dt), grp in hub_df.groupby(['ORIGIN', 'FL_DATE']):
        n = len(grp)
        n_canc = int(grp['CANCELLED'].sum())
        n_div = int(grp['DIVERTED'].sum()) if 'DIVERTED' in grp.columns else 0
        ops = grp[grp['CANCELLED'] == 0]
        n_ops = len(ops)

        rec = {
            'date': dt, 'hub': hub,
            'n_flights': n, 'n_operating': n_ops,
            'n_cancelled': n_canc, 'n_diverted': n_div,
            'cancel_rate': n_canc / n if n > 0 else np.nan,
            'divert_rate': n_div / n if n > 0 else np.nan,
        }

        if n_ops > 0:
            for col in ['DEP_DELAY','ARR_DELAY']:
                if col in ops.columns:
                    rec[f'avg_{col.lower()}'] = ops[col].mean()
            if 'DEP_DELAY_30' in ops.columns:
                rec['frac_delay_30'] = ops['DEP_DELAY_30'].mean()
                rec['frac_delay_60'] = ops['DEP_DELAY_60'].mean()
            for cause in ['WEATHER_DELAY','NAS_DELAY','CARRIER_DELAY','LATE_AIRCRAFT_DELAY']:
                if cause in ops.columns:
                    vals = ops[cause].fillna(0)
                    rec[f'avg_{cause.lower()}'] = vals.mean()
                    rec[f'pct_{cause.lower()}'] = (vals > 0).mean()
        else:
            rec.update({
                'avg_dep_delay': np.nan, 'avg_arr_delay': np.nan,
                'frac_delay_30': np.nan, 'frac_delay_60': np.nan,
            })

        records.append(rec)

    panel = pd.DataFrame(records)
    panel['date'] = pd.to_datetime(panel['date'])
    return panel.sort_values(['hub','date']).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def run_data_analysis(flights, panel):
    """Comprehensive data suitability and quality analysis."""

    print("\n" + "=" * 78)
    print("DATA SUITABILITY & QUALITY ANALYSIS")
    print("=" * 78)

    # ── 1. REALNESS VERIFICATION ──
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  1. DATA AUTHENTICITY CHECK                                │")
    print("└─────────────────────────────────────────────────────────────┘")

    has_weekends = (panel['date'].dt.dayofweek >= 5).any()
    has_nan = panel.isna().any().any()
    divert_unique = panel['divert_rate'].nunique()
    unique_dates = panel['date'].nunique()
    date_range = (panel['date'].max() - panel['date'].min()).days

    print(f"  Weekend flights present:     {'YES ✓' if has_weekends else 'NO ✗ (SYNTHETIC!)'}")
    print(f"  Contains NaN values:         {'YES ✓' if has_nan else 'NO ✗ (SYNTHETIC!)'}")
    print(f"  Unique divert_rate values:   {divert_unique} {'✓' if divert_unique > 5 else '✗ (SYNTHETIC!)'}")
    print(f"  Unique dates:                {unique_dates} (expected ~{date_range} calendar days)")
    print(f"  Date coverage:               {unique_dates/date_range*100:.1f}%")

    if not has_weekends or not has_nan or divert_unique < 5:
        print("\n  ⚠ WARNING: This appears to be SYNTHETIC data, not real BTS data!")
        print("  The ZIPs were downloaded but the old script cached synthetic output.")
        print("  Fix: delete data/processed/hub_daily_panel.csv and re-run with --force")
    else:
        print("\n  ✓ Data appears to be REAL BTS on-time performance data")

    # ── 2. MISSING VALUES BY HUB ──
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  2. MISSING VALUES BY AIRPORT HUB                         │")
    print("└─────────────────────────────────────────────────────────────┘")

    key_cols = ['avg_dep_delay', 'avg_arr_delay', 'cancel_rate',
                'frac_delay_30', 'frac_delay_60']
    available_key = [c for c in key_cols if c in panel.columns]

    print(f"\n  {'Hub':<6}", end='')
    for c in available_key:
        print(f"  {c:>15}", end='')
    print(f"  {'total_rows':>10}")
    print("  " + "-" * (6 + 17 * len(available_key) + 12))

    for hub in sorted(panel['hub'].unique()):
        h = panel[panel['hub'] == hub]
        print(f"  {hub:<6}", end='')
        for c in available_key:
            n_na = h[c].isna().sum()
            pct = n_na / len(h) * 100
            print(f"  {n_na:>6} ({pct:>4.1f}%)", end='')
        print(f"  {len(h):>10}")

    # Overall
    print(f"\n  {'TOTAL':<6}", end='')
    for c in available_key:
        n_na = panel[c].isna().sum()
        pct = n_na / len(panel) * 100
        print(f"  {n_na:>6} ({pct:>4.1f}%)", end='')
    print(f"  {len(panel):>10}")

    # ── 3. MISSING VALUES BY YEAR ──
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  3. MISSING VALUES BY YEAR                                │")
    print("└─────────────────────────────────────────────────────────────┘")

    panel['year'] = panel['date'].dt.year
    for year in sorted(panel['year'].unique()):
        ydf = panel[panel['year'] == year]
        print(f"\n  {year}: {len(ydf)} rows, {ydf['date'].nunique()} unique dates")
        for c in available_key:
            n_na = ydf[c].isna().sum()
            if n_na > 0:
                print(f"    {c}: {n_na} NaN ({n_na/len(ydf)*100:.1f}%)")
        if all(ydf[c].isna().sum() == 0 for c in available_key):
            print(f"    No missing values in key columns")

    # ── 4. AIRLINE / CARRIER ANALYSIS (from raw flights) ──
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  4. AIRLINE COVERAGE & MISSING DATA                       │")
    print("└─────────────────────────────────────────────────────────────┘")

    if 'OP_UNIQUE_CARRIER' in flights.columns:
        carrier_stats = flights.groupby('OP_UNIQUE_CARRIER').agg(
            n_flights=('FL_DATE', 'count'),
            pct_dep_nan=('DEP_DELAY', lambda x: x.isna().mean() * 100),
            pct_arr_nan=('ARR_DELAY', lambda x: x.isna().mean() * 100),
            pct_cancelled=('CANCELLED', 'mean'),
            mean_dep_delay=('DEP_DELAY', 'mean'),
        ).round(2)
        carrier_stats = carrier_stats.sort_values('n_flights', ascending=False)

        print(f"\n  {'Carrier':<10} {'Flights':>10} {'DEP NaN%':>10} {'ARR NaN%':>10} "
              f"{'Cancel%':>10} {'Avg Delay':>10}")
        print("  " + "-" * 62)
        for carrier, row in carrier_stats.iterrows():
            print(f"  {carrier:<10} {row['n_flights']:>10,.0f} {row['pct_dep_nan']:>9.1f}% "
                  f"{row['pct_arr_nan']:>9.1f}% {row['pct_cancelled']*100:>9.2f}% "
                  f"{row['mean_dep_delay']:>10.1f}")
    else:
        print("  Carrier column not available in flight data")

    # ── 5. DELAY CAUSE FIELD COVERAGE ──
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  5. DELAY CAUSE FIELD COVERAGE                            │")
    print("└─────────────────────────────────────────────────────────────┘")

    cause_cols = ['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY',
                  'SECURITY_DELAY','LATE_AIRCRAFT_DELAY']

    print(f"\n  Note: BTS only reports delay causes for flights arriving 15+ min late.")
    print(f"  NaN in these fields usually means the flight was on time.\n")

    for col in cause_cols:
        if col in flights.columns:
            n_total = len(flights)
            n_nan = flights[col].isna().sum()
            n_zero = (flights[col] == 0).sum()
            n_pos = (flights[col] > 0).sum()
            print(f"  {col:<25} NaN: {n_nan/n_total*100:>5.1f}%  "
                  f"Zero: {n_zero/n_total*100:>5.1f}%  "
                  f">0: {n_pos/n_total*100:>5.1f}%")

    # ── 6. DATE GAPS & COVERAGE ──
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  6. DATE GAPS & COVERAGE BY HUB                           │")
    print("└─────────────────────────────────────────────────────────────┘")

    full_range = pd.date_range(panel['date'].min(), panel['date'].max(), freq='D')
    print(f"\n  Full calendar range: {len(full_range)} days")
    print(f"  Dates in panel: {panel['date'].nunique()}")

    for hub in sorted(panel['hub'].unique()):
        hub_dates = set(panel[panel['hub'] == hub]['date'].dt.normalize())
        missing_dates = set(full_range) - hub_dates
        if missing_dates:
            sorted_missing = sorted(missing_dates)
            # Find contiguous gaps
            gaps = []
            start = sorted_missing[0]
            prev = start
            for d in sorted_missing[1:]:
                if (d - prev).days > 1:
                    gaps.append((start, prev, (prev - start).days + 1))
                    start = d
                prev = d
            gaps.append((start, prev, (prev - start).days + 1))

            big_gaps = [g for g in gaps if g[2] >= 3]
            print(f"  {hub}: {len(missing_dates)} missing days, "
                  f"{len(big_gaps)} gaps ≥3 days", end='')
            if big_gaps:
                print(f" — largest: {max(g[2] for g in big_gaps)} days")
            else:
                print()
        else:
            print(f"  {hub}: complete coverage (0 missing days)")

    # ── 7. DISTRIBUTIONAL STATISTICS BY YEAR ──
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  7. DELAY DISTRIBUTIONS BY YEAR                           │")
    print("└─────────────────────────────────────────────────────────────┘")

    for year in sorted(panel['year'].unique()):
        ydf = panel[panel['year'] == year]
        dep = ydf['avg_dep_delay'].dropna()
        canc = ydf['cancel_rate'].dropna()
        print(f"\n  {year}:")
        print(f"    Dep delay:   mean={dep.mean():.2f}  std={dep.std():.2f}  "
              f"median={dep.median():.2f}  P95={dep.quantile(0.95):.2f}  max={dep.max():.2f}")
        print(f"    Cancel rate: mean={canc.mean():.4f}  std={canc.std():.4f}  "
              f"max={canc.max():.4f}")
        print(f"    Flights/day: mean={ydf['n_flights'].mean():.0f}  "
              f"min={ydf['n_flights'].min()}  max={ydf['n_flights'].max()}")

    # ── 8. OUTLIER DETECTION ──
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  8. OUTLIERS & ANOMALIES                                  │")
    print("└─────────────────────────────────────────────────────────────┘")

    for col in ['avg_dep_delay', 'cancel_rate', 'n_flights']:
        if col not in panel.columns:
            continue
        s = panel[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        outliers = panel[(panel[col] < lower) | (panel[col] > upper)]
        print(f"\n  {col}: IQR={iqr:.4f}, bounds=[{lower:.4f}, {upper:.4f}]")
        print(f"    Extreme outliers (3×IQR): {len(outliers)} rows ({len(outliers)/len(panel)*100:.2f}%)")
        if len(outliers) > 0 and len(outliers) <= 10:
            for _, row in outliers.iterrows():
                print(f"      {row['date'].date()} {row['hub']}: {col}={row[col]:.4f}")

    # ── 9. CLEANING RECOMMENDATIONS ──
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  9. CLEANING RECOMMENDATIONS                              │")
    print("└─────────────────────────────────────────────────────────────┘")

    issues = []

    # Check if data is synthetic
    if not has_weekends:
        issues.append("CRITICAL: Data is SYNTHETIC. Delete hub_daily_panel.csv and reprocess ZIPs with --force")

    # NaN analysis
    total_nan = panel[available_key].isna().sum().sum()
    total_cells = len(panel) * len(available_key)
    nan_pct = total_nan / total_cells * 100
    if nan_pct > 5:
        issues.append(f"HIGH NaN rate ({nan_pct:.1f}%): Consider imputation for delay fields")
    elif nan_pct > 1:
        issues.append(f"Moderate NaN rate ({nan_pct:.1f}%): Forward-fill or drop NaN rows")
    else:
        issues.append(f"Low NaN rate ({nan_pct:.1f}%): Minimal cleaning needed")

    # Date gaps
    missing_total = len(full_range) - panel['date'].nunique()
    if missing_total > 0:
        issues.append(f"Date gaps: {missing_total} missing calendar days — "
                      f"{'expected (business-day data)' if not has_weekends else 'investigate'}")

    # Outliers
    dep_outliers = panel[panel['avg_dep_delay'] > panel['avg_dep_delay'].quantile(0.99)]
    if len(dep_outliers) > 0:
        issues.append(f"Delay outliers: {len(dep_outliers)} rows above P99 "
                      f"({panel['avg_dep_delay'].quantile(0.99):.1f} min) — keep as real stress events")

    # Flight count consistency
    for hub in panel['hub'].unique():
        h = panel[panel['hub'] == hub]
        cv = h['n_flights'].std() / h['n_flights'].mean()
        if cv > 0.3:
            issues.append(f"{hub}: High flight count volatility (CV={cv:.2f}) — check for data gaps")

    print()
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    panel.drop(columns=['year'], inplace=True, errors='ignore')

    return panel


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("BTS DATA EXTRACTION + QUALITY ANALYSIS")
    print("=" * 78)

    # Find all ZIPs
    zip_paths = sorted(RAW_DIR.glob('*.zip'))
    zip_paths = [z for z in zip_paths if z.stat().st_size > 1_000_000]

    print(f"\n  Found {len(zip_paths)} ZIPs in {RAW_DIR}/")

    if not zip_paths:
        print(f"  ERROR: No ZIP files found in {RAW_DIR}")
        print(f"  Make sure your downloaded OnTime_*.zip files are in {RAW_DIR}/")
        sys.exit(1)

    for z in zip_paths[:5]:
        print(f"    {z.name} ({z.stat().st_size / 1e6:.1f} MB)")
    if len(zip_paths) > 5:
        print(f"    ... and {len(zip_paths) - 5} more")

    # ── Extract all ZIPs ──
    print(f"\n  Extracting CSVs from {len(zip_paths)} ZIPs...")

    all_flights = []
    for i, zp in enumerate(zip_paths, 1):
        print(f"    [{i:>2}/{len(zip_paths)}] {zp.name}... ", end='', flush=True)
        df = extract_csv_from_zip(zp, hub_filter=HUB_AIRPORTS)
        if df is not None and not df.empty:
            df = clean_flights(df)
            print(f"-> {len(df):,} hub flights")
            all_flights.append(df)
        else:
            print("-> EMPTY")

    if not all_flights:
        print("\n  ERROR: No flight data extracted!")
        print("  Run: python diagnose_zip.py ./data/raw/OnTime_2023_01.zip")
        sys.exit(1)

    flights = pd.concat(all_flights, ignore_index=True)
    print(f"\n  Total flights extracted: {len(flights):,}")
    print(f"  Columns: {list(flights.columns)}")
    print(f"  Date range: {flights['FL_DATE'].min()} -> {flights['FL_DATE'].max()}")

    # ── Aggregate ──
    print(f"\n  Building hub x daily panel...")
    panel = aggregate_hub_daily(flights, HUB_AIRPORTS)
    print(f"  Panel shape: {panel.shape}")

    # ── Save ──
    panel_path = PROC_DIR / 'hub_daily_panel.csv'
    panel.to_csv(panel_path, index=False)
    print(f"  Saved: {panel_path} ({panel_path.stat().st_size / 1e6:.2f} MB)")

    # ── Run analysis ──
    panel = run_data_analysis(flights, panel)

    print(f"\n{'='*78}")
    print("COMPLETE")
    print(f"{'='*78}")
    print(f"  Panel: {panel_path}")
    print(f"  Shape: {panel.shape[0]:,} rows x {panel.shape[1]} cols")
    print(f"  Next:  python 02_signal_pipeline.py --data-dir {PROC_DIR}")


if __name__ == '__main__':
    main()