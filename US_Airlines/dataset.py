#!/usr/bin/env python3
"""
01_bts_data_pull.py — BTS On-Time Performance Data Ingestion
=============================================================
Downloads flight-level on-time performance data from the US Bureau of
Transportation Statistics (transtats.bts.gov) and aggregates it into
a clean hub-level daily panel ready for signal construction.

USAGE
-----
  python 01_bts_data_pull.py                 # download missing + extract + aggregate
  python 01_bts_data_pull.py --skip-download # just extract existing ZIPs
  python 01_bts_data_pull.py --force         # reprocess even if panel CSV exists

OUTPUT
------
  data/raw/                              - downloaded ZIP files
  data/processed/hub_daily_panel.csv     - clean hub x date panel (main output)
  data/processed/flight_counts.csv       - daily flight counts per hub
  data/processed/data_quality_report.txt - QA summary
"""

import os
import sys
import io
import time
import zipfile
import argparse
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

HUB_AIRPORTS = {
    'ATL': 'Atlanta Hartsfield-Jackson',
    'ORD': "Chicago O'Hare",
    'DFW': 'Dallas/Fort Worth',
    'DEN': 'Denver International',
    'LAX': 'Los Angeles International',
    'JFK': 'New York JFK',
    'CLT': 'Charlotte Douglas',
    'LAS': 'Las Vegas Harry Reid',
    'PHX': 'Phoenix Sky Harbor',
    'IAH': 'Houston George Bush',
}

BTS_BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present"
    "_{year}_{month}.zip"
)

KEEP_COLUMNS = [
    'FL_DATE', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST',
    'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'ARR_DELAY',
    'CANCELLED', 'DIVERTED',
    'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY',
    'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'DISTANCE',
]


# ─────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────

def download_bts_month(year: int, month: int, output_dir: Path) -> Path | None:
    """Download one month. Skips if file already exists and is valid."""
    import urllib.request
    import urllib.error

    url = BTS_BASE_URL.format(year=year, month=month)
    filename = f"OnTime_{year}_{month:02d}.zip"
    filepath = output_dir / filename

    if filepath.exists() and filepath.stat().st_size > 1_000_000:
        with open(filepath, 'rb') as f:
            if f.read(2) == b'PK':
                print(f"    [SKIP] {filename} ({filepath.stat().st_size / 1e6:.1f} MB)")
                return filepath

    print(f"    [GET]  {url}")
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': (
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.transtats.bts.gov/',
        })
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        elapsed = time.time() - t0

        if len(data) < 10000 or data[:2] != b'PK':
            print(f"           -> Invalid response ({len(data)} bytes)")
            return _download_with_curl(url, filepath)

        filepath.write_bytes(data)
        print(f"           -> {len(data) / 1e6:.1f} MB in {elapsed:.1f}s")
        return filepath
    except Exception as e:
        print(f"           -> Error: {e}")
        return _download_with_curl(url, filepath)


def _download_with_curl(url: str, filepath: Path) -> Path | None:
    """Fallback download using system curl."""
    import subprocess
    print(f"           -> Trying curl fallback...")
    try:
        subprocess.run([
            'curl', '-L', '-s', '-S', '--max-time', '300',
            '-o', str(filepath),
            '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            '-H', 'Referer: https://www.transtats.bts.gov/',
            url
        ], capture_output=True, text=True, timeout=320)

        if filepath.exists() and filepath.stat().st_size > 10000:
            with open(filepath, 'rb') as f:
                if f.read(2) == b'PK':
                    print(f"           -> curl OK: {filepath.stat().st_size / 1e6:.1f} MB")
                    return filepath
        if filepath.exists():
            filepath.unlink()
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# ZIP EXTRACTION & CSV PARSING
# ─────────────────────────────────────────────────────────────────────

def read_bts_zip(zip_path: Path, hub_filter: set[str] | None = None) -> pd.DataFrame:
    """
    Open a BTS ZIP, find the CSV inside, parse into DataFrame.
    Handles: trailing commas, nested ZIPs, column name variations, encoding.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            all_names = zf.namelist()
            csv_names = [n for n in all_names if n.lower().endswith('.csv')]

            # No CSV — check for nested ZIP
            if not csv_names:
                nested = [n for n in all_names if n.lower().endswith('.zip')]
                if nested:
                    print("[nested zip] ", end='')
                    with zf.open(nested[0]) as inner_f:
                        inner_zf = zipfile.ZipFile(io.BytesIO(inner_f.read()))
                    csv_names = [n for n in inner_zf.namelist() if n.lower().endswith('.csv')]
                    if csv_names:
                        with inner_zf.open(csv_names[0]) as f:
                            return _parse_bts_csv(f, hub_filter)

                # Check files without .csv extension
                for name in all_names:
                    if '__MACOSX' in name or name.startswith('.'):
                        continue
                    with zf.open(name) as f:
                        header = f.readline().decode('latin-1', errors='replace')
                    if 'FL_DATE' in header or 'ORIGIN' in header:
                        with zf.open(name) as f:
                            return _parse_bts_csv(f, hub_filter)

                print(f"[no CSV, contents: {all_names[:3]}] ", end='')
                return pd.DataFrame()

            with zf.open(csv_names[0]) as f:
                return _parse_bts_csv(f, hub_filter)

    except zipfile.BadZipFile:
        print("[bad ZIP file] ", end='')
        return pd.DataFrame()
    except Exception as e:
        print(f"[error: {e}] ", end='')
        return pd.DataFrame()


def _parse_bts_csv(file_obj, hub_filter: set[str] | None = None) -> pd.DataFrame:
    """Parse a raw BTS CSV into a clean DataFrame."""
    try:
        df = pd.read_csv(file_obj, encoding='latin-1', low_memory=False, on_bad_lines='skip')
    except UnicodeDecodeError:
        file_obj.seek(0)
        df = pd.read_csv(file_obj, encoding='utf-8', low_memory=False, on_bad_lines='skip')

    if df.empty:
        return df

    # Drop trailing-comma "Unnamed" columns
    unnamed = [c for c in df.columns if 'Unnamed' in str(c)]
    if unnamed:
        df = df.drop(columns=unnamed)

    df.columns = df.columns.str.strip()

    # Column name variations across BTS schema years
    for old, new in [('UNIQUE_CARRIER', 'OP_UNIQUE_CARRIER'),
                     ('REPORTING_AIRLINE', 'OP_UNIQUE_CARRIER')]:
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    available = [c for c in KEEP_COLUMNS if c in df.columns]
    if not available:
        print(f"[no expected columns, got: {list(df.columns)[:8]}] ", end='')
        return pd.DataFrame()
    df = df[available].copy()

    # Filter to hub airports to save memory
    if hub_filter and 'ORIGIN' in df.columns:
        if 'DEST' in df.columns:
            mask = df['ORIGIN'].isin(hub_filter) | df['DEST'].isin(hub_filter)
        else:
            mask = df['ORIGIN'].isin(hub_filter)
        df = df[mask]

    return df


# ─────────────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────────────

def clean_flight_records(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, coerce numerics, add delay flags."""
    if df.empty:
        return df

    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    df = df.dropna(subset=['FL_DATE'])

    for col in ['DEP_DELAY', 'ARR_DELAY', 'CARRIER_DELAY', 'WEATHER_DELAY',
                'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col, default in [('CANCELLED', 0), ('DIVERTED', 0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
        else:
            df[col] = default

    # Fill NaN delays with 0 for non-cancelled flights
    non_cancelled = df['CANCELLED'] == 0
    for col in ['DEP_DELAY', 'ARR_DELAY']:
        if col in df.columns:
            df.loc[non_cancelled, col] = df.loc[non_cancelled, col].fillna(0)

    if 'DEP_DELAY' in df.columns:
        df['DEP_DELAY_30'] = (df['DEP_DELAY'] >= 30).astype(int)
        df['DEP_DELAY_60'] = (df['DEP_DELAY'] >= 60).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────────────

def aggregate_hub_daily(df: pd.DataFrame, hub_airports: set[str]) -> pd.DataFrame:
    """Aggregate flight-level records into hub x date panel."""
    hub_df = df[df['ORIGIN'].isin(hub_airports)].copy()
    if hub_df.empty:
        return pd.DataFrame()

    records = []
    for (hub, dt), grp in hub_df.groupby(['ORIGIN', 'FL_DATE']):
        n_total = len(grp)
        n_cancelled = int(grp['CANCELLED'].sum())
        n_diverted = int(grp.get('DIVERTED', pd.Series([0])).sum())
        operating = grp[grp['CANCELLED'] == 0]
        n_operating = len(operating)

        rec = {
            'date': dt, 'hub': hub,
            'n_flights': n_total, 'n_operating': n_operating,
            'n_cancelled': n_cancelled, 'n_diverted': n_diverted,
            'cancel_rate': n_cancelled / n_total if n_total > 0 else np.nan,
            'divert_rate': n_diverted / n_total if n_total > 0 else np.nan,
        }

        if n_operating > 0:
            for col in ['DEP_DELAY', 'ARR_DELAY']:
                if col in operating.columns:
                    rec[f'avg_{col.lower()}'] = operating[col].mean()
            if 'DEP_DELAY_30' in operating.columns:
                rec['frac_delay_30'] = operating['DEP_DELAY_30'].mean()
                rec['frac_delay_60'] = operating['DEP_DELAY_60'].mean()
            for cause in ['WEATHER_DELAY', 'NAS_DELAY', 'CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY']:
                if cause in operating.columns:
                    vals = operating[cause].fillna(0)
                    rec[f'avg_{cause.lower()}'] = vals.mean()
                    rec[f'pct_{cause.lower()}'] = (vals > 0).mean()
        else:
            rec.update({'avg_dep_delay': np.nan, 'avg_arr_delay': np.nan,
                        'frac_delay_30': np.nan, 'frac_delay_60': np.nan})

        records.append(rec)

    panel = pd.DataFrame(records)
    panel['date'] = pd.to_datetime(panel['date'])
    return panel.sort_values(['hub', 'date']).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────

def write_quality_report(panel: pd.DataFrame, output_dir: Path):
    lines = [
        "=" * 70,
        "BTS HUB DAILY PANEL — DATA QUALITY REPORT",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 70,
        f"\nDate range: {panel['date'].min().date()} to {panel['date'].max().date()}",
        f"Total records: {len(panel):,}",
        f"Hubs: {sorted(panel['hub'].unique())}",
        f"Unique dates: {panel['date'].nunique()}",
    ]

    lines.append("\n--- Coverage by Hub ---")
    for hub in sorted(panel['hub'].unique()):
        h = panel[panel['hub'] == hub]
        na_dep = h['avg_dep_delay'].isna().mean() * 100
        lines.append(
            f"  {hub}: {len(h)} days, avg flights={h['n_flights'].mean():.0f}, "
            f"NaN dep_delay={na_dep:.1f}%"
        )

    lines.append("\n--- Delay Statistics ---")
    for col in ['avg_dep_delay', 'avg_arr_delay', 'cancel_rate', 'frac_delay_30']:
        if col in panel.columns:
            s = panel[col].dropna()
            lines.append(
                f"  {col}: mean={s.mean():.3f}, std={s.std():.3f}, "
                f"min={s.min():.3f}, max={s.max():.3f}, skew={s.skew():.2f}"
            )

    lines.append("\n--- Top 10 Stress Days ---")
    daily = panel.groupby('date')['avg_dep_delay'].mean().nlargest(10)
    for dt, val in daily.items():
        hubs = panel[(panel['date'] == dt) & (panel['avg_dep_delay'] > val * 0.8)]
        lines.append(f"  {dt.date()}: mean_delay={val:.1f}min, hubs: {', '.join(hubs['hub'])}")

    report = '\n'.join(lines)
    (output_dir / 'data_quality_report.txt').write_text(report)
    print(report)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='BTS On-Time data ingestion')
    parser.add_argument('--start-year', type=int, default=2021)
    parser.add_argument('--end-year', type=int, default=2023)
    parser.add_argument('--output-dir', type=str, default='./data')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloads, process existing ZIPs only')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocess even if panel CSV exists')
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    raw_dir = base_dir / 'raw'
    proc_dir = base_dir / 'processed'
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    hub_set = set(HUB_AIRPORTS.keys())

    print("=" * 70)
    print("BTS ON-TIME PERFORMANCE DATA PULL")
    print("=" * 70)
    print(f"  Years:  {args.start_year}-{args.end_year}")
    print(f"  Hubs:   {sorted(hub_set)}")
    print(f"  Output: {base_dir.resolve()}")

    # ── STEP 1: Find / download ZIPs ──
    print(f"\n  STEP 1: Checking for data in {raw_dir}/")

    expected = []
    for y in range(args.start_year, args.end_year + 1):
        for m in range(1, 13):
            if date(y, m, 1) <= date.today():
                expected.append((y, m))

    existing = []
    missing = []
    for y, m in expected:
        fp = raw_dir / f"OnTime_{y}_{m:02d}.zip"
        if fp.exists() and fp.stat().st_size > 1_000_000:
            existing.append(fp)
        else:
            missing.append((y, m))

    # Pick up any other ZIPs in directory (manually downloaded)
    for p in raw_dir.glob('*.zip'):
        if p.stat().st_size > 1_000_000 and p not in existing:
            existing.append(p)

    print(f"    Expected: {len(expected)} months")
    print(f"    Found:    {len(existing)} ZIPs")
    print(f"    Missing:  {len(missing)}")

    if missing and not args.skip_download:
        print(f"\n    Downloading {len(missing)} missing months...")
        for y, m in missing:
            path = download_bts_month(y, m, raw_dir)
            if path:
                existing.append(path)
            time.sleep(1.0)

    zip_paths = sorted(set(existing))

    if not zip_paths:
        print(f"\n  ERROR: No ZIP files found in {raw_dir}")
        print(f"  Either download them or check --output-dir")
        sys.exit(1)

    print(f"    Ready to process: {len(zip_paths)} ZIPs")

    # ── Check cached panel ──
    panel_csv = proc_dir / 'hub_daily_panel.csv'
    if panel_csv.exists() and not args.force:
        print(f"\n  Panel already exists: {panel_csv} ({panel_csv.stat().st_size/1e6:.2f} MB)")
        print(f"  Use --force to reprocess from ZIPs")
        panel = pd.read_csv(panel_csv, parse_dates=['date'])
        print(f"  Loaded: {len(panel):,} rows, {panel['hub'].nunique()} hubs, "
              f"{panel['date'].nunique()} dates")
        return

    # ── STEP 2: Extract & parse ──
    print(f"\n  STEP 2: Extracting CSVs from {len(zip_paths)} ZIPs...")

    all_flights = []
    for i, zp in enumerate(zip_paths, 1):
        print(f"    [{i:>2}/{len(zip_paths)}] {zp.name}... ", end='', flush=True)
        df = read_bts_zip(zp, hub_filter=hub_set)
        if not df.empty:
            df = clean_flight_records(df)
            print(f"-> {len(df):,} hub flights")
            all_flights.append(df)
        else:
            print("-> EMPTY")

    if not all_flights:
        print("\n  ERROR: Could not parse flight data from any ZIP!")
        print("  Try: python diagnose_zip.py ./data/raw/OnTime_2023_01.zip")
        sys.exit(1)

    flights = pd.concat(all_flights, ignore_index=True)
    print(f"\n  Total hub flights: {len(flights):,}")
    print(f"  Columns: {list(flights.columns)}")
    print(f"  Dates: {flights['FL_DATE'].min()} -> {flights['FL_DATE'].max()}")
    print(f"  Hubs: {sorted(flights['ORIGIN'].unique())}")

    # ── STEP 3: Aggregate ──
    print(f"\n  STEP 3: Building hub x daily panel...")
    panel = aggregate_hub_daily(flights, hub_set)
    print(f"  Shape: {panel.shape}")
    print(f"\n  First 5 rows:")
    print(panel.head(5).to_string(index=False))

    # ── STEP 4: Save ──
    print(f"\n  STEP 4: Saving...")
    panel.to_csv(panel_csv, index=False)
    print(f"  -> {panel_csv} ({panel_csv.stat().st_size / 1e6:.2f} MB)")

    try:
        pq = proc_dir / 'hub_daily_panel.parquet'
        panel.to_parquet(pq, index=False, engine='pyarrow')
        print(f"  -> {pq} ({pq.stat().st_size / 1e6:.2f} MB)")
    except Exception:
        pass

    panel.groupby('hub')['n_flights'].agg(['mean','std','min','max']).to_csv(
        proc_dir / 'flight_counts.csv')

    print()
    write_quality_report(panel, proc_dir)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print(f"  Panel: {panel_csv}")
    print(f"  Shape: {panel.shape[0]:,} rows x {panel.shape[1]} cols")
    print(f"  Next:  python 02_signal_pipeline.py --data-dir {proc_dir}")


if __name__ == '__main__':
    main()