#!/usr/bin/env python3
"""Incremental data collector using the dual-source Tushare adapter.

Only fetches dates not present in the existing local panel.
Converts promax volume from "lots" (手) to "shares" (股) by multiplying 100.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantlab.data.tushare_adapter import TushareAdapter

PANEL_DIR = ROOT / "data/smallcap_v4"
ASSETS_DIR = ROOT / "data/smallcap_assets"
OUT_DIR = ROOT / "data/tushare_incremental"


def latest_panel_dir() -> Path | None:
    """Find the most recent panel directory by actual data date (not directory name)."""
    if not PANEL_DIR.exists():
        return None
    best_dir, best_date = None, None
    for d in PANEL_DIR.iterdir():
        if not d.is_dir():
            continue
        close_path = d / "close.parquet"
        if not close_path.exists():
            continue
        try:
            df = pd.read_parquet(close_path)
            last_date = df.index.max()
            if best_date is None or last_date > best_date:
                best_date = last_date
                best_dir = d
        except Exception:
            continue
    return best_dir


def existing_calendar(panel_dir: Path) -> pd.DatetimeIndex:
    """Read the existing trading calendar from the panel."""
    close = pd.read_parquet(panel_dir / "adj_close.parquet")
    return close.index


def canonical_code(ts_code: str) -> str:
    """Convert Tushare ts_code to local format."""
    code, market = ts_code.lower().split(".")
    if market == "sh":
        return f"sh.{code}"
    elif market == "sz":
        return f"sz.{code}"
    elif market == "bj":
        return f"bj.{code}"
    return ts_code


def fetch_daily_incremental(adapter: TushareAdapter, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch full-market daily data for a date range via datahubco.

    Returns a long-format DataFrame with columns:
    date, code, open, high, low, close, pre_close, volume, amount
    where volume is in shares (股), not lots (手).
    """
    # Get calendar from promax
    cal_df = adapter.trade_cal("SSE", start_date, end_date)
    trade_dates = cal_df[cal_df["is_open"]]["cal_date"].dt.strftime("%Y%m%d").tolist()

    all_rows = []
    for td in trade_dates:
        print(f"Fetching {td} ...", flush=True)
        df = adapter.daily_trade_date(td)
        if df.empty:
            continue
        df["date"] = df["trade_date"]
        df["code"] = df["ts_code"].apply(canonical_code)
        df["volume"] = df["vol"] * 100  # convert lots to shares
        df["amount"] = df["amount"]
        all_rows.append(df[["date", "code", "open", "high", "low", "close", "pre_close", "volume", "amount"]])

    if not all_rows:
        return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "pre_close", "volume", "amount"])

    return pd.concat(all_rows, ignore_index=True)


def build_panel(long_df: pd.DataFrame, existing_codes: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Convert long-format DataFrame to wide panel (one per field)."""
    if long_df.empty:
        return {}

    long_df["date"] = pd.to_datetime(long_df["date"])
    codes = sorted(long_df["code"].unique())
    if existing_codes:
        codes = sorted(set(codes) | set(existing_codes))

    panels = {}
    for field in ["open", "high", "low", "close", "pre_close", "volume", "amount"]:
        pivot = long_df.pivot(index="date", columns="code", values=field)
        pivot = pivot.reindex(columns=codes)
        panels[field] = pivot

    return panels


def update_tradestatus(adapter: TushareAdapter, codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Build tradestatus frame from suspend_d data.

    Returns DataFrame indexed by date, columns by code, values: 1.0 = trading, NaN = suspended/unknown.
    """
    cal_df = adapter.trade_cal("SSE", start_date, end_date)
    dates = cal_df[cal_df["is_open"]]["cal_date"].tolist()

    # Initialize with 1.0 (trading)
    tradestatus = pd.DataFrame(1.0, index=dates, columns=codes)

    for code in codes:
        ts_code = code.replace("sh.", "").replace("sz.", "").replace("bj.", "") + {
            "sh.": ".SH", "sz.": ".SZ", "bj.": ".BJ"
        }.get(code[:3], ".SH")
        try:
            susp = adapter.suspend(ts_code, start_date, end_date)
            if not susp.empty:
                for _, row in susp.iterrows():
                    if row["suspend_type"] == "S":
                        date = pd.Timestamp(row["trade_date"])
                        if date in tradestatus.index:
                            tradestatus.loc[date, code] = np.nan
        except Exception as e:
            print(f"  suspend check failed for {code}: {e}")

    return tradestatus


def update_delistings(adapter: TushareAdapter, codes: list[str]) -> pd.DataFrame:
    """Check stock_basic for delisted stocks and build delistings frame."""
    rows = []
    for code in codes:
        ts_code = code.replace("sh.", "").replace("sz.", "").replace("bj.", "") + {
            "sh.": ".SH", "sz.": ".SZ", "bj.": ".BJ"
        }.get(code[:3], ".SH")
        try:
            info = adapter.stock_basic(ts_code=ts_code)
            if not info.empty:
                row = info.iloc[0]
                list_status = row.get("list_status", "")
                delist_date = row.get("delist_date", "")
                if list_status == "D" and delist_date:
                    rows.append({"code": code.split(".")[1], "delist_date": delist_date, "last_bar": ""})
        except Exception:
            pass
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="Start date YYYYMMDD (defaults to day after latest panel)")
    parser.add_argument("--end", help="End date YYYYMMDD (defaults to today)")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    adapter = TushareAdapter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Determine date range
    panel_dir = latest_panel_dir()
    if panel_dir:
        cal = existing_calendar(panel_dir)
        last_date = cal.max().strftime("%Y%m%d")
        print(f"Latest panel: {panel_dir.name}, last date: {last_date}")
        start = args.start or (cal.max() + pd.Timedelta(days=1)).strftime("%Y%m%d")
    else:
        start = args.start or "20261013"

    end = args.end or pd.Timestamp.now().strftime("%Y%m%d")

    if start > end:
        print(f"No new dates to fetch: start={start} > end={end}")
        return

    print(f"Fetching incremental data: {start} to {end}")

    # Fetch daily data
    long_df = fetch_daily_incremental(adapter, start, end)
    if long_df.empty:
        print("No data returned.")
        return

    print(f"Fetched {len(long_df)} rows, {long_df['date'].nunique()} dates, {long_df['code'].nunique()} codes")

    if args.dry_run:
        print("Dry run - not writing files.")
        return

    # Build and save panels
    existing_codes = None
    if panel_dir:
        existing = pd.read_parquet(panel_dir / "adj_close.parquet")
        existing_codes = existing.columns.tolist()

    panels = build_panel(long_df, existing_codes)

    for field, df in panels.items():
        path = out / f"{field}.parquet"
        df.to_parquet(path)
        print(f"Wrote {path}: {df.shape}")

    # Save raw long format for audit
    long_df.to_parquet(out / "daily_long.parquet")
    print(f"Wrote {out / 'daily_long.parquet'}: {long_df.shape}")

    # Save metadata
    meta = {
        "source": "tushare_dual",
        "datahubco": "daily_trade_date (full market)",
        "promax": "suspend_d, trade_cal, stock_basic",
        "date_range": [start, end],
        "n_dates": int(long_df["date"].nunique()),
        "n_codes": int(long_df["code"].nunique()),
        "volume_unit": "shares (lots * 100)",
        "panel_based_on": str(panel_dir) if panel_dir else None,
    }
    (out / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Wrote manifest: {meta}")


if __name__ == "__main__":
    main()
