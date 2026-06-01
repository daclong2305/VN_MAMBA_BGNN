from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume", "Value"]
STOCK_FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "oc_return",
    "hl_range",
    "volume_change_1d",
    "value_change_1d",
    "rolling_vol_5",
    "rolling_vol_20",
    "ma_ratio_5",
    "ma_ratio_20",
    "volume_ma_ratio_5",
    "volume_ma_ratio_20",
    "market_ret_1d",
    "market_ret_5d",
    "market_vol_20",
    "relative_ret_1d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Kaggle VN30 per-symbol CSV files into panel datasets."
    )
    parser.add_argument("--input-dir", default="Dataset/VN30")
    parser.add_argument("--output-dir", default="Dataset")
    parser.add_argument("--benchmark", default="VN30")
    parser.add_argument("--min-date", default=None)
    parser.add_argument("--max-date", default=None)
    parser.add_argument(
        "--drop-incomplete-feature-rows",
        action="store_true",
        help="Drop rows with NaN feature values after rolling feature generation.",
    )
    return parser.parse_args()


def read_symbol_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"Symbol", "TradingDate", "Open", "High", "Low", "Close", "Volume", "Value"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out = df.rename(columns={"Symbol": "Ticker", "TradingDate": "Date"}).copy()
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    out["Ticker"] = out["Ticker"].fillna(path.stem).astype(str).str.upper()

    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Value"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[RAW_COLUMNS]
    out = out.dropna(subset=["Date", "Ticker", "Close"])
    out = out.drop_duplicates(subset=["Ticker", "Date"], keep="last")
    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(series / series.shift(periods))


def add_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for ticker, g in df.groupby("Ticker", sort=True):
        g = g.sort_values("Date").copy()
        close = g["Close"].replace(0, np.nan)
        open_ = g["Open"].replace(0, np.nan)
        high = g["High"].replace(0, np.nan)
        low = g["Low"].replace(0, np.nan)
        volume = g["Volume"].replace(0, np.nan)
        value = g["Value"].replace(0, np.nan)

        g["ret_1d"] = log_return(close, 1)
        g["ret_5d"] = log_return(close, 5)
        g["ret_20d"] = log_return(close, 20)
        g["oc_return"] = np.log(close / open_)
        g["hl_range"] = (high - low) / close
        g["volume_change_1d"] = log_return(volume, 1)
        g["value_change_1d"] = log_return(value, 1)
        g["rolling_vol_5"] = g["ret_1d"].rolling(5, min_periods=5).std()
        g["rolling_vol_20"] = g["ret_1d"].rolling(20, min_periods=20).std()
        g["ma_ratio_5"] = close / close.rolling(5, min_periods=5).mean() - 1.0
        g["ma_ratio_20"] = close / close.rolling(20, min_periods=20).mean() - 1.0
        g["volume_ma_ratio_5"] = volume / volume.rolling(5, min_periods=5).mean() - 1.0
        g["volume_ma_ratio_20"] = volume / volume.rolling(20, min_periods=20).mean() - 1.0
        g["target_return_1d"] = np.log(close.shift(-1) / close)
        g["target_direction_1d"] = np.sign(g["target_return_1d"]).astype("float")
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def build_market_features(raw_panel: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    market = raw_panel[raw_panel["Ticker"] == benchmark].sort_values("Date").copy()
    if market.empty:
        raise ValueError(f"Benchmark ticker {benchmark!r} was not found.")

    close = market["Close"].replace(0, np.nan)
    market["market_close"] = close
    market["market_ret_1d"] = log_return(close, 1)
    market["market_ret_5d"] = log_return(close, 5)
    market["market_vol_20"] = market["market_ret_1d"].rolling(20, min_periods=20).std()
    return market[["Date", "market_close", "market_ret_1d", "market_ret_5d", "market_vol_20"]]


def write_metadata(
    raw_panel: pd.DataFrame,
    feature_panel: pd.DataFrame,
    balanced_feature_panel: pd.DataFrame,
    output_dir: Path,
    benchmark: str,
    raw_path: Path,
    feature_path: Path,
    balanced_feature_path: Path,
) -> None:
    stock_rows = raw_panel[raw_panel["Ticker"] != benchmark]
    per_ticker = (
        stock_rows.groupby("Ticker")["Date"]
        .agg(rows="count", start="min", end="max")
        .reset_index()
    )
    per_ticker["start"] = per_ticker["start"].dt.strftime("%Y-%m-%d")
    per_ticker["end"] = per_ticker["end"].dt.strftime("%Y-%m-%d")

    metadata = {
        "source_dir": "Dataset/VN30",
        "benchmark": benchmark,
        "raw_panel_path": str(raw_path.as_posix()),
        "feature_panel_path": str(feature_path.as_posix()),
        "balanced_feature_panel_path": str(balanced_feature_path.as_posix()),
        "raw_rows": int(len(raw_panel)),
        "feature_rows": int(len(feature_panel)),
        "balanced_feature_rows": int(len(balanced_feature_panel)),
        "num_stock_tickers": int(stock_rows["Ticker"].nunique()),
        "stock_tickers": sorted(stock_rows["Ticker"].unique().tolist()),
        "date_start": raw_panel["Date"].min().strftime("%Y-%m-%d"),
        "date_end": raw_panel["Date"].max().strftime("%Y-%m-%d"),
        "feature_columns": STOCK_FEATURE_COLUMNS,
        "target_columns": ["target_return_1d", "target_direction_1d"],
        "recommended_tensor_shape": {
            "X": "(samples, lookback, num_stocks, num_features)",
            "Y": "(samples, num_stocks)",
        },
        "notes": [
            "Rows with Ticker == VN30 are benchmark/index rows and are excluded from stock-node feature panel.",
            "Rolling features use current and past observations only.",
            "target_return_1d is next trading day's log return per stock.",
            "Correlation graphs for experiments should be computed from train dates only.",
            "The balanced feature panel keeps only dates where all 30 stock nodes are present.",
            "Large single-day returns may indicate corporate actions if source prices are not adjusted.",
        ],
        "per_ticker_coverage": per_ticker.to_dict(orient="records"),
    }

    metadata_path = output_dir / "vn30_panel_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [read_symbol_file(path) for path in sorted(input_dir.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    raw_panel = pd.concat(frames, ignore_index=True)
    if args.min_date:
        raw_panel = raw_panel[raw_panel["Date"] >= pd.Timestamp(args.min_date)]
    if args.max_date:
        raw_panel = raw_panel[raw_panel["Date"] <= pd.Timestamp(args.max_date)]
    raw_panel = raw_panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    raw_path = output_dir / "vn30_panel_raw.csv"
    raw_panel.to_csv(raw_path, index=False, date_format="%Y-%m-%d")

    benchmark = args.benchmark.upper()
    stock_panel = raw_panel[raw_panel["Ticker"] != benchmark].copy()
    feature_panel = add_stock_features(stock_panel)
    market_features = build_market_features(raw_panel, benchmark)
    feature_panel = feature_panel.merge(market_features, on="Date", how="left")
    feature_panel["relative_ret_1d"] = feature_panel["ret_1d"] - feature_panel["market_ret_1d"]

    feature_panel = feature_panel.replace([np.inf, -np.inf], np.nan)
    feature_panel = feature_panel.dropna(subset=["target_return_1d"])
    if args.drop_incomplete_feature_rows:
        feature_panel = feature_panel.dropna(subset=STOCK_FEATURE_COLUMNS)

    feature_panel = feature_panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    feature_path = output_dir / "vn30_panel_features.csv"
    feature_panel.to_csv(feature_path, index=False, date_format="%Y-%m-%d")

    ticker_count_by_date = feature_panel.groupby("Date")["Ticker"].nunique()
    full_dates = ticker_count_by_date[ticker_count_by_date == stock_panel["Ticker"].nunique()].index
    balanced_feature_panel = feature_panel[feature_panel["Date"].isin(full_dates)].copy()
    balanced_feature_panel = balanced_feature_panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    balanced_feature_path = output_dir / "vn30_panel_features_balanced.csv"
    balanced_feature_panel.to_csv(balanced_feature_path, index=False, date_format="%Y-%m-%d")

    write_metadata(
        raw_panel,
        feature_panel,
        balanced_feature_panel,
        output_dir,
        benchmark,
        raw_path,
        feature_path,
        balanced_feature_path,
    )

    print(f"Raw panel saved: {raw_path} ({len(raw_panel):,} rows)")
    print(f"Feature panel saved: {feature_path} ({len(feature_panel):,} rows)")
    print(f"Balanced feature panel saved: {balanced_feature_path} ({len(balanced_feature_panel):,} rows)")
    print(f"Metadata saved: {output_dir / 'vn30_panel_metadata.json'}")
    print(f"Stock tickers: {stock_panel['Ticker'].nunique()}")
    print(f"Date range: {raw_panel['Date'].min().date()} to {raw_panel['Date'].max().date()}")


if __name__ == "__main__":
    main()
