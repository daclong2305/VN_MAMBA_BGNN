from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class VN30PanelData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    test_dates: list[str]
    tickers: list[str]
    feature_columns: list[str]
    train_dates: list[str]
    val_dates: list[str]
    scaler_mean: torch.Tensor
    scaler_std: torch.Tensor


def load_metadata(metadata_path: str | Path = "Dataset/vn30_panel_metadata.json") -> dict:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _split_indices(num_samples: int, train_ratio: float, val_ratio: float) -> tuple[slice, slice, slice]:
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= num_samples:
        raise ValueError(
            f"Invalid split for {num_samples} samples: train_end={train_end}, val_end={val_end}"
        )
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, num_samples)


def _make_tensor_panel(
    panel_path: str | Path,
    tickers: Sequence[str],
    feature_columns: Sequence[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(panel_path, parse_dates=["Date"])
    required = {"Date", "Ticker", target_column, *feature_columns}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{panel_path} is missing columns: {sorted(missing)}")

    df = df[df["Ticker"].isin(tickers)].copy()
    df["Ticker"] = pd.Categorical(df["Ticker"], categories=list(tickers), ordered=True)
    df = df.sort_values(["Date", "Ticker"])

    counts = df.groupby("Date", observed=True)["Ticker"].nunique()
    bad_dates = counts[counts != len(tickers)]
    if len(bad_dates) > 0:
        first_bad = bad_dates.index[0].strftime("%Y-%m-%d")
        raise ValueError(
            f"Panel is not balanced: {first_bad} has {int(bad_dates.iloc[0])} tickers, "
            f"expected {len(tickers)}. Use vn30_panel_features_balanced.csv."
        )

    dates = sorted(df["Date"].drop_duplicates())
    features = []
    targets = []
    for date in dates:
        day = df[df["Date"] == date].sort_values("Ticker")
        features.append(day[list(feature_columns)].to_numpy(dtype=np.float32))
        targets.append(day[target_column].to_numpy(dtype=np.float32))

    feature_arr = np.stack(features, axis=0)  # (T,S,F)
    target_arr = np.stack(targets, axis=0)  # (T,S)
    date_labels = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]
    return feature_arr, target_arr, date_labels


def _make_windows(
    features: np.ndarray,
    targets: np.ndarray,
    dates: Sequence[str],
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, ys, y_dates = [], [], []
    for end_idx in range(lookback - 1, len(features)):
        xs.append(features[end_idx - lookback + 1 : end_idx + 1])
        ys.append(targets[end_idx])
        y_dates.append(dates[end_idx])
    return np.stack(xs, axis=0), np.stack(ys, axis=0), y_dates


def _fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def build_vn30_panel_loaders(
    panel_path: str | Path = "Dataset/vn30_panel_features_balanced.csv",
    metadata_path: str | Path = "Dataset/vn30_panel_metadata.json",
    lookback: int = 20,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    target_column: str = "target_return_1d",
    feature_columns: Sequence[str] | None = None,
) -> VN30PanelData:
    metadata = load_metadata(metadata_path)
    tickers = metadata["stock_tickers"]
    selected_features = list(feature_columns or metadata["feature_columns"])

    features, targets, dates = _make_tensor_panel(panel_path, tickers, selected_features, target_column)
    x, y, sample_dates = _make_windows(features, targets, dates, lookback)

    train_slice, val_slice, test_slice = _split_indices(len(x), train_ratio, val_ratio)
    x_train, y_train = x[train_slice], y[train_slice]
    x_val, y_val = x[val_slice], y[val_slice]
    x_test, y_test = x[test_slice], y[test_slice]

    mean, std = _fit_standardizer(x_train)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    return VN30PanelData(
        train_loader=_make_loader(x_train, y_train, batch_size, shuffle=True),
        val_loader=_make_loader(x_val, y_val, batch_size, shuffle=False),
        test_loader=_make_loader(x_test, y_test, batch_size, shuffle=False),
        x_train=torch.tensor(x_train, dtype=torch.float32),
        y_train=torch.tensor(y_train, dtype=torch.float32),
        x_val=torch.tensor(x_val, dtype=torch.float32),
        y_val=torch.tensor(y_val, dtype=torch.float32),
        x_test=torch.tensor(x_test, dtype=torch.float32),
        y_test=torch.tensor(y_test, dtype=torch.float32),
        train_dates=list(sample_dates[train_slice]),
        val_dates=list(sample_dates[val_slice]),
        test_dates=list(sample_dates[test_slice]),
        tickers=tickers,
        feature_columns=selected_features,
        scaler_mean=torch.tensor(mean, dtype=torch.float32),
        scaler_std=torch.tensor(std, dtype=torch.float32),
    )


def build_correlation_graph(
    y_train: torch.Tensor,
    absolute: bool = True,
    self_loop_weight: float = 1.0,
) -> torch.Tensor:
    """Build a row-normalized static stock graph from train target returns only."""
    returns = y_train.detach().cpu().numpy()
    corr = np.corrcoef(returns, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if absolute:
        corr = np.abs(corr)
    else:
        corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, self_loop_weight)
    corr = corr / np.clip(corr.sum(axis=1, keepdims=True), 1e-8, None)
    return torch.tensor(corr, dtype=torch.float32)


if __name__ == "__main__":
    data = build_vn30_panel_loaders()
    batch_x, batch_y = next(iter(data.train_loader))
    print(f"X batch: {tuple(batch_x.shape)}")
    print(f"Y batch: {tuple(batch_y.shape)}")
    print(f"Tickers: {len(data.tickers)}")
    print(f"Features: {len(data.feature_columns)}")
    print(f"Train/Val/Test: {len(data.x_train)}/{len(data.x_val)}/{len(data.x_test)}")
