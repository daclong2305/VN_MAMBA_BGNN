from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from stock_node_models import StockGraphModelConfig, create_model
from vn30_stock_graph_dataset import build_correlation_graph, build_vn30_panel_loaders


DEFAULT_MODELS = [
    "lstm",
    "transformer",
    "original_mamba_bgnn",
    "stock_mamba_no_graph",
    "stock_mamba_static",
    "stock_mamba_adaptive",
    "stock_mamba_hybrid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VN30 stock-node graph experiments.")
    parser.add_argument("--panel-path", default="Dataset/vn30_panel_features_balanced.csv")
    parser.add_argument("--metadata-path", default="Dataset/vn30_panel_metadata.json")
    parser.add_argument("--output-dir", default="vn30_stock_graph_results")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--quick", action="store_true", help="Run a fast smoke experiment.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--transaction-cost", type=float, default=0.0015)
    parser.add_argument(
        "--rebalance-every",
        type=int,
        default=1,
        help="Rebalance every N test days. 1 reproduces the original daily top-k backtest.",
    )
    parser.add_argument(
        "--hold-k",
        type=int,
        default=0,
        help="Keep existing names while they remain in the top hold-k ranks. 0 disables the hold buffer.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Only open new positions whose predicted return is at least this value.",
    )
    parser.add_argument(
        "--turnover-sweep",
        action="store_true",
        help="Also evaluate common lower-turnover portfolio rules without retraining models.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.gaussian_nll_loss(mu, y, log_var.exp().clamp_min(1e-8), full=True, reduction="mean")


def direction_accuracy(mu: torch.Tensor, y: torch.Tensor) -> float:
    mask = y != 0
    if mask.sum() == 0:
        return 0.0
    return (torch.sign(mu[mask]) == torch.sign(y[mask])).float().mean().item()


def rank_ic_by_day(mu: torch.Tensor, y: torch.Tensor) -> float:
    vals = []
    for pred_day, true_day in zip(mu, y):
        if pred_day.numel() < 2:
            continue
        pred_rank = torch.argsort(torch.argsort(pred_day)).float()
        true_rank = torch.argsort(torch.argsort(true_day)).float()
        pred_rank = pred_rank - pred_rank.mean()
        true_rank = true_rank - true_rank.mean()
        denom = pred_rank.norm() * true_rank.norm()
        if denom > 0:
            vals.append((pred_rank * true_rank).sum().item() / denom.item())
    return float(np.mean(vals)) if vals else 0.0


def compute_metrics(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    mu = mu.detach().cpu()
    y = y.detach().cpu()
    log_var = log_var.detach().cpu()
    flat_mu = mu.reshape(-1)
    flat_y = y.reshape(-1)
    rmse = torch.sqrt(torch.mean((flat_mu - flat_y) ** 2)).item()
    mae = torch.mean(torch.abs(flat_mu - flat_y)).item()
    corr = torch.corrcoef(torch.stack([flat_mu, flat_y]))[0, 1].item() if flat_mu.numel() > 1 else 0.0
    nll = gaussian_nll(mu, log_var, y).item()
    return {
        "rmse": rmse,
        "mae": mae,
        "ic": 0.0 if np.isnan(corr) else corr,
        "rank_ic_by_day": rank_ic_by_day(mu, y),
        "directional_accuracy": direction_accuracy(mu, y),
        "nll": nll,
    }


def collect_predictions(model: nn.Module, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    preds, logvars, trues = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            mu, log_var = model(x)
            preds.append(mu.cpu())
            logvars.append(log_var.cpu())
            trues.append(y.cpu())
    return torch.cat(preds, dim=0), torch.cat(logvars, dim=0), torch.cat(trues, dim=0)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> tuple[nn.Module, float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = float("inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            mu, log_var = model(x)
            loss = gaussian_nll(mu, log_var, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            train_losses.append(loss.item())

        val_mu, val_log_var, val_y = collect_predictions(model, val_loader, device)
        val_loss = gaussian_nll(val_mu, val_log_var, val_y).item()
        print(f"  epoch {epoch:03d} train_nll={np.mean(train_losses):.6f} val_nll={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print("  early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def backtest_topk(
    pred: torch.Tensor,
    true_returns: torch.Tensor,
    dates: list[str],
    tickers: list[str],
    top_k: int = 5,
    transaction_cost: float = 0.0015,
    rebalance_every: int = 1,
    hold_k: int | None = None,
    min_score: float | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")
    if hold_k is not None and hold_k < top_k:
        raise ValueError("hold_k must be >= top_k when provided")

    pred_np = pred.detach().cpu().numpy()
    ret_np = true_returns.detach().cpu().numpy()
    prev_weights = np.zeros(pred_np.shape[1], dtype=np.float32)
    current_holdings: list[int] = []
    rows = []
    portfolio_returns = []

    for i, date in enumerate(dates):
        should_rebalance = i == 0 or (i % rebalance_every == 0)
        if should_rebalance:
            order = np.argsort(pred_np[i])[::-1]
            rank = {stock_idx: rank_idx + 1 for rank_idx, stock_idx in enumerate(order)}

            kept = []
            if hold_k is not None:
                kept = [
                    stock_idx
                    for stock_idx in current_holdings
                    if rank.get(stock_idx, len(order) + 1) <= hold_k
                    and (min_score is None or pred_np[i, stock_idx] >= min_score)
                ]

            chosen = list(kept)
            for stock_idx in order:
                if len(chosen) >= top_k:
                    break
                if stock_idx in chosen:
                    continue
                if min_score is not None and pred_np[i, stock_idx] < min_score:
                    continue
                chosen.append(int(stock_idx))

            if len(chosen) < top_k:
                for stock_idx in order:
                    if len(chosen) >= top_k:
                        break
                    if stock_idx not in chosen:
                        chosen.append(int(stock_idx))
            current_holdings = chosen[:top_k]
        else:
            chosen = current_holdings

        weights = np.zeros(pred_np.shape[1], dtype=np.float32)
        weights[chosen] = 1.0 / top_k
        gross = float(np.sum(weights * ret_np[i]))
        turnover = float(np.sum(np.abs(weights - prev_weights)))
        trading_cost = transaction_cost * turnover
        net = gross - trading_cost
        portfolio_returns.append(net)
        rows.append(
            {
                "Date": date,
                "is_rebalance_day": should_rebalance,
                "gross_return": gross,
                "turnover": turnover,
                "trading_cost": trading_cost,
                "net_return": net,
                "selected_tickers": ",".join(tickers[j] for j in chosen),
            }
        )
        prev_weights = weights

    rets = np.asarray(portfolio_returns, dtype=np.float64)
    cumulative = np.cumprod(1.0 + rets)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative / running_max - 1.0
    sharpe = 0.0 if rets.std() == 0 else float(rets.mean() / rets.std() * np.sqrt(252))
    metrics = {
        "top_k": top_k,
        "rebalance_every": rebalance_every,
        "hold_k": 0 if hold_k is None else hold_k,
        "min_score": "" if min_score is None else min_score,
        "total_net_return": float(cumulative[-1] - 1.0),
        "annualized_sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "mean_daily_net_return": float(rets.mean()),
        "average_turnover": float(np.mean([r["turnover"] for r in rows])),
        "total_trading_cost": float(np.sum([r["trading_cost"] for r in rows])),
        "positive_day_ratio": float(np.mean([r["net_return"] > 0 for r in rows])),
    }
    return metrics, pd.DataFrame(rows)


def turnover_sweep_variants(top_k: int) -> list[dict]:
    hold_2x = max(top_k * 2, top_k)
    return [
        {"variant": "daily_topk", "rebalance_every": 1, "hold_k": None, "min_score": None},
        {"variant": "weekly_topk", "rebalance_every": 5, "hold_k": None, "min_score": None},
        {"variant": "daily_hold_buffer", "rebalance_every": 1, "hold_k": hold_2x, "min_score": None},
        {"variant": "weekly_hold_buffer", "rebalance_every": 5, "hold_k": hold_2x, "min_score": None},
        {"variant": "weekly_positive_only", "rebalance_every": 5, "hold_k": hold_2x, "min_score": 0.0},
    ]


def save_predictions(
    output_dir: Path,
    model_name: str,
    pred: torch.Tensor,
    log_var: torch.Tensor,
    true: torch.Tensor,
    dates: list[str],
    tickers: list[str],
) -> None:
    rows = []
    pred_np = pred.numpy()
    sigma_np = np.exp(0.5 * log_var.numpy())
    true_np = true.numpy()
    for i, date in enumerate(dates):
        for j, ticker in enumerate(tickers):
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "y": true_np[i, j],
                    "mu": pred_np[i, j],
                    "sigma": sigma_np[i, j],
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / f"{model_name}_test_predictions.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.patience = 2
        args.models = args.models[:2] if args.models == DEFAULT_MODELS else args.models

    set_seed(args.seed)
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = build_vn30_panel_loaders(
        panel_path=args.panel_path,
        metadata_path=args.metadata_path,
        lookback=args.lookback,
        batch_size=args.batch_size,
    )
    static_adj = build_correlation_graph(data.y_train)
    config = StockGraphModelConfig(
        num_stocks=len(data.tickers),
        num_features=len(data.feature_columns),
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"X train: {tuple(data.x_train.shape)} | y train: {tuple(data.y_train.shape)}")
    print(f"Tickers: {len(data.tickers)} | Features: {len(data.feature_columns)}")
    print(f"Output: {output_dir}")

    all_results = []
    backtest_results = []
    turnover_sweep_results = []
    adjacency_df = pd.DataFrame(static_adj.numpy(), index=data.tickers, columns=data.tickers)
    adjacency_df.to_csv(output_dir / "static_correlation_adjacency.csv")

    for model_name in args.models:
        print(f"\n=== Training {model_name} ===")
        model = create_model(model_name, config, static_adjacency=static_adj)
        model, best_val = train_model(
            model,
            data.train_loader,
            data.val_loader,
            device,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        pred, log_var, true = collect_predictions(model, data.test_loader, device)
        metrics = compute_metrics(pred, log_var, true)
        bt_metrics, bt_df = backtest_topk(
            pred,
            true,
            data.test_dates,
            data.tickers,
            top_k=args.top_k,
            transaction_cost=args.transaction_cost,
            rebalance_every=args.rebalance_every,
            hold_k=None if args.hold_k == 0 else args.hold_k,
            min_score=args.min_score,
        )
        row = {"model": model_name, "best_val_nll": best_val, **metrics, **bt_metrics}
        all_results.append(row)
        backtest_results.append({"model": model_name, **bt_metrics})
        bt_suffix = f"top{args.top_k}_reb{args.rebalance_every}_hold{args.hold_k}"
        bt_df.to_csv(output_dir / f"{model_name}_backtest_{bt_suffix}.csv", index=False)

        if args.turnover_sweep:
            for variant in turnover_sweep_variants(args.top_k):
                sweep_metrics, sweep_df = backtest_topk(
                    pred,
                    true,
                    data.test_dates,
                    data.tickers,
                    top_k=args.top_k,
                    transaction_cost=args.transaction_cost,
                    rebalance_every=variant["rebalance_every"],
                    hold_k=variant["hold_k"],
                    min_score=variant["min_score"],
                )
                turnover_sweep_results.append(
                    {"model": model_name, "variant": variant["variant"], **sweep_metrics}
                )
                sweep_df.to_csv(
                    output_dir / f"{model_name}_backtest_{variant['variant']}_top{args.top_k}.csv",
                    index=False,
                )
        save_predictions(output_dir, model_name, pred, log_var, true, data.test_dates, data.tickers)
        print(json.dumps(row, indent=2))

    results_df = pd.DataFrame(all_results).sort_values("rank_ic_by_day", ascending=False)
    results_df.to_csv(output_dir / "vn30_model_comparison.csv", index=False)
    pd.DataFrame(backtest_results).to_csv(output_dir / "vn30_backtest_summary.csv", index=False)
    if turnover_sweep_results:
        turnover_sweep_df = pd.DataFrame(turnover_sweep_results)
        turnover_sweep_df.to_csv(output_dir / "vn30_turnover_sweep.csv", index=False)

    experiment_config = {
        "args": vars(args),
        "tickers": data.tickers,
        "feature_columns": data.feature_columns,
        "train_period": [data.train_dates[0], data.train_dates[-1]],
        "val_period": [data.val_dates[0], data.val_dates[-1]],
        "test_period": [data.test_dates[0], data.test_dates[-1]],
        "tensor_shapes": {
            "x_train": list(data.x_train.shape),
            "y_train": list(data.y_train.shape),
            "x_val": list(data.x_val.shape),
            "y_val": list(data.y_val.shape),
            "x_test": list(data.x_test.shape),
            "y_test": list(data.y_test.shape),
        },
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(experiment_config, indent=2), encoding="utf-8")
    print("\n=== Summary ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
