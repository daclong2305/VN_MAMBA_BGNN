from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from stock_node_models import StockGraphModelConfig, create_model
from vn30_stock_graph_dataset import build_correlation_graph, build_vn30_panel_loaders


DEFAULT_MODELS = [
    "lstm",
    "transformer",
    "frets",
    "stockmixer",
    "agcrn",
    "fouriergnn",
    "mambastock",
    "original_mamba_bgnn_full",
    "stock_mamba_hybrid",
]
BASELINE_PROVENANCE_PATH = Path(__file__).with_name("baseline_provenance.json")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument(
        "--rank-loss-weight",
        type=float,
        default=0.05,
        help="Weight for pairwise within-day ranking loss. Set 0 to disable.",
    )
    parser.add_argument(
        "--rank-loss-temperature",
        type=float,
        default=0.01,
        help="Temperature for pairwise ranking logits; smaller values emphasize rank gaps.",
    )
    parser.add_argument(
        "--rank-loss-models",
        nargs="+",
        default=DEFAULT_MODELS.copy(),
        help="Model names that receive the ranking loss term. Defaults to every final-comparison model.",
    )
    parser.add_argument(
        "--portfolio-loss-weight",
        type=float,
        default=0.0,
        help="Weight for soft top-k excess-return portfolio loss. Set 0 to disable.",
    )
    parser.add_argument(
        "--portfolio-loss-temperature",
        type=float,
        default=0.02,
        help="Softmax temperature for differentiable portfolio weights.",
    )
    parser.add_argument(
        "--portfolio-loss-models",
        nargs="+",
        default=DEFAULT_MODELS.copy(),
        help="Model names that receive the soft portfolio loss term.",
    )
    parser.add_argument(
        "--topk-loss-weight",
        type=float,
        default=0.0,
        help="Weight for differentiable top-k Sharpe/return loss. Set 0 to disable.",
    )
    parser.add_argument(
        "--topk-loss-temperature",
        type=float,
        default=0.01,
        help="Temperature for differentiable top-k portfolio weights.",
    )
    parser.add_argument(
        "--topk-loss-models",
        nargs="+",
        default=DEFAULT_MODELS.copy(),
        help="Model names that receive the differentiable top-k portfolio loss.",
    )
    parser.add_argument(
        "--listwise-rank-loss-weight",
        type=float,
        default=0.05,
        help="Weight for listwise cross-sectional ranking loss. Set 0 to disable.",
    )
    parser.add_argument(
        "--listwise-rank-loss-temperature",
        type=float,
        default=0.01,
        help="Temperature for listwise ranking target and prediction distributions.",
    )
    parser.add_argument(
        "--listwise-rank-loss-models",
        nargs="+",
        default=DEFAULT_MODELS.copy(),
        help="Model names that receive the listwise ranking loss term. Defaults to every final-comparison model.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=["loss", "forecast_rank", "val_sharpe", "val_return", "rank_ic"],
        default="forecast_rank",
        help="Validation criterion used for early-stopping checkpoint selection.",
    )
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--quick", action="store_true", help="Run a fast smoke experiment.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--graph-correlation",
        choices=["positive", "absolute"],
        default="positive",
        help="Static graph construction from train returns. positive ignores negative correlations.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--transaction-cost", type=float, default=0.0015)
    parser.add_argument(
        "--score-mode",
        choices=["mu", "risk_adjusted", "positive_confidence", "rank_zscore"],
        default="mu",
        help="Prediction score used for portfolio selection.",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=0.25,
        help="Penalty applied to sigma when score-mode uses uncertainty.",
    )
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


def log_returns_to_arithmetic(log_returns: torch.Tensor) -> torch.Tensor:
    """Convert log returns to arithmetic returns for economic/portfolio metrics."""
    return torch.expm1(log_returns)


def pairwise_rank_loss(mu: torch.Tensor, y: torch.Tensor, temperature: float = 0.01) -> torch.Tensor:
    pred_diff = (mu.unsqueeze(2) - mu.unsqueeze(1)) / max(temperature, 1e-8)
    true_diff = y.unsqueeze(2) - y.unsqueeze(1)
    target = torch.sign(true_diff)
    mask = target != 0
    if not torch.any(mask):
        return mu.new_tensor(0.0)
    return F.softplus(-target[mask] * pred_diff[mask]).mean()


def listwise_rank_loss(mu: torch.Tensor, y: torch.Tensor, temperature: float = 0.01) -> torch.Tensor:
    pred_logits = (mu - mu.mean(dim=1, keepdim=True)) / max(temperature, 1e-8)
    target_logits = (y - y.mean(dim=1, keepdim=True)) / max(temperature, 1e-8)
    target_probs = F.softmax(target_logits, dim=1)
    pred_log_probs = F.log_softmax(pred_logits, dim=1)
    return -(target_probs * pred_log_probs).sum(dim=1).mean()


def soft_portfolio_loss(mu: torch.Tensor, y: torch.Tensor, temperature: float = 0.02) -> torch.Tensor:
    scores = mu - mu.mean(dim=1, keepdim=True)
    weights = F.softmax(scores / max(temperature, 1e-8), dim=1)
    arithmetic_returns = log_returns_to_arithmetic(y)
    excess_returns = arithmetic_returns - arithmetic_returns.mean(dim=1, keepdim=True)
    return -(weights * excess_returns).sum(dim=1).mean()


def soft_topk_weights(scores: torch.Tensor, top_k: int, temperature: float = 0.01) -> torch.Tensor:
    """Differentiable equal-weight approximation of repeated top-k selection."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    k = min(top_k, scores.shape[1])
    logits = (scores - scores.mean(dim=1, keepdim=True)) / max(temperature, 1e-8)
    remaining = torch.ones_like(logits)
    selected = torch.zeros_like(logits)
    for _ in range(k):
        step_logits = logits + remaining.clamp_min(1e-6).log()
        probs = F.softmax(step_logits, dim=1)
        selected = selected + probs
        remaining = remaining * (1.0 - probs).clamp_min(1e-6)
    weights = selected / float(k)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)


def soft_topk_portfolio_loss(
    mu: torch.Tensor,
    y: torch.Tensor,
    top_k: int,
    temperature: float = 0.01,
    sharpe_weight: float = 0.25,
    downside_weight: float = 0.25,
) -> torch.Tensor:
    weights = soft_topk_weights(mu, top_k=top_k, temperature=temperature)
    arithmetic_returns = log_returns_to_arithmetic(y)
    portfolio_returns = (weights * arithmetic_returns).sum(dim=1)
    benchmark_returns = arithmetic_returns.mean(dim=1)
    excess_returns = portfolio_returns - benchmark_returns
    mean_excess = excess_returns.mean()
    sharpe = mean_excess / excess_returns.std(unbiased=False).clamp_min(1e-4)
    downside = F.relu(-excess_returns).mean()
    return -mean_excess - sharpe_weight * sharpe + downside_weight * downside


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


def topk_ranking_metrics(mu: torch.Tensor, y: torch.Tensor, top_k: int = 5) -> dict[str, float]:
    """Ranking metrics on log-return targets, with economic returns reported as arithmetic returns."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    k = min(top_k, mu.shape[1])
    overlaps = []
    hit_rates = []
    top_returns = []
    bottom_returns = []
    oracle_top_returns = []
    for pred_day, true_day in zip(mu, y):
        true_day_arithmetic = log_returns_to_arithmetic(true_day)
        pred_order = torch.argsort(pred_day, descending=True)
        true_order = torch.argsort(true_day, descending=True)
        pred_top = pred_order[:k]
        pred_bottom = pred_order[-k:]
        true_top = set(true_order[:k].tolist())
        overlap = sum(1 for idx in pred_top.tolist() if idx in true_top)
        overlaps.append(overlap / float(k))
        hit_rates.append(float(overlap > 0))
        top_returns.append(true_day_arithmetic[pred_top].mean().item())
        bottom_returns.append(true_day_arithmetic[pred_bottom].mean().item())
        oracle_top_returns.append(true_day_arithmetic[true_order[:k]].mean().item())
    top_return = float(np.mean(top_returns)) if top_returns else 0.0
    bottom_return = float(np.mean(bottom_returns)) if bottom_returns else 0.0
    oracle_top_return = float(np.mean(oracle_top_returns)) if oracle_top_returns else 0.0
    return {
        "topk_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "topk_hit_rate": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "pred_topk_mean_return": top_return,
        "pred_bottomk_mean_return": bottom_return,
        "top_bottom_return_spread": top_return - bottom_return,
        "oracle_topk_mean_return": oracle_top_return,
        "topk_return_capture": 0.0 if abs(oracle_top_return) < 1e-12 else top_return / oracle_top_return,
    }


def compute_metrics(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor, top_k: int = 5) -> dict[str, float]:
    mu = mu.detach().cpu()
    y = y.detach().cpu()
    log_var = log_var.detach().cpu()
    flat_mu = mu.reshape(-1)
    flat_y = y.reshape(-1)
    rmse = torch.sqrt(torch.mean((flat_mu - flat_y) ** 2)).item()
    mae = torch.mean(torch.abs(flat_mu - flat_y)).item()
    corr = torch.corrcoef(torch.stack([flat_mu, flat_y]))[0, 1].item() if flat_mu.numel() > 1 else 0.0
    nll = gaussian_nll(mu, log_var, y).item()
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "ic": 0.0 if np.isnan(corr) else corr,
        "rank_ic_by_day": rank_ic_by_day(mu, y),
        "directional_accuracy": direction_accuracy(mu, y),
        "nll": nll,
    }
    metrics.update(topk_ranking_metrics(mu, y, top_k=top_k))
    return metrics


def score_predictions(
    pred: torch.Tensor,
    log_var: torch.Tensor | None = None,
    score_mode: str = "mu",
    risk_aversion: float = 0.25,
) -> torch.Tensor:
    if score_mode == "mu":
        return pred
    if score_mode == "rank_zscore":
        centered = pred - pred.mean(dim=1, keepdim=True)
        return centered / pred.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    if log_var is None:
        raise ValueError(f"score_mode={score_mode} requires log_var")
    sigma = torch.exp(0.5 * log_var).clamp_min(1e-8)
    if score_mode == "risk_adjusted":
        return pred - risk_aversion * sigma
    if score_mode == "positive_confidence":
        return pred / sigma
    raise ValueError(f"Unknown score_mode: {score_mode}")


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


def validation_portfolio_metrics(
    pred: torch.Tensor,
    log_var: torch.Tensor,
    true_returns: torch.Tensor,
    top_k: int,
    transaction_cost: float,
    rebalance_every: int,
    hold_k: int | None,
    min_score: float | None,
    score_mode: str,
    risk_aversion: float,
) -> dict[str, float]:
    scores = score_predictions(pred, log_var, score_mode=score_mode, risk_aversion=risk_aversion)
    dummy_dates = [str(i) for i in range(scores.shape[0])]
    dummy_tickers = [str(i) for i in range(scores.shape[1])]
    metrics, _ = backtest_topk(
        scores,
        true_returns,
        dummy_dates,
        dummy_tickers,
        top_k=top_k,
        transaction_cost=transaction_cost,
        rebalance_every=rebalance_every,
        hold_k=hold_k,
        min_score=min_score,
    )
    return metrics


def checkpoint_score(
    checkpoint_metric: str,
    val_loss: torch.Tensor,
    val_mu: torch.Tensor,
    val_log_var: torch.Tensor,
    val_y: torch.Tensor,
    top_k: int,
    transaction_cost: float,
    rebalance_every: int,
    hold_k: int | None,
    min_score: float | None,
    score_mode: str,
    risk_aversion: float,
) -> tuple[float, dict[str, float]]:
    val_metrics = compute_metrics(val_mu, val_log_var, val_y, top_k=top_k)
    val_bt = validation_portfolio_metrics(
        val_mu,
        val_log_var,
        val_y,
        top_k=top_k,
        transaction_cost=transaction_cost,
        rebalance_every=rebalance_every,
        hold_k=hold_k,
        min_score=min_score,
        score_mode=score_mode,
        risk_aversion=risk_aversion,
    )
    if checkpoint_metric == "loss":
        return -float(val_loss.item()), val_bt
    if checkpoint_metric == "forecast_rank":
        score = -float(val_metrics["nll"]) + 0.25 * float(val_metrics["rank_ic_by_day"])
        return score, {**val_bt, **val_metrics}
    if checkpoint_metric == "val_sharpe":
        return float(val_bt["annualized_sharpe"]), val_bt
    if checkpoint_metric == "val_return":
        return float(val_bt["total_net_return"]), val_bt
    if checkpoint_metric == "rank_ic":
        return rank_ic_by_day(val_mu, val_y), val_bt
    raise ValueError(f"Unknown checkpoint_metric: {checkpoint_metric}")


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    rank_loss_weight: float = 0.0,
    rank_loss_temperature: float = 0.01,
    listwise_rank_loss_weight: float = 0.0,
    listwise_rank_loss_temperature: float = 0.01,
    portfolio_loss_weight: float = 0.0,
    portfolio_loss_temperature: float = 0.02,
    topk_loss_weight: float = 0.0,
    topk_loss_temperature: float = 0.01,
    top_k: int = 5,
    transaction_cost: float = 0.0015,
    rebalance_every: int = 1,
    hold_k: int | None = None,
    min_score: float | None = None,
    score_mode: str = "mu",
    risk_aversion: float = 0.25,
    checkpoint_metric: str = "loss",
) -> tuple[nn.Module, float, dict[str, float]]:
    model.to(device)
    # The MAMBA-BGNN experiment protocol reports Adam; use it for every model.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_score = -float("inf")
    best_val_nll = float("inf")
    stale = 0
    epoch = 0
    train_epoch_times = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_epoch_started = time.perf_counter()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            mu, log_var = model(x)
            nll = gaussian_nll(mu, log_var, y)
            rank_loss = pairwise_rank_loss(mu, y, rank_loss_temperature) if rank_loss_weight > 0 else mu.new_tensor(0.0)
            listwise_loss = (
                listwise_rank_loss(mu, y, listwise_rank_loss_temperature)
                if listwise_rank_loss_weight > 0
                else mu.new_tensor(0.0)
            )
            portfolio_loss = (
                soft_portfolio_loss(mu, y, portfolio_loss_temperature)
                if portfolio_loss_weight > 0
                else mu.new_tensor(0.0)
            )
            topk_loss = (
                soft_topk_portfolio_loss(mu, y, top_k=top_k, temperature=topk_loss_temperature)
                if topk_loss_weight > 0
                else mu.new_tensor(0.0)
            )
            loss = (
                nll
                + rank_loss_weight * rank_loss
                + listwise_rank_loss_weight * listwise_loss
                + portfolio_loss_weight * portfolio_loss
                + topk_loss_weight * topk_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            train_losses.append(loss.item())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_epoch_times.append(time.perf_counter() - train_epoch_started)

        val_mu, val_log_var, val_y = collect_predictions(model, val_loader, device)
        val_nll = gaussian_nll(val_mu, val_log_var, val_y)
        val_rank = pairwise_rank_loss(val_mu, val_y, rank_loss_temperature) if rank_loss_weight > 0 else val_mu.new_tensor(0.0)
        val_listwise = (
            listwise_rank_loss(val_mu, val_y, listwise_rank_loss_temperature)
            if listwise_rank_loss_weight > 0
            else val_mu.new_tensor(0.0)
        )
        val_portfolio = (
            soft_portfolio_loss(val_mu, val_y, portfolio_loss_temperature)
            if portfolio_loss_weight > 0
            else val_mu.new_tensor(0.0)
        )
        val_topk = (
            soft_topk_portfolio_loss(val_mu, val_y, top_k=top_k, temperature=topk_loss_temperature)
            if topk_loss_weight > 0
            else val_mu.new_tensor(0.0)
        )
        val_loss = (
            val_nll
            + rank_loss_weight * val_rank
            + listwise_rank_loss_weight * val_listwise
            + portfolio_loss_weight * val_portfolio
            + topk_loss_weight * val_topk
        )
        current_score, val_bt = checkpoint_score(
            checkpoint_metric,
            val_loss,
            val_mu,
            val_log_var,
            val_y,
            top_k=top_k,
            transaction_cost=transaction_cost,
            rebalance_every=rebalance_every,
            hold_k=hold_k,
            min_score=min_score,
            score_mode=score_mode,
            risk_aversion=risk_aversion,
        )
        print(
            f"  epoch {epoch:03d} train_loss={np.mean(train_losses):.6f} "
            f"val_nll={val_nll.item():.6f} val_rank={val_rank.item():.6f} "
            f"val_listwise={val_listwise.item():.6f} val_portfolio={val_portfolio.item():.6f} "
            f"val_topk={val_topk.item():.6f} "
            f"val_loss={val_loss.item():.6f} val_sharpe={val_bt['annualized_sharpe']:.4f} "
            f"val_return={val_bt['total_net_return']:.4f} "
            f"val_rank_ic={val_bt.get('rank_ic_by_day', rank_ic_by_day(val_mu, val_y)):.4f}"
        )

        if current_score > best_score:
            best_score = current_score
            best_val_nll = val_nll.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print("  early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_wall_time_sec = time.perf_counter() - fit_started
    training_time_sec = float(np.sum(train_epoch_times))
    training_stats = {
        "epochs_ran": epoch,
        "training_time_sec": training_time_sec,
        "training_time_sec_per_epoch": training_time_sec / max(len(train_epoch_times), 1),
        "fit_wall_time_sec": fit_wall_time_sec,
    }
    return model, best_val_nll, training_stats


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
    """Backtest a top-k portfolio.

    ``true_returns`` are the dataset's next-day log returns. They are converted to
    arithmetic returns before computing portfolio P&L, trading-cost-adjusted net
    returns, cumulative return, Sharpe, and drawdown.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")
    if hold_k is not None and hold_k < top_k:
        raise ValueError("hold_k must be >= top_k when provided")

    pred_np = pred.detach().cpu().numpy()
    ret_np = np.expm1(true_returns.detach().cpu().numpy())
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
    static_adj = build_correlation_graph(data.y_train, absolute=args.graph_correlation == "absolute")
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
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model, best_val, training_stats = train_model(
            model,
            data.train_loader,
            data.val_loader,
            device,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            rank_loss_weight=args.rank_loss_weight if model_name in args.rank_loss_models else 0.0,
            rank_loss_temperature=args.rank_loss_temperature,
            listwise_rank_loss_weight=(
                args.listwise_rank_loss_weight if model_name in args.listwise_rank_loss_models else 0.0
            ),
            listwise_rank_loss_temperature=args.listwise_rank_loss_temperature,
            portfolio_loss_weight=args.portfolio_loss_weight if model_name in args.portfolio_loss_models else 0.0,
            portfolio_loss_temperature=args.portfolio_loss_temperature,
            topk_loss_weight=args.topk_loss_weight if model_name in args.topk_loss_models else 0.0,
            topk_loss_temperature=args.topk_loss_temperature,
            top_k=args.top_k,
            transaction_cost=args.transaction_cost,
            rebalance_every=args.rebalance_every,
            hold_k=None if args.hold_k == 0 else args.hold_k,
            min_score=args.min_score,
            score_mode=args.score_mode,
            risk_aversion=args.risk_aversion,
            checkpoint_metric=args.checkpoint_metric,
        )
        pred, log_var, true = collect_predictions(model, data.test_loader, device)
        metrics = compute_metrics(pred, log_var, true, top_k=args.top_k)
        portfolio_score = score_predictions(
            pred,
            log_var,
            score_mode=args.score_mode,
            risk_aversion=args.risk_aversion,
        )
        bt_metrics, bt_df = backtest_topk(
            portfolio_score,
            true,
            data.test_dates,
            data.tickers,
            top_k=args.top_k,
            transaction_cost=args.transaction_cost,
            rebalance_every=args.rebalance_every,
            hold_k=None if args.hold_k == 0 else args.hold_k,
            min_score=args.min_score,
        )
        row = {
            "model": model_name,
            "num_parameters": num_parameters,
            **training_stats,
            "best_val_nll": best_val,
            **metrics,
            **bt_metrics,
        }
        all_results.append(row)
        backtest_results.append({"model": model_name, **bt_metrics})
        bt_suffix = f"top{args.top_k}_reb{args.rebalance_every}_hold{args.hold_k}"
        bt_df.to_csv(output_dir / f"{model_name}_backtest_{bt_suffix}.csv", index=False)

        if args.turnover_sweep:
            for variant in turnover_sweep_variants(args.top_k):
                sweep_metrics, sweep_df = backtest_topk(
                    portfolio_score,
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
        model.to("cpu")
        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "num_stocks": config.num_stocks,
                    "num_features": config.num_features,
                    "lookback": config.lookback,
                    "hidden_dim": config.hidden_dim,
                    "dropout": config.dropout,
                },
                "num_parameters": num_parameters,
                "best_validation_score": best_val,
                "training_stats": training_stats,
            },
            output_dir / f"{model_name}_best.pt",
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    results_df["forecast_rank_score"] = (
        -results_df["nll"] + 0.25 * results_df["rank_ic_by_day"] + 0.05 * results_df["directional_accuracy"]
    )
    results_df = results_df.sort_values("forecast_rank_score", ascending=False)
    results_df.to_csv(output_dir / "vn30_model_comparison.csv", index=False)
    forecast_columns = [
        "model",
        "num_parameters",
        "epochs_ran",
        "training_time_sec",
        "training_time_sec_per_epoch",
        "fit_wall_time_sec",
        "forecast_rank_score",
        "best_val_nll",
        "nll",
        "rmse",
        "mae",
        "ic",
        "rank_ic_by_day",
        "directional_accuracy",
        "topk_overlap",
        "topk_hit_rate",
        "pred_topk_mean_return",
        "pred_bottomk_mean_return",
        "top_bottom_return_spread",
        "topk_return_capture",
    ]
    results_df[forecast_columns].to_csv(output_dir / "vn30_forecast_ranking_summary.csv", index=False)
    pd.DataFrame(backtest_results).to_csv(output_dir / "vn30_backtest_summary.csv", index=False)
    if turnover_sweep_results:
        turnover_sweep_df = pd.DataFrame(turnover_sweep_results)
        turnover_sweep_df.to_csv(output_dir / "vn30_turnover_sweep.csv", index=False)

    baseline_provenance = json.loads(BASELINE_PROVENANCE_PATH.read_text(encoding="utf-8"))
    experiment_config = {
        "args": vars(args),
        "data_artifacts": {
            "panel_sha256": sha256_file(args.panel_path),
            "metadata_sha256": sha256_file(args.metadata_path),
        },
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
        "baseline_provenance": baseline_provenance,
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(experiment_config, indent=2), encoding="utf-8")
    print("\n=== Summary ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
