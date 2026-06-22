from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vn30_official_baseline_cores import (
    AGCRNCore,
    FourierGNNCore,
    FreTSCore,
    MambaStockCore,
    StockMixerCore,
)

if TYPE_CHECKING:
    from stock_node_models import StockGraphModelConfig


class DeterministicBaselineAdapter(nn.Module):
    """Give deterministic baselines the common ``(mu, log_var)`` interface.

    The scalar variance is deliberately independent of the input and stocks. It
    lets every deterministic baseline use the same Gaussian training objective
    without adding a second, architecture-specific forecasting network.
    """

    def __init__(self, initial_log_var: float = -7.0):
        super().__init__()
        self.log_var = nn.Parameter(torch.tensor(float(initial_log_var)))

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.forward_mean(x)
        return mu, self.log_var.clamp(-10.0, 5.0).expand_as(mu)


class FreTSBaseline(DeterministicBaselineAdapter):
    """Official FreTS core with VN30 stocks as multivariate channels.

    VN30 adapter: a shared linear projection maps each stock's indicators to one
    temporal signal, then the official frequency channel/temporal learners model
    all stocks jointly and forecast one return per stock.
    """

    def __init__(self, config: StockGraphModelConfig):
        super().__init__()
        self.feature_encoder = nn.Linear(config.num_features, 1)
        self.core = FreTSCore(
            seq_len=config.lookback,
            num_channels=config.num_stocks,
            pred_len=1,
            embed_size=128,
            hidden_size=256,
            use_channel_mixing=True,
        )

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        panel_signal = self.feature_encoder(x).squeeze(-1)
        return self.core(panel_signal).squeeze(1)


class FourierGNNBaseline(DeterministicBaselineAdapter):
    """Official FourierGNN core with VN30 stocks as graph nodes."""

    def __init__(self, config: StockGraphModelConfig):
        super().__init__()
        self.feature_encoder = nn.Linear(config.num_features, 1)
        self.core = FourierGNNCore(
            seq_len=config.lookback,
            num_nodes=config.num_stocks,
            pred_len=1,
            embed_size=128,
            hidden_size=256,
            projection_width=8,
        )

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        panel_signal = self.feature_encoder(x).squeeze(-1)
        return self.core(panel_signal).squeeze(-1)


class AGCRNBaseline(DeterministicBaselineAdapter):
    """Official AGCRN encoder/predictor with VN30 stocks as graph nodes."""

    def __init__(self, config: StockGraphModelConfig):
        super().__init__()
        self.core = AGCRNCore(
            num_nodes=config.num_stocks,
            input_dim=config.num_features,
            hidden_dim=64,
            output_dim=1,
            horizon=1,
            num_layers=2,
            embed_dim=10,
            cheb_k=2,
        )

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)[:, 0, :, 0]


class StockMixerBaseline(DeterministicBaselineAdapter):
    """Batch-aware clean-room StockMixer with stocks as cross-sectional nodes."""

    def __init__(self, config: StockGraphModelConfig):
        super().__init__()
        if config.lookback < 2:
            raise ValueError("StockMixer requires lookback >= 2")
        self.core = StockMixerCore(
            num_stocks=config.num_stocks,
            time_steps=config.lookback,
            channels=config.num_features,
            stock_hidden_dim=20,
        )

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x.permute(0, 2, 1, 3))


class MambaStockBaseline(DeterministicBaselineAdapter):
    """Clean-room MambaStock temporal model shared independently by stocks."""

    def __init__(self, config: StockGraphModelConfig):
        super().__init__()
        self.core = MambaStockCore(
            input_dim=config.num_features,
            hidden_dim=16,
            num_layers=2,
        )

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        b, length, stocks, features = x.shape
        per_stock = x.permute(0, 2, 1, 3).reshape(b * stocks, length, features)
        return self.core(per_stock).reshape(b, stocks)


PANEL_BASELINE_NAMES = (
    "frets",
    "stockmixer",
    "agcrn",
    "fouriergnn",
    "mambastock",
)


def create_panel_baseline(name: str, config: StockGraphModelConfig) -> nn.Module:
    canonical = name.lower().replace("-", "").replace("_", "")
    factories = {
        "frets": FreTSBaseline,
        "stockmixer": StockMixerBaseline,
        "agcrn": AGCRNBaseline,
        "fouriergnn": FourierGNNBaseline,
        "mambastock": MambaStockBaseline,
    }
    if canonical not in factories:
        raise ValueError(f"Unknown panel baseline: {name}")
    return factories[canonical](config)
