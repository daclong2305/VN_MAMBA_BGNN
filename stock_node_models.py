from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_bgnn import MAMBA_BayesMAGAC, ModelArgs


@dataclass(frozen=True)
class StockGraphModelConfig:
    num_stocks: int
    num_features: int
    lookback: int
    hidden_dim: int = 64
    dropout: float = 0.1


class PerStockLSTM(nn.Module):
    """Shared per-stock LSTM baseline. Input: (B,L,S,F), output: (B,S)."""

    def __init__(self, config: StockGraphModelConfig, num_layers: int = 2):
        super().__init__()
        self.num_stocks = config.num_stocks
        self.lstm = nn.LSTM(
            input_size=config.num_features,
            hidden_size=config.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=config.dropout if num_layers > 1 else 0.0,
        )
        self.mean_head = nn.Linear(config.hidden_dim, 1)
        self.logvar_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, l, s, f = x.shape
        stock_seq = x.permute(0, 2, 1, 3).reshape(b * s, l, f)
        out, _ = self.lstm(stock_seq)
        last = out[:, -1]
        mu = self.mean_head(last).view(b, s)
        log_var = self.logvar_head(last).view(b, s).clamp(-10.0, 5.0)
        return mu, log_var


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PerStockTransformer(nn.Module):
    """Shared per-stock temporal Transformer baseline."""

    def __init__(self, config: StockGraphModelConfig, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_stocks = config.num_stocks
        self.input_proj = nn.Linear(config.num_features, config.hidden_dim)
        self.pos = PositionalEncoding(config.hidden_dim, max_len=config.lookback + 4)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=nhead,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.mean_head = nn.Linear(config.hidden_dim, 1)
        self.logvar_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, l, s, f = x.shape
        stock_seq = x.permute(0, 2, 1, 3).reshape(b * s, l, f)
        h = self.pos(self.input_proj(stock_seq))
        h = self.encoder(h)[:, -1]
        mu = self.mean_head(h).view(b, s)
        log_var = self.logvar_head(h).view(b, s).clamp(-10.0, 5.0)
        return mu, log_var


class OriginalMambaBGNNPerStock(nn.Module):
    """
    Compatibility baseline for the original paper-code idea.

    It treats features as graph nodes for each stock independently, then reshapes
    back to multi-stock outputs. This is intentionally not a stock-node graph.
    """

    def __init__(
        self,
        config: StockGraphModelConfig,
        R: int = 2,
        K: int = 2,
        heads: int = 2,
        d_e: int = 8,
        d_state: int = 64,
        mc_train: int = 1,
        mc_eval: int = 3,
        drop_edge_p: float = 0.1,
        mc_dropout_p: float = 0.2,
    ):
        super().__init__()
        args = ModelArgs(d_model=config.num_features, seq_len=config.lookback, d_state=d_state)
        self.num_stocks = config.num_stocks
        self.model = MAMBA_BayesMAGAC(
            args,
            R=R,
            K=K,
            d_e=d_e,
            heads=heads,
            mc_train=mc_train,
            mc_eval=mc_eval,
            drop_edge_p=drop_edge_p,
            mc_dropout_p=mc_dropout_p,
        )
        self._init_like_original_code()

    def _init_like_original_code(self) -> None:
        for param in self.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, l, s, f = x.shape
        stock_seq = x.permute(0, 2, 1, 3).reshape(b * s, l, f)
        mu, log_var = self.model(stock_seq)
        return mu.view(b, s), log_var.view(b, s).clamp(-10.0, 5.0)


class OriginalMambaBGNNFullPerStock(OriginalMambaBGNNPerStock):
    """
    Fuller original-paper baseline for VN30 adaptation.

    It keeps the original feature-node formulation but uses the heavier
    BiMamba + Bayesian MAGAC settings from mamba_bgnn.py as closely as the
    per-stock VN30 wrapper allows.
    """

    def __init__(self, config: StockGraphModelConfig):
        super().__init__(
            config,
            R=3,
            K=3,
            heads=4,
            d_e=10,
            d_state=128,
            mc_train=3,
            mc_eval=10,
            drop_edge_p=0.1,
            mc_dropout_p=0.2,
        )


class TemporalMambaEncoder(nn.Module):
    """Uses the existing Mamba block stack as a temporal encoder per stock."""

    def __init__(self, config: StockGraphModelConfig):
        super().__init__()
        args = ModelArgs(
            d_model=config.num_features,
            seq_len=config.lookback,
            d_proj_E=config.hidden_dim,
            d_proj_H=config.hidden_dim,
            d_proj_U=config.hidden_dim,
            d_state=config.hidden_dim,
        )
        from mamba_bgnn import BIMambaBlock

        self.encoder = BIMambaBlock(args, R=2, dropout=config.dropout)
        self.proj = nn.Linear(config.num_features, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, s, f = x.shape
        stock_seq = x.permute(0, 2, 1, 3).reshape(b * s, l, f)
        h = self.encoder(stock_seq)[:, -1]
        h = self.proj(h).view(b, s, -1)
        return h


class StockGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        a = adjacency.to(device=h.device, dtype=h.dtype)
        if a.dim() == 2:
            neighbor = torch.einsum("ij,bjh->bih", a, h)
        elif a.dim() == 3:
            neighbor = torch.einsum("bij,bjh->bih", a, h)
        else:
            raise ValueError(f"Expected 2D or 3D adjacency, got shape {tuple(a.shape)}")
        out = F.relu(self.self_proj(h) + self.neighbor_proj(neighbor))
        return self.norm(h + self.dropout(out))


class AdaptiveBayesianStockGraph(nn.Module):
    """Learned stock graph with MC-dropout node embeddings for graph uncertainty."""

    def __init__(self, num_stocks: int, emb_dim: int = 16, mc_train: int = 1, mc_eval: int = 5, dropout: float = 0.15):
        super().__init__()
        self.node_embedding = nn.Parameter(torch.randn(num_stocks, emb_dim) * 0.1)
        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.mc_train = mc_train
        self.mc_eval = mc_eval

    def _sample_adjacency(self) -> torch.Tensor:
        emb = self.dropout(self.node_embedding)
        q = self.query(emb)
        k = self.key(emb)
        scores = q @ k.T / math.sqrt(q.shape[-1])
        eye_bias = torch.eye(scores.shape[0], device=scores.device, dtype=scores.dtype)
        return F.softmax(scores + eye_bias, dim=-1)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        samples = self.mc_train if self.training else self.mc_eval
        adjs = torch.stack([self._sample_adjacency() for _ in range(samples)], dim=0)
        mean_adj = adjs.mean(dim=0)
        adj_var = adjs.var(dim=0, unbiased=False) if samples > 1 else torch.zeros_like(mean_adj)
        return mean_adj, adj_var


class InputConditionedStockGraph(nn.Module):
    """Builds a batch-specific stock relation graph from temporal hidden states."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.query(self.dropout(h))
        k = self.key(self.dropout(h))
        scores = torch.einsum("bih,bjh->bij", q, k) / math.sqrt(q.shape[-1])
        eye_bias = torch.eye(scores.shape[-1], device=h.device, dtype=h.dtype).unsqueeze(0)
        return F.softmax(scores + eye_bias, dim=-1)


class StockNodeMambaBGNN(nn.Module):
    """
    Proposed stock-node graph model.

    Mamba encodes each stock's temporal features, then graph layers exchange
    information across stock nodes. Output is per-stock probabilistic return.
    """

    def __init__(
        self,
        config: StockGraphModelConfig,
        static_adjacency: torch.Tensor | None = None,
        graph_mode: str = "static",
        num_graph_layers: int = 2,
        mc_train: int = 1,
        mc_eval: int = 5,
    ):
        super().__init__()
        if graph_mode not in {"none", "static", "adaptive", "hybrid"}:
            raise ValueError("graph_mode must be one of: none, static, adaptive, hybrid")
        self.graph_mode = graph_mode
        self.num_stocks = config.num_stocks
        self.register_buffer("static_adjacency", static_adjacency if static_adjacency is not None else torch.eye(config.num_stocks))
        self.temporal = TemporalMambaEncoder(config)
        self.stock_embedding = nn.Parameter(torch.randn(config.num_stocks, config.hidden_dim) * 0.02)
        self.recent_feature_proj = nn.Sequential(
            nn.LayerNorm(config.num_features),
            nn.Linear(config.num_features, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.recent_feature_gate = nn.Parameter(torch.tensor(-0.5))
        self.graph_layers = nn.ModuleList(
            [StockGraphLayer(config.hidden_dim, config.dropout) for _ in range(num_graph_layers)]
        )
        self.cross_stock_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_stock_norm = nn.LayerNorm(config.hidden_dim)
        self.cross_stock_ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        )
        self.cross_stock_ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.adaptive_graph = AdaptiveBayesianStockGraph(
            config.num_stocks, emb_dim=16, mc_train=mc_train, mc_eval=mc_eval, dropout=config.dropout
        )
        self.dynamic_graph = InputConditionedStockGraph(config.hidden_dim, config.dropout)
        self.mean_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.graph_var_scale = nn.Parameter(torch.tensor(0.05))
        self.graph_mix_logits = nn.Parameter(torch.tensor([0.0, -0.5, 0.0]))
        self.graph_output_logit = nn.Parameter(torch.tensor(-1.0))
        self.alpha_output_logit = nn.Parameter(torch.tensor(-2.0))

    def _adjacency(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.graph_mode == "none":
            adj = torch.eye(self.static_adjacency.shape[0], device=self.static_adjacency.device)
            return adj, torch.zeros_like(adj)
        if self.graph_mode == "static":
            return self.static_adjacency, torch.zeros_like(self.static_adjacency)
        adaptive_adj, adj_var = self.adaptive_graph()
        if self.graph_mode == "adaptive":
            return adaptive_adj, adj_var
        dynamic_adj = self.dynamic_graph(h)
        static_adj = self.static_adjacency.unsqueeze(0).to(dynamic_adj.device)
        adaptive_adj = adaptive_adj.unsqueeze(0).to(dynamic_adj.device)
        mix = F.softmax(self.graph_mix_logits, dim=0).to(dynamic_adj.device)
        hybrid = mix[0] * static_adj + mix[1] * adaptive_adj + mix[2] * dynamic_adj
        hybrid = hybrid / hybrid.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return hybrid, adj_var

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recent = self.recent_feature_proj(x[:, -1])
        recent_gate = torch.sigmoid(self.recent_feature_gate)
        h0 = self.temporal(x) + recent_gate * recent + self.stock_embedding.unsqueeze(0)
        h = h0
        attn_out, _ = self.cross_stock_attention(h, h, h, need_weights=False)
        h = self.cross_stock_norm(h + attn_out)
        h = self.cross_stock_ffn_norm(h + self.cross_stock_ffn(h))
        adj, adj_var = self._adjacency(h)
        for layer in self.graph_layers:
            h = layer(h, adj)
        graph_gate = torch.sigmoid(self.graph_output_logit)
        h = (1.0 - graph_gate) * h0 + graph_gate * h
        raw_mu = self.mean_head(h).squeeze(-1)
        alpha_gate = torch.sigmoid(self.alpha_output_logit)
        mu = raw_mu + alpha_gate * (raw_mu - raw_mu.mean(dim=1, keepdim=True))
        base_log_var = self.logvar_head(h).squeeze(-1).clamp(-10.0, 5.0)
        graph_uncertainty = adj_var.mean(dim=1).unsqueeze(0).to(mu.device)
        var = base_log_var.exp() + F.softplus(self.graph_var_scale) * graph_uncertainty + 1e-6
        return mu, var.log().clamp(-10.0, 5.0)


def create_model(
    name: str,
    config: StockGraphModelConfig,
    static_adjacency: torch.Tensor | None = None,
) -> nn.Module:
    name = name.lower()
    if name == "lstm":
        return PerStockLSTM(config)
    if name == "transformer":
        return PerStockTransformer(config)
    if name == "original_mamba_bgnn":
        return OriginalMambaBGNNPerStock(config)
    if name == "original_mamba_bgnn_full":
        return OriginalMambaBGNNFullPerStock(config)
    if name == "stock_mamba_no_graph":
        return StockNodeMambaBGNN(config, static_adjacency=static_adjacency, graph_mode="none")
    if name == "stock_mamba_static":
        return StockNodeMambaBGNN(config, static_adjacency=static_adjacency, graph_mode="static")
    if name == "stock_mamba_adaptive":
        return StockNodeMambaBGNN(config, static_adjacency=static_adjacency, graph_mode="adaptive")
    if name == "stock_mamba_hybrid":
        return StockNodeMambaBGNN(config, static_adjacency=static_adjacency, graph_mode="hybrid")
    raise ValueError(f"Unknown model name: {name}")
