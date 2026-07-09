from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# FreTS core ported from aikunyi/FreTS at commit 6de28ab19f83955087e2690cdfbb29b065ab0b9c.
# Upstream license: Apache-2.0. The device handling and constructor interface are adapted.
class FreTSCore(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_channels: int,
        pred_len: int = 1,
        embed_size: int = 128,
        hidden_size: int = 256,
        use_channel_mixing: bool = True,
        sparsity_threshold: float = 0.01,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.pred_len = pred_len
        self.embed_size = embed_size
        self.use_channel_mixing = use_channel_mixing
        self.sparsity_threshold = sparsity_threshold
        scale = 0.02

        self.embeddings = nn.Parameter(torch.randn(1, embed_size))
        self.r1 = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.i1 = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.rb1 = nn.Parameter(scale * torch.randn(embed_size))
        self.ib1 = nn.Parameter(scale * torch.randn(embed_size))
        self.r2 = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.i2 = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.rb2 = nn.Parameter(scale * torch.randn(embed_size))
        self.ib2 = nn.Parameter(scale * torch.randn(embed_size))
        self.fc = nn.Sequential(
            nn.Linear(seq_len * embed_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, pred_len),
        )

    def token_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).unsqueeze(-1) * self.embeddings

    def frequency_mlp(
        self,
        x: torch.Tensor,
        real_weight: torch.Tensor,
        imag_weight: torch.Tensor,
        real_bias: torch.Tensor,
        imag_bias: torch.Tensor,
    ) -> torch.Tensor:
        real = F.relu(
            torch.einsum("bijd,dd->bijd", x.real, real_weight)
            - torch.einsum("bijd,dd->bijd", x.imag, imag_weight)
            + real_bias
        )
        imag = F.relu(
            torch.einsum("bijd,dd->bijd", x.imag, real_weight)
            + torch.einsum("bijd,dd->bijd", x.real, imag_weight)
            + imag_bias
        )
        value = F.softshrink(
            torch.stack([real, imag], dim=-1), lambd=self.sparsity_threshold
        )
        return torch.view_as_complex(value)

    def temporal_learner(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=2, norm="ortho")
        mixed = self.frequency_mlp(
            spectrum, self.r2, self.i2, self.rb2, self.ib2
        )
        return torch.fft.irfft(mixed, n=self.seq_len, dim=2, norm="ortho")

    def channel_learner(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x.permute(0, 2, 1, 3), dim=2, norm="ortho")
        mixed = self.frequency_mlp(
            spectrum, self.r1, self.i1, self.rb1, self.ib1
        )
        restored = torch.fft.irfft(
            mixed, n=self.num_channels, dim=2, norm="ortho"
        )
        return restored.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C); output: (B,pred_len,C)
        h = self.token_embedding(x)
        residual = h
        if self.use_channel_mixing:
            h = self.channel_learner(h)
        h = self.temporal_learner(h) + residual
        return self.fc(h.reshape(x.shape[0], self.num_channels, -1)).permute(0, 2, 1)


# FourierGNN core ported from aikunyi/FourierGNN at commit
# f239290b6f6881c53c5dda094bb0a7255eb0ea2d. Upstream license: MIT.
class FourierGNNCore(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_nodes: int,
        pred_len: int = 1,
        embed_size: int = 128,
        hidden_size: int = 256,
        projection_width: int = 8,
        sparsity_threshold: float = 0.01,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.embed_size = embed_size
        self.sparsity_threshold = sparsity_threshold
        scale = 0.02

        self.embeddings = nn.Parameter(torch.randn(1, embed_size))
        self.w1 = nn.Parameter(scale * torch.randn(2, embed_size, embed_size))
        self.b1 = nn.Parameter(scale * torch.randn(2, embed_size))
        self.w2 = nn.Parameter(scale * torch.randn(2, embed_size, embed_size))
        self.b2 = nn.Parameter(scale * torch.randn(2, embed_size))
        self.w3 = nn.Parameter(scale * torch.randn(2, embed_size, embed_size))
        self.b3 = nn.Parameter(scale * torch.randn(2, embed_size))
        self.temporal_projection = nn.Parameter(torch.randn(seq_len, projection_width))
        self.fc = nn.Sequential(
            nn.Linear(embed_size * projection_width, 64),
            nn.LeakyReLU(),
            nn.Linear(64, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, pred_len),
        )

    def _complex_layer(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_real = F.relu(
            torch.einsum("bli,ii->bli", real, weight[0])
            - torch.einsum("bli,ii->bli", imag, weight[1])
            + bias[0]
        )
        out_imag = F.relu(
            torch.einsum("bli,ii->bli", imag, weight[0])
            + torch.einsum("bli,ii->bli", real, weight[1])
            + bias[1]
        )
        return out_real, out_imag

    def fourier_graph_operator(self, x: torch.Tensor) -> torch.Tensor:
        r1, i1 = self._complex_layer(x.real, x.imag, self.w1, self.b1)
        skip1 = F.softshrink(torch.stack([r1, i1], dim=-1), self.sparsity_threshold)
        r2, i2 = self._complex_layer(r1, i1, self.w2, self.b2)
        skip2 = F.softshrink(torch.stack([r2, i2], dim=-1), self.sparsity_threshold)
        skip2 = skip2 + skip1
        r3, i3 = self._complex_layer(r2, i2, self.w3, self.b3)
        out = F.softshrink(torch.stack([r3, i3], dim=-1), self.sparsity_threshold)
        return torch.view_as_complex(out + skip2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,N); output: (B,N,pred_len)
        batch = x.shape[0]
        flattened = x.permute(0, 2, 1).contiguous().reshape(batch, -1)
        embedded = flattened.unsqueeze(-1) * self.embeddings
        spectrum = torch.fft.rfft(embedded, dim=1, norm="ortho")
        mixed = self.fourier_graph_operator(spectrum) + spectrum
        restored = torch.fft.irfft(
            mixed, n=self.num_nodes * self.seq_len, dim=1, norm="ortho"
        )
        restored = restored.reshape(batch, self.num_nodes, self.seq_len, self.embed_size)
        restored = restored.permute(0, 1, 3, 2)
        projected = torch.matmul(restored, self.temporal_projection)
        return self.fc(projected.reshape(batch, self.num_nodes, -1))


# AGCRN core ported from LeiBAI/AGCRN at commit
# 7fbbf2aeb099242098a3cf482b55cd45d7295c28. Upstream license: MIT.
class AdaptiveVertexWiseGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cheb_k: int, embed_dim: int):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, cheb_k, in_dim, out_dim))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, out_dim))

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        num_nodes = node_embeddings.shape[0]
        supports = F.softmax(F.relu(node_embeddings @ node_embeddings.T), dim=1)
        support_set = [torch.eye(num_nodes, device=x.device, dtype=x.dtype), supports]
        for _ in range(2, self.cheb_k):
            support_set.append(2 * supports @ support_set[-1] - support_set[-2])
        support_stack = torch.stack(support_set, dim=0)
        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = node_embeddings @ self.bias_pool
        x_graph = torch.einsum("knm,bmc->bknc", support_stack, x).permute(0, 2, 1, 3)
        return torch.einsum("bnki,nkio->bno", x_graph, weights) + bias


class AGCRNCellCore(nn.Module):
    def __init__(self, num_nodes: int, in_dim: int, hidden_dim: int, cheb_k: int, embed_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.gate = AdaptiveVertexWiseGraphConv(
            in_dim + hidden_dim, 2 * hidden_dim, cheb_k, embed_dim
        )
        self.update = AdaptiveVertexWiseGraphConv(
            in_dim + hidden_dim, hidden_dim, cheb_k, embed_dim
        )

    def forward(
        self, x: torch.Tensor, state: torch.Tensor, node_embeddings: torch.Tensor
    ) -> torch.Tensor:
        gates = torch.sigmoid(
            self.gate(torch.cat([x, state.to(x.device)], dim=-1), node_embeddings)
        )
        z, r = torch.split(gates, self.hidden_dim, dim=-1)
        candidate = torch.cat([x, z * state], dim=-1)
        candidate_state = torch.tanh(self.update(candidate, node_embeddings))
        return r * state + (1.0 - r) * candidate_state

    def init_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device, dtype=dtype)


class AGCRNEncoderCore(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        cheb_k: int,
        embed_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.cells = nn.ModuleList(
            [AGCRNCellCore(num_nodes, in_dim, hidden_dim, cheb_k, embed_dim)]
            + [
                AGCRNCellCore(num_nodes, hidden_dim, hidden_dim, cheb_k, embed_dim)
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        current = x
        for cell in self.cells:
            state = cell.init_hidden(current.shape[0], current.device, current.dtype)
            states = []
            for step in range(current.shape[1]):
                state = cell(current[:, step], state, node_embeddings)
                states.append(state)
            current = torch.stack(states, dim=1)
        return current


class AGCRNCore(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        horizon: int = 1,
        num_layers: int = 2,
        embed_dim: int = 10,
        cheb_k: int = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.horizon = horizon
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.encoder = AGCRNEncoderCore(
            num_nodes, input_dim, hidden_dim, cheb_k, embed_dim, num_layers
        )
        self.end_conv = nn.Conv2d(
            1, horizon * output_dim, kernel_size=(1, hidden_dim), bias=True
        )
        self._reset_like_upstream()

    def _reset_like_upstream(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(source, self.node_embeddings)[:, -1:, :, :]
        output = self.end_conv(encoded).squeeze(-1)
        output = output.reshape(-1, self.horizon, self.output_dim, self.num_nodes)
        return output.permute(0, 1, 3, 2)


# Clean-room, batch-aware reimplementation of the public StockMixer architecture
# at SJTU-DMTai/StockMixer commit cce13598afd3ff33ae317700a85ae08db0554652.
# Upstream repository did not declare a license when inspected.
class TriangularTemporalProjection(nn.Module):
    def __init__(self, time_steps: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(i + 1, 1) for i in range(time_steps)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = [layer(x[..., : i + 1]) for i, layer in enumerate(self.layers)]
        return torch.cat(pieces, dim=-1)


class StockMixerChannelBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.first = nn.Linear(channels, channels)
        self.second = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(F.gelu(self.first(x)))


class StockMixer2DCore(nn.Module):
    def __init__(self, time_steps: int, channels: int):
        super().__init__()
        self.norm1 = nn.LayerNorm([time_steps, channels])
        self.norm2 = nn.LayerNorm([time_steps, channels])
        self.time_mixer = TriangularTemporalProjection(time_steps)
        self.channel_mixer = StockMixerChannelBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.time_mixer(self.norm1(x).transpose(-2, -1)).transpose(-2, -1)
        hidden = self.norm2(temporal + x)
        return hidden + self.channel_mixer(hidden)


class StockMixerMultiResolutionCore(nn.Module):
    def __init__(self, time_steps: int, channels: int, scale_steps: int):
        super().__init__()
        self.full_scale = StockMixer2DCore(time_steps, channels)
        self.coarse_scale = StockMixer2DCore(scale_steps, channels)

    def forward(self, full: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [full, self.full_scale(full), self.coarse_scale(coarse)], dim=-2
        )


class CrossStockMixerCore(nn.Module):
    def __init__(self, num_stocks: int, hidden_dim: int = 20):
        super().__init__()
        self.norm = nn.LayerNorm(num_stocks)
        self.first = nn.Linear(num_stocks, hidden_dim)
        self.second = nn.Linear(hidden_dim, num_stocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stock_last = x.transpose(-2, -1)
        mixed = self.second(F.hardswish(self.first(self.norm(stock_last))))
        return mixed.transpose(-2, -1)


class StockMixerCore(nn.Module):
    def __init__(
        self,
        num_stocks: int,
        time_steps: int,
        channels: int,
        stock_hidden_dim: int = 20,
    ):
        super().__init__()
        coarse_steps = (time_steps - 2) // 2 + 1
        self.multiscale = StockMixerMultiResolutionCore(
            time_steps, channels, coarse_steps
        )
        self.channel_readout = nn.Linear(channels, 1)
        self.coarse_conv = nn.Conv1d(
            in_channels=channels, out_channels=channels, kernel_size=2, stride=2
        )
        mixed_steps = time_steps * 2 + coarse_steps
        self.stock_mixer = CrossStockMixerCore(num_stocks, stock_hidden_dim)
        self.direct_readout = nn.Linear(mixed_steps, 1)
        self.stock_readout = nn.Linear(mixed_steps, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Thin batching extension: upstream input is (S,T,C), here (B,S,T,C).
        b, s, t, c = x.shape
        coarse = self.coarse_conv(x.reshape(b * s, t, c).transpose(1, 2))
        coarse = coarse.transpose(1, 2).reshape(b, s, -1, c)
        temporal = self.multiscale(x, coarse)
        temporal = self.channel_readout(temporal).squeeze(-1)
        stock_context = self.stock_mixer(temporal)
        return (
            self.direct_readout(temporal) + self.stock_readout(stock_context)
        ).squeeze(-1)


# Clean-room reimplementation of the MambaStock temporal core at commit
# ff0b16cb3e57dfa66c6955e7c2b80eaf64058f83. Upstream declared no license.
@dataclass
class MambaStockConfig:
    d_model: int = 16
    num_layers: int = 2
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4
    dt_rank: int | str = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4

    def __post_init__(self) -> None:
        self.inner_dim = self.expand_factor * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class MambaStockRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class MambaStockBlockCore(nn.Module):
    def __init__(self, config: MambaStockConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, 2 * config.inner_dim, bias=False)
        self.conv1d = nn.Conv1d(
            config.inner_dim,
            config.inner_dim,
            kernel_size=config.d_conv,
            groups=config.inner_dim,
            padding=config.d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(
            config.inner_dim, config.dt_rank + 2 * config.d_state, bias=False
        )
        self.dt_proj = nn.Linear(config.dt_rank, config.inner_dim, bias=True)
        nn.init.uniform_(
            self.dt_proj.weight,
            -(config.dt_rank ** -0.5),
            config.dt_rank ** -0.5,
        )
        dt = torch.exp(
            torch.rand(config.inner_dim)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        state = torch.arange(1, config.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(state.repeat(config.inner_dim, 1)))
        self.D = nn.Parameter(torch.ones(config.inner_dim))
        self.out_proj = nn.Linear(config.inner_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        main, gate = self.in_proj(x).chunk(2, dim=-1)
        main = self.conv1d(main.transpose(1, 2))[:, :, :length].transpose(1, 2)
        main = F.silu(main)
        delta_bc = self.x_proj(main)
        delta, b_state, c_state = torch.split(
            delta_bc,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta))
        a_state = -torch.exp(self.A_log.float())
        delta_a = torch.exp(delta.unsqueeze(-1) * a_state)
        input_term = delta.unsqueeze(-1) * b_state.unsqueeze(2) * main.unsqueeze(-1)
        hidden = main.new_zeros(
            main.shape[0], self.config.inner_dim, self.config.d_state
        )
        outputs = []
        for step in range(length):
            hidden = delta_a[:, step] * hidden + input_term[:, step]
            outputs.append((hidden @ c_state[:, step].unsqueeze(-1)).squeeze(-1))
        scanned = torch.stack(outputs, dim=1) + self.D * main
        return self.out_proj(scanned * F.silu(gate))


class MambaStockResidualCore(nn.Module):
    def __init__(self, config: MambaStockConfig):
        super().__init__()
        self.norm = MambaStockRMSNorm(config.d_model)
        self.mixer = MambaStockBlockCore(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class MambaStockCore(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        num_layers: int = 2,
    ):
        super().__init__()
        config = MambaStockConfig(d_model=hidden_dim, num_layers=num_layers)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MambaStockResidualCore(config) for _ in range(num_layers)]
        )
        self.final_norm = MambaStockRMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return torch.tanh(self.output_proj(h[:, -1])).squeeze(-1)
