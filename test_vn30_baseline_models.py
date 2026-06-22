import torch

from run_vn30_stock_graph_experiment import DEFAULT_MODELS
from stock_node_models import StockGraphModelConfig, create_model
from vn30_baseline_models import DeterministicBaselineAdapter


EXPECTED_FINAL_MODELS = [
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


def test_final_model_registry_contains_full_comparison_only():
    assert DEFAULT_MODELS == EXPECTED_FINAL_MODELS
    assert "original_mamba_bgnn" not in DEFAULT_MODELS
    assert "stock_mamba_static" not in DEFAULT_MODELS
    assert "stock_mamba_adaptive" not in DEFAULT_MODELS


def test_all_final_models_return_probabilistic_stock_predictions():
    torch.manual_seed(11)
    config = StockGraphModelConfig(
        num_stocks=4,
        num_features=6,
        lookback=5,
        hidden_dim=16,
        dropout=0.0,
    )
    adjacency = torch.eye(config.num_stocks)
    x = torch.randn(2, config.lookback, config.num_stocks, config.num_features)

    for name in DEFAULT_MODELS:
        model = create_model(name, config, static_adjacency=adjacency).eval()
        with torch.no_grad():
            mu, log_var = model(x)
        assert mu.shape == (2, config.num_stocks), name
        assert log_var.shape == (2, config.num_stocks), name
        assert torch.isfinite(mu).all(), name
        assert torch.isfinite(log_var).all(), name


def test_official_baseline_adapters_have_finite_gradients():
    torch.manual_seed(17)
    config = StockGraphModelConfig(
        num_stocks=4,
        num_features=6,
        lookback=5,
        hidden_dim=16,
        dropout=0.0,
    )
    x = torch.randn(2, config.lookback, config.num_stocks, config.num_features)
    y = torch.randn(2, config.num_stocks)

    for name in ["frets", "stockmixer", "agcrn", "fouriergnn", "mambastock"]:
        model = create_model(name, config)
        mu, log_var = model(x)
        loss = (mu - y).square().mean() + 0.01 * log_var.square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None for g in gradients), name
        assert all(torch.isfinite(g).all() for g in gradients if g is not None), name


def test_deterministic_baselines_use_the_same_homoscedastic_variance_adapter():
    config = StockGraphModelConfig(4, 6, 5, hidden_dim=16, dropout=0.0)
    x = torch.randn(2, 5, 4, 6)

    for name in ["frets", "stockmixer", "agcrn", "fouriergnn", "mambastock"]:
        model = create_model(name, config).eval()
        assert isinstance(model, DeterministicBaselineAdapter), name
        with torch.no_grad():
            direct_mean = model.forward_mean(x)
            mu, log_var = model(x)
        assert torch.allclose(mu, direct_mean), name
        assert torch.allclose(log_var, torch.full_like(log_var, log_var[0, 0])), name


def test_cross_stock_baselines_keep_stocks_as_channels_or_nodes():
    config = StockGraphModelConfig(7, 6, 5, hidden_dim=16, dropout=0.0)
    frets = create_model("frets", config)
    fouriergnn = create_model("fouriergnn", config)
    agcrn = create_model("agcrn", config)
    stockmixer = create_model("stockmixer", config)

    assert frets.core.num_channels == config.num_stocks
    assert fouriergnn.core.num_nodes == config.num_stocks
    assert agcrn.core.num_nodes == config.num_stocks
    assert stockmixer.core.stock_mixer.norm.normalized_shape == (config.num_stocks,)


def test_official_baseline_eval_is_repeatable():
    torch.manual_seed(23)
    config = StockGraphModelConfig(4, 6, 5, hidden_dim=16, dropout=0.0)
    x = torch.randn(2, 5, 4, 6)

    for name in ["frets", "stockmixer", "agcrn", "fouriergnn", "mambastock"]:
        model = create_model(name, config).eval()
        with torch.no_grad():
            first = model(x)[0]
            second = model(x)[0]
        assert torch.equal(first, second), name
