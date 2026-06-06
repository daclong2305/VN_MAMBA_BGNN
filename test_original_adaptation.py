import torch

from mamba_bgnn import MAMBA_BayesMAGAC, ModelArgs
from stock_node_models import (
    OriginalMambaBGNNFullPerStock,
    OriginalMambaBGNNPerStock,
    StockGraphModelConfig,
)


def test_original_wrapper_matches_direct_mamba_bgnn_call():
    torch.manual_seed(7)
    config = StockGraphModelConfig(num_stocks=4, num_features=6, lookback=5, hidden_dim=16)
    wrapper = OriginalMambaBGNNPerStock(
        config,
        R=2,
        K=2,
        heads=2,
        d_e=8,
        d_state=32,
        mc_train=1,
        mc_eval=1,
        drop_edge_p=0.0,
        mc_dropout_p=0.0,
    )
    direct = MAMBA_BayesMAGAC(
        ModelArgs(d_model=config.num_features, seq_len=config.lookback, d_state=32),
        R=2,
        K=2,
        d_e=8,
        heads=2,
        mc_train=1,
        mc_eval=1,
        drop_edge_p=0.0,
        mc_dropout_p=0.0,
    )
    direct.load_state_dict(wrapper.model.state_dict())
    wrapper.eval()
    direct.eval()

    x = torch.randn(3, config.lookback, config.num_stocks, config.num_features)
    wrapped_mu, wrapped_log_var = wrapper(x)

    stock_seq = x.permute(0, 2, 1, 3).reshape(
        x.shape[0] * config.num_stocks,
        config.lookback,
        config.num_features,
    )
    direct_mu, direct_log_var = direct(stock_seq)
    direct_mu = direct_mu.view(x.shape[0], config.num_stocks)
    direct_log_var = direct_log_var.view(x.shape[0], config.num_stocks)

    assert torch.allclose(wrapped_mu, direct_mu, atol=1e-6)
    assert torch.allclose(wrapped_log_var, direct_log_var, atol=1e-6)


def test_full_adapted_baseline_uses_original_core_model():
    config = StockGraphModelConfig(num_stocks=30, num_features=17, lookback=20, hidden_dim=64)
    model = OriginalMambaBGNNFullPerStock(config)

    assert isinstance(model.model, MAMBA_BayesMAGAC)
    assert model.model.bi_mamba.R == 3
    assert model.model.agc_bayes.K == 3
    assert model.model.agc_bayes.H == 4
    assert model.model.agc_bayes.mc_train == 3
    assert model.model.agc_bayes.mc_eval == 10
