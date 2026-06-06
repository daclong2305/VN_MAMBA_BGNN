import torch

from run_vn30_stock_graph_experiment import (
    listwise_rank_loss,
    score_predictions,
    soft_topk_portfolio_loss,
    soft_topk_weights,
    topk_ranking_metrics,
    validation_portfolio_metrics,
)


def test_soft_topk_weights_are_valid_probabilities():
    scores = torch.tensor([[0.3, 0.1, -0.2, 0.5], [0.0, 0.2, 0.4, -0.1]])
    weights = soft_topk_weights(scores, top_k=2, temperature=0.05)

    assert weights.shape == scores.shape
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(scores.shape[0]), atol=1e-6)


def test_soft_topk_portfolio_loss_is_finite():
    mu = torch.randn(6, 5)
    y = torch.randn(6, 5) * 0.01
    loss = soft_topk_portfolio_loss(mu, y, top_k=2)

    assert torch.isfinite(loss)


def test_listwise_rank_loss_is_finite():
    mu = torch.randn(6, 5)
    y = torch.randn(6, 5) * 0.01
    loss = listwise_rank_loss(mu, y, temperature=0.01)

    assert torch.isfinite(loss)


def test_topk_ranking_metrics_are_bounded():
    mu = torch.tensor([[0.3, 0.1, -0.2, 0.5], [0.0, 0.2, 0.4, -0.1]])
    y = torch.tensor([[0.2, -0.1, 0.0, 0.3], [0.1, 0.4, 0.2, -0.2]])
    metrics = topk_ranking_metrics(mu, y, top_k=2)

    assert 0.0 <= metrics["topk_overlap"] <= 1.0
    assert 0.0 <= metrics["topk_hit_rate"] <= 1.0
    assert metrics["top_bottom_return_spread"] > 0.0


def test_score_predictions_modes_keep_shape():
    pred = torch.tensor([[0.01, -0.02, 0.03]])
    log_var = torch.zeros_like(pred) - 4.0

    for mode in ["mu", "risk_adjusted", "positive_confidence", "rank_zscore"]:
        score = score_predictions(pred, log_var, score_mode=mode, risk_aversion=0.1)
        assert score.shape == pred.shape
        assert torch.isfinite(score).all()


def test_validation_portfolio_metrics_smoke():
    pred = torch.tensor(
        [
            [0.03, 0.02, -0.01],
            [0.01, 0.04, 0.00],
            [0.05, -0.01, 0.02],
        ]
    )
    log_var = torch.zeros_like(pred) - 4.0
    true = torch.tensor(
        [
            [0.01, 0.00, -0.01],
            [0.00, 0.02, -0.01],
            [0.03, -0.02, 0.01],
        ]
    )

    metrics = validation_portfolio_metrics(
        pred,
        log_var,
        true,
        top_k=1,
        transaction_cost=0.0015,
        rebalance_every=1,
        hold_k=None,
        min_score=None,
        score_mode="mu",
        risk_aversion=0.25,
    )

    assert metrics["positive_day_ratio"] >= 0.0
    assert metrics["average_turnover"] > 0.0
