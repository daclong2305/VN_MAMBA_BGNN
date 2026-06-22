# VN30 Stock Forecasting and Ranking

This document describes the VN30 panel experiment for probabilistic stock return forecasting and cross-sectional stock ranking. This is the updated code path for evaluating the proposed stock-node Mamba graph model on VN30.

## Code Files

- `vn30_stock_graph_dataset.py`: balanced VN30 panel loading, chronological split, feature standardization, and static correlation graph construction.
- `vn30_official_baseline_cores.py`: traced ports/reimplementations of the official baseline cores.
- `vn30_baseline_models.py`: thin VN30 input/output adapters for those cores.
- `baseline_provenance.json`: pinned upstream commits, source hashes, licenses, adaptation status, and deviations.
- `stock_node_models.py`: model registry, LSTM/Transformer, the full original MAMBA-BGNN adaptation, and the proposed stock-node hybrid model.
- `run_vn30_stock_graph_experiment.py`: training loop, forecasting/ranking metrics, prediction export, optional secondary backtest, and experiment config export.

## Problem Setting

The VN30 task uses a balanced stock panel with shape:

```text
(samples, lookback, stocks, features)
```

The model predicts next-day returns for all VN30 constituents. The main objective is not direct portfolio optimization, but:

1. Probabilistic return forecasting.
2. Cross-sectional ranking of stocks within each trading day.
3. Evaluating whether stock-level graph learning improves over temporal baselines and the original MAMBA-BGNN adaptation.

Backtest outputs are still generated, but they should be treated as secondary economic analysis rather than the central claim.

## Models

| Model | Description |
|-------|-------------|
| `stock_mamba_hybrid` | Proposed model. Mamba encodes each stock's temporal features, then static/adaptive/dynamic stock graphs exchange information across stocks. |
| `original_mamba_bgnn_full` | Original MAMBA-BGNN core adapted per stock. It treats features as graph nodes inside each stock and does not model cross-stock relations. |
| `mambastock` | Clean-room MambaStock temporal core, shared per stock and without a stock graph. |
| `agcrn` | Official AGCRN core; VN30 stocks are nodes and indicators are node features. |
| `fouriergnn` | Official FourierGNN core; a shared feature projection feeds VN30 stocks as graph nodes. |
| `stockmixer` | Clean-room, batch-aware StockMixer retaining indicator, temporal, and cross-stock mixing. |
| `frets` | Official FreTS core; a shared feature projection feeds VN30 stocks as multivariate channels. |
| `transformer` | Shared per-stock temporal Transformer baseline. |
| `lstm` | Shared per-stock LSTM baseline. |

`original_mamba_bgnn_full` uses the original `MAMBA_BayesMAGAC` core with the full original-style settings:

```text
R=3, K=3, heads=4, d_e=10, d_state=128, mc_train=3, mc_eval=10
```

This makes it a fair baseline for asking whether the original feature-node MAMBA-BGNN idea is sufficient, or whether a stock-node graph is more suitable for VN30 panel data.

### Baseline provenance and fidelity

| Model | Integration status | Pinned upstream | License status |
|---|---|---|---|
| FreTS | Official core port + thin VN30 adapter | `aikunyi/FreTS@6de28ab` | Apache-2.0 |
| FourierGNN | Official core port + thin VN30 adapter | `aikunyi/FourierGNN@f239290` | MIT |
| AGCRN | Official core port + thin VN30 adapter | `LeiBAI/AGCRN@7fbbf2a` | MIT |
| StockMixer | Clean-room reimplementation + batching/lookback adapter | `SJTU-DMTai/StockMixer@cce1359` | No upstream license found |
| MambaStock | Clean-room reimplementation + window adapter | `zshicode/MambaStock@ff0b16c` | No upstream license found |
| LSTM / Transformer | Local adapted controls | No exact author implementation was available | Project license |

The full source-file SHA-256 values and exact adapter decisions are recorded in `baseline_provenance.json` and embedded into every generated `experiment_config.json`, together with hashes of the panel and metadata files. These models are **independent official-code adaptations**, not an exact reproduction of Table 1 in the MAMBA-BGNN paper. Exact reproduction would additionally require the authors' private baseline configs, preprocessing artifacts, seeds, and checkpoints.

FreTS, FourierGNN, AGCRN, and StockMixer now all see the actual VN30 stock dimension. MambaStock remains deliberately per-stock because its source architecture is a temporal model rather than a cross-stock graph model. All deterministic controls use one trainable homoscedastic variance scalar so NLL can be computed without giving different baselines different uncertainty heads.

With the current VN30 defaults (`30` stocks, `17` features, lookback `20`, proposed hidden size `64`), trainable parameter counts are:

| Model | Parameters |
|---|---:|
| LSTM-adapted | 54,594 |
| Transformer-adapted | 68,162 |
| FreTS-adapted | 722,068 |
| StockMixer-adapted | 5,585 |
| AGCRN-adapted | 806,766 |
| FourierGNN-adapted | 181,876 |
| MambaStock-adapted | 7,074 |
| Original MAMBA-BGNN Full adaptation | 106,537 |
| Stock Mamba Hybrid | 147,125 |

These counts are intentionally not forced to match the MAMBA-BGNN Table 1 counts. They change with VN30 feature/node dimensions and lookback; for example, the official FreTS final projection grows from a 5-step to a 20-step input. StockMixer and MambaStock also retain their public-repository defaults, which do not explain the larger configurations apparently used for Table 1. Since the original paper does not publish complete per-baseline configurations, matching its parameter counts by silently changing hidden sizes would not constitute faithful reproduction. Every actual run logs its own count in the result CSV.

## Forecasting and Ranking Metrics

The VN30 pipeline reports standard forecasting metrics:

- `NLL`: Gaussian negative log-likelihood.
- `RMSE`, `MAE`: point prediction errors.
- `IC`: Pearson correlation between predicted and realized returns.
- `rank_ic_by_day`: average daily rank correlation.
- `directional_accuracy`: sign prediction accuracy.

It also reports top-k ranking metrics:

- `topk_overlap`: overlap between predicted top-k and realized top-k stocks.
- `topk_hit_rate`: fraction of days where predicted top-k contains at least one realized top-k stock.
- `pred_topk_mean_return`: realized mean return of predicted top-k stocks.
- `pred_bottomk_mean_return`: realized mean return of predicted bottom-k stocks.
- `top_bottom_return_spread`: realized return spread between predicted top-k and bottom-k groups.
- `topk_return_capture`: predicted top-k return divided by oracle top-k return.

The main output file for this setting is:

```text
vn30_forecast_ranking_summary.csv
```

## Recommended Final Evaluation

Run the 5-seed final evaluation with PowerShell:

```powershell
$seeds = 26,42,123,2024,3407
foreach ($s in $seeds) {
  .venv\Scripts\python.exe run_vn30_stock_graph_experiment.py `
    --output-dir vn30_full_baseline_final_5seed `
    --models lstm transformer frets stockmixer agcrn fouriergnn mambastock original_mamba_bgnn_full stock_mamba_hybrid `
    --seed $s `
    --batch-size 16 `
    --hidden-dim 64 `
    --epochs 60 `
    --patience 10 `
    --top-k 5
}
```

The default VN30 settings are aligned with forecasting and ranking:

```text
optimizer=Adam
checkpoint_metric=forecast_rank
rank_loss_weight=0.05
listwise_rank_loss_weight=0.05
portfolio_loss_weight=0.0
topk_loss_weight=0.0
score_mode=mu
graph_correlation=positive
```

Pairwise and listwise ranking losses are applied to every final-comparison model by default so that the architecture comparison uses a common objective. The lightweight `original_mamba_bgnn` registry entry is intentionally excluded; final experiments use only `original_mamba_bgnn_full`.

## Output Structure

Each seed creates one timestamped directory under the selected output directory:

```text
vn30_full_baseline_final_5seed/
  YYYYMMDD_HHMMSS/
    experiment_config.json
    static_correlation_adjacency.csv
    vn30_forecast_ranking_summary.csv
    vn30_model_comparison.csv
    vn30_backtest_summary.csv
    {model}_test_predictions.csv
    {model}_best.pt
    {model}_backtest_top5_reb1_hold0.csv
```

For the main paper table, use `vn30_forecast_ranking_summary.csv`. Use `vn30_backtest_summary.csv` only as secondary analysis.

## Previous 5-Seed Results (Before Full Baseline Expansion)

The table below records the earlier four-model run. It should not be used as the final full-baseline table after adding FreTS, StockMixer, AGCRN, FourierGNN, and MambaStock; rerun the command above to produce the final comparison.

| Model | NLL (lower) | RMSE (lower) | MAE (lower) | IC (higher) | RankIC (higher) | Direction Acc. (higher) |
|-------|------------:|-------------:|------------:|------------:|----------------:|------------------------:|
| Original MAMBA-BGNN Full | -2.1772 | 0.022977 | 0.015913 | 0.0169 | 0.0162 | 0.5075 |
| LSTM | -2.4290 | 0.022687 | 0.015417 | 0.0461 | 0.0104 | 0.5058 |
| Transformer | -2.4696 | 0.022441 | 0.015282 | 0.0452 | 0.0284 | 0.5118 |
| **Stock Mamba Hybrid** | **-2.4884** | **0.022296** | **0.015138** | **0.1122** | **0.0553** | **0.5219** |

Top-k ranking metrics also favor the proposed model:

| Model | Top-k Overlap (higher) | Hit Rate (higher) | Top-bottom Spread (higher) | Return Capture (higher) |
|-------|-----------------------:|------------------:|---------------------------:|------------------------:|
| Original MAMBA-BGNN Full | 0.2176 | 0.7093 | 0.00145 | 0.0793 |
| LSTM | 0.2256 | 0.7093 | 0.00232 | 0.1052 |
| Transformer | 0.1640 | 0.5936 | 0.00123 | 0.0684 |
| **Stock Mamba Hybrid** | **0.2347** | **0.7450** | **0.00380** | **0.1325** |

## Interpretation

The proposed `stock_mamba_hybrid` improves both forecasting quality and ranking quality. It achieves the best average NLL, RMSE, MAE, IC, RankIC, and directional accuracy over 5 seeds.

Compared with `original_mamba_bgnn_full`, the proposed model improves:

- NLL by about 14.29%.
- RMSE by about 2.97%.
- MAE by about 4.87%.
- RankIC by about 241.32%.
- Top-bottom return spread by about 162.32%.

These results support the main conclusion that modeling stocks as graph nodes and learning cross-stock relations is more suitable for VN30 panel forecasting than applying the original feature-node MAMBA-BGNN independently to each stock.

## Suggested Paper Claim

A safe final claim is:

> The proposed Stock Mamba Hybrid improves probabilistic return forecasting and cross-sectional stock ranking on VN30 by combining per-stock temporal Mamba encoding with stock-level graph learning. Across 5 random seeds, it outperforms LSTM, Transformer, and the original MAMBA-BGNN full adaptation on the main forecasting and ranking metrics.
