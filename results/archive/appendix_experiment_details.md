# Appendix: Hyperparameters and Compute Budget

## Hyperparameter Configuration

Source config: `experiment_config.json`

- `seeds`: `[42, 123, 777]`
- `snr_db`: `10`
- `centralized_epochs.cnn`: `20`
- `centralized_epochs.resunet`: `40`
- `batch_size`: `32`
- `lr.cnn`: `0.001`
- `lr.resunet`: `0.0005`
- `lr.federated`: `0.001`
- `fl_rounds`: `[10, 20, 30, 50]`
- `local_epochs`: `[1, 2, 3, 5]`
- `fedprox_mu`: `[0.001, 0.005, 0.01]`

## Compute Budget Summary (from available run logs)

Computed from `results/raw/*.json` durations on 2026-04-15.

- Total logged runs: `7`
- Total logged duration: `3502.064 sec` (~`58.37 min`)

By setting/model/data mode:

- Centralized / CNN / N/A: `631.270 sec`
- Centralized / ResUNet / N/A: `2429.249 sec`
- FedAvg / CNN / IID: `441.546 sec`

Per-run durations observed:

- `cnn_centralized_snr10_seed42`: `322.044 sec`
- `cnn_centralized_snr10_seed123`: `97.718 sec`
- `cnn_centralized_snr10_seed777`: `211.507 sec`
- `resunet_centralized_snr10_seed42`: `1256.996 sec`
- `resunet_centralized_snr10_seed123`: `1172.253 sec`
- `resunet_centralized_snr10_seed777`: `0.000 sec` (likely interrupted/incomplete run metadata)
- `cnn_fedavg_iid_r20_e2_seed42`: `441.546 sec`

## Reproducibility Pointers

- Raw logs: `results/raw/`
- Aggregated metrics: `results/summary/aggregated_metrics.csv`
- Final comparison table: `results/summary/final_comparison_table.csv`
- Coverage report: `results/summary/coverage_report.csv`
- Figure manifest: `results/figures/figure_manifest.csv`
- Clean rerun figure manifest: `results/figures_clean/figure_manifest.csv`
