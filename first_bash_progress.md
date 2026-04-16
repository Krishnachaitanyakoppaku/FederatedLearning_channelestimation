# Progress Report: First Bash Command

- Generated on: 2026-04-15 21:52:32
- Run logs detected: `37` JSON files in `results/raw/`

## What the first bash command is expected to produce
- Phase 2 centralized runs: `6` total (`3 seeds x 2 models`)
- Phase 3 federated matrix: `240` total
  - FedAvg IID: `48` (`3 seeds x 4 rounds x 4 local_epochs`)
  - FedAvg Non-IID: `48` (`3 seeds x 4 rounds x 4 local_epochs`)
  - FedProx Non-IID: `144` (`3 seeds x 4 rounds x 4 local_epochs x 3 mu`)
- Extra ablation block from same command: `16` total
- Final reporting commands: aggregation + publication package outputs

## Completion status by step

| Step | Expected | Completed | Remaining |
|---|---:|---:|---:|
| Phase 2 (Centralized) | 6 | 6 | 0 |
| Phase 3 FedAvg IID | 48 | 7 | 41 |
| Phase 3 FedAvg Non-IID | 48 | 6 | 42 |
| Phase 3 FedProx Non-IID | 144 | 18 | 126 |
| Extra ablation block | 16 | 0 | 16 |

**Overall (runs only):** Expected `262`, Completed `37`, Remaining `225`

## Round/Epoch progress snapshot for Phase 3

### Seed 42
| rounds | local_epochs | FedAvg IID | FedAvg Non-IID | FedProx mu completed |
|---:|---:|:---:|:---:|---:|
| 10 | 1 | Done | Done | 3/3 |
| 10 | 2 | Done | Done | 3/3 |
| 10 | 3 | Done | Done | 3/3 |
| 10 | 5 | Done | Done | 3/3 |
| 20 | 1 | Done | Done | 3/3 |
| 20 | 2 | Done | Done | 3/3 |
| 20 | 3 | Done | Pending | 0/3 |
| 20 | 5 | Pending | Pending | 0/3 |
| 30 | 1 | Pending | Pending | 0/3 |
| 30 | 2 | Pending | Pending | 0/3 |
| 30 | 3 | Pending | Pending | 0/3 |
| 30 | 5 | Pending | Pending | 0/3 |
| 50 | 1 | Pending | Pending | 0/3 |
| 50 | 2 | Pending | Pending | 0/3 |
| 50 | 3 | Pending | Pending | 0/3 |
| 50 | 5 | Pending | Pending | 0/3 |
- Seed 42 summary: IID `7/16`, Non-IID `6/16`, FedProx mu runs `18/48`

### Seed 123
| rounds | local_epochs | FedAvg IID | FedAvg Non-IID | FedProx mu completed |
|---:|---:|:---:|:---:|---:|
| 10 | 1 | Pending | Pending | 0/3 |
| 10 | 2 | Pending | Pending | 0/3 |
| 10 | 3 | Pending | Pending | 0/3 |
| 10 | 5 | Pending | Pending | 0/3 |
| 20 | 1 | Pending | Pending | 0/3 |
| 20 | 2 | Pending | Pending | 0/3 |
| 20 | 3 | Pending | Pending | 0/3 |
| 20 | 5 | Pending | Pending | 0/3 |
| 30 | 1 | Pending | Pending | 0/3 |
| 30 | 2 | Pending | Pending | 0/3 |
| 30 | 3 | Pending | Pending | 0/3 |
| 30 | 5 | Pending | Pending | 0/3 |
| 50 | 1 | Pending | Pending | 0/3 |
| 50 | 2 | Pending | Pending | 0/3 |
| 50 | 3 | Pending | Pending | 0/3 |
| 50 | 5 | Pending | Pending | 0/3 |
- Seed 123 summary: IID `0/16`, Non-IID `0/16`, FedProx mu runs `0/48`

### Seed 777
| rounds | local_epochs | FedAvg IID | FedAvg Non-IID | FedProx mu completed |
|---:|---:|:---:|:---:|---:|
| 10 | 1 | Pending | Pending | 0/3 |
| 10 | 2 | Pending | Pending | 0/3 |
| 10 | 3 | Pending | Pending | 0/3 |
| 10 | 5 | Pending | Pending | 0/3 |
| 20 | 1 | Pending | Pending | 0/3 |
| 20 | 2 | Pending | Pending | 0/3 |
| 20 | 3 | Pending | Pending | 0/3 |
| 20 | 5 | Pending | Pending | 0/3 |
| 30 | 1 | Pending | Pending | 0/3 |
| 30 | 2 | Pending | Pending | 0/3 |
| 30 | 3 | Pending | Pending | 0/3 |
| 30 | 5 | Pending | Pending | 0/3 |
| 50 | 1 | Pending | Pending | 0/3 |
| 50 | 2 | Pending | Pending | 0/3 |
| 50 | 3 | Pending | Pending | 0/3 |
| 50 | 5 | Pending | Pending | 0/3 |
- Seed 777 summary: IID `0/16`, Non-IID `0/16`, FedProx mu runs `0/48`

## Reporting/artifact outputs expected at the end
- Required report files currently present: `10/10`
- [x] `results/summary/aggregated_metrics.csv` (Done)
- [x] `results/summary/final_comparison_table.csv` (Done)
- [x] `results/summary/coverage_report.csv` (Done)
- [x] `results/summary/main_comparison_table.csv` (Done)
- [x] `results/summary/ablation_table.csv` (Done)
- [x] `results/summary/significance_tests.csv` (Done)
- [x] `results/summary/reliability_table.csv` (Done)
- [x] `results/summary/communication_efficiency_table.csv` (Done)
- [x] `results/summary/complexity_and_latency_table.csv` (Done)
- [x] `results/figures/paper/figure_manifest.csv` (Done)

## Immediate next run target from your stop point
- You had started `FedAvg IID` for `(seed=42, rounds=20, local_epochs=3)`.
- Next logical continuation is to run remaining Phase 3 combinations with `--resume`.
