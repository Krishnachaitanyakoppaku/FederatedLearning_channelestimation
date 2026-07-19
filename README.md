# DeepMIMO Federated Wireless Channel Estimation

This repository contains a complete, publication-oriented research pipeline for wireless channel estimation using centralized and federated deep learning.

It includes:
- dataset generation from DeepMIMO O1_60,
- controlled noise injection and train/test preparation,
- centralized baselines (CNN, ResUNet),
- federated training (FedAvg, FedProx under IID/Non-IID),
- run-level structured logging with resume support,
- publication-grade summaries, statistics, and figures,
- IEEE conference manuscript sources and submission package.

The project is designed for reproducible experimentation and manuscript-ready evidence generation.

---

## Contents

1. [Research Overview](#research-overview)
2. [What Was Implemented](#what-was-implemented)
3. [Repository Structure](#repository-structure)
4. [Dataset Pipeline (Full Detail)](#dataset-pipeline-full-detail)
5. [Model Architectures](#model-architectures)
6. [Federated Setup and Non-IID Definitions](#federated-setup-and-non-iid-definitions)
7. [Experiment Protocols](#experiment-protocols)
8. [Current Results and Conclusions](#current-results-and-conclusions)
9. [Publication Artifacts](#publication-artifacts)
10. [How to Reproduce End-to-End](#how-to-reproduce-end-to-end)
11. [Submission Package and LaTeX Build](#submission-package-and-latex-build)
12. [Limitations and Future Work](#limitations-and-future-work)

---

## Research Overview

### Problem
Wireless channel estimation quality is critical for communication reliability, but centralized training assumes data pooling at a central server. In realistic deployments, user/client data is distributed and often non-identically distributed (Non-IID), making federated learning behavior non-trivial.

### Main questions investigated
1. Can federated learning approach centralized NMSE performance?
2. How much does Non-IID distribution hurt channel estimation?
3. Does FedProx improve heterogeneity robustness over FedAvg?
4. Do trends hold across both CNN and ResUNet architectures?
5. What is the communication-vs-accuracy trade-off?

### Study framing
This work is a systematic, reproducible benchmark study (not a novel optimizer proposal), with strong emphasis on transparency and conference-quality reporting.

---

## What Was Implemented

### Core training/evaluation components
- **Centralized CNN** (`Centralised Learning/CNN/train_and_evaluate.py`)
- **Centralized ResUNet** (`Centralised Learning/ResUNet/resunet_model.py`)
- **Federated CNN FedAvg** (`Federated Learning/CNN_Fedavg/federated_train.py`)
- **Federated CNN FedProx/Non-IID** (`Federated Learning/CNN_FedProx/federated_train.py`)
- **Federated ResUNet FedProx (+ IID/Non-IID via flags)** (`Federated Learning/resunet_fedprox/fedprox_resunet.py`)

### Reproducibility and reporting system
- Structured run logs (`results/raw/*.json`, `results/raw/*_history.csv`)
- Checkpoint/resume mechanism (`results/checkpoints/`)
- Aggregation and coverage checks (`aggregate_results.py`, `validate_smoke.py`)
- Publication figures and tables (`generate_roadmap_assets.py`, notebook)

### Manuscript production
- IEEE manuscript source: `IEEE_Conference_Paper.tex`
- Submission-ready package: `submission_package/main.tex`, `submission_package/main.pdf`, `submission_package/figures/`

---

## Repository Structure

```text
Project/
|-- generate_dataset.py
|-- add_noise.py
|-- prepare_data.py
|-- ls_estimator.py
|-- run_experiments.py
|-- aggregate_results.py
|-- generate_paper_plots.py
|-- generate_roadmap_assets.py
|-- run_publication_scenarios.sh
|-- run_resunet_claim_support.sh
|-- roadmap_figure_generation.ipynb
|-- IEEE_Conference_Paper.tex
|-- PUBLICATION_ROADMAP.md
|
|-- dataset/
|   |-- H_clean.npy
|   |-- H_clean_ri.npy
|   |-- H_noisy_{0,5,10,15,20}dB.npy
|   |-- H_noisy_{0,5,10,15,20}dB_ri.npy
|   `-- cnn/
|       |-- 0dB/ ... 20dB/
|       `-- each contains: X_train.npy, Y_train.npy, X_test.npy, Y_test.npy, norm_stats.npz
|
|-- Centralised Learning/
|   |-- CNN/
|   |   |-- cnn_model.py
|   |   `-- train_and_evaluate.py
|   `-- ResUNet/
|       `-- resunet_model.py
|
|-- Federated Learning/
|   |-- CNN_Fedavg/
|   |   |-- split_clients.py
|   |   |-- local_train.py
|   |   `-- federated_train.py
|   |-- CNN_FedProx/
|   |   |-- non_iid_split.py
|   |   |-- local_train.py
|   |   `-- federated_train.py
|   |-- ResUNet/
|   |   `-- fed_resunet.py
|   `-- resunet_fedprox/
|       `-- fedprox_resunet.py
|
|-- results/
|   |-- raw/
|   |-- checkpoints/
|   |-- summary/
|   `-- figures/paper/
|
`-- submission_package/
    |-- main.tex
    |-- main.pdf
    `-- figures/
```

---

## Dataset Pipeline (Full Detail)

### 1) DeepMIMO channel extraction (`generate_dataset.py`)

Pipeline performed:
- Loads DeepMIMO scenario `O1_60`
- Selects base station index `0`
- Subsamples `6000` users (uniformly spaced indices)
- Configures channel generation:
  - BS antenna shape: `8x8`
  - UE antenna shape: `1x1`
  - frequency-domain channels
  - OFDM bandwidth: `0.5e6`
  - subcarriers: `64`
  - selected subcarriers: all 64
- Computes channels and saves to:
  - `o1_60_matrix.npy`

Resulting tensor shape from script comments/logs:
- users × rx_ant × tx_ant × subcarriers
- then downstream squeezed to remove singleton rx axis for processing.

### 2) Noise injection (`add_noise.py`)

Input:
- `o1_60_matrix.npy` (complex channels)

Noise protocol:
- Circularly symmetric complex Gaussian AWGN
- SNR levels: `[0, 5, 10, 15, 20] dB`
- Fixed RNG seed: `42`
- Per-sample power normalization to enforce target SNR per user

Saved outputs:
- clean complex: `dataset/H_clean.npy`
- clean real/imag stack: `dataset/H_clean_ri.npy`
- noisy complex per SNR: `dataset/H_noisy_{snr}dB.npy`
- noisy real/imag per SNR: `dataset/H_noisy_{snr}dB_ri.npy`

### 3) CNN-ready split and normalization (`prepare_data.py`)

For each SNR:
- Loads noisy input and clean target in `ri` format
- Applies global normalization using mean/std over concatenated X+Y
- Uses reproducible shuffle (`seed=42`)
- Splits into `80/20` train/test
- Saves to `dataset/cnn/{snr}dB/`:
  - `X_train.npy`, `Y_train.npy`, `X_test.npy`, `Y_test.npy`, `norm_stats.npz`

This produces the standardized supervised learning dataset used by CNN and FL scripts.

### 4) Classical baseline utility (`ls_estimator.py`)

- Implements LS approximation by direct noisy observation pass-through (`H_ls = H_noisy`)
- Computes NMSE against clean channels over SNR levels.
- Used as a classical baseline reference in manuscript discussion.

---

## Model Architectures

### CNN (`Centralised Learning/CNN/cnn_model.py`)

Input/output:
- Input: `(B, 2, 64, 64)` real/imag noisy channel
- Output: `(B, 2, 64, 64)` estimated clean channel

Architecture:
- Conv(2→32) + BN + ReLU
- One residual block (32 channels)
- Conv(32→2)
- Residual-learning output: `H_est = H_noisy + delta`

Parameter count:
- `19,778` trainable parameters (computed in environment)

### ResUNet (`Centralised Learning/ResUNet/resunet_model.py` and FL variants)

Architecture family:
- Encoder-decoder with skip connections
- Residual bottleneck design
- Higher capacity compared to CNN

Parameter count:
- `763,234` trainable parameters (computed in environment)

---

## Federated Setup and Non-IID Definitions

### FedAvg (IID) for CNN
- Script: `Federated Learning/CNN_Fedavg/split_clients.py`
- Mechanism:
  - global shuffle of train indices
  - equal partition to clients

### FedAvg/FedProx Non-IID for CNN

Two non-IID split mechanisms are present in repo:

1. `Federated Learning/CNN_Fedavg/split_clients.py --non_iid`
   - sorts samples by clean-channel magnitude norm
   - assigns contiguous chunks per client

2. `Federated Learning/CNN_FedProx/non_iid_split.py`
   - loads source pools from SNRs `[0,5,10,15,20]`
   - rotates source pool by client ID and takes fixed samples per client
   - produces client distribution skew across source SNRs

### ResUNet federated (recommended path: `fedprox_resunet.py`)

Supports both IID and Non-IID in one script:
- IID: default split
- Non-IID: `--non_iid`
- Non-IID controls:
  - `--non_iid_unique_ratio` (default in script-based runs: `0.6`)
  - `--non_iid_seed`

### Resume/checkpoint protocol

Federated scripts support:
- `--run_id`
- `--checkpoint_dir`
- `--resume`

This allows interrupted long runs to continue from latest checkpoint.

---

## Experiment Protocols

## 1) Phased run interface

```bash
python run_experiments.py --phase 1
python run_experiments.py --phase 2
python run_experiments.py --phase 3
python run_experiments.py --phase 4
```

## 2) Publication scenario script

```bash
./run_publication_scenarios.sh
```

## 3) ResUNet add-on validation batch (20 runs)

```bash
./run_resunet_claim_support.sh
```

Protocol covered by this script:
- Centralized: 5 seeds
- FedAvg-IID: 5 seeds (via `fedprox_resunet.py` with `mu=0`)
- FedAvg-Non-IID: 5 seeds (`mu=0`, `--non_iid`)
- FedProx-Non-IID: 5 seeds (`mu=0.001`, `--non_iid`)

---

## Current Results and Conclusions

All values below come from generated summary artifacts in `results/summary/`.

### CNN core benchmark (10 dB)

From `paper_table_2_statistical_summary.csv`:
- Centralized: mean `-19.4227 dB`, std `0.1617`, N=3
- FedAvg-IID: mean `-19.1190 dB`, std `0.6087`, N=48
- FedAvg-Non-IID: mean `-18.8483 dB`, std `0.5734`, N=48
- FedProx-Non-IID: mean `-18.5683 dB`, std `0.5498`, N=144

From `paper_table_1_best_configs.csv`:
- Best centralized: `-19.6215 dB`
- Best FedAvg-IID: `-20.0381 dB` (R50, E5)
- Best FedAvg-Non-IID: `-19.7338 dB`
- Best FedProx-Non-IID: `-19.6479 dB`

Interpretation:
- Non-IID degrades mean performance vs IID.
- Best FedAvg-IID run exceeds best centralized run.
- In mean metrics, centralized remains strongest in current CNN summary.

### ResUNet add-on benchmark (20 runs)

From `resunet_addon_summary.md` / `resunet_addon_group_stats.csv`:
- Centralized: mean `-22.1210 dB`, std `0.3934`, best `-22.5281`
- FedAvg-IID: mean `-22.8969 dB`, std `0.1327`, best `-23.0406`
- FedAvg-Non-IID: mean `-21.1819 dB`, std `0.3369`, best `-21.6237`
- FedProx-Non-IID: mean `-20.4661 dB`, std `0.1393`, best `-20.6357`

Derived deltas (`resunet_addon_derived_deltas.csv`):
- FedAvg Non-IID degradation vs IID: `+1.7150 dB`
- FedAvg-IID vs centralized gap: `-0.7759 dB` (FedAvg-IID better)
- FedProx Non-IID vs FedAvg Non-IID gap: `+0.7158 dB`

### Communication snapshot (CNN)

From `paper_table_4_communication_cost.csv`:
- FedAvg-IID (R10,E1): 50 events, `-17.7319 dB`
- FedAvg-IID (R30,E3): 150 events, `-19.5375 dB`
- FedAvg-IID (R50,E5): 250 events, `-20.0190 dB`
- Centralized(20ep): 0 federated events, `-19.4227 dB`

### High-level conclusions

1. Heterogeneity is a dominant factor: IID consistently outperforms Non-IID.
2. FedAvg-IID can achieve highly competitive performance.
3. FedProx behavior is regime-dependent and not universally superior in this dataset setup.
4. ResUNet shows stronger absolute NMSE than CNN but with larger communication/runtime implications.

---

## Publication Artifacts

### Core summary tables
- `results/summary/paper_table_1_best_configs.csv`
- `results/summary/paper_table_2_statistical_summary.csv`
- `results/summary/paper_table_3_hyperparameter_ablation.csv`
- `results/summary/paper_table_4_communication_cost.csv`
- `results/summary/statistical_tests_research_grade.csv`
- `results/summary/claim_evidence_matrix.md`

### ResUNet add-on summaries
- `results/summary/resunet_addon_run_table.csv`
- `results/summary/resunet_addon_group_stats.csv`
- `results/summary/resunet_addon_derived_deltas.csv`
- `results/summary/resunet_addon_cnn_resunet_comparison.csv`
- `results/summary/resunet_addon_summary.md`

### Figures
- Core paper figures: `results/figures/paper/figure1_*` ... `figure12_*`
- Add-on figures:
  - `results/figures/paper/resunet_addon_nmse_bar.*`
  - `results/figures/paper/resunet_addon_nmse_box.*`
  - `results/figures/paper/resunet_addon_gap_analysis.*`
  - `results/figures/paper/resunet_addon_runtime_bar.*`
  - `results/figures/paper/resunet_addon_cnn_vs_resunet.*`

---

## How to Reproduce End-to-End

```bash
# 1) Setup
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

# 2) Dataset build
python generate_dataset.py
python add_noise.py
for snr in 0 5 10 15 20; do python prepare_data.py --snr "$snr"; done

# 3) Main experiment phases
python run_experiments.py --phase 1
python run_experiments.py --phase 2
python run_experiments.py --phase 3
python run_experiments.py --phase 4

# 4) Aggregate + publication assets
python aggregate_results.py --raw_dir results/raw --out results/summary/aggregated_metrics.csv --paper_table results/summary/final_comparison_table.csv --coverage_out results/summary/coverage_report.csv
python generate_roadmap_assets.py --phase all --paper_model CNN

# 5) ResUNet add-on (optional but recommended for architecture validation)
./run_resunet_claim_support.sh

# 6) Notebook regeneration (includes appended add-on cells)
jupyter notebook roadmap_figure_generation.ipynb
```

---

## Submission Package and LaTeX Build

### Manuscript sources
- Main source: `IEEE_Conference_Paper.tex`
- Submission copy: `submission_package/main.tex`

### Build command

```bash
cd submission_package
tectonic main.tex
```

### Output
- `submission_package/main.pdf`

---

## Limitations and Future Work

Current limitations explicitly tracked in manuscript and workflow:
- Full federated multi-SNR sweep (0/5/10/15/20 dB) is not yet complete.
- MMSE baseline is not fully integrated in matched reproducible pipeline.
- Dataset diversity currently centered on DeepMIMO O1_60.
- ResUNet is currently an add-on validation rather than full hyperparameter grid.

Planned extensions:
- full multi-SNR federated matrix,
- MMSE integration with proper covariance calibration,
- additional DeepMIMO scenarios,
- architecture-component ablation.

---

## External Literature Used in Comparison Framing

- FedAvg foundation: arXiv `1602.05629`
- FedProx foundation: arXiv `1812.06127`
- FL channel-estimation benchmark: arXiv `2008.10846`
- FL survey/open problems: arXiv `1912.04977`

Reference PDFs are stored in `papers/` (or `Papers/` depending on local folder naming).

---

## Practical Notes

- Always run scripts from repository root to avoid relative-path issues.
- Keep run IDs stable for resume-safe long experiments.
- Do not overwrite summary/figure assets used in manuscript claims; create additive files.
- Before paper submission, verify every claim maps to a specific table/figure file path.
