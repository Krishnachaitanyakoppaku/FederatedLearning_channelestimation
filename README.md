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
- **Centralized CNN** (`models/centralized/CNN/train_and_evaluate.py`)
- **Centralized ResUNet** (`models/centralized/ResUNet/resunet_model.py`)
- **Federated CNN FedAvg** (`models/federated/CNN_FedAvg/federated_train.py`)
- **Federated CNN FedProx/Non-IID** (`models/federated/CNN_FedProx/federated_train.py`)
- **Federated ResUNet FedProx (+ IID/Non-IID via flags)** (`models/federated/ResUNet_FedProx/fedprox_resunet.py`)

### Reproducibility and reporting system
- Structured run logs (`results/raw/*.json`, `results/raw/*_history.csv`)
- Aggregation and coverage checks (`scripts/aggregate_results.py`, `scripts/validate_smoke.py`)
- Publication figures and tables (`scripts/generate_roadmap_assets.py`, plotting scripts)

### Manuscript production
- IEEE manuscript source: `IEEE_Conference_Paper.tex`
- Submission-ready package: `submission_package/main.tex`, `submission_package/main.pdf`, `submission_package/figures/`

---

## Repository Structure

```text
Project/
|-- README.md
|-- requirements.txt
|-- requirements-lock.txt
|-- experiment_config.json
|
|-- scripts/
|   |-- generate_dataset.py
|   |-- add_noise.py
|   |-- prepare_data.py
|   |-- ls_estimator.py
|   |-- compute_mmse_baseline.py
|   |-- run_experiments.py
|   |-- aggregate_results.py
|   |-- generate_paper_plots.py
|   |-- generate_publication_package.py
|   |-- generate_roadmap_assets.py
|   `-- validate_smoke.py
|
|-- data/
|   |-- raw/
|   |   |-- o1_60_matrix.npy
|   |   |-- H_clean.npy
|   |   |-- H_clean_ri.npy
|   |   |-- H_noisy_{0,5,10,15,20}dB.npy
|   |   `-- H_noisy_{0,5,10,15,20}dB_ri.npy
|   `-- splits/
|       |-- 0dB/ ... 20dB/
|       `-- each contains: X_train.npy, Y_train.npy, X_test.npy, Y_test.npy, norm_stats.npz
|
|-- models/
|   |-- centralized/
|   |   |-- CNN/
|   |   |   |-- cnn_model.py
|   |   |   `-- train_and_evaluate.py
|   |   `-- ResUNet/
|   |       `-- resunet_model.py
|   `-- federated/
|       |-- CNN_FedAvg/
|       |   |-- split_clients.py
|       |   |-- local_train.py
|       |   `-- federated_train.py
|       |-- CNN_FedProx/
|       |   |-- non_iid_split.py
|       |   |-- local_train.py
|       |   `-- federated_train.py
|       |-- ResUNet_FedAvg/
|       |   `-- fed_resunet.py
|       `-- ResUNet_FedProx/
|           `-- fedprox_resunet.py
|
|-- research_core/
|   |-- __init__.py
|   |-- config.py
|   |-- logging_utils.py
|   `-- metrics.py
|
|-- results/
|   |-- raw/
|   |-- summary/
|   |-- logs/
|   |-- models/
|   `-- figures/
|
`-- paper/
    |-- IEEE_Conference_Paper.tex
    |-- main.tex
    |-- main.pdf
    |-- main.bib
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
- clean complex: `data/raw/H_clean.npy`
- clean real/imag stack: `data/raw/H_clean_ri.npy`
- noisy complex per SNR: `data/raw/H_noisy_{snr}dB.npy`
- noisy real/imag per SNR: `data/raw/H_noisy_{snr}dB_ri.npy`

### 3) CNN-ready split and normalization (`prepare_data.py`)

For each SNR:
- Loads noisy input and clean target in `ri` format
- Applies global normalization using mean/std over concatenated X+Y
- Uses reproducible shuffle (`seed=42`)
- Splits into `80/20` train/test
- Saves to `data/splits/{snr}dB/`:
  - `X_train.npy`, `Y_train.npy`, `X_test.npy`, `Y_test.npy`, `norm_stats.npz`

This produces the standardized supervised learning dataset used by CNN and FL scripts.

### 4) Classical baseline utility (`ls_estimator.py`)

- Implements LS approximation by direct noisy observation pass-through (`H_ls = H_noisy`)
- Computes NMSE against clean channels over SNR levels.
- Used as a classical baseline reference in manuscript discussion.

---

## Model Architectures

### CNN (`models/centralized/CNN/cnn_model.py`)

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

### ResUNet (`models/centralized/ResUNet/resunet_model.py` and FL variants)

Architecture family:
- Encoder-decoder with skip connections
- Residual bottleneck design
- Higher capacity compared to CNN

Parameter count:
- `763,234` trainable parameters (computed in environment)

---

## Federated Setup and Non-IID Definitions

### FedAvg (IID) for CNN
- Script: `models/federated/CNN_FedAvg/split_clients.py`
- Mechanism:
  - global shuffle of train indices
  - equal partition to clients

### FedAvg/FedProx Non-IID for CNN

Two non-IID split mechanisms are present in repo:

1. `models/federated/CNN_FedAvg/split_clients.py --non_iid`
   - sorts samples by clean-channel magnitude norm
   - assigns contiguous chunks per client

2. `models/federated/CNN_FedProx/non_iid_split.py`
   - loads source pools from SNRs `[0,5,10,15,20]`
   - rotates source pool by client ID and takes fixed samples per client
   - produces client distribution skew across source SNRs

### ResUNet federated (recommended path: `models/federated/ResUNet_FedProx/fedprox_resunet.py`)

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

To execute the systematic experimental grid, we use a unified interface structured around **four phases**:

### **Phase 1: Verification & Configuration**
Loads the environment settings from `experiment_config.json` and prints the parameter grid (seeds, target SNRs, communication rounds, client local epochs). Useful to sanity-check configurations before launching computations.
```bash
python scripts/run_experiments.py --phase 1
```

### **Phase 2: Centralized Baseline Benchmarks**
Sweeps through each random seed to train the centralized baseline models (CNN and ResUNet) on the combined dataset. This provides the performance ceilings (optimal NMSE) under unified training conditions.
```bash
python scripts/run_experiments.py --phase 2
```

### **Phase 3: Federated Grid Sweeps**
Performs the complete federated learning parameter sweep:
1. Shards the dataset into IID divisions (`split_clients.py`) and Non-IID divisions (`non_iid_split.py`).
2. Sweeps through seeds, global rounds, and local epochs to train the FedAvg-IID, FedAvg-Non-IID, and FedProx-Non-IID frameworks.
```bash
python scripts/run_experiments.py --phase 3
```

### **Phase 4: Structured Data Aggregation**
Collects the raw JSON telemetry logs generated by the centralized and federated training runs in `results/raw/` and compiles them into clean comparison sheets in `results/summary/`.
```bash
python scripts/run_experiments.py --phase 4
```



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

Follow these step-by-step instructions to recreate the dataset, execute the grid sweep, aggregate results, and compile the final paper figures:

### 1) Environment Setup
Recreate the virtual environment using Python 3.12 (Homebrew installation is recommended to avoid version mismatch):
```bash
# Clean previous setup and create new environment
rm -rf myenv
/opt/homebrew/bin/python3.12 -m venv myenv
source myenv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Dataset Pipeline Generation
Extract channels from the DeepMIMO simulator, inject CSCG noise per sample, and create split normalization folders:
```bash
# Extract O1_60 ray-tracing channels
python scripts/generate_dataset.py

# Inject CSCG noise at [0, 5, 10, 15, 20] dB SNR
python scripts/add_noise.py

# Create 80/20 train/test splits for each SNR
for snr in 0 5 10 15 20; do
    python scripts/prepare_data.py --snr "$snr"
done
```

### 3) Run Experiment Grid (Phases 1-4)
Execute the configured sweep phases sequentially:
```bash
# Verify configs
python scripts/run_experiments.py --phase 1

# Train Centralized CNN and ResUNet models
python scripts/run_experiments.py --phase 2

# Split client datasets (IID/Non-IID) and run federated sweep
python scripts/run_experiments.py --phase 3

# Consolidate raw logs
python scripts/run_experiments.py --phase 4
```

### 4) Generate Manuscript Tables and Figures
Compile the raw metrics into final comparison tables and generate all publication-grade figures:
```bash
# Generate aggregated metrics and coverage reports
python scripts/aggregate_results.py \
    --raw_dir results/raw \
    --out results/summary/aggregated_metrics.csv \
    --paper_table results/summary/final_comparison_table.csv \
    --coverage_out results/summary/coverage_report.csv

# Generate publication plots (Pareto fronts, SNR sweeps, latency)
python scripts/generate_publication_package.py \
    --raw_dir results/raw \
    --summary_dir results/summary \
    --dataset_dir data/splits/10dB

# Generate additional roadmap assets
python scripts/generate_roadmap_assets.py \
    --phase all \
    --paper_model CNN
```

---

## Submission Package and LaTeX Build

### Manuscript sources
- Main source: `paper/IEEE_Conference_Paper.tex`
- Submission copy: `paper/main.tex`

### Build command

```bash
cd paper
tectonic main.tex
```

### Output
- `paper/main.pdf`

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
