# DeepMIMO Channel Estimation and Federated Learning

This project performs MIMO channel estimation on DeepMIMO O1_60 data using:
- centralised learning (CNN and ResUNet), and
- federated learning (FedAvg/FedProx for CNN, plus ResUNet federated variants).

## Project Structure

```text
Project/
|-- generate_dataset.py
|-- add_noise.py
|-- prepare_data.py
|-- ls_estimator.py
|-- dataset/
|   |-- H_clean.npy, H_noisy_*dB.npy, *_ri.npy
|   `-- cnn/
|       |-- 0dB/ 5dB/ 10dB/ 15dB/ 20dB/
|       `-- each: X_train.npy, Y_train.npy, X_test.npy, Y_test.npy, norm_stats.npz
|-- deepmimo_scenarios/
|-- Centralised Learning/
|   |-- CNN/
|   |   |-- cnn_model.py
|   |   |-- train_and_evaluate.py
|   |   `-- weights/
|   `-- ResUNet/
|       |-- resunet_model.py
|       `-- resunet_model.pth (generated)
`-- Federated Learning/
    |-- CNN_Fedavg/
    |   |-- split_clients.py
    |   |-- local_train.py
    |   |-- federated_train.py
    |   |-- evaluate_federated.py
    |   `-- clients/ (generated)
    |-- CNN_FedProx/
    |   |-- non_iid_split.py
    |   |-- local_train.py
    |   |-- federated_train.py
    |   |-- tune_fedprox.py / test_*.py
    |   `-- clients_non_iid/ (generated)
    |-- ResUNet/
    |   `-- fed_resunet.py
    `-- resunet_fedprox/
        `-- fedprox_resunet.py
```

## Setup

1) Create and activate a virtual environment:

```bash
python3 -m venv myenv
source myenv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Download DeepMIMO O1_60 scenario:

```python
import deepmimo as dm
dm.download("O1_60")
```

## Pipeline

Run from `Project/` root.

### 1) Generate and preprocess data

```bash
python generate_dataset.py
python add_noise.py
for snr in 0 5 10 15 20; do python prepare_data.py --snr "$snr"; done
```

### 2) Centralised baseline (CNN)

```bash
python "Centralised Learning/CNN/train_and_evaluate.py" --all
```

### 3) Federated CNN

FedAvg (IID):

```bash
python "Federated Learning/CNN_Fedavg/split_clients.py"
python "Federated Learning/CNN_Fedavg/federated_train.py"
```

FedProx/FedAvg comparison on non-IID split:

```bash
python "Federated Learning/CNN_FedProx/non_iid_split.py"
python "Federated Learning/CNN_FedProx/federated_train.py"
```

### 4) Optional ResUNet runs

```bash
python "Centralised Learning/ResUNet/resunet_model.py" --all
python "Federated Learning/ResUNet/fed_resunet.py"
python "Federated Learning/resunet_fedprox/fedprox_resunet.py" --non_iid
```

## Research Workflow (Phased)

The repository now includes a phased execution pipeline aligned with research-paper reporting norms.

### Phase 1: Protocol and configuration check

```bash
python run_experiments.py --phase 1
```

### Phase 2: Centralized baselines (multi-seed)

```bash
python run_experiments.py --phase 2
```

### Phase 3: Federated sweeps (rounds, local epochs, mu)

```bash
python run_experiments.py --phase 3
```

### Phase 4: Aggregate results for final tables

```bash
python run_experiments.py --phase 4
```

## Outputs for Paper

- Raw run logs: `results/raw/*.json`, `results/raw/*_history.csv`
- Aggregated metrics: `results/summary/aggregated_metrics.csv`
- Final comparison table: `results/summary/final_comparison_table.csv`
- Figures (mandatory plots):

```bash
python generate_paper_plots.py
```

Saved in `results/figures/`.
