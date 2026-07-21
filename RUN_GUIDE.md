# Wireless CSI Channel Estimation: Complete Execution Guide

This guide provides step-by-step instructions on how to set up, prepare, and run each component of the massive MIMO channel estimation pipeline individually.

---

## 1. Downloading the DeepMIMO Dataset

The pipeline uses the **O1_60** ray-tracing scenario from the DeepMIMO database.

1. Visit the official DeepMIMO website: [deepmimo.net](https://deepmimo.net)
2. Navigate to the **Scenarios** page and locate the **O1 (Outdoor Urban Street - 60 GHz)** scenario.
3. Download the scenario data files.
4. Extract the zip folder and rename the folder to `O1_60`.
5. Place the `O1_60` directory into the scenarios folder in the project:
   ```text
   Project/deepmimo_scenarios/O1_60/
   ```

---

## 2. Preparing and Splitting the Dataset

Before running any models, you need to extract the raw channel matrices, corrupt them with noise, and split them into training and testing partitions.

Run all commands from the **`Project/`** directory.

### Step A: Extract Raw Channels
Reads the O1_60 scenario parameters (8x8 BS antenna grid, 1 UE antenna, 64 subcarriers) and extracts channel vectors for 6,000 users.
```bash
myenv/bin/python scripts/generate_dataset.py
```
* **Output**: `Project/data/raw/o1_60_matrix.npy` (~187 MB).

### Step B: Noise Injection (AWGN)
Applies Circularly Symmetric Complex Gaussian (CSCG) noise per sample at multiple SNR levels (`0, 5, 10, 15, 20 dB`) and stacks them in Real/Imaginary (`_ri`) split formats.
```bash
myenv/bin/python scripts/add_noise.py
```
* **Output**: Real/imaginary stacked matrices (`H_clean_ri.npy`, `H_noisy_10dB_ri.npy`, etc.) saved to `Project/data/raw/`.

### Step C: Train-Test Splits
Partitions the data into an 80/20 train/test split (4,800 train samples, 1,200 test samples) and standardizes them.
```bash
# Repeat for each SNR level (0, 5, 10, 15, 20)
for snr in 0 5 10 15 20; do
    myenv/bin/python scripts/prepare_data.py --snr "$snr"
done
```
* **Output**: Data tensors saved in `Project/data/splits/{snr}dB/` (containing `X_train.npy`, `Y_train.npy`, `X_test.npy`, `Y_test.npy`, and `norm_stats.npz`).

---

## 3. Running Individual Models

You can train and evaluate any model architecture (Centralized or Federated) independently.

### Option A: Centralized CNN Baseline
Trains a lightweight CNN with residual learning directly on the unified dataset at a target SNR.
```bash
myenv/bin/python models/centralized/CNN/train_and_evaluate.py \
    --snr 10 \
    --epochs 20 \
    --batch 32 \
    --lr 0.001
```
* **Key Flags**:
  * `--snr`: Target SNR level.
  * `--epochs`: Number of epochs to train.
  * `--all`: Evaluates training across all SNR levels sequentially.
* **Output**: Trained weights saved to `models/centralized/CNN/weights/cnn_10dB.pth` and log json to `results/raw/`.

### Option B: Centralized ResUNet Baseline
Trains the high-capacity ResUNet encoder-decoder architecture with skip-connections.
```bash
myenv/bin/python models/centralized/ResUNet/resunet_model.py \
    --snr 10 \
    --epochs 30 \
    --batch 32 \
    --lr 0.001
```
* **Output**: Trained weights saved to `models/centralized/ResUNet/resunet_model.pth`.

---

### Option C: Federated CNN (FedAvg IID)
To run federated training, you must first partition the splits into client-side shards, then run the trainer.

#### 1. Split Client Shards (IID)
```bash
myenv/bin/python models/federated/CNN_FedAvg/split_clients.py \
    --x_path data/splits/10dB/X_train.npy \
    --y_path data/splits/10dB/Y_train.npy \
    --clients 5 \
    --output_dir models/federated/CNN_FedAvg/clients
```

#### 2. Run Federated Averaging
```bash
myenv/bin/python models/federated/CNN_FedAvg/federated_train.py \
    --clients 5 \
    --rounds 20 \
    --local_epochs 2 \
    --batch 32 \
    --lr 0.001 \
    --client_dir models/federated/CNN_FedAvg/clients \
    --x_test data/splits/10dB/X_test.npy \
    --y_test data/splits/10dB/Y_test.npy \
    --snr 10
```

---

### Option D: Federated CNN (FedAvg/FedProx Non-IID)

#### 1. Split Client Shards (Non-IID)
Creates statistical heterogeneity across client devices by rotating SNR partitions.
```bash
myenv/bin/python models/federated/CNN_FedProx/non_iid_split.py \
    --snr 10 \
    --clients 5 \
    --samples_per_client 960 \
    --output_dir models/federated/CNN_FedProx/clients_non_iid
```

#### 2. Run FedAvg (on Non-IID clients)
```bash
myenv/bin/python models/federated/CNN_FedProx/federated_train.py \
    --clients 5 \
    --rounds 20 \
    --local_epochs 2 \
    --batch 32 \
    --lr 0.001 \
    --client_dir models/federated/CNN_FedProx/clients_non_iid \
    --x_test data/splits/10dB/X_test.npy \
    --y_test data/splits/10dB/Y_test.npy \
    --snr 10
```

#### 3. Run FedProx (with proximal coefficient $\mu$)
Adds the proximal term to penalize local weight drift from the global model.
```bash
myenv/bin/python models/federated/CNN_FedProx/federated_train.py \
    --fedprox \
    --mu 0.01 \
    --clients 5 \
    --rounds 20 \
    --local_epochs 2 \
    --batch 32 \
    --lr 0.001 \
    --client_dir models/federated/CNN_FedProx/clients_non_iid \
    --x_test data/splits/10dB/X_test.npy \
    --y_test data/splits/10dB/Y_test.npy \
    --snr 10
```

---

### Option E: Federated ResUNet (FedAvg/FedProx IID & Non-IID)
The ResUNet federated training consolidates splitting and execution parameters into a single script.

#### 1. Run ResUNet FedAvg IID
```bash
myenv/bin/python models/federated/ResUNet_FedProx/fedprox_resunet.py \
    --snr 10 \
    --rounds 50 \
    --local_epochs 5 \
    --batch 64 \
    --lr 0.001 \
    --mu 0.0
```

#### 2. Run ResUNet FedProx Non-IID
```bash
myenv/bin/python models/federated/ResUNet_FedProx/fedprox_resunet.py \
    --non_iid \
    --non_iid_unique_ratio 0.6 \
    --snr 10 \
    --rounds 50 \
    --local_epochs 5 \
    --batch 64 \
    --lr 0.001 \
    --mu 0.001
```

---

## 4. Compiling the Figures & Summary Reports

Once individual training runs or the phased experiment sweep has finished:

### 1. Compile comparison metrics
Creates tables listing mean, standard deviation, and best configurations:
```bash
python scripts/aggregate_results.py \
    --raw_dir results/raw \
    --out results/summary/aggregated_metrics.csv \
    --paper_table results/summary/final_comparison_table.csv \
    --coverage_out results/summary/coverage_report.csv
```

### 2. Generate charts
Constructs validation plots, communication sweeps, and latency benchmarks:
```bash
python scripts/generate_publication_package.py \
    --raw_dir results/raw \
    --summary_dir results/summary \
    --dataset_dir data/splits/10dB
```
Generated plots will be saved to `results/figures/`.
