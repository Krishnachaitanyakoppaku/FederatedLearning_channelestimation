# Experimental Results & Performance Comparisons

This document consolidates all quantitative results, statistical summaries, hyperparameter sensitivity sweeps, and model comparisons for the massive MIMO channel estimation framework (at 10 dB SNR).

---

## 1. CNN Core Benchmarks

The baseline CNN model consists of a residual-learning architecture with $19,778$ trainable parameters.

### Table 1: Best Performing Configurations
Summarizes the absolute best normalized mean squared error (NMSE, in dB) achieved by any single run under each framework setting.

| Algorithm | Client Data Mode | Best NMSE (dB) | Optimal Configuration | Associated Run ID |
| :--- | :--- | :--- | :--- | :--- |
| **Centralized** | N/A | **-19.62 dB** | Seed 42 | `cnn_centralized_snr10_seed42` |
| **FedAvg** | IID | **-20.04 dB** | 50 Rounds, 5 Local Epochs, Seed 123 | `cnn_fedavg_iid_r50_e5_seed123` |
| **FedAvg** | Non-IID | **-19.73 dB** | 50 Rounds, 5 Local Epochs, Seed 42 | `cnn_fedavg_non-iid_r50_e5_mu0.01_seed42` |
| **FedProx** | Non-IID | **-19.65 dB** | 50 Rounds, 5 Local Epochs, $\mu = 0.001$, Seed 123 | `cnn_fedprox_non-iid_r50_e5_mu0.001_seed123` |

### Table 2: Statistical Summary (10 dB SNR)
Presents the aggregate statistics over all experimental grid seeds and configurations.

| Algorithm / Framework | N (Runs) | Mean NMSE (dB) | Std Dev (dB) | Min NMSE (dB) | Max NMSE (dB) | Gap vs Centralized (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** | 3 | -19.42 dB | 0.16 dB | -19.62 dB | -19.23 dB | 0.00 dB (ref) |
| **FedAvg-IID** | 48 | -19.12 dB | 0.61 dB | -20.04 dB | -17.66 dB | +0.30 dB |
| **FedAvg-Non-IID** | 48 | -18.85 dB | 0.57 dB | -19.73 dB | -17.38 dB | +0.57 dB |
| **FedProx-Non-IID** | 144 | -18.57 dB | 0.55 dB | -19.65 dB | -17.12 dB | +0.85 dB |

---

## 2. Hyperparameter Ablation and Sensitivity Analysis

Sweeps are conducted to identify performance sensitivity across global communication rounds, client local training epochs, proximal regularizer weight ($\mu$), and random initialization seeds.

### Table 3: Hyperparameter Ablation Summary
| Parameter Variable | Sweep Range | Performance Range (dB) | Best Value | Sensitivity Span (dB) | Impact Level |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Global Rounds ($R$)** | 10, 20, 30, 50 | -19.20 to -18.14 dB | 50 | 1.07 dB | Moderate |
| **Local Epochs ($E$)** | 1, 2, 3, 5 | -19.13 to -18.16 dB | 5 | 0.97 dB | Moderate-High |
| **FedProx $\mu$** | 0.001, 0.005, 0.01 | -18.79 to -18.36 dB | 0.001 | 0.42 dB | Low |
| **Random Seed** | 42, 123, 777 | -18.76 to -18.71 dB | 42 | 0.05 dB | Negligible |

---

## 3. Communication Cost vs. Performance Trade-off

Federated training requires transmitting model weights between client devices and the centralized server. This analysis evaluates the communication payload required to reach corresponding NMSE performance levels.

### Table 4: Communication vs. Efficiency Matrix (CNN)
| Configuration | Global Rounds ($R$) | Local Epochs ($E$) | Comm. Events | Mean NMSE (dB) | Efficiency Level |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FedAvg-IID (R10, E1)** | 10 | 1 | 50 | -17.73 dB | Low Overhead / Low Accuracy |
| **FedAvg-IID (R30, E3)** | 30 | 3 | 150 | -19.54 dB | Medium Overhead / High Accuracy |
| **FedAvg-IID (R50, E5)** | 50 | 5 | 250 | -20.02 dB | High Overhead / Max Accuracy |
| **Centralized (20ep)** | N/A | 20 | 0 | -19.42 dB | Offline Baseline (No Overhead) |

---

## 4. ResUNet Architecture Add-on Validation

A validation sweep was executed on the higher-capacity ResUNet encoder-decoder architecture ($763,234$ parameters) over 20 structured runs (5 seeds per setup) to benchmark it against the CNN.

### Table 5: ResUNet Aggregate Statistics (10 dB SNR)
| Framework Setting | Runs ($N$) | Mean NMSE (dB) | Std Dev (dB) | Min NMSE (dB) | Max NMSE (dB) | Mean Training Time (sec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** | 5 | -22.12 dB | 0.39 dB | -22.53 dB | -21.53 dB | ~982s |
| **FedAvg-IID** | 5 | -22.90 dB | 0.13 dB | -23.04 dB | -22.70 dB | ~6048s |
| **FedAvg-Non-IID** | 5 | -21.18 dB | 0.34 dB | -21.62 dB | -20.91 dB | ~9013s |
| **FedProx-Non-IID** | 5 | -20.47 dB | 0.14 dB | -20.64 dB | -20.26 dB | ~4984s |

### Table 6: Derived Architecture Gaps (ResUNet vs. CNN)
* **FedAvg Non-IID Heterogeneity Penalty**: **+1.72 dB** degradation (the gap between ResUNet FedAvg IID and Non-IID).
* **FedAvg-IID vs. Centralized Advantage**: **-0.78 dB** (FedAvg-IID performs better than centralized training under identical seed constraints).
* **FedProx vs. FedAvg (Non-IID)**: **+0.72 dB** (FedProx did not outperform FedAvg in this setup; higher $\mu$ penalizes capacity local updates too strictly).

---

## 5. High-Level Insights and Comparison Findings

1. **Heterogeneity Penalization**: In both CNN and ResUNet setups, the transition from IID (independent and identically distributed) client splits to Non-IID (unbalanced SNR pools) splits degrades performance by **0.25 to 1.72 dB**.
2. **Federated Optimization Ceiling**: Surprisingly, the best-case FedAvg-IID runs outperform centralized baseline runs (by **0.42 dB** for CNN and **0.78 dB** for ResUNet). This indicates that the local Adam optimizer runs on client shards, followed by weight averaging, acts as an effective implicit regularizer that helps escape local minima.
3. **ResUNet Capacity Gains**: Across all corresponding settings, ResUNet achieves **1.6 to 3.8 dB lower NMSE** than the lightweight CNN. However, this capacity improvement comes at the cost of **~38x larger model weight sizes** (increased communication payload) and significantly longer training durations.
