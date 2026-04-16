# CNN Experiments Test Report

- Generated on: 2026-04-17 02:09:18
- Source: `results/raw/cnn_*.json`
- Total CNN runs found: `243`

## 1) What We Ran (Quick View)

| Category | Value |
|---|---|
| Centralized CNN runs | 3 |
| Federated CNN runs | 240 |
| FedAvg IID runs | 48 |
| FedAvg Non-IID runs | 48 |
| FedProx Non-IID runs | 144 |
| Best NMSE seen (overall) | -20.0381 dB (`cnn_fedavg_iid_r50_e5_seed123`) |

## 2) Main Configuration Space

- Seeds: `[42, 123, 777]`
- SNR(dB): `[10]`
- Clients (federated): `[5]`
- Global rounds tested: `[10, 20, 30, 50]`
- Local epochs tested: `[1, 2, 3, 5]`
- FedProx mu tested: `['0.001', '0.005', '0.01']`

## 3) Best Result per Setting

| Setting | Data mode | Best NMSE (dB) | Run ID |
|---|---|---:|---|
| Centralized | N/A | -19.6215 | `cnn_centralized_snr10_seed42` |
| FedAvg | IID | -20.0381 | `cnn_fedavg_iid_r50_e5_seed123` |
| FedAvg | Non-IID | -19.7338 | `cnn_fedavg_non-iid_r50_e5_mu0.01_seed42` |
| FedProx | Non-IID | -19.6479 | `cnn_fedprox_non-iid_r50_e5_mu0.001_seed123` |

## 4) Centralized CNN Runs (Detailed)

| Run ID | Seed | SNR(dB) | Epochs | LR | Batch | Final NMSE (dB) |
|---|---:|---:|---:|---:|---:|---:|
| `cnn_centralized_snr10_seed42` | 42 | 10 | 20 | 0.001 | 32 | -19.6215 |
| `cnn_centralized_snr10_seed123` | 123 | 10 | 20 | 0.001 | 32 | -19.4210 |
| `cnn_centralized_snr10_seed777` | 777 | 10 | 20 | 0.001 | 32 | -19.2255 |

## 5) Federated CNN Runs (All Detailed Tests)

| Run ID | Setting | Mode | Seed | Clients | Rounds | Local Epochs | mu | Final NMSE (dB) | Best NMSE (dB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cnn_fedavg_iid_r10_e1_seed42` | FedAvg | IID | 42 | 5 | 10 | 1 | - | -17.7981 | -17.7981 |
| `cnn_fedavg_iid_r10_e2_seed42` | FedAvg | IID | 42 | 5 | 10 | 2 | - | -18.4549 | -18.4549 |
| `cnn_fedavg_iid_r10_e3_seed42` | FedAvg | IID | 42 | 5 | 10 | 3 | - | -18.9154 | -18.9154 |
| `cnn_fedavg_iid_r10_e5_seed42` | FedAvg | IID | 42 | 5 | 10 | 5 | - | -19.2582 | -19.2582 |
| `cnn_fedavg_iid_r20_e1_seed42` | FedAvg | IID | 42 | 5 | 20 | 1 | - | -18.3613 | -18.3613 |
| `cnn_fedavg_iid_r20_e2_seed42` | FedAvg | IID | 42 | 5 | 20 | 2 | - | -19.0481 | -19.0481 |
| `cnn_fedavg_iid_r20_e3_seed42` | FedAvg | IID | 42 | 5 | 20 | 3 | - | -19.3213 | -19.3213 |
| `cnn_fedavg_iid_r20_e5_seed42` | FedAvg | IID | 42 | 5 | 20 | 5 | - | -19.6673 | -19.6673 |
| `cnn_fedavg_iid_r30_e1_seed42` | FedAvg | IID | 42 | 5 | 30 | 1 | - | -18.6709 | -18.6709 |
| `cnn_fedavg_iid_r30_e2_seed42` | FedAvg | IID | 42 | 5 | 30 | 2 | - | -19.2721 | -19.2721 |
| `cnn_fedavg_iid_r30_e3_seed42` | FedAvg | IID | 42 | 5 | 30 | 3 | - | -19.5606 | -19.5606 |
| `cnn_fedavg_iid_r30_e5_seed42` | FedAvg | IID | 42 | 5 | 30 | 5 | - | -19.8526 | -19.8526 |
| `cnn_fedavg_iid_r50_e1_seed42` | FedAvg | IID | 42 | 5 | 50 | 1 | - | -19.0139 | -19.0139 |
| `cnn_fedavg_iid_r50_e2_seed42` | FedAvg | IID | 42 | 5 | 50 | 2 | - | -19.5105 | -19.5105 |
| `cnn_fedavg_iid_r50_e3_seed42` | FedAvg | IID | 42 | 5 | 50 | 3 | - | -19.7467 | -19.7467 |
| `cnn_fedavg_iid_r50_e5_seed42` | FedAvg | IID | 42 | 5 | 50 | 5 | - | -20.0167 | -20.0167 |
| `cnn_fedavg_iid_r10_e1_seed123` | FedAvg | IID | 123 | 5 | 10 | 1 | - | -17.7359 | -17.7359 |
| `cnn_fedavg_iid_r10_e2_seed123` | FedAvg | IID | 123 | 5 | 10 | 2 | - | -18.4077 | -18.4077 |
| `cnn_fedavg_iid_r10_e3_seed123` | FedAvg | IID | 123 | 5 | 10 | 3 | - | -18.7743 | -18.7743 |
| `cnn_fedavg_iid_r10_e5_seed123` | FedAvg | IID | 123 | 5 | 10 | 5 | - | -19.3502 | -19.3502 |
| `cnn_fedavg_iid_r20_e1_seed123` | FedAvg | IID | 123 | 5 | 20 | 1 | - | -18.2705 | -18.2705 |
| `cnn_fedavg_iid_r20_e2_seed123` | FedAvg | IID | 123 | 5 | 20 | 2 | - | -19.0142 | -19.0142 |
| `cnn_fedavg_iid_r20_e3_seed123` | FedAvg | IID | 123 | 5 | 20 | 3 | - | -19.3427 | -19.3427 |
| `cnn_fedavg_iid_r20_e5_seed123` | FedAvg | IID | 123 | 5 | 20 | 5 | - | -19.6824 | -19.6824 |
| `cnn_fedavg_iid_r30_e1_seed123` | FedAvg | IID | 123 | 5 | 30 | 1 | - | -18.6172 | -18.6172 |
| `cnn_fedavg_iid_r30_e2_seed123` | FedAvg | IID | 123 | 5 | 30 | 2 | - | -19.2772 | -19.2772 |
| `cnn_fedavg_iid_r30_e3_seed123` | FedAvg | IID | 123 | 5 | 30 | 3 | - | -19.5589 | -19.5589 |
| `cnn_fedavg_iid_r30_e5_seed123` | FedAvg | IID | 123 | 5 | 30 | 5 | - | -19.8585 | -19.8585 |
| `cnn_fedavg_iid_r50_e1_seed123` | FedAvg | IID | 123 | 5 | 50 | 1 | - | -18.9609 | -18.9609 |
| `cnn_fedavg_iid_r50_e2_seed123` | FedAvg | IID | 123 | 5 | 50 | 2 | - | -19.5032 | -19.5032 |
| `cnn_fedavg_iid_r50_e3_seed123` | FedAvg | IID | 123 | 5 | 50 | 3 | - | -19.7657 | -19.7657 |
| `cnn_fedavg_iid_r50_e5_seed123` | FedAvg | IID | 123 | 5 | 50 | 5 | - | -20.0381 | -20.0381 |
| `cnn_fedavg_iid_r10_e1_seed777` | FedAvg | IID | 777 | 5 | 10 | 1 | - | -17.6616 | -17.6616 |
| `cnn_fedavg_iid_r10_e2_seed777` | FedAvg | IID | 777 | 5 | 10 | 2 | - | -18.3297 | -18.3297 |
| `cnn_fedavg_iid_r10_e3_seed777` | FedAvg | IID | 777 | 5 | 10 | 3 | - | -18.7218 | -18.7218 |
| `cnn_fedavg_iid_r10_e5_seed777` | FedAvg | IID | 777 | 5 | 10 | 5 | - | -19.2376 | -19.2376 |
| `cnn_fedavg_iid_r20_e1_seed777` | FedAvg | IID | 777 | 5 | 20 | 1 | - | -18.2835 | -18.2835 |
| `cnn_fedavg_iid_r20_e2_seed777` | FedAvg | IID | 777 | 5 | 20 | 2 | - | -18.8711 | -18.8711 |
| `cnn_fedavg_iid_r20_e3_seed777` | FedAvg | IID | 777 | 5 | 20 | 3 | - | -19.2739 | -19.2739 |
| `cnn_fedavg_iid_r20_e5_seed777` | FedAvg | IID | 777 | 5 | 20 | 5 | - | -19.6500 | -19.6500 |
| `cnn_fedavg_iid_r30_e1_seed777` | FedAvg | IID | 777 | 5 | 30 | 1 | - | -18.5655 | -18.5655 |
| `cnn_fedavg_iid_r30_e2_seed777` | FedAvg | IID | 777 | 5 | 30 | 2 | - | -19.1581 | -19.1581 |
| `cnn_fedavg_iid_r30_e3_seed777` | FedAvg | IID | 777 | 5 | 30 | 3 | - | -19.4929 | -19.4929 |
| `cnn_fedavg_iid_r30_e5_seed777` | FedAvg | IID | 777 | 5 | 30 | 5 | - | -19.8090 | -19.8090 |
| `cnn_fedavg_iid_r50_e1_seed777` | FedAvg | IID | 777 | 5 | 50 | 1 | - | -18.9116 | -18.9116 |
| `cnn_fedavg_iid_r50_e2_seed777` | FedAvg | IID | 777 | 5 | 50 | 2 | - | -19.4238 | -19.4238 |
| `cnn_fedavg_iid_r50_e3_seed777` | FedAvg | IID | 777 | 5 | 50 | 3 | - | -19.6945 | -19.6945 |
| `cnn_fedavg_iid_r50_e5_seed777` | FedAvg | IID | 777 | 5 | 50 | 5 | - | -20.0022 | -20.0022 |
| `cnn_fedavg_non-iid_r10_e1_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 10 | 1 | - | -17.5007 | -17.5007 |
| `cnn_fedavg_non-iid_r10_e2_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 10 | 2 | - | -18.1887 | -18.1887 |
| `cnn_fedavg_non-iid_r10_e3_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 10 | 3 | - | -18.5704 | -18.5704 |
| `cnn_fedavg_non-iid_r10_e5_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 10 | 5 | - | -19.0116 | -19.0116 |
| `cnn_fedavg_non-iid_r20_e1_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 20 | 1 | - | -18.1116 | -18.1116 |
| `cnn_fedavg_non-iid_r20_e2_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 20 | 2 | - | -18.8611 | -18.8611 |
| `cnn_fedavg_non-iid_r20_e3_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 20 | 3 | - | -19.0361 | -19.0361 |
| `cnn_fedavg_non-iid_r20_e5_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 20 | 5 | - | -19.3305 | -19.3305 |
| `cnn_fedavg_non-iid_r30_e1_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 30 | 1 | - | -18.4165 | -18.4165 |
| `cnn_fedavg_non-iid_r30_e2_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 30 | 2 | - | -19.0991 | -19.0991 |
| `cnn_fedavg_non-iid_r30_e3_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 30 | 3 | - | -19.3013 | -19.3013 |
| `cnn_fedavg_non-iid_r30_e5_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 30 | 5 | - | -19.5004 | -19.5004 |
| `cnn_fedavg_non-iid_r50_e1_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 50 | 1 | - | -18.7872 | -18.7872 |
| `cnn_fedavg_non-iid_r50_e2_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 50 | 2 | - | -19.2952 | -19.2952 |
| `cnn_fedavg_non-iid_r50_e3_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 50 | 3 | - | -19.4897 | -19.4897 |
| `cnn_fedavg_non-iid_r50_e5_mu0.01_seed42` | FedAvg | Non-IID | 42 | 5 | 50 | 5 | - | -19.7338 | -19.7338 |
| `cnn_fedavg_non-iid_r10_e1_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 10 | 1 | - | -17.4353 | -17.4353 |
| `cnn_fedavg_non-iid_r10_e2_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 10 | 2 | - | -18.1455 | -18.1455 |
| `cnn_fedavg_non-iid_r10_e3_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 10 | 3 | - | -18.6160 | -18.6160 |
| `cnn_fedavg_non-iid_r10_e5_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 10 | 5 | - | -18.8584 | -18.8584 |
| `cnn_fedavg_non-iid_r20_e1_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 20 | 1 | - | -18.1244 | -18.1244 |
| `cnn_fedavg_non-iid_r20_e2_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 20 | 2 | - | -18.6875 | -18.6875 |
| `cnn_fedavg_non-iid_r20_e3_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 20 | 3 | - | -19.0420 | -19.0420 |
| `cnn_fedavg_non-iid_r20_e5_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 20 | 5 | - | -19.1940 | -19.1940 |
| `cnn_fedavg_non-iid_r30_e1_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 30 | 1 | - | -18.4044 | -18.4044 |
| `cnn_fedavg_non-iid_r30_e2_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 30 | 2 | - | -18.9716 | -18.9716 |
| `cnn_fedavg_non-iid_r30_e3_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 30 | 3 | - | -19.1891 | -19.1891 |
| `cnn_fedavg_non-iid_r30_e5_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 30 | 5 | - | -19.3826 | -19.3826 |
| `cnn_fedavg_non-iid_r50_e1_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 50 | 1 | - | -18.8255 | -18.8255 |
| `cnn_fedavg_non-iid_r50_e2_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 50 | 2 | - | -19.2642 | -19.2642 |
| `cnn_fedavg_non-iid_r50_e3_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 50 | 3 | - | -19.3929 | -19.3929 |
| `cnn_fedavg_non-iid_r50_e5_mu0.01_seed123` | FedAvg | Non-IID | 123 | 5 | 50 | 5 | - | -19.6339 | -19.6339 |
| `cnn_fedavg_non-iid_r10_e1_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 10 | 1 | - | -17.3778 | -17.3778 |
| `cnn_fedavg_non-iid_r10_e2_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 10 | 2 | - | -18.1480 | -18.1480 |
| `cnn_fedavg_non-iid_r10_e3_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 10 | 3 | - | -18.4891 | -18.4891 |
| `cnn_fedavg_non-iid_r10_e5_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 10 | 5 | - | -18.8664 | -18.8664 |
| `cnn_fedavg_non-iid_r20_e1_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 20 | 1 | - | -18.1407 | -18.1407 |
| `cnn_fedavg_non-iid_r20_e2_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 20 | 2 | - | -18.7550 | -18.7550 |
| `cnn_fedavg_non-iid_r20_e3_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 20 | 3 | - | -19.0533 | -19.0533 |
| `cnn_fedavg_non-iid_r20_e5_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 20 | 5 | - | -19.2181 | -19.2181 |
| `cnn_fedavg_non-iid_r30_e1_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 30 | 1 | - | -18.4045 | -18.4045 |
| `cnn_fedavg_non-iid_r30_e2_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 30 | 2 | - | -19.0076 | -19.0076 |
| `cnn_fedavg_non-iid_r30_e3_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 30 | 3 | - | -19.2584 | -19.2584 |
| `cnn_fedavg_non-iid_r30_e5_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 30 | 5 | - | -19.4545 | -19.4545 |
| `cnn_fedavg_non-iid_r50_e1_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 50 | 1 | - | -18.7763 | -18.7763 |
| `cnn_fedavg_non-iid_r50_e2_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 50 | 2 | - | -19.2909 | -19.2909 |
| `cnn_fedavg_non-iid_r50_e3_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 50 | 3 | - | -19.5105 | -19.5105 |
| `cnn_fedavg_non-iid_r50_e5_mu0.01_seed777` | FedAvg | Non-IID | 777 | 5 | 50 | 5 | - | -19.5678 | -19.5678 |
| `cnn_fedprox_non-iid_r10_e1_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 10 | 1 | 0.001 | -17.4548 | -17.4548 |
| `cnn_fedprox_non-iid_r10_e1_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 10 | 1 | 0.005 | -17.3574 | -17.3574 |
| `cnn_fedprox_non-iid_r10_e1_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 10 | 1 | 0.01 | -17.2936 | -17.2936 |
| `cnn_fedprox_non-iid_r10_e2_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 10 | 2 | 0.001 | -18.1334 | -18.1334 |
| `cnn_fedprox_non-iid_r10_e2_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 10 | 2 | 0.005 | -17.9331 | -17.9331 |
| `cnn_fedprox_non-iid_r10_e2_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 10 | 2 | 0.01 | -17.7479 | -17.7479 |
| `cnn_fedprox_non-iid_r10_e3_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 10 | 3 | 0.001 | -18.4852 | -18.4852 |
| `cnn_fedprox_non-iid_r10_e3_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 10 | 3 | 0.005 | -18.1681 | -18.1681 |
| `cnn_fedprox_non-iid_r10_e3_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 10 | 3 | 0.01 | -17.9221 | -17.9221 |
| `cnn_fedprox_non-iid_r10_e5_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 10 | 5 | 0.001 | -18.8035 | -18.8035 |
| `cnn_fedprox_non-iid_r10_e5_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 10 | 5 | 0.005 | -18.3370 | -18.3370 |
| `cnn_fedprox_non-iid_r10_e5_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 10 | 5 | 0.01 | -18.0487 | -18.0487 |
| `cnn_fedprox_non-iid_r20_e1_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 20 | 1 | 0.001 | -18.1057 | -18.1057 |
| `cnn_fedprox_non-iid_r20_e1_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 20 | 1 | 0.005 | -18.0283 | -18.0283 |
| `cnn_fedprox_non-iid_r20_e1_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 20 | 1 | 0.01 | -17.9236 | -17.9236 |
| `cnn_fedprox_non-iid_r20_e2_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 20 | 2 | 0.001 | -18.8141 | -18.8141 |
| `cnn_fedprox_non-iid_r20_e2_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 20 | 2 | 0.005 | -18.5642 | -18.5642 |
| `cnn_fedprox_non-iid_r20_e2_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 20 | 2 | 0.01 | -18.3216 | -18.3216 |
| `cnn_fedprox_non-iid_r20_e3_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 20 | 3 | 0.001 | -19.0238 | -19.0238 |
| `cnn_fedprox_non-iid_r20_e3_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 20 | 3 | 0.005 | -18.7492 | -18.7492 |
| `cnn_fedprox_non-iid_r20_e3_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 20 | 3 | 0.01 | -18.4286 | -18.4286 |
| `cnn_fedprox_non-iid_r20_e5_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 20 | 5 | 0.001 | -19.1712 | -19.1712 |
| `cnn_fedprox_non-iid_r20_e5_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 20 | 5 | 0.005 | -18.7598 | -18.7598 |
| `cnn_fedprox_non-iid_r20_e5_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 20 | 5 | 0.01 | -18.4883 | -18.4883 |
| `cnn_fedprox_non-iid_r30_e1_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 30 | 1 | 0.001 | -18.4180 | -18.4180 |
| `cnn_fedprox_non-iid_r30_e1_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 30 | 1 | 0.005 | -18.2805 | -18.2805 |
| `cnn_fedprox_non-iid_r30_e1_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 30 | 1 | 0.01 | -18.1818 | -18.1818 |
| `cnn_fedprox_non-iid_r30_e2_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 30 | 2 | 0.001 | -19.0582 | -19.0582 |
| `cnn_fedprox_non-iid_r30_e2_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 30 | 2 | 0.005 | -18.8669 | -18.8669 |
| `cnn_fedprox_non-iid_r30_e2_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 30 | 2 | 0.01 | -18.6650 | -18.6650 |
| `cnn_fedprox_non-iid_r30_e3_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 30 | 3 | 0.001 | -19.2498 | -19.2498 |
| `cnn_fedprox_non-iid_r30_e3_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 30 | 3 | 0.005 | -18.9763 | -18.9763 |
| `cnn_fedprox_non-iid_r30_e3_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 30 | 3 | 0.01 | -18.7247 | -18.7247 |
| `cnn_fedprox_non-iid_r30_e5_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 30 | 5 | 0.001 | -19.3488 | -19.3488 |
| `cnn_fedprox_non-iid_r30_e5_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 30 | 5 | 0.005 | -19.0342 | -19.0342 |
| `cnn_fedprox_non-iid_r30_e5_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 30 | 5 | 0.01 | -18.8115 | -18.8115 |
| `cnn_fedprox_non-iid_r50_e1_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 50 | 1 | 0.001 | -18.7641 | -18.7641 |
| `cnn_fedprox_non-iid_r50_e1_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 50 | 1 | 0.005 | -18.6609 | -18.6609 |
| `cnn_fedprox_non-iid_r50_e1_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 50 | 1 | 0.01 | -18.5513 | -18.5513 |
| `cnn_fedprox_non-iid_r50_e2_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 50 | 2 | 0.001 | -19.2763 | -19.2763 |
| `cnn_fedprox_non-iid_r50_e2_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 50 | 2 | 0.005 | -19.0888 | -19.0888 |
| `cnn_fedprox_non-iid_r50_e2_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 50 | 2 | 0.01 | -18.9065 | -18.9065 |
| `cnn_fedprox_non-iid_r50_e3_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 50 | 3 | 0.001 | -19.4109 | -19.4109 |
| `cnn_fedprox_non-iid_r50_e3_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 50 | 3 | 0.005 | -19.2277 | -19.2277 |
| `cnn_fedprox_non-iid_r50_e3_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 50 | 3 | 0.01 | -18.9646 | -18.9646 |
| `cnn_fedprox_non-iid_r50_e5_mu0.001_seed42` | FedProx | Non-IID | 42 | 5 | 50 | 5 | 0.001 | -19.5797 | -19.5797 |
| `cnn_fedprox_non-iid_r50_e5_mu0.005_seed42` | FedProx | Non-IID | 42 | 5 | 50 | 5 | 0.005 | -19.2424 | -19.2424 |
| `cnn_fedprox_non-iid_r50_e5_mu0.01_seed42` | FedProx | Non-IID | 42 | 5 | 50 | 5 | 0.01 | -19.0556 | -19.0556 |
| `cnn_fedprox_non-iid_r10_e1_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 10 | 1 | 0.001 | -17.3671 | -17.3671 |
| `cnn_fedprox_non-iid_r10_e1_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 10 | 1 | 0.005 | -17.3215 | -17.3215 |
| `cnn_fedprox_non-iid_r10_e1_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 10 | 1 | 0.01 | -17.1904 | -17.1904 |
| `cnn_fedprox_non-iid_r10_e2_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 10 | 2 | 0.001 | -18.1046 | -18.1046 |
| `cnn_fedprox_non-iid_r10_e2_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 10 | 2 | 0.005 | -17.9570 | -17.9570 |
| `cnn_fedprox_non-iid_r10_e2_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 10 | 2 | 0.01 | -17.7645 | -17.7645 |
| `cnn_fedprox_non-iid_r10_e3_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 10 | 3 | 0.001 | -18.5031 | -18.5031 |
| `cnn_fedprox_non-iid_r10_e3_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 10 | 3 | 0.005 | -18.2336 | -18.2336 |
| `cnn_fedprox_non-iid_r10_e3_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 10 | 3 | 0.01 | -18.0028 | -18.0028 |
| `cnn_fedprox_non-iid_r10_e5_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 10 | 5 | 0.001 | -18.7579 | -18.7579 |
| `cnn_fedprox_non-iid_r10_e5_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 10 | 5 | 0.005 | -18.3902 | -18.3902 |
| `cnn_fedprox_non-iid_r10_e5_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 10 | 5 | 0.01 | -18.1024 | -18.1024 |
| `cnn_fedprox_non-iid_r20_e1_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 20 | 1 | 0.001 | -18.0936 | -18.0936 |
| `cnn_fedprox_non-iid_r20_e1_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 20 | 1 | 0.005 | -18.0110 | -18.0110 |
| `cnn_fedprox_non-iid_r20_e1_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 20 | 1 | 0.01 | -17.8330 | -17.8330 |
| `cnn_fedprox_non-iid_r20_e2_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 20 | 2 | 0.001 | -18.6427 | -18.6427 |
| `cnn_fedprox_non-iid_r20_e2_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 20 | 2 | 0.005 | -18.4551 | -18.4551 |
| `cnn_fedprox_non-iid_r20_e2_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 20 | 2 | 0.01 | -18.2398 | -18.2398 |
| `cnn_fedprox_non-iid_r20_e3_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 20 | 3 | 0.001 | -18.9832 | -18.9832 |
| `cnn_fedprox_non-iid_r20_e3_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 20 | 3 | 0.005 | -18.6802 | -18.6802 |
| `cnn_fedprox_non-iid_r20_e3_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 20 | 3 | 0.01 | -18.4348 | -18.4348 |
| `cnn_fedprox_non-iid_r20_e5_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 20 | 5 | 0.001 | -19.1116 | -19.1116 |
| `cnn_fedprox_non-iid_r20_e5_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 20 | 5 | 0.005 | -18.6751 | -18.6751 |
| `cnn_fedprox_non-iid_r20_e5_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 20 | 5 | 0.01 | -18.4734 | -18.4734 |
| `cnn_fedprox_non-iid_r30_e1_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 30 | 1 | 0.001 | -18.4232 | -18.4232 |
| `cnn_fedprox_non-iid_r30_e1_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 30 | 1 | 0.005 | -18.3079 | -18.3079 |
| `cnn_fedprox_non-iid_r30_e1_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 30 | 1 | 0.01 | -18.1971 | -18.1971 |
| `cnn_fedprox_non-iid_r30_e2_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 30 | 2 | 0.001 | -18.9036 | -18.9036 |
| `cnn_fedprox_non-iid_r30_e2_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 30 | 2 | 0.005 | -18.7132 | -18.7132 |
| `cnn_fedprox_non-iid_r30_e2_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 30 | 2 | 0.01 | -18.5154 | -18.5154 |
| `cnn_fedprox_non-iid_r30_e3_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 30 | 3 | 0.001 | -19.1336 | -19.1336 |
| `cnn_fedprox_non-iid_r30_e3_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 30 | 3 | 0.005 | -18.8935 | -18.8935 |
| `cnn_fedprox_non-iid_r30_e3_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 30 | 3 | 0.01 | -18.6988 | -18.6988 |
| `cnn_fedprox_non-iid_r30_e5_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 30 | 5 | 0.001 | -19.3432 | -19.3432 |
| `cnn_fedprox_non-iid_r30_e5_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 30 | 5 | 0.005 | -18.9753 | -18.9753 |
| `cnn_fedprox_non-iid_r30_e5_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 30 | 5 | 0.01 | -18.7538 | -18.7538 |
| `cnn_fedprox_non-iid_r50_e1_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 50 | 1 | 0.001 | -18.8187 | -18.8187 |
| `cnn_fedprox_non-iid_r50_e1_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 50 | 1 | 0.005 | -18.7023 | -18.7023 |
| `cnn_fedprox_non-iid_r50_e1_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 50 | 1 | 0.01 | -18.5808 | -18.5808 |
| `cnn_fedprox_non-iid_r50_e2_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 50 | 2 | 0.001 | -19.1860 | -19.1860 |
| `cnn_fedprox_non-iid_r50_e2_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 50 | 2 | 0.005 | -19.0279 | -19.0279 |
| `cnn_fedprox_non-iid_r50_e2_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 50 | 2 | 0.01 | -18.8278 | -18.8278 |
| `cnn_fedprox_non-iid_r50_e3_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 50 | 3 | 0.001 | -19.3710 | -19.3710 |
| `cnn_fedprox_non-iid_r50_e3_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 50 | 3 | 0.005 | -19.1679 | -19.1679 |
| `cnn_fedprox_non-iid_r50_e3_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 50 | 3 | 0.01 | -18.9650 | -18.9650 |
| `cnn_fedprox_non-iid_r50_e5_mu0.001_seed123` | FedProx | Non-IID | 123 | 5 | 50 | 5 | 0.001 | -19.6479 | -19.6479 |
| `cnn_fedprox_non-iid_r50_e5_mu0.005_seed123` | FedProx | Non-IID | 123 | 5 | 50 | 5 | 0.005 | -19.2843 | -19.2843 |
| `cnn_fedprox_non-iid_r50_e5_mu0.01_seed123` | FedProx | Non-IID | 123 | 5 | 50 | 5 | 0.01 | -19.0835 | -19.0835 |
| `cnn_fedprox_non-iid_r10_e1_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 10 | 1 | 0.001 | -17.2981 | -17.2981 |
| `cnn_fedprox_non-iid_r10_e1_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 10 | 1 | 0.005 | -17.2284 | -17.2284 |
| `cnn_fedprox_non-iid_r10_e1_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 10 | 1 | 0.01 | -17.1230 | -17.1230 |
| `cnn_fedprox_non-iid_r10_e2_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 10 | 2 | 0.001 | -18.1009 | -18.1009 |
| `cnn_fedprox_non-iid_r10_e2_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 10 | 2 | 0.005 | -17.8910 | -17.8910 |
| `cnn_fedprox_non-iid_r10_e2_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 10 | 2 | 0.01 | -17.6933 | -17.6933 |
| `cnn_fedprox_non-iid_r10_e3_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 10 | 3 | 0.001 | -18.4051 | -18.4051 |
| `cnn_fedprox_non-iid_r10_e3_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 10 | 3 | 0.005 | -18.0843 | -18.0843 |
| `cnn_fedprox_non-iid_r10_e3_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 10 | 3 | 0.01 | -17.8730 | -17.8730 |
| `cnn_fedprox_non-iid_r10_e5_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 10 | 5 | 0.001 | -18.7414 | -18.7414 |
| `cnn_fedprox_non-iid_r10_e5_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 10 | 5 | 0.005 | -18.3606 | -18.3606 |
| `cnn_fedprox_non-iid_r10_e5_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 10 | 5 | 0.01 | -18.0812 | -18.0812 |
| `cnn_fedprox_non-iid_r20_e1_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 20 | 1 | 0.001 | -18.0989 | -18.0989 |
| `cnn_fedprox_non-iid_r20_e1_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 20 | 1 | 0.005 | -17.9487 | -17.9487 |
| `cnn_fedprox_non-iid_r20_e1_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 20 | 1 | 0.01 | -17.8696 | -17.8696 |
| `cnn_fedprox_non-iid_r20_e2_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 20 | 2 | 0.001 | -18.7150 | -18.7150 |
| `cnn_fedprox_non-iid_r20_e2_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 20 | 2 | 0.005 | -18.4888 | -18.4888 |
| `cnn_fedprox_non-iid_r20_e2_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 20 | 2 | 0.01 | -18.3033 | -18.3033 |
| `cnn_fedprox_non-iid_r20_e3_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 20 | 3 | 0.001 | -18.9606 | -18.9606 |
| `cnn_fedprox_non-iid_r20_e3_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 20 | 3 | 0.005 | -18.6785 | -18.6785 |
| `cnn_fedprox_non-iid_r20_e3_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 20 | 3 | 0.01 | -18.4531 | -18.4531 |
| `cnn_fedprox_non-iid_r20_e5_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 20 | 5 | 0.001 | -19.1020 | -19.1020 |
| `cnn_fedprox_non-iid_r20_e5_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 20 | 5 | 0.005 | -18.7675 | -18.7675 |
| `cnn_fedprox_non-iid_r20_e5_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 20 | 5 | 0.01 | -18.5571 | -18.5571 |
| `cnn_fedprox_non-iid_r30_e1_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 30 | 1 | 0.001 | -18.4105 | -18.4105 |
| `cnn_fedprox_non-iid_r30_e1_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 30 | 1 | 0.005 | -18.3309 | -18.3309 |
| `cnn_fedprox_non-iid_r30_e1_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 30 | 1 | 0.01 | -18.1785 | -18.1785 |
| `cnn_fedprox_non-iid_r30_e2_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 30 | 2 | 0.001 | -18.9484 | -18.9484 |
| `cnn_fedprox_non-iid_r30_e2_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 30 | 2 | 0.005 | -18.7558 | -18.7558 |
| `cnn_fedprox_non-iid_r30_e2_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 30 | 2 | 0.01 | -18.5974 | -18.5974 |
| `cnn_fedprox_non-iid_r30_e3_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 30 | 3 | 0.001 | -19.1919 | -19.1919 |
| `cnn_fedprox_non-iid_r30_e3_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 30 | 3 | 0.005 | -18.9435 | -18.9435 |
| `cnn_fedprox_non-iid_r30_e3_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 30 | 3 | 0.01 | -18.7460 | -18.7460 |
| `cnn_fedprox_non-iid_r30_e5_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 30 | 5 | 0.001 | -19.3257 | -19.3257 |
| `cnn_fedprox_non-iid_r30_e5_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 30 | 5 | 0.005 | -18.9975 | -18.9975 |
| `cnn_fedprox_non-iid_r30_e5_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 30 | 5 | 0.01 | -18.8094 | -18.8094 |
| `cnn_fedprox_non-iid_r50_e1_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 50 | 1 | 0.001 | -18.7636 | -18.7636 |
| `cnn_fedprox_non-iid_r50_e1_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 50 | 1 | 0.005 | -18.6519 | -18.6519 |
| `cnn_fedprox_non-iid_r50_e1_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 50 | 1 | 0.01 | -18.5481 | -18.5481 |
| `cnn_fedprox_non-iid_r50_e2_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 50 | 2 | 0.001 | -19.2482 | -19.2482 |
| `cnn_fedprox_non-iid_r50_e2_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 50 | 2 | 0.005 | -19.0541 | -19.0541 |
| `cnn_fedprox_non-iid_r50_e2_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 50 | 2 | 0.01 | -18.8944 | -18.8944 |
| `cnn_fedprox_non-iid_r50_e3_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 50 | 3 | 0.001 | -19.4931 | -19.4931 |
| `cnn_fedprox_non-iid_r50_e3_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 50 | 3 | 0.005 | -19.1972 | -19.1972 |
| `cnn_fedprox_non-iid_r50_e3_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 50 | 3 | 0.01 | -19.0055 | -19.0055 |
| `cnn_fedprox_non-iid_r50_e5_mu0.001_seed777` | FedProx | Non-IID | 777 | 5 | 50 | 5 | 0.001 | -19.4652 | -19.4652 |
| `cnn_fedprox_non-iid_r50_e5_mu0.005_seed777` | FedProx | Non-IID | 777 | 5 | 50 | 5 | 0.005 | -19.1767 | -19.1767 |
| `cnn_fedprox_non-iid_r50_e5_mu0.01_seed777` | FedProx | Non-IID | 777 | 5 | 50 | 5 | 0.01 | -19.0214 | -19.0214 |
