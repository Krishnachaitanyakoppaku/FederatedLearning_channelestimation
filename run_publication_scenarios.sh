#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Publication scenario runner
# ------------------------------------------------------------
# Scenarios covered:
# 1) Complete ResUNet for one SNR (centralized + federated scripts)
# 2) Complete CNN for remaining SNRs (centralized + federated matrix)
# 3) ResUNet for remaining SNRs (centralized)
#
# Usage:
#   bash run_publication_scenarios.sh all
#   bash run_publication_scenarios.sh resunet_one
#   bash run_publication_scenarios.sh cnn_remaining
#   bash run_publication_scenarios.sh resunet_remaining
#
# Optional env overrides:
#   PROJECT=/abs/path/to/Project
#   SNR_ONE_RESUNET=10
#   REMAINING_SNRS="0 5 15 20"
# ------------------------------------------------------------

PROJECT="${PROJECT:-/Users/chaitu/Desktop/Wireless paper/Project}"
PY="$PROJECT/myenv/bin/python"
CFG="$PROJECT/experiment_config.json"
CKPT="$PROJECT/results/checkpoints"

SEEDS=(42 123 777)
ROUNDS=(10 20 30 50)
LOCAL_EPOCHS=(1 2 3 5)
MUS=(0.001 0.005 0.01)

SNR_ONE_RESUNET="${SNR_ONE_RESUNET:-10}"
read -r -a REMAINING_SNRS <<< "${REMAINING_SNRS:-0 5 15 20}"

MODE="${1:-all}"

echo "[INFO] Project: $PROJECT"
echo "[INFO] Mode: $MODE"

run_resunet_one_snr_complete() {
  local snr="$1"
  echo "[SCENARIO 1] ResUNet complete for one SNR=${snr}"

  # 1A) Centralized ResUNet (3 seeds)
  for seed in "${SEEDS[@]}"; do
    "$PY" "$PROJECT/Centralised Learning/ResUNet/resunet_model.py" \
      --snr "$snr" \
      --epochs 40 \
      --batch 32 \
      --lr 0.0005 \
      --seed "$seed" \
      --run_id "resunet_centralized_snr${snr}_seed${seed}" \
      --config_path "$CFG"
  done

  # 1B) Federated ResUNet scripts in this repo are hardcoded to 10 dB dataset files.
  # Run federated ResUNet only when SNR=10.
  if [[ "$snr" != "10" ]]; then
    echo "[WARN] Skipping federated ResUNet for SNR=$snr (current scripts are 10 dB fixed)."
    return
  fi

  # FedAvg (IID + Non-IID)
  for seed in "${SEEDS[@]}"; do
    for r in "${ROUNDS[@]}"; do
      for e in "${LOCAL_EPOCHS[@]}"; do
        "$PY" "$PROJECT/Federated Learning/ResUNet/fed_resunet.py" \
          --rounds "$r" \
          --clients 5 \
          --local_epochs "$e" \
          --batch 16 \
          --lr 0.0005 \
          --snr "$snr" \
          --seed "$seed" \
          --run_id "resunet_fedavg_iid_snr${snr}_r${r}_e${e}_seed${seed}" \
          --config_path "$CFG" \
          --checkpoint_dir "$CKPT" \
          --save "$PROJECT/Federated Learning/ResUNet/fed_resunet_iid_snr${snr}_r${r}_e${e}_seed${seed}.pth" \
          --resume

        "$PY" "$PROJECT/Federated Learning/ResUNet/fed_resunet.py" \
          --rounds "$r" \
          --clients 5 \
          --local_epochs "$e" \
          --batch 16 \
          --lr 0.0005 \
          --snr "$snr" \
          --seed "$seed" \
          --non_iid \
          --run_id "resunet_fedavg_non-iid_snr${snr}_r${r}_e${e}_seed${seed}" \
          --config_path "$CFG" \
          --checkpoint_dir "$CKPT" \
          --save "$PROJECT/Federated Learning/ResUNet/fed_resunet_non-iid_snr${snr}_r${r}_e${e}_seed${seed}.pth" \
          --resume
      done
    done
  done

  # FedProx (IID + Non-IID)
  for seed in "${SEEDS[@]}"; do
    for r in "${ROUNDS[@]}"; do
      for e in "${LOCAL_EPOCHS[@]}"; do
        for mu in "${MUS[@]}"; do
          "$PY" "$PROJECT/Federated Learning/resunet_fedprox/fedprox_resunet.py" \
            --rounds "$r" \
            --clients 5 \
            --local_epochs "$e" \
            --batch 16 \
            --lr 0.0005 \
            --mu "$mu" \
            --snr "$snr" \
            --seed "$seed" \
            --run_id "resunet_fedprox_iid_snr${snr}_r${r}_e${e}_mu${mu}_seed${seed}" \
            --config_path "$CFG" \
            --checkpoint_dir "$CKPT" \
            --save "$PROJECT/Federated Learning/resunet_fedprox/fedprox_resunet_iid_snr${snr}_r${r}_e${e}_mu${mu}_seed${seed}.pth" \
            --resume

          "$PY" "$PROJECT/Federated Learning/resunet_fedprox/fedprox_resunet.py" \
            --rounds "$r" \
            --clients 5 \
            --local_epochs "$e" \
            --batch 16 \
            --lr 0.0005 \
            --mu "$mu" \
            --snr "$snr" \
            --seed "$seed" \
            --non_iid \
            --run_id "resunet_fedprox_non-iid_snr${snr}_r${r}_e${e}_mu${mu}_seed${seed}" \
            --config_path "$CFG" \
            --checkpoint_dir "$CKPT" \
            --save "$PROJECT/Federated Learning/resunet_fedprox/fedprox_resunet_non-iid_snr${snr}_r${r}_e${e}_mu${mu}_seed${seed}.pth" \
            --resume
        done
      done
    done
  done
}

run_cnn_for_snr() {
  local snr="$1"
  echo "[SCENARIO 2] CNN full run for SNR=${snr}"

  local x_train="$PROJECT/dataset/cnn/${snr}dB/X_train.npy"
  local y_train="$PROJECT/dataset/cnn/${snr}dB/Y_train.npy"
  local x_test="$PROJECT/dataset/cnn/${snr}dB/X_test.npy"
  local y_test="$PROJECT/dataset/cnn/${snr}dB/Y_test.npy"

  local iid_clients_dir="$PROJECT/Federated Learning/CNN_Fedavg/clients_snr${snr}"
  local noniid_clients_dir="$PROJECT/Federated Learning/CNN_FedProx/clients_non_iid_snr${snr}"

  # 2A) Centralized CNN (3 seeds)
  for seed in "${SEEDS[@]}"; do
    "$PY" "$PROJECT/Centralised Learning/CNN/train_and_evaluate.py" \
      --snr "$snr" \
      --epochs 20 \
      --batch 32 \
      --lr 0.001 \
      --seed "$seed" \
      --run_id "cnn_centralized_snr${snr}_seed${seed}" \
      --config_path "$CFG"
  done

  # 2B) Build federated client splits for this SNR
  "$PY" "$PROJECT/Federated Learning/CNN_Fedavg/split_clients.py" \
    --x_path "$x_train" \
    --y_path "$y_train" \
    --clients 5 \
    --output_dir "$iid_clients_dir"

  "$PY" "$PROJECT/Federated Learning/CNN_FedProx/non_iid_split.py" \
    --snr "$snr" \
    --clients 5 \
    --samples_per_client 960 \
    --output_dir "$noniid_clients_dir"

  # 2C) Federated matrix (FedAvg IID, FedAvg Non-IID, FedProx Non-IID)
  for seed in "${SEEDS[@]}"; do
    for r in "${ROUNDS[@]}"; do
      for e in "${LOCAL_EPOCHS[@]}"; do
        "$PY" "$PROJECT/Federated Learning/CNN_Fedavg/federated_train.py" \
          --clients 5 \
          --rounds "$r" \
          --local_epochs "$e" \
          --batch 32 \
          --lr 0.001 \
          --seed "$seed" \
          --snr "$snr" \
          --x_test "$x_test" \
          --y_test "$y_test" \
          --data_mode IID \
          --client_dir "$iid_clients_dir" \
          --run_id "cnn_fedavg_iid_snr${snr}_r${r}_e${e}_seed${seed}" \
          --config_path "$CFG" \
          --checkpoint_dir "$CKPT" \
          --resume

        "$PY" "$PROJECT/Federated Learning/CNN_FedProx/federated_train.py" \
          --rounds "$r" \
          --local_epochs "$e" \
          --batch 32 \
          --lr 0.001 \
          --seed "$seed" \
          --snr "$snr" \
          --x_test "$x_test" \
          --y_test "$y_test" \
          --data_mode Non-IID \
          --client_dir "$noniid_clients_dir" \
          --run_id "cnn_fedavg_non-iid_snr${snr}_r${r}_e${e}_mu0.01_seed${seed}" \
          --config_path "$CFG" \
          --checkpoint_dir "$CKPT" \
          --resume

        for mu in "${MUS[@]}"; do
          "$PY" "$PROJECT/Federated Learning/CNN_FedProx/federated_train.py" \
            --fedprox \
            --mu "$mu" \
            --rounds "$r" \
            --local_epochs "$e" \
            --batch 32 \
            --lr 0.001 \
            --seed "$seed" \
            --snr "$snr" \
            --x_test "$x_test" \
            --y_test "$y_test" \
            --data_mode Non-IID \
            --client_dir "$noniid_clients_dir" \
            --run_id "cnn_fedprox_non-iid_snr${snr}_r${r}_e${e}_mu${mu}_seed${seed}" \
            --config_path "$CFG" \
            --checkpoint_dir "$CKPT" \
            --resume
        done
      done
    done
  done
}

run_resunet_remaining_snrs() {
  echo "[SCENARIO 3] ResUNet for remaining SNRs (centralized)"
  for snr in "${REMAINING_SNRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      "$PY" "$PROJECT/Centralised Learning/ResUNet/resunet_model.py" \
        --snr "$snr" \
        --epochs 40 \
        --batch 32 \
        --lr 0.0005 \
        --seed "$seed" \
        --run_id "resunet_centralized_snr${snr}_seed${seed}" \
        --config_path "$CFG"
    done
  done
}

case "$MODE" in
  all)
    run_resunet_one_snr_complete "$SNR_ONE_RESUNET"
    for snr in "${REMAINING_SNRS[@]}"; do
      run_cnn_for_snr "$snr"
    done
    run_resunet_remaining_snrs
    ;;
  resunet_one)
    run_resunet_one_snr_complete "$SNR_ONE_RESUNET"
    ;;
  cnn_remaining)
    for snr in "${REMAINING_SNRS[@]}"; do
      run_cnn_for_snr "$snr"
    done
    ;;
  resunet_remaining)
    run_resunet_remaining_snrs
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Use one of: all | resunet_one | cnn_remaining | resunet_remaining"
    exit 1
    ;;
esac

echo "[DONE] Mode '$MODE' completed."
