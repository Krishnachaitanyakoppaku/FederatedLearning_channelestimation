#!/usr/bin/env bash
set -euo pipefail

# Minimal ResUNet support study (20 runs total):
# - Centralized: 5 seeds
# - FedAvg IID: 5 seeds (implemented via fedprox_resunet.py with mu=0, no --non_iid)
# - FedAvg Non-IID: 5 seeds (mu=0, --non_iid)
# - FedProx Non-IID: 5 seeds (mu=0.001, --non_iid)
#
# Resume behavior:
# - Federated runs use --resume and checkpoint_dir, so interrupted runs continue.
# - Centralized script has no native resume; this launcher SKIPS completed run_ids by checking
#   existing structured JSON logs for an ended run.

PROJECT="${PROJECT:-/Users/chaitu/Desktop/Wireless paper/Project}"
PY="${PY:-$PROJECT/myenv/bin/python}"
CFG="${CFG:-$PROJECT/experiment_config.json}"

RAW_LOG_DIR="${RAW_LOG_DIR:-$PROJECT/results/raw}"
CKPT_DIR="${CKPT_DIR:-$PROJECT/results/checkpoints}"
MODEL_DIR="${MODEL_DIR:-$PROJECT/results/models/resunet_claim_support}"
STDOUT_DIR="${STDOUT_DIR:-$PROJECT/results/logs/resunet_claim_support}"

SEEDS=(42 123 777 2024 3407)

SNR="${SNR:-10}"
CENT_EPOCHS="${CENT_EPOCHS:-40}"
CENT_BATCH="${CENT_BATCH:-32}"
CENT_LR="${CENT_LR:-0.0005}"

ROUNDS="${ROUNDS:-50}"
CLIENTS="${CLIENTS:-5}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
FED_BATCH="${FED_BATCH:-16}"
FED_LR="${FED_LR:-0.0005}"
FEDPROX_MU="${FEDPROX_MU:-0.001}"

mkdir -p "$RAW_LOG_DIR" "$CKPT_DIR" "$MODEL_DIR" "$STDOUT_DIR"

is_run_complete() {
  local run_id="$1"
  local json_path="$RAW_LOG_DIR/${run_id}.json"
  if [[ ! -f "$json_path" ]]; then
    return 1
  fi

  "$PY" - "$json_path" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
except Exception:
    sys.exit(1)

ended = payload.get("ended_at") is not None
summary = payload.get("summary")
has_final = isinstance(summary, dict) and ("final_nmse_db" in summary or "final_nmse_linear" in summary)
sys.exit(0 if (ended and has_final) else 1)
PY
}

run_centralized() {
  local seed="$1"
  local run_id="resunet_centralized_snr${SNR}_seed${seed}"
  local out_log="$STDOUT_DIR/${run_id}.log"
  local model_out="$MODEL_DIR/${run_id}.pth"

  if is_run_complete "$run_id"; then
    echo "[SKIP] Completed centralized run: $run_id"
    return 0
  fi

  echo "[RUN ] $run_id"
  "$PY" "$PROJECT/Centralised Learning/ResUNet/resunet_model.py" \
    --snr "$SNR" \
    --epochs "$CENT_EPOCHS" \
    --batch "$CENT_BATCH" \
    --lr "$CENT_LR" \
    --seed "$seed" \
    --run_id "$run_id" \
    --log_dir "$RAW_LOG_DIR" \
    --config_path "$CFG" \
    2>&1 | tee "$out_log"

  if [[ -f "$PROJECT/resunet_model.pth" ]]; then
    cp "$PROJECT/resunet_model.pth" "$model_out"
  fi
}

run_fed() {
  local run_id="$1"
  local out_log="$2"
  shift 2

  if is_run_complete "$run_id"; then
    echo "[SKIP] Completed federated run: $run_id"
    return 0
  fi

  echo "[RUN ] $run_id"
  "$PY" "$PROJECT/Federated Learning/resunet_fedprox/fedprox_resunet.py" \
    --rounds "$ROUNDS" \
    --clients "$CLIENTS" \
    --local_epochs "$LOCAL_EPOCHS" \
    --batch "$FED_BATCH" \
    --lr "$FED_LR" \
    --snr "$SNR" \
    --run_id "$run_id" \
    --log_dir "$RAW_LOG_DIR" \
    --config_path "$CFG" \
    --checkpoint_dir "$CKPT_DIR" \
    --save "$MODEL_DIR/${run_id}.pth" \
    --resume \
    "$@" \
    2>&1 | tee "$out_log"
}

echo "[INFO] Starting ResUNet claim-support plan"

echo "[INFO] Phase 1/4: Centralized (5 runs)"
for seed in "${SEEDS[@]}"; do
  run_centralized "$seed"
done

echo "[INFO] Phase 2/4: FedAvg IID via mu=0 (5 runs)"
for seed in "${SEEDS[@]}"; do
  run_id="resunet_fedavg_iid_snr${SNR}_r${ROUNDS}_e${LOCAL_EPOCHS}_seed${seed}"
  out_log="$STDOUT_DIR/${run_id}.log"
  run_fed "$run_id" "$out_log" --mu 0 --seed "$seed"
done

echo "[INFO] Phase 3/4: FedAvg Non-IID via mu=0 (5 runs)"
for seed in "${SEEDS[@]}"; do
  run_id="resunet_fedavg_non-iid_snr${SNR}_r${ROUNDS}_e${LOCAL_EPOCHS}_seed${seed}"
  out_log="$STDOUT_DIR/${run_id}.log"
  run_fed "$run_id" "$out_log" --mu 0 --non_iid --non_iid_unique_ratio 0.6 --non_iid_seed 123 --seed "$seed"
done

echo "[INFO] Phase 4/4: FedProx Non-IID (5 runs)"
for seed in "${SEEDS[@]}"; do
  run_id="resunet_fedprox_non-iid_snr${SNR}_r${ROUNDS}_e${LOCAL_EPOCHS}_mu${FEDPROX_MU}_seed${seed}"
  out_log="$STDOUT_DIR/${run_id}.log"
  run_fed "$run_id" "$out_log" --mu "$FEDPROX_MU" --non_iid --non_iid_unique_ratio 0.6 --non_iid_seed 123 --seed "$seed"
done

echo "[DONE] ResUNet claim-support runs completed."
echo "[INFO] Raw structured logs: $RAW_LOG_DIR"
echo "[INFO] Checkpoints: $CKPT_DIR"
echo "[INFO] Stdout logs: $STDOUT_DIR"
echo "[INFO] Model artifacts: $MODEL_DIR"
