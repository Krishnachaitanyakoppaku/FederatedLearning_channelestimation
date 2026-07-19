"""Federated Averaging trainer for CNN channel estimation experiments."""

from __future__ import annotations

import argparse
import copy
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_model import ChannelDataset, ChannelEstimatorCNN, count_parameters
from local_train import train_local_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from research_core.logging_utils import ExperimentLogger, RunMetadata
from research_core.logging_utils import (
    detect_torch_version,
    file_sha256,
    get_git_commit,
    runtime_platform,
    runtime_python_version,
)
from research_core.metrics import nmse_from_two_channel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated Averaging for CNN channel estimation")
    parser.add_argument("--clients", type=int, default=5, help="Number of federated clients")
    parser.add_argument("--rounds", type=int, default=20, help="Global communication rounds")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local epochs per client")
    parser.add_argument("--batch", type=int, default=32, help="Local batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial client learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.97, help="Round-wise LR decay factor")
    parser.add_argument("--client_dir", type=str, default="clients", help="Client split directory")
    parser.add_argument("--x_test", type=str, default="../../dataset/cnn/10dB/X_test.npy", help="Test input path")
    parser.add_argument("--y_test", type=str, default="../../dataset/cnn/10dB/Y_test.npy", help="Test target path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--snr", type=int, default=10, help="SNR used for this run metadata")
    parser.add_argument("--run_id", type=str, default="", help="Run identifier for structured logging")
    parser.add_argument("--log_dir", type=str, default=str(PROJECT_ROOT / "results" / "raw"), help="Directory for structured logs")
    parser.add_argument("--config_path", type=str, default="", help="Optional config path for reproducibility logging")
    parser.add_argument("--save", type=str, default="fed_model.pth", help="Best model checkpoint path")
    parser.add_argument("--data_mode", type=str, default="IID", choices=["IID", "Non-IID"], help="Data split regime")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for per-round resumable checkpoints",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    return parser.parse_args()


def federated_average_weighted(client_weights, sample_counts):
    if not client_weights:
        raise ValueError("No client weights received for aggregation.")

    total_samples = sum(sample_counts)
    if total_samples <= 0:
        raise ValueError("Total client samples is zero; verify client splits.")

    fractions = [n / total_samples for n in sample_counts]
    avg_dict = copy.deepcopy(client_weights[0])
    for key in avg_dict.keys():
        avg_dict[key] = avg_dict[key].float() * fractions[0]
        for i in range(1, len(client_weights)):
            avg_dict[key] += client_weights[i][key].float() * fractions[i]
        avg_dict[key] = avg_dict[key].to(client_weights[0][key].dtype)
    return avg_dict


def evaluate_global_model(model, x_test_path, y_test_path, device):
    model.eval()
    dataset = ChannelDataset(x_test_path, y_test_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    preds = []
    true = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_pred = model(x_batch.to(device)).cpu().numpy()
            preds.append(y_pred)
            true.append(y_batch.numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(true, axis=0)
    metrics = nmse_from_two_channel(y_true, y_pred)
    model.train()
    return metrics


def checkpoint_path(checkpoint_dir: str, run_id: str, round_idx: int) -> str:
    return os.path.join(checkpoint_dir, f"{run_id}_round{round_idx:04d}.pth")


def save_checkpoint(
    checkpoint_dir: str,
    run_id: str,
    round_idx: int,
    global_model: torch.nn.Module,
    best_state: dict,
    best_nmse_db: float,
    current_lr: float,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "round_idx": round_idx,
        "global_state": {k: v.cpu() for k, v in global_model.state_dict().items()},
        "best_state": {k: v.cpu() for k, v in best_state.items()},
        "best_nmse_db": float(best_nmse_db),
        "current_lr": float(current_lr),
    }
    torch.save(payload, checkpoint_path(checkpoint_dir, run_id, round_idx))


def load_latest_checkpoint(checkpoint_dir: str, run_id: str):
    pattern = os.path.join(checkpoint_dir, f"{run_id}_round*.pth")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    latest = candidates[-1]
    return torch.load(latest, map_location="cpu")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    global_model = ChannelEstimatorCNN().to(device)
    n_params = count_parameters(global_model)
    print(f"Global Model: {n_params:,} parameters")
    assert n_params < 100_000, f"Model too large: {n_params:,}"

    if not os.path.isdir(args.client_dir) or len(os.listdir(args.client_dir)) == 0:
        raise FileNotFoundError(
            f"Client data not found in '{args.client_dir}'. Run split_clients.py first."
        )

    for client_id in range(args.clients):
        x_expected = os.path.join(args.client_dir, f"X_client_{client_id}.npy")
        y_expected = os.path.join(args.client_dir, f"Y_client_{client_id}.npy")
        if not (os.path.exists(x_expected) and os.path.exists(y_expected)):
            raise FileNotFoundError(
                f"Missing expected client files for index {client_id}: "
                f"{x_expected}, {y_expected}"
            )

    run_id = args.run_id or f"cnn_fedavg_{args.data_mode.lower()}_r{args.rounds}_e{args.local_epochs}_seed{args.seed}"
    logger = ExperimentLogger(args.log_dir)
    start_round = 1
    resumed = False

    if args.resume:
        ckpt = load_latest_checkpoint(args.checkpoint_dir, run_id)
        if ckpt is not None:
            global_model.load_state_dict(ckpt["global_state"])
            best_state = ckpt["best_state"]
            best_nmse_db = float(ckpt["best_nmse_db"])
            current_lr = float(ckpt["current_lr"])
            start_round = int(ckpt["round_idx"]) + 1
            resumed = True
            print(f"[RESUME] Loaded checkpoint for run_id={run_id} at round {start_round - 1}")
        else:
            print(f"[RESUME] No checkpoint found for run_id={run_id}; starting fresh")

    if not resumed:
        logger.start_run(
            RunMetadata(
                run_id=run_id,
                model="CNN",
                setting="FedAvg",
                data_mode=args.data_mode,
                snr_db=args.snr,
                seed=args.seed,
                local_epochs=args.local_epochs,
                global_rounds=args.rounds,
                mu=None,
                lr=args.lr,
                batch_size=args.batch,
                config_path=args.config_path or None,
                config_sha256=file_sha256(args.config_path) if args.config_path else None,
                git_commit=get_git_commit(str(PROJECT_ROOT)),
                python_version=runtime_python_version(),
                torch_version=detect_torch_version(),
                device=str(device),
                platform=runtime_platform(),
                extra={"num_clients": args.clients},
            )
        )

        initial = evaluate_global_model(global_model, args.x_test, args.y_test, device)
        print(
            f"[Round 0] NMSE={initial['nmse_db']:.2f} dB "
            f"(linear={initial['nmse_linear']:.6f})"
        )

        best_nmse_db = initial["nmse_db"]
        best_state = copy.deepcopy({k: v.cpu() for k, v in global_model.state_dict().items()})
        current_lr = args.lr
        save_checkpoint(args.checkpoint_dir, run_id, 0, global_model, best_state, best_nmse_db, current_lr)

    if start_round > args.rounds:
        print(f"[RESUME] All requested rounds already completed for run_id={run_id}")

    for round_idx in range(start_round, args.rounds + 1):
        print(f"\nRound {round_idx}/{args.rounds} | lr={current_lr:.6f}")
        client_weights = []
        sample_counts = []

        for client_id in range(args.clients):
            x_path = os.path.join(args.client_dir, f"X_client_{client_id}.npy")
            y_path = os.path.join(args.client_dir, f"Y_client_{client_id}.npy")

            local_model = ChannelEstimatorCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())

            w_local, n_samples = train_local_model(
                model=local_model,
                client_id=client_id,
                x_path=x_path,
                y_path=y_path,
                epochs=args.local_epochs,
                batch_size=args.batch,
                lr=current_lr,
                device=device,
            )
            if n_samples > 0:
                client_weights.append(w_local)
                sample_counts.append(n_samples)

        avg_weights = federated_average_weighted(client_weights, sample_counts)
        global_model.load_state_dict(avg_weights)
        global_model.to(device)

        metrics = evaluate_global_model(global_model, args.x_test, args.y_test, device)
        logger.log_step(
            run_id,
            "round",
            round_idx,
            {"nmse_db": float(metrics["nmse_db"]), "nmse_linear": float(metrics["nmse_linear"])},
        )

        improved = " *best" if metrics["nmse_db"] < best_nmse_db else ""
        print(
            f"Round {round_idx} NMSE={metrics['nmse_db']:.2f} dB "
            f"(linear={metrics['nmse_linear']:.6f}){improved}"
        )

        if metrics["nmse_db"] < best_nmse_db:
            best_nmse_db = metrics["nmse_db"]
            best_state = copy.deepcopy({k: v.cpu() for k, v in global_model.state_dict().items()})

        current_lr *= args.lr_decay
        save_checkpoint(args.checkpoint_dir, run_id, round_idx, global_model, best_state, best_nmse_db, current_lr)

    torch.save(best_state, args.save)
    logger.end_run(run_id, {"final_nmse_db": best_nmse_db})

    print("\nFederated training complete")
    print(f"Best NMSE: {best_nmse_db:.2f} dB")
    print(f"Saved model: {args.save}")


if __name__ == "__main__":
    main()
