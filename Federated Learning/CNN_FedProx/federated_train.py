"""FedAvg/FedProx experiment runner with structured research logging."""

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

from cnn_model import ChannelDataset, ChannelEstimatorCNN
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
    parser = argparse.ArgumentParser(description="FedAvg/FedProx training for CNN channel estimation")
    parser.add_argument("--name", type=str, default="FedProx Non-IID", help="Experiment label")
    parser.add_argument("--client_dir", type=str, default="clients_non_iid", help="Client split directory")
    parser.add_argument("--rounds", type=int, default=20, help="Global communication rounds")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local epochs")
    parser.add_argument("--batch", type=int, default=32, help="Local batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Local learning rate")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal coefficient")
    parser.add_argument("--fedprox", action="store_true", help="Enable FedProx (otherwise FedAvg)")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--x_test", type=str, default="../../dataset/cnn/10dB/X_test.npy", help="Test input path")
    parser.add_argument("--y_test", type=str, default="../../dataset/cnn/10dB/Y_test.npy", help="Test target path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--snr", type=int, default=10, help="SNR used for this run metadata")
    parser.add_argument("--run_id", type=str, default="", help="Run identifier for structured logging")
    parser.add_argument("--log_dir", type=str, default=str(PROJECT_ROOT / "results" / "raw"), help="Structured log output directory")
    parser.add_argument("--config_path", type=str, default="", help="Optional config path for reproducibility logging")
    parser.add_argument("--save", type=str, default="fedprox_model.pth", help="Checkpoint path for best model")
    parser.add_argument("--data_mode", type=str, default="Non-IID", choices=["IID", "Non-IID"], help="Data regime")
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
        raise ValueError("Total client samples is zero. Check client dataset paths/splits.")

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
    best_state: dict | None,
    best_nmse_db: float,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "round_idx": round_idx,
        "global_state": {k: v.cpu() for k, v in global_model.state_dict().items()},
        "best_state": {k: v.cpu() for k, v in best_state.items()} if best_state is not None else None,
        "best_nmse_db": float(best_nmse_db),
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

    if not os.path.isdir(args.client_dir):
        raise FileNotFoundError(f"Client directory not found: {args.client_dir}")

    for client_slot in range(args.clients):
        x_expected = os.path.join(args.client_dir, f"X_client_{client_slot}.npy")
        y_expected = os.path.join(args.client_dir, f"Y_client_{client_slot}.npy")
        if not (os.path.exists(x_expected) and os.path.exists(y_expected)):
            raise FileNotFoundError(
                f"Missing expected client files for index {client_slot}: "
                f"{x_expected}, {y_expected}"
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    global_model = ChannelEstimatorCNN().to(device)

    setting_name = "FedProx" if args.fedprox else "FedAvg"
    run_id = (
        args.run_id
        or f"cnn_{setting_name.lower()}_{args.data_mode.lower()}_r{args.rounds}_e{args.local_epochs}_mu{args.mu}_seed{args.seed}"
    )
    logger = ExperimentLogger(args.log_dir)
    start_round = 1
    resumed = False

    if args.resume:
        ckpt = load_latest_checkpoint(args.checkpoint_dir, run_id)
        if ckpt is not None:
            global_model.load_state_dict(ckpt["global_state"])
            best_state = ckpt.get("best_state")
            best_nmse_db = float(ckpt["best_nmse_db"])
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
                setting=setting_name,
                data_mode=args.data_mode,
                snr_db=args.snr,
                seed=args.seed,
                local_epochs=args.local_epochs,
                global_rounds=args.rounds,
                mu=args.mu if args.fedprox else None,
                lr=args.lr,
                batch_size=args.batch,
                config_path=args.config_path or None,
                config_sha256=file_sha256(args.config_path) if args.config_path else None,
                git_commit=get_git_commit(str(PROJECT_ROOT)),
                python_version=runtime_python_version(),
                torch_version=detect_torch_version(),
                device=str(device),
                platform=runtime_platform(),
                extra={"num_clients": args.clients, "name": args.name},
            )
        )

        initial = evaluate_global_model(global_model, args.x_test, args.y_test, device)
        print(
            f"[Round 0] NMSE={initial['nmse_db']:.2f} dB "
            f"(linear={initial['nmse_linear']:.6f})"
        )
        best_nmse_db = float(initial["nmse_db"])
        best_state = copy.deepcopy({k: v.cpu() for k, v in global_model.state_dict().items()})
        save_checkpoint(args.checkpoint_dir, run_id, 0, global_model, best_state, best_nmse_db)

    if start_round > args.rounds:
        print(f"[RESUME] All requested rounds already completed for run_id={run_id}")

    for round_idx in range(start_round, args.rounds + 1):
        print(f"\nRound {round_idx}/{args.rounds}")
        client_weights = []
        sample_counts = []
        global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}

        for client_slot in range(args.clients):
            x_path = os.path.join(args.client_dir, f"X_client_{client_slot}.npy")
            y_path = os.path.join(args.client_dir, f"Y_client_{client_slot}.npy")

            local_model = ChannelEstimatorCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())

            w_local, n_samples = train_local_model(
                model=local_model,
                client_id=client_slot,
                x_path=x_path,
                y_path=y_path,
                global_weights=global_weights if args.fedprox else None,
                epochs=args.local_epochs,
                batch_size=args.batch,
                lr=args.lr,
                device=device,
                use_fedprox=args.fedprox,
                mu=args.mu,
            )

            if n_samples > 0:
                client_weights.append(w_local)
                sample_counts.append(n_samples)

        if not client_weights:
            raise RuntimeError("No valid clients were available for aggregation.")

        avg_weights = federated_average_weighted(client_weights, sample_counts)
        global_model.load_state_dict(avg_weights)

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

        save_checkpoint(args.checkpoint_dir, run_id, round_idx, global_model, best_state, best_nmse_db)

    if best_state is not None:
        torch.save(best_state, args.save)

    logger.end_run(run_id, {"final_nmse_db": best_nmse_db})
    print(f"\n{setting_name} complete | Best NMSE={best_nmse_db:.2f} dB | Saved={args.save}")


if __name__ == "__main__":
    main()
