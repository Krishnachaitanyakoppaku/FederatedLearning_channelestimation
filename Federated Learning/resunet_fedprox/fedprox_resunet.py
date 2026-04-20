import argparse
import copy
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from research_core.logging_utils import (  # noqa: E402
    ExperimentLogger,
    RunMetadata,
    detect_torch_version,
    file_sha256,
    get_git_commit,
    runtime_platform,
    runtime_python_version,
)


DATASET_DIR = PROJECT_ROOT / "dataset"


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(2, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(64, 128)
        self.bottleneck = ResidualBlock(128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.final_conv = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        bn = self.bottleneck(e3)
        u1 = self.upconv1(bn)
        c1 = torch.cat([e2, u1], dim=1)
        d1 = self.dec1(c1)
        u2 = self.upconv2(d1)
        c2 = torch.cat([e1, u2], dim=1)
        d2 = self.dec2(c2)
        return self.final_conv(d2)


def parse_args():
    parser = argparse.ArgumentParser(description="Federated ResUNet with FedProx")
    parser.add_argument("--rounds", "--epochs", type=int, default=15, dest="rounds", help="Global federated rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local epochs per client")
    parser.add_argument("--batch", type=int, default=16, help="Local batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Local learning rate")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal coefficient")
    parser.add_argument("--non_iid", action="store_true", help="Use Non-IID data distribution")
    parser.add_argument(
        "--non_iid_unique_ratio",
        type=float,
        default=0.6,
        help="For Non-IID split: per-client ratio of unique sorted samples (rest is shared)",
    )
    parser.add_argument(
        "--non_iid_seed",
        type=int,
        default=123,
        help="Random seed used for Non-IID shared sampling/shuffling",
    )
    parser.add_argument("--snr", type=int, default=10, help="SNR level in dB")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=str, default="fedprox_resunet_model.pth", help="Best model save path")
    parser.add_argument("--run_id", type=str, default="", help="Run identifier for structured logging")
    parser.add_argument("--log_dir", type=str, default=str(PROJECT_ROOT / "results" / "raw"), help="Structured log output directory")
    parser.add_argument("--config_path", type=str, default="", help="Optional config path for reproducibility logging")
    parser.add_argument("--checkpoint_dir", type=str, default=str(PROJECT_ROOT / "results" / "checkpoints"), help="Checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint for this run_id")
    return parser.parse_args()


def checkpoint_path(checkpoint_dir: str, run_id: str, round_idx: int) -> str:
    return os.path.join(checkpoint_dir, f"{run_id}_round{round_idx:04d}.pth")


def save_checkpoint(checkpoint_dir: str, run_id: str, round_idx: int, global_model: nn.Module, best_state: dict, best_nmse_db: float):
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "round_idx": round_idx,
        "global_state": global_model.state_dict(),
        "best_state": best_state,
        "best_nmse_db": float(best_nmse_db),
    }
    torch.save(payload, checkpoint_path(checkpoint_dir, run_id, round_idx))


def load_latest_checkpoint(checkpoint_dir: str, run_id: str):
    pattern = os.path.join(checkpoint_dir, f"{run_id}_round*.pth")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return torch.load(files[-1], map_location="cpu")


def load_and_split_data(
    num_clients: int,
    snr_db: int,
    non_iid: bool = False,
    non_iid_unique_ratio: float = 0.6,
    non_iid_seed: int = 123,
):
    print("[INFO] Loading raw dataset...")
    x_path = DATASET_DIR / f"H_noisy_{snr_db}dB_ri.npy"
    y_path = DATASET_DIR / "H_clean_ri.npy"
    if not (x_path.exists() and y_path.exists()):
        raise FileNotFoundError(f"Required dataset files not found: {x_path} / {y_path}")

    x = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)
    print(f"[INFO] Loaded Dataset. Shape: {x.shape}")

    global_mean = np.mean(np.concatenate([x, y], axis=0)).astype(np.float32)
    global_std = np.std(np.concatenate([x, y], axis=0)).astype(np.float32)
    x = (x - global_mean) / global_std
    y = (y - global_mean) / global_std
    norm_stats = {"mean": float(global_mean), "std": float(global_std)}

    n_samples = x.shape[0]
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_samples)
    split_idx = int(n_samples * 0.8)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    n_train = len(x_train)
    chunk_size = n_train // num_clients
    if non_iid:
        if not (0.0 <= non_iid_unique_ratio <= 1.0):
            raise ValueError("--non_iid_unique_ratio must be in [0, 1]")

        n_unique = int(chunk_size * non_iid_unique_ratio)
        n_shared = chunk_size - n_unique
        print(
            "[INFO] Creating Non-IID data split "
            f"({n_unique}/{chunk_size} unique + {n_shared}/{chunk_size} shared, "
            f"ratio={non_iid_unique_ratio:.3f}, seed={non_iid_seed})"
        )
        norms = np.linalg.norm(y_train.reshape(n_train, -1), axis=1)
        sort_idx = np.argsort(norms)
        x_sorted = x_train[sort_idx]
        y_sorted = y_train[sort_idx]
        client_data = []
        rng_split = np.random.default_rng(non_iid_seed)
        for i in range(num_clients):
            unique_start = i * (n_train // num_clients)
            unique_end = unique_start + n_unique
            x_unique = x_sorted[unique_start:unique_end]
            y_unique = y_sorted[unique_start:unique_end]
            shared_idx = rng_split.choice(n_train, size=n_shared, replace=False)
            x_shared = x_train[shared_idx]
            y_shared = y_train[shared_idx]
            x_client = np.concatenate([x_unique, x_shared], axis=0)
            y_client = np.concatenate([y_unique, y_shared], axis=0)
            shuffle_idx = rng_split.permutation(len(x_client))
            x_client = x_client[shuffle_idx]
            y_client = y_client[shuffle_idx]
            client_data.append((x_client, y_client))
    else:
        print("[INFO] Creating IID data split")
        client_data = []
        for i in range(num_clients):
            start = i * chunk_size
            end = start + chunk_size
            client_data.append((x_train[start:end], y_train[start:end]))

    print(f"[INFO] Train: {n_train} samples across {num_clients} clients, Test: {len(x_test)} samples")
    return client_data, (x_test, y_test), norm_stats


def fedavg_aggregate(client_weights_list, client_data_sizes):
    n_total = sum(client_data_sizes)
    global_weights = copy.deepcopy(client_weights_list[0])
    weight_factor = client_data_sizes[0] / n_total
    for key in global_weights.keys():
        global_weights[key] = global_weights[key].float() * weight_factor
    for i in range(1, len(client_weights_list)):
        weight_factor = client_data_sizes[i] / n_total
        for key in global_weights.keys():
            global_weights[key] += client_weights_list[i][key].float() * weight_factor
    return global_weights


def client_update_fedprox(global_model, client_data, epochs, batch_size, lr, mu, device):
    x, y = client_data
    dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = copy.deepcopy(global_model).to(device)
    model.train()
    global_weight_list = [p.detach().clone().to(device) for p in global_model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            residual = model(x_batch)
            output = x_batch - residual
            mse_loss = criterion(output, y_batch)
            prox_term = 0.0
            if mu > 0.0:
                for local_param, global_param in zip(model.parameters(), global_weight_list):
                    prox_term += ((local_param - global_param) ** 2).sum()
            total_loss = mse_loss + (mu / 2.0) * prox_term
            total_loss.backward()
            optimizer.step()

    del global_weight_list
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    model.to("cpu")
    return model.state_dict()


def evaluate(model, test_data, norm_stats, batch_size, device):
    x, y = test_data
    dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            residual = model(x_batch)
            y_pred = x_batch - residual
            all_preds.append(y_pred.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = y
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    y_pred = y_pred * std + mean
    y_true = y_true * std + mean
    h_pred = y_pred[:, 0, :, :] + 1j * y_pred[:, 1, :, :]
    h_true = y_true[:, 0, :, :] + 1j * y_true[:, 1, :, :]
    nmse_num = np.linalg.norm(h_true - h_pred) ** 2
    nmse_den = np.linalg.norm(h_true) ** 2
    nmse = float(nmse_num / nmse_den)
    nmse_db = float(10 * np.log10(nmse))
    return nmse, nmse_db


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mode_str = "Non-IID" if args.non_iid else "IID"
    mode_token = "non-iid" if args.non_iid else "iid"
    print(f"\n[INFO] Initializing FedProx ResUNet on {device}")

    client_data, test_data, norm_stats = load_and_split_data(
        args.clients,
        args.snr,
        non_iid=args.non_iid,
        non_iid_unique_ratio=args.non_iid_unique_ratio,
        non_iid_seed=args.non_iid_seed,
    )
    global_model = ResUNet()

    run_id = (
        args.run_id
        or f"resunet_fedprox_{mode_token}_snr{args.snr}_r{args.rounds}_e{args.local_epochs}_mu{args.mu}_seed{args.seed}"
    )
    logger = ExperimentLogger(args.log_dir)

    start_round = 1
    best_nmse_db = float("inf")
    best_state = copy.deepcopy(global_model.state_dict())

    if args.resume:
        ckpt = load_latest_checkpoint(args.checkpoint_dir, run_id)
        if ckpt is not None:
            global_model.load_state_dict(ckpt["global_state"])
            best_state = ckpt.get("best_state", best_state)
            best_nmse_db = float(ckpt.get("best_nmse_db", best_nmse_db))
            start_round = int(ckpt["round_idx"]) + 1
            print(f"[RESUME] Loaded checkpoint for run_id={run_id} at round {start_round - 1}")
        else:
            print(f"[RESUME] No checkpoint found for run_id={run_id}; starting fresh")

    if start_round == 1:
        logger.start_run(
            RunMetadata(
                run_id=run_id,
                model="ResUNet",
                setting="FedProx",
                data_mode=mode_str,
                snr_db=args.snr,
                seed=args.seed,
                local_epochs=args.local_epochs,
                global_rounds=args.rounds,
                mu=args.mu,
                lr=args.lr,
                batch_size=args.batch,
                config_path=args.config_path or None,
                config_sha256=file_sha256(args.config_path) if args.config_path else None,
                git_commit=get_git_commit(str(PROJECT_ROOT)),
                python_version=runtime_python_version(),
                torch_version=detect_torch_version(),
                device=str(device),
                platform=runtime_platform(),
                extra={
                    "num_clients": args.clients,
                    "non_iid_unique_ratio": args.non_iid_unique_ratio if args.non_iid else None,
                    "non_iid_seed": args.non_iid_seed if args.non_iid else None,
                },
            )
        )
        nmse, nmse_db = evaluate(global_model.to(device), test_data, norm_stats, args.batch, device)
        global_model.to("cpu")
        best_nmse_db = nmse_db
        best_state = copy.deepcopy(global_model.state_dict())
        logger.log_step(run_id, "round", 0, {"nmse_db": nmse_db, "nmse_linear": nmse})
        save_checkpoint(args.checkpoint_dir, run_id, 0, global_model, best_state, best_nmse_db)
        print(f"[Round 0] NMSE={nmse_db:.2f} dB (linear={nmse:.6f})")

    if start_round > args.rounds:
        print(f"[RESUME] All requested rounds already completed for run_id={run_id}")
        logger.end_run(run_id, {"final_nmse_db": best_nmse_db, "final_nmse_linear": float(10 ** (best_nmse_db / 10))})
        torch.save(best_state, args.save)
        return

    for round_idx in range(start_round, args.rounds + 1):
        client_weights = []
        client_sizes = []
        for i in range(args.clients):
            print(f"Round {round_idx} | Client {i + 1}/{args.clients} training...")
            weights = client_update_fedprox(global_model, client_data[i], args.local_epochs, args.batch, args.lr, args.mu, device)
            client_weights.append(weights)
            client_sizes.append(len(client_data[i][0]))

        updated_weights = fedavg_aggregate(client_weights, client_sizes)
        global_model.load_state_dict(updated_weights)
        global_model.to(device)
        nmse, nmse_db = evaluate(global_model, test_data, norm_stats, args.batch, device)
        global_model.to("cpu")

        if nmse_db < best_nmse_db:
            best_nmse_db = nmse_db
            best_state = copy.deepcopy(global_model.state_dict())

        logger.log_step(run_id, "round", round_idx, {"nmse_db": nmse_db, "nmse_linear": nmse})
        save_checkpoint(args.checkpoint_dir, run_id, round_idx, global_model, best_state, best_nmse_db)
        print(f"Round {round_idx} -> NMSE={nmse_db:.2f} dB")

    logger.end_run(run_id, {"final_nmse_db": best_nmse_db, "final_nmse_linear": float(10 ** (best_nmse_db / 10))})
    torch.save(best_state, args.save)
    print("-----------------------------------------")
    print(f"FEDPROX ({mode_str}) TEST RESULTS:")
    print(f"Samples tested: {len(test_data[0])}")
    print(f"Best NMSE (dB): {best_nmse_db:.2f}")
    print(f"Saved model: {args.save}")
    print("-----------------------------------------")


if __name__ == "__main__":
    main()
