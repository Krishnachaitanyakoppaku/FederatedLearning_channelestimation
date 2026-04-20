import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import sys
from pathlib import Path

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
from research_core.metrics import nmse

# -------------------------------------------------------------
# DIRECTORIES & CONSTANTS
# -------------------------------------------------------------
SNR_LEVELS = [0, 5, 10, 15, 20]
DATASET_DIR = PROJECT_ROOT / "dataset" / "cnn"

# -------------------------------------------------------------
# DATASET
# -------------------------------------------------------------
class ChannelDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = torch.tensor(np.load(x_path), dtype=torch.float32)
        self.y = torch.tensor(np.load(y_path), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def build_dataloaders(snr_db, batch_size):
    data_dir = DATASET_DIR / f"{snr_db}dB"
    
    train_ds = ChannelDataset(
        x_path=data_dir / "X_train.npy",
        y_path=data_dir / "Y_train.npy",
    )
    test_ds = ChannelDataset(
        x_path=data_dir / "X_test.npy",
        y_path=data_dir / "Y_test.npy",
    )
    
    stats = np.load(data_dir / "norm_stats.npz")
    norm_stats = {"mean": float(stats["mean"]), "std": float(stats["std"])}
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, norm_stats

# -------------------------------------------------------------
# BUILDING BLOCKS
# -------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

# -------------------------------------------------------------
# MODEL ARCHITECTURE (ResUNet)
# -------------------------------------------------------------
class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()
        # Encoder
        self.enc1 = ConvBlock(2, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = ConvBlock(64, 128)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(128)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        
        # Final Layer
        self.final_conv = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        
        # Bottleneck
        bn = self.bottleneck(e3)
        
        # Decoder
        u1 = self.upconv1(bn)
        c1 = torch.cat([e2, u1], dim=1) # skip connection
        d1 = self.dec1(c1)
        
        u2 = self.upconv2(d1)
        c2 = torch.cat([e1, u2], dim=1) # skip connection
        d2 = self.dec2(c2)
        
        # Final Layer (no activation)
        out = self.final_conv(d2)
        return out

# -------------------------------------------------------------
# EVALUATION (NMSE Calculation)
# -------------------------------------------------------------
def evaluate(model, test_loader, device, norm_stats):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            residual = model(x_batch)
            y_pred = x_batch - residual
            all_preds.append(y_pred.cpu().numpy())
            all_labels.append(y_batch.numpy())
            
    Y_pred = np.concatenate(all_preds, axis=0) # (N, 2, 64, 64)
    Y_true = np.concatenate(all_labels, axis=0)
    
    # Original scale
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    Y_pred = Y_pred * std + mean
    Y_true = Y_true * std + mean
    
    # Complex form
    H_pred = Y_pred[:, 0, :, :] + 1j * Y_pred[:, 1, :, :]
    H_true = Y_true[:, 0, :, :] + 1j * Y_true[:, 1, :, :]
    
    nmse_metrics = nmse(H_true, H_pred)
    return nmse_metrics["nmse_linear"], nmse_metrics["nmse_db"]

# -------------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------------
def train_model(snr_db, epochs, lr, batch_size, device, logger=None, run_id=None):
    print(f"\n=========================================")
    print(f" TRAINING ResUNet FOR SNR = {snr_db} dB")
    print(f"=========================================")
    
    train_loader, test_loader, norm_stats = build_dataloaders(snr_db, batch_size)
    
    model = ResUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            residual = model(x_batch)
            y_pred = x_batch - residual
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        
        # Validation NMSE
        val_nmse, val_nmse_db = evaluate(model, test_loader, device, norm_stats)
        
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.6f} - Validation NMSE: {val_nmse_db:.2f} dB")
        if logger is not None and run_id is not None:
            logger.log_step(
                run_id,
                "epoch",
                epoch,
                {"train_mse_loss": float(epoch_loss), "val_nmse_db": float(val_nmse_db)},
            )
        
    print("\n-----------------------------------------")
    print("TEST SET RESULTS:")
    print(f"Samples tested: {len(test_loader.dataset)}")
    print(f"NMSE (linear): {val_nmse:.6f}")
    print(f"NMSE (dB): {val_nmse_db:.2f} dB")
    print("-----------------------------------------")
    
    return model, val_nmse, val_nmse_db

# -------------------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ResUNet for Channel Estimation")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--snr", type=int, default=10, help="SNR to train on (default 10 dB)")
    parser.add_argument("--all", action="store_true", help="Train and plot across all SNRs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_id", type=str, default="", help="Run identifier for structured logs")
    parser.add_argument("--log_dir", type=str, default=str(PROJECT_ROOT / "results" / "raw"), help="Directory for structured logs")
    parser.add_argument("--config_path", type=str, default="", help="Optional config path for reproducibility logging")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.all:
        nmses_db = []
        logger = ExperimentLogger(args.log_dir)
        for snr in SNR_LEVELS:
            run_id = args.run_id or f"resunet_centralized_snr{snr}_seed{args.seed}"
            logger.start_run(
                RunMetadata(
                    run_id=run_id,
                    model="ResUNet",
                    setting="Centralized",
                    data_mode="N/A",
                    snr_db=snr,
                    seed=args.seed,
                    local_epochs=None,
                    global_rounds=None,
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
                )
            )
            model, _, val_nmse_db = train_model(snr, args.epochs, args.lr, args.batch, device, logger=logger, run_id=run_id)
            nmses_db.append(val_nmse_db)
            logger.end_run(run_id, {"final_nmse_db": float(val_nmse_db)})
            
            # Save for each SNR
            torch.save(model.state_dict(), f"resunet_model_{snr}dB.pth")
            
        # Plot NMSE vs SNR
        plt.figure(figsize=(8, 6))
        plt.plot(SNR_LEVELS, nmses_db, marker='o', linestyle='-', linewidth=2, color='r', label='ResUNet')
        plt.xlabel('SNR (dB)')
        plt.ylabel('NMSE (dB)')
        plt.title('NMSE vs SNR for ResUNet Channel Estimation')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig("nmse_vs_snr_resunet.png")
        print("\n[✓] Plot saved as nmse_vs_snr_resunet.png")
    else:
        run_id = args.run_id or f"resunet_centralized_snr{args.snr}_seed{args.seed}"
        logger = ExperimentLogger(args.log_dir)
        logger.start_run(
            RunMetadata(
                run_id=run_id,
                model="ResUNet",
                setting="Centralized",
                data_mode="N/A",
                snr_db=args.snr,
                seed=args.seed,
                local_epochs=None,
                global_rounds=None,
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
            )
        )
        model, _, val_nmse_db = train_model(args.snr, args.epochs, args.lr, args.batch, device, logger=logger, run_id=run_id)
        logger.end_run(run_id, {"final_nmse_db": float(val_nmse_db)})
        save_path = "resunet_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f"\n[✓] Trained model saved as: {save_path}")

if __name__ == "__main__":
    main()
