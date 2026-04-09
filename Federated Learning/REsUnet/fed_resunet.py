import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------
# DIRECTORIES & CONSTANTS
# -------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../dataset"))

# -------------------------------------------------------------
# BUILDING BLOCKS & MODEL ARCHITECTURE
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
# DATA LOADING & SPLITTING
# -------------------------------------------------------------
def load_and_split_data(num_clients):
    print("[INFO] Loading raw dataset...")
    X_path = os.path.join(DATASET_DIR, "H_noisy_10dB_ri.npy")
    Y_path = os.path.join(DATASET_DIR, "H_clean_ri.npy")

    if not (os.path.exists(X_path) and os.path.exists(Y_path)):
        raise FileNotFoundError(
            f"Required dataset files not found under {DATASET_DIR}. "
            "Run data generation/preparation first."
        )
        
    X = np.load(X_path)
    Y = np.load(Y_path)
    
    print(f"[INFO] Loaded Dataset. Shape: {X.shape}")
    
    # Global Normalization
    global_mean = np.mean(np.concatenate([X, Y], axis=0)).astype(np.float32)
    global_std = np.std(np.concatenate([X, Y], axis=0)).astype(np.float32)
    
    X = ((X - global_mean) / global_std).astype(np.float32)
    Y = ((Y - global_mean) / global_std).astype(np.float32)
    
    norm_stats = {"mean": float(global_mean), "std": float(global_std)}
    
    # 80/20 train/test split
    n_samples = X.shape[0]
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_samples)
    
    split_idx = int(n_samples * 0.8)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    # Split training data across clients (IID)
    client_data = []
    chunk_size = len(X_train) // num_clients
    
    for i in range(num_clients):
        start = i * chunk_size
        end = start + chunk_size
        client_data.append((X_train[start:end], Y_train[start:end]))
        
    print(f"[INFO] Dataset split -> 80% Train ({len(X_train)} samples across {num_clients} clients), 20% Test ({len(X_test)} samples)")
    return client_data, (X_test, Y_test), norm_stats

# -------------------------------------------------------------
# FEDERATED LOGIC & EVALUATION
# -------------------------------------------------------------
def fedavg(client_weights_list, client_data_sizes):
    N_total = sum(client_data_sizes)
    global_weights = copy.deepcopy(client_weights_list[0])
    
    # Initialize with the weighted first client
    weight_factor = client_data_sizes[0] / N_total
    for key in global_weights.keys():
        global_weights[key] = global_weights[key] * weight_factor
        
    # Add other clients iteratively
    for i in range(1, len(client_weights_list)):
        weight_factor = client_data_sizes[i] / N_total
        for key in global_weights.keys():
            global_weights[key] += client_weights_list[i][key] * weight_factor
            
    return global_weights

def client_update(global_model, client_data, epochs, batch_size, lr, device):
    X, Y = client_data
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Copy global model
    model = copy.deepcopy(global_model).to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            # Residual Learning: Output = noisy - res_model(noisy)
            residual = model(x_batch)
            y_pred = x_batch - residual
            
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
    # Move model back to CPU to save VRAM and extract weights
    model.to("cpu")
    return model.state_dict()

def evaluate(model, test_data, norm_stats, batch_size, device):
    X, Y = test_data
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            residual = model(x_batch)
            y_pred = x_batch - residual
            all_preds.append(y_pred.cpu().numpy())
            
    Y_pred = np.concatenate(all_preds, axis=0) # (N, 2, 64, 64)
    Y_true = Y  
    
    # Restore original scale
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    
    Y_pred = Y_pred * std + mean
    Y_true = Y_true * std + mean
    
    # Convert to complex form
    H_pred = Y_pred[:, 0, :, :] + 1j * Y_pred[:, 1, :, :]
    H_true = Y_true[:, 0, :, :] + 1j * Y_true[:, 1, :, :]
    
    # Calculate NMSE
    nmse_num = np.linalg.norm(H_true - H_pred) ** 2
    nmse_den = np.linalg.norm(H_true) ** 2
    nmse = float(nmse_num / nmse_den)
    nmse_db = float(10 * np.log10(nmse))
    
    return nmse, nmse_db

# -------------------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Federated ResUNet with FedAvg")
    parser.add_argument("--rounds", type=int, default=15, help="Global federated rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local epochs per client")
    parser.add_argument("--batch", type=int, default=16, help="Local batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Local learning rate")
    args = parser.parse_args()

    # Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"\n[INFO] Starting Federated ResUNet Training (FedAvg) on {device}")
    
    # Prepare Data
    client_data, test_data, norm_stats = load_and_split_data(args.clients)
    
    # Initialize Global Model
    global_model = ResUNet()
    
    print("\n=========================================")
    print(f" FEDERATED TRAINING SETTINGS:")
    print(f" Global Rounds  : {args.rounds}")
    print(f" Clients        : {args.clients}")
    print(f" Local Epochs   : {args.local_epochs}")
    print(f" Local Batch    : {args.batch}")
    print(f" Learning Rate  : {args.lr}")
    print("=========================================")
    
    for round_idx in range(1, args.rounds + 1):
        print(f"\n[Round {round_idx}/{args.rounds}]")
        client_weights = []
        client_sizes = []
        
        # Local Client Training
        for i in range(args.clients):
            print(f"  -> Training Client {i+1} locally...")
            weights = client_update(global_model, client_data[i], args.local_epochs, args.batch, args.lr, device)
            client_weights.append(weights)
            client_sizes.append(len(client_data[i][0]))
            
        # FedAvg Aggregation
        print(f"  -> Aggregating Client Weights (FedAvg)...")
        updated_weights = fedavg(client_weights, client_sizes)
        global_model.load_state_dict(updated_weights)
        
        # Evaluation
        global_model.to(device)
        nmse, nmse_db = evaluate(global_model, test_data, norm_stats, args.batch, device)
        global_model.to("cpu") # save VRAM memory off cycle
        
        print(f"Round {round_idx} -> NMSE = {nmse_db:.2f} dB")
        
    print("\n-----------------------------------------")
    print("FEDERATED TEST RESULTS:")
    print(f"Samples tested: {len(test_data[0])}")
    print(f"NMSE (linear): {nmse:.6f}")
    print(f"NMSE (dB): {nmse_db:.2f} dB")
    print("-----------------------------------------")
    
    save_path = "fed_resunet_model.pth"
    torch.save(global_model.state_dict(), save_path)
    print(f"\n[✓] Final global model saved as: {save_path}")

if __name__ == "__main__":
    main()
