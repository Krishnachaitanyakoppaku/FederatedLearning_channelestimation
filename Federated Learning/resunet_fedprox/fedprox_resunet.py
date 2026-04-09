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
# BUILDING BLOCKS & MODEL ARCHITECTURE (ResUNet)
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
        
        # Final Layer (no activation)
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
        c1 = torch.cat([e2, u1], dim=1) # skip
        d1 = self.dec1(c1)
        
        u2 = self.upconv2(d1)
        c2 = torch.cat([e1, u2], dim=1) # skip
        d2 = self.dec2(c2)
        
        out = self.final_conv(d2)
        return out

# -------------------------------------------------------------
# DATA LOADING & SPLITTING (IID & Non-IID)
# -------------------------------------------------------------
def load_and_split_data(num_clients, non_iid=False):
    """
    Load dataset and distribute across clients.
    
    Non-IID strategy (moderate skew):
      - Sort samples by channel norm
      - Each client gets 60% from a unique contiguous block (biased)
      - Each client gets 40% sampled uniformly from the full pool (shared)
      This creates heterogeneity without completely destroying diversity.
    """
    print("[INFO] Loading raw dataset...")
    X_path = os.path.join(DATASET_DIR, "H_noisy_10dB_ri.npy")
    Y_path = os.path.join(DATASET_DIR, "H_clean_ri.npy")

    if not (os.path.exists(X_path) and os.path.exists(Y_path)):
        raise FileNotFoundError(
            f"Required dataset files not found under {DATASET_DIR}. "
            "Run data generation/preparation first."
        )
        
    X = np.load(X_path).astype(np.float32)
    Y = np.load(Y_path).astype(np.float32)
    
    print(f"[INFO] Loaded Dataset. Shape: {X.shape}")
    
    # Global Normalization
    global_mean = np.mean(np.concatenate([X, Y], axis=0)).astype(np.float32)
    global_std = np.std(np.concatenate([X, Y], axis=0)).astype(np.float32)
    
    X = (X - global_mean) / global_std
    Y = (Y - global_mean) / global_std
    
    norm_stats = {"mean": float(global_mean), "std": float(global_std)}
    
    # 80/20 train/test split with fixed seed
    n_samples = X.shape[0]
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_samples)
    
    split_idx = int(n_samples * 0.8)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    n_train = len(X_train)
    chunk_size = n_train // num_clients
    
    if non_iid:
        print("[INFO] Creating Non-IID data split (moderate skew: 60% unique + 40% shared)...")
        
        # Sort by channel norm to create ordered blocks
        norms = np.linalg.norm(Y_train.reshape(n_train, -1), axis=1)
        sort_idx = np.argsort(norms)
        X_sorted = X_train[sort_idx]
        Y_sorted = Y_train[sort_idx]
        
        # Each client: 60% unique (contiguous sorted block) + 40% shared (random from full pool)
        unique_ratio = 0.6
        shared_ratio = 0.4
        n_unique = int(chunk_size * unique_ratio)
        n_shared = chunk_size - n_unique
        
        client_data = []
        rng_split = np.random.default_rng(123)
        
        for i in range(num_clients):
            # Unique block: contiguous segment from sorted data
            unique_start = i * (n_train // num_clients)
            unique_end = unique_start + n_unique
            X_unique = X_sorted[unique_start:unique_end]
            Y_unique = Y_sorted[unique_start:unique_end]
            
            # Shared pool: random samples from the entire training set
            shared_idx = rng_split.choice(n_train, size=n_shared, replace=False)
            X_shared = X_train[shared_idx]
            Y_shared = Y_train[shared_idx]
            
            # Combine and shuffle
            X_client = np.concatenate([X_unique, X_shared], axis=0)
            Y_client = np.concatenate([Y_unique, Y_shared], axis=0)
            shuffle_idx = rng_split.permutation(len(X_client))
            X_client = X_client[shuffle_idx]
            Y_client = Y_client[shuffle_idx]
            
            client_data.append((X_client, Y_client))
            print(f"       Client {i+1}: {n_unique} unique + {n_shared} shared = {len(X_client)} total")
    else:
        print("[INFO] Creating IID data split (Random uniform allocation)...")
        client_data = []
        for i in range(num_clients):
            start = i * chunk_size
            end = start + chunk_size
            client_data.append((X_train[start:end], Y_train[start:end]))
        
    print(f"[INFO] Train: {n_train} samples across {num_clients} clients, Test: {len(X_test)} samples")
    return client_data, (X_test, Y_test), norm_stats

# -------------------------------------------------------------
# FEDERATED LOGIC: FEDPROX
# -------------------------------------------------------------
def fedavg_aggregate(client_weights_list, client_data_sizes):
    """ Weighted average of client weights (Same as FedAvg) """
    N_total = sum(client_data_sizes)
    global_weights = copy.deepcopy(client_weights_list[0])
    
    weight_factor = client_data_sizes[0] / N_total
    for key in global_weights.keys():
        global_weights[key] = global_weights[key].float() * weight_factor
        
    for i in range(1, len(client_weights_list)):
        weight_factor = client_data_sizes[i] / N_total
        for key in global_weights.keys():
            global_weights[key] += client_weights_list[i][key].float() * weight_factor
            
    return global_weights

def client_update_fedprox(global_model, client_data, epochs, batch_size, lr, mu, device):
    """
    Local client training with FedProx proximal term.
    
    Loss = MSE(output, target) + (mu / 2) * sum_params( ||w_local - w_global||^2 )
    
    CRITICAL: global parameters are .detach()'ed to prevent gradient flow back.
    """
    X, Y = client_data
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Deep copy global model for local training
    model = copy.deepcopy(global_model).to(device)
    model.train()
    
    # Freeze global weights on device — .detach() is CRITICAL
    global_weight_list = [p.detach().clone().to(device) for p in global_model.parameters()]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Residual Learning: H_pred = H_noisy - model(H_noisy)
            residual = model(x_batch)
            output = x_batch - residual
            
            mse_loss = criterion(output, y_batch)
            
            # FedProx proximal term: (mu / 2) * sum( ||w_local - w_global||^2 )
            prox_term = 0.0
            if mu > 0.0:
                for local_param, global_param in zip(model.parameters(), global_weight_list):
                    prox_term += ((local_param - global_param) ** 2).sum()
            
            total_loss = mse_loss + (mu / 2.0) * prox_term
            
            total_loss.backward()
            optimizer.step()
            
    # Free VRAM
    del global_weight_list
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    # Return weights on CPU for aggregation
    model.to("cpu")
    return model.state_dict()

# -------------------------------------------------------------
# EVALUATION (NMSE)
# -------------------------------------------------------------
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
            
    Y_pred = np.concatenate(all_preds, axis=0)
    Y_true = Y  
    
    # Denormalize
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    Y_pred = Y_pred * std + mean
    Y_true = Y_true * std + mean
    
    # Complex form
    H_pred = Y_pred[:, 0, :, :] + 1j * Y_pred[:, 1, :, :]
    H_true = Y_true[:, 0, :, :] + 1j * Y_true[:, 1, :, :]
    
    # NMSE
    nmse_num = np.linalg.norm(H_true - H_pred) ** 2
    nmse_den = np.linalg.norm(H_true) ** 2
    nmse = float(nmse_num / nmse_den)
    nmse_db = float(10 * np.log10(nmse))
    
    return nmse, nmse_db

# -------------------------------------------------------------
# MAIN SERVER LOOP
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Federated ResUNet with FedProx")
    parser.add_argument("--rounds", "--epochs", type=int, default=15, help="Global federated rounds", dest="rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local epochs per client")
    parser.add_argument("--batch", type=int, default=16, help="Local batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Local learning rate")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal coefficient (mu=0 is FedAvg)")
    parser.add_argument("--non_iid", action="store_true", help="Use Non-IID data distribution")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"\n[INFO] Initializing FedProx on {device}")
    
    # Data Distribution
    client_data, test_data, norm_stats = load_and_split_data(args.clients, non_iid=args.non_iid)
    
    global_model = ResUNet()
    
    mode_str = "Non-IID" if args.non_iid else "IID"
    print(f"\n=========================================")
    print(f" FEDPROX TRAINING CONFIGURATION")
    print(f" Data Mode      : {mode_str}")
    print(f" Global Rounds  : {args.rounds}")
    print(f" Clients        : {args.clients}")
    print(f" Local Epochs   : {args.local_epochs}")
    print(f" Batch Size     : {args.batch}")
    print(f" Learning Rate  : {args.lr}")
    print(f" Proximal (mu)  : {args.mu}")
    print(f"=========================================")
    
    best_nmse_db = float("inf")
    best_nmse = None
    
    for round_idx in range(1, args.rounds + 1):
        client_weights = []
        client_sizes = []
        
        # Sequential client training to optimize memory
        for i in range(args.clients):
            print(f"Round {round_idx} | Client {i+1}/{args.clients} training...")
            weights = client_update_fedprox(
                global_model, client_data[i],
                args.local_epochs, args.batch, args.lr, args.mu, device
            )
            client_weights.append(weights)
            client_sizes.append(len(client_data[i][0]))
            
        # FedAvg Aggregation Rule (used in FedProx too)
        updated_weights = fedavg_aggregate(client_weights, client_sizes)
        global_model.load_state_dict(updated_weights)
        
        # Evaluate global model
        global_model.to(device)
        nmse, nmse_db = evaluate(global_model, test_data, norm_stats, args.batch, device)
        global_model.to("cpu")
        
        if nmse_db < best_nmse_db:
            best_nmse_db = nmse_db
            best_nmse = nmse
        
        print(f"Round {round_idx} → NMSE = {nmse_db:.2f} dB\n")
        
    print("-----------------------------------------")
    print(f"FEDPROX ({mode_str}) TEST RESULTS:")
    print(f"Samples tested: {len(test_data[0])}")
    if best_nmse is None:
        best_nmse = nmse

    print(f"NMSE (linear): {nmse:.6f}")
    print(f"NMSE (dB): {nmse_db:.2f} dB")
    print(f"Best NMSE (dB): {best_nmse_db:.2f} dB")
    print("-----------------------------------------")
    
    iid_str = "non_iid" if args.non_iid else "iid"
    save_path = f"fedprox_resunet_{iid_str}_mu{args.mu}.pth"
    torch.save(global_model.state_dict(), save_path)
    print(f"\n[✓] Global model saved: {save_path}")

if __name__ == "__main__":
    main()
