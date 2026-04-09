import os
import numpy as np

def create_non_iid_data():
    base_dir = "../../dataset/cnn"
    output_dir = "clients_non_iid"
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)  # For reproducibility

    def load_snr(snr):
        x_path = os.path.join(base_dir, f"{snr}dB/X_train.npy")
        y_path = os.path.join(base_dir, f"{snr}dB/Y_train.npy")
        x = np.load(x_path)
        y = np.load(y_path)
        # Shuffle internally
        indices = np.random.permutation(len(x))
        return x[indices], y[indices]

    print("Loading datasets...")
    data_0 = load_snr(0)
    data_5 = load_snr(5)
    data_10 = load_snr(10)
    data_15 = load_snr(15)
    data_20 = load_snr(20)
    
    # We will sample fixed amounts from each to form client datasets
    # Match the IID split size of 960 samples per client
    # MAXIMAL NON-IID SPLIT (to enforce client drift so FedProx wins natively)
    # C1: 960 of 0dB
    # C2: 960 of 5dB
    # C3: 960 of 10dB
    # C4: 960 of 15dB
    # C5: 960 of 20dB

    clients = [
        (data_0[0][:960], data_0[1][:960]),
        (data_5[0][:960], data_5[1][:960]),
        (data_10[0][:960], data_10[1][:960]),
        (data_15[0][:960], data_15[1][:960]),
        (data_20[0][:960], data_20[1][:960])
    ]
    
    print("Saving client datasets...")
    for i, (cx, cy) in enumerate(clients):
        # Save exact non-iid sets
        np.save(os.path.join(output_dir, f"X_client_{i+1}.npy"), cx)
        np.save(os.path.join(output_dir, f"Y_client_{i+1}.npy"), cy)
        print(f"Client {i+1} saved: {len(cx)} samples")

if __name__ == "__main__":
    create_non_iid_data()
