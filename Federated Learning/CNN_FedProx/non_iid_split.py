import argparse
import os

import numpy as np

def create_non_iid_data(snr: int, num_clients: int, samples_per_client: int, output_dir: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "dataset", "cnn"))
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)  # For reproducibility

    def load_snr(snr_level):
        x_path = os.path.join(base_dir, f"{snr_level}dB/X_train.npy")
        y_path = os.path.join(base_dir, f"{snr_level}dB/Y_train.npy")
        x = np.load(x_path)
        y = np.load(y_path)
        # Shuffle internally
        indices = np.random.permutation(len(x))
        return x[indices], y[indices]

    print("Loading datasets...")
    if num_clients <= 0:
        raise ValueError("--clients must be > 0")
    if samples_per_client <= 0:
        raise ValueError("--samples_per_client must be > 0")

    source_snrs = [0, 5, 10, 15, 20]
    datasets = [load_snr(level) for level in source_snrs]

    clients = []
    for client_id in range(num_clients):
        source_idx = client_id % len(datasets)
        x_src, y_src = datasets[source_idx]

        if len(x_src) < samples_per_client:
            raise ValueError(
                f"Requested {samples_per_client} samples per client but only {len(x_src)} "
                f"available in source SNR {source_snrs[source_idx]}dB"
            )
        clients.append((x_src[:samples_per_client], y_src[:samples_per_client]))
    
    print("Saving client datasets...")
    for i, (cx, cy) in enumerate(clients):
        # Save exact non-iid sets
        np.save(os.path.join(output_dir, f"X_client_{i}.npy"), cx)
        np.save(os.path.join(output_dir, f"Y_client_{i}.npy"), cy)
        print(f"Client {i} saved: {len(cx)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create non-IID client datasets")
    parser.add_argument("--snr", type=int, default=10, help="Primary run SNR (metadata only)")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--samples_per_client", type=int, default=960, help="Samples per client")
    parser.add_argument("--output_dir", type=str, default="clients_non_iid", help="Output directory")
    args = parser.parse_args()

    create_non_iid_data(
        snr=args.snr,
        num_clients=args.clients,
        samples_per_client=args.samples_per_client,
        output_dir=args.output_dir,
    )
