import numpy as np
import os
import argparse

def split_data(x_path, y_path, num_clients=5, iid=True, output_dir='clients'):
    """
    Split the training data into K clients.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {x_path} and {y_path}...")
    X = np.load(x_path)
    Y = np.load(y_path)
    
    assert len(X) == len(Y), "X and Y must have the same number of samples"
    
    num_samples = len(X)
    samples_per_client = num_samples // num_clients
    
    # We will shuffle the dataset for IID or sort for NON-IID
    indices = np.arange(num_samples)
    
    if iid:
        print("Performing IID split...")
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
    else:
        print("Performing NON-IID split (sorting by average magnitude to simulate SNR/label skew)...")
        # Simulating label skew by sorting by the structural magnitude of clean channel
        mags = np.linalg.norm(Y.reshape(num_samples, -1), axis=1)
        indices = np.argsort(mags)
        
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        if i == num_clients - 1:
            end_idx = num_samples  # Give the rest to the last client if not exactly divisible
            
        client_indices = indices[start_idx:end_idx]
        
        X_client = X[client_indices]
        Y_client = Y[client_indices]
        
        x_save_path = os.path.join(output_dir, f'X_client_{i}.npy')
        y_save_path = os.path.join(output_dir, f'Y_client_{i}.npy')
        
        np.save(x_save_path, X_client)
        np.save(y_save_path, Y_client)
        
        print(f"Client {i}: Saved {len(client_indices)} samples to {x_save_path} and {y_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data for Federated Learning")
    parser.add_argument('--x_path', type=str, default='../../dataset/cnn/10dB/X_train.npy', help='Path to noisy training data (default 10dB/X_train)')
    parser.add_argument('--y_path', type=str, default='../../dataset/cnn/10dB/Y_train.npy', help='Path to clean training data (default 10dB/Y_train)')
    parser.add_argument('--clients', type=int, default=5, help='Number of clients')
    parser.add_argument('--non_iid', action='store_true', help='Use Non-IID splitting')
    parser.add_argument('--output_dir', type=str, default='clients', help='Output directory for client data')
    
    args = parser.parse_args()
    
    split_data(args.x_path, args.y_path, num_clients=args.clients, iid=not args.non_iid, output_dir=args.output_dir)
