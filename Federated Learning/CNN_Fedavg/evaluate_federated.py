import torch
import numpy as np
from torch.utils.data import DataLoader
from cnn_model import ChannelEstimatorCNN, ChannelDataset

def compute_nmse(H_true_np, H_pred_np):
    """
    Computes NMSE (Normalized Mean Square Error)
    NMSE = ||H_true - H_pred||^2 / ||H_true||^2
    """
    # H_true_np shape: (N, 64, 64) complex numbers
    error = H_true_np - H_pred_np
    
    mse_per_sample = np.linalg.norm(error.reshape(error.shape[0], -1), axis=1)**2
    power_per_sample = np.linalg.norm(H_true_np.reshape(H_true_np.shape[0], -1), axis=1)**2
    
    nmse_linear = np.mean(mse_per_sample / power_per_sample)
    nmse_db = 10 * np.log10(nmse_linear)
    
    return nmse_linear, nmse_db

def evaluate(model_path, x_test_path, y_test_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    model = ChannelEstimatorCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    dataset = ChannelDataset(x_test_path, y_test_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_preds_real = []
    all_preds_imag = []
    
    all_true_real = []
    all_true_imag = []
    
    print("Running inference...")
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_pred = model(X_batch).cpu().numpy()
            Y_true = Y_batch.numpy()
            
            # Predict
            # Separate real and imaginary for both prediction and ground truth
            # Channel 0 is real, channel 1 is imag
            all_preds_real.append(Y_pred[:, 0, :, :])
            all_preds_imag.append(Y_pred[:, 1, :, :])
            
            all_true_real.append(Y_true[:, 0, :, :])
            all_true_imag.append(Y_true[:, 1, :, :])
            
    # Concatenate results
    preds_real = np.concatenate(all_preds_real, axis=0)
    preds_imag = np.concatenate(all_preds_imag, axis=0)
    
    true_real = np.concatenate(all_true_real, axis=0)
    true_imag = np.concatenate(all_true_imag, axis=0)
    
    # Convert to complex arrays for standard NMSE evaluation (2 channels -> complex form)
    H_pred_complex = preds_real + 1j * preds_imag
    H_true_complex = true_real + 1j * true_imag
    
    print("Computing NMSE...")
    nmse_lin, nmse_db = compute_nmse(H_true_complex, H_pred_complex)
    
    print(f"\nFinal Evaluation Results:")
    print(f"-------------------------")
    print(f"NMSE (linear): {nmse_lin:.6f}")
    print(f"NMSE (dB):     {nmse_db:.2f} dB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='fed_model.pth', help='Path to trained federated model')
    parser.add_argument('--x_test', type=str, default='../../dataset/cnn/10dB/X_test.npy', help='Path to test noisy channel')
    parser.add_argument('--y_test', type=str, default='../../dataset/cnn/10dB/Y_test.npy', help='Path to test clean channel')
    
    args = parser.parse_args()
    
    evaluate(args.model, args.x_test, args.y_test)
