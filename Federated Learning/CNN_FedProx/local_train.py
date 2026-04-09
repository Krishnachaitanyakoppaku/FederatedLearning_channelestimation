import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cnn_model import ChannelDataset
import copy

def train_local_model(model, client_id, x_path, y_path, global_weights,
                      epochs=2, batch_size=32, lr=0.001, device='cpu', 
                      use_fedprox=False, mu=0.01):
    """
    Train a model locally on one client's data for a few epochs.
    Allows standard FedAvg training as well as FedProx.
    """
    print(f"--- Client {client_id} Local Training ---")

    # Load client data
    try:
        dataset = ChannelDataset(x_path, y_path)
    except Exception as e:
        print(f"Error loading data for client {client_id}: {e}")
        return {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    n_samples = len(dataset)

    # Move global weights to device if using FedProx
    if use_fedprox and global_weights is not None:
        global_weights_tensor = {k: v.to(device) for k, v in global_weights.items()}

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)

            if use_fedprox and global_weights is not None:
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    if name in global_weights_tensor:
                        proximal_term += torch.norm(param - global_weights_tensor[name])**2
                loss += mu * proximal_term

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"  Client {client_id} | Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

    # Return updated weights on CPU
    local_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return local_weights, n_samples
