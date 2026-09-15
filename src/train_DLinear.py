import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.db_DLinear import DLinear
from src.data_processing import load_and_split_data
from src.utils_file import plot_loss_curve, save_model_weights
from tqdm import tqdm


def train():
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 31

    data_folder = "../data/cleaned_test01"
    train_loader, val_loader, test_loader = load_and_split_data(data_folder, seq_length=10, batch_size=batch_size)

    class Config:
        def __init__(self):
            self.seq_len = 10
            self.pred_len = 10
            self.enc_in = 23
            self.c_out = 7
            self.batch_size = batch_size
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    configs = Config()
    model = DLinear(configs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    save_dir = "../checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    param_info = (
        f"db_DLinear_seq_len_{configs.seq_len}_pred_len_{configs.pred_len}_enc_in_{configs.enc_in}_"
        f"ep_{num_epochs}_lr_{learning_rate}"
    )
    save_path = os.path.join(save_dir, param_info)
    os.makedirs(save_path, exist_ok=True)

    loss_file = os.path.join(save_path, 'losses.txt')
    with open(loss_file, 'w') as f:
        f.write("Epoch,Train Loss,Validation Loss\n")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = []

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch") as train_bar:
            for input_data, target_data, _, _ in train_bar:
                input_data, target_data = input_data.to(device), target_data.to(device)

                optimizer.zero_grad()

                output = model(input_data)

                target_data = target_data[:, :configs.pred_len, :]

                loss = criterion(output, target_data)
                loss.backward()
                optimizer.step()

                train_loss_epoch.append(loss.item())
                train_bar.set_postfix(loss=loss.item())

        train_losses.append(sum(train_loss_epoch) / len(train_loss_epoch))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", unit="batch") as val_bar:
                for input_data, target_data, _, _ in val_bar:
                    input_data, target_data = input_data.to(device), target_data.to(device)

                    output = model(input_data)
                    loss = criterion(output, target_data[:, :configs.pred_len, :])
                    val_loss += loss.item()
                    val_bar.set_postfix(val_loss=loss.item())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.8f}, Val Loss: {val_loss:.8f}")

        with open(loss_file, 'a') as f:
            f.write(f"{epoch + 1},{train_losses[-1]:.8f},{val_loss:.8f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))

        save_model_weights(model, os.path.join(save_path, f"weights_epoch_{epoch + 1:02d}.pt"))

    plot_loss_curve(train_losses, val_losses, os.path.join(save_path, 'loss_curve.png'))


if __name__ == '__main__':
    train()