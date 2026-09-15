import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.db_Autoformer import Model_Autoformer
from src.data_processing import load_and_split_data
from src.utils_file import plot_loss_curve, save_model_weights
from tqdm import tqdm


def train_autoformer():
    batch_size = 1024
    learning_rate = 0.1
    num_epochs = 200

    data_folder = "../data/cleaned_test01"
    train_loader, val_loader, test_loader = load_and_split_data(data_folder, seq_length=10, batch_size=batch_size)

    class Config:
        def __init__(self):
            self.enc_in = 23
            self.d_model = 256
            self.embed = 256
            self.freq = 32
            self.dropout = 0.1
            self.dec_in = 23
            self.c_out = 7
            self.pred_len = 10
            self.output_attention = False
            self.factor = 4
            self.n_heads = 8
            self.e_layers = 6
            self.d_layers = 6
            self.d_ff = 1024
            self.activation = 'gelu'
            self.batch_size = batch_size
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.seq_length = 10
    configs = Config()

    model = Model_Autoformer(configs)

    device = configs.device
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    save_dir = "../checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    param_info = (
        f"db_autoformer_enc_in_{configs.enc_in}_d_model_{configs.d_model}_d_ff_{configs.d_ff}_ep_{num_epochs}_"
        f"e_layers_{configs.e_layers}_d_layers_{configs.d_layers}_embed_{configs.embed}_lr_{learning_rate}"
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

                x_dec = torch.zeros(input_data.size(0), configs.pred_len, configs.dec_in).to(device)

                optimizer.zero_grad()

                output = model(input_data, x_dec)

                target_data = target_data[:, :configs.pred_len, :]

                loss = criterion(output, target_data)
                loss.backward()
                optimizer.step()

                train_loss_epoch.append(loss.item())
                train_bar.set_postfix(loss=loss.item())

        avg_train_loss = sum(train_loss_epoch) / len(train_loss_epoch)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", unit="batch") as val_bar:
                for input_data, target_data, _, _ in val_bar:
                    input_data, target_data = input_data.to(device), target_data.to(device)

                    x_dec = torch.zeros(input_data.size(0), configs.pred_len, configs.dec_in).to(device)
                    output = model(input_data, x_dec)

                    loss = criterion(output, target_data[:, :configs.pred_len, :])
                    val_loss += loss.item()
                    val_bar.set_postfix(val_loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")
        with open(loss_file, 'a') as f:
            f.write(f"{epoch + 1},{avg_train_loss:.8f},{avg_val_loss:.8f}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))

        save_model_weights(model, os.path.join(save_path, f"weights_epoch_{epoch + 1:02d}.pt"))

    plot_loss_curve(train_losses, val_losses, os.path.join(save_path, 'loss_curve.png'))


if __name__ == '__main__':
    train_autoformer()
