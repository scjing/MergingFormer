import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, test_loader, configs, save_path):
    model.eval()

    mse_list, rmse_list, mae_list, mape_list, r2_list = [], [], [], [], []
    mse_speed_list, rmse_speed_list, mae_speed_list, mape_speed_list, r2_speed_list = [], [], [], [], []
    mse_speedY_list, rmse_speedY_list, mae_speedY_list, mape_speedY_list, r2_speedY_list = [], [], [], [], []
    criterion = torch.nn.MSELoss()

    device = configs.device

    with torch.no_grad():
        for input_data, target_data, _, _ in test_loader:
            input_data, target_data = input_data.to(device), target_data.to(device)

            x_dec = torch.zeros(input_data.size(0), configs.pred_len, configs.enc_in).to(device)

            output = model(input_data, x_dec)

            target_data = target_data[:, :configs.pred_len, :]

            output_np = output.cpu().numpy()
            target_np = target_data.cpu().numpy()

            for feature_idx in range(output_np.shape[-1]):
                y_pred = output_np[:, :, feature_idx].flatten()
                y_true = target_np[:, :, feature_idx].flatten()

                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
                r2 = r2_score(y_true, y_pred)

                mse_list.append(mse)
                rmse_list.append(rmse)
                mae_list.append(mae)
                mape_list.append(mape)
                r2_list.append(r2)

            for idx in [5]:
                y_pred = output_np[:, :, idx].flatten()
                y_true = target_np[:, :, idx].flatten()

                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
                r2 = r2_score(y_true, y_pred)

                mse_speed_list.append(mse)
                rmse_speed_list.append(rmse)
                mae_speed_list.append(mae)
                mape_speed_list.append(mape)
                r2_speed_list.append(r2)

            for idx in [4]:
                y_pred = output_np[:, :, idx].flatten()
                y_true = target_np[:, :, idx].flatten()

                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
                r2 = r2_score(y_true, y_pred)

                mse_speedY_list.append(mse)
                rmse_speedY_list.append(rmse)
                mae_speedY_list.append(mae)
                mape_speedY_list.append(mape)
                r2_speedY_list.append(r2)

    avg_mse = np.mean(mse_list)
    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    avg_mape = np.mean(mape_list)
    avg_r2 = np.mean(r2_list)

    avg_mse_speed = np.mean(mse_speed_list)
    avg_rmse_speed = np.mean(rmse_speed_list)
    avg_mae_speed = np.mean(mae_speed_list)
    avg_mape_speed = np.mean(mape_speed_list)
    avg_r2_speed = np.mean(r2_speed_list)

    avg_mse_speedY = np.mean(mse_speedY_list)
    avg_rmse_speedY = np.mean(rmse_speedY_list)
    avg_mae_speedY = np.mean(mae_speedY_list)
    avg_mape_speedY = np.mean(mape_speedY_list)
    avg_r2_speedY = np.mean(r2_speedY_list)

    print(f"Evaluation Results:")
    print(f"  MSE: {avg_mse:.8f}")
    print(f"  RMSE: {avg_rmse:.8f}")
    print(f"  MAE: {avg_mae:.8f}")
    print(f"  MAPE: {avg_mape:.8f}")
    print(f"  R^2: {avg_r2:.8f}")

    print(f"\nSpeed Evaluation Results:")
    print(f"  MSE: {avg_mse_speed:.8f}")
    print(f"  RMSE: {avg_rmse_speed:.8f}")
    print(f"  MAE: {avg_mae_speed:.8f}")
    print(f"  MAPE: {avg_mape_speed:.8f}")
    print(f"  R^2: {avg_r2_speed:.8f}")

    print(f"\nSpeedY Evaluation Results:")
    print(f"  MSE: {avg_mse_speedY:.8f}")
    print(f"  RMSE: {avg_rmse_speedY:.8f}")
    print(f"  MAE: {avg_mae_speedY:.8f}")
    print(f"  MAPE: {avg_mape_speedY:.8f}")
    print(f"  R^2: {avg_r2_speedY:.8f}")

    eval_file = os.path.join(save_path, 'evaluation_results.txt')
    with open(eval_file, 'w') as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"MSE: {avg_mse:.8f}\n")
        f.write(f"RMSE: {avg_rmse:.8f}\n")
        f.write(f"MAE: {avg_mae:.8f}\n")
        f.write(f"MAPE: {avg_mape:.8f}\n")
        f.write(f"R^2: {avg_r2:.8f}\n")

        f.write(f"\nSpeed Evaluation Metrics:\n")
        f.write(f"MSE: {avg_mse_speed:.8f}\n")
        f.write(f"RMSE: {avg_rmse_speed:.8f}\n")
        f.write(f"MAE: {avg_mae_speed:.8f}\n")
        f.write(f"MAPE: {avg_mape_speed:.8f}\n")
        f.write(f"R^2: {avg_r2_speed:.8f}\n")

        f.write(f"\nSpeedY Evaluation Metrics:\n")
        f.write(f"MSE: {avg_mse_speedY:.8f}\n")
        f.write(f"RMSE: {avg_rmse_speedY:.8f}\n")
        f.write(f"MAE: {avg_mae_speedY:.8f}\n")
        f.write(f"MAPE: {avg_mape_speedY:.8f}\n")
        f.write(f"R^2: {avg_r2_speedY:.8f}\n")

    return avg_mse, avg_rmse, avg_mae, avg_mape, avg_r2, avg_mse_speed, avg_rmse_speed, avg_mae_speed, avg_mape_speed, avg_r2_speed, avg_mse_speedY, avg_rmse_speedY, avg_mae_speedY, avg_mape_speedY, avg_r2_speedY


if __name__ == '__main__':
    import argparse
    from models.MyTransformer import Model
    from src.data_processing import load_and_split_data

    parser = argparse.ArgumentParser(description="Evaluate the Transformer model and visualize results.")
    parser.add_argument('--data_folder', type=str, default="../data/111cleaned_test01",
                        help="Path to the folder containing the preprocessed data.")
    parser.add_argument('--config_name', type=str, default="enc_in_23_d_model_128_d_ff_1024_ep_30_e_layers_6_d_layers_6_embed_64_lr_0.01",
                        help="Name of the model configuration folder.")
    parser.add_argument('--specific_file', type=str, default="car_75_data.csv",
                        help="Name of the specific file for visualization.")
    args = parser.parse_args()


    class Config:
        def __init__(self):
            self.enc_in = 23
            self.d_model = 128
            self.embed = 64
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
            self.batch_size = 32
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.seq_length = 10


    configs = Config()

    model = Model(configs)
    model_path = os.path.join("../checkpoints", args.config_name, "best_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(configs.device)

    train_loader, val_loader, test_loader = load_and_split_data(
        args.data_folder,
        seq_length=configs.seq_length,
        batch_size=configs.batch_size
    )

    save_path = os.path.join("../checkpoints", args.config_name)
    os.makedirs(save_path, exist_ok=True)

    print("Starting evaluation...")
    evaluate_model(model, test_loader, configs, save_path)
    print("Evaluation completed.")