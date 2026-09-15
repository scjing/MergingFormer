import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse



def compute_ADE_FDE(pred, actual):
    """
    计算 ADE 和 FDE
    :param pred: 预测值序列（list 或 array）
    :param actual: 实际值序列（list 或 array）
    :return: ade, fde
    """
    pred = np.array(pred)
    actual = np.array(actual)
    min_len = min(len(pred), len(actual))
    pred = pred[:min_len]
    actual = actual[:min_len]

    ade = np.mean(np.abs(pred - actual))
    fde = np.abs(pred[-1] - actual[-1])
    return ade, fde



def evaluate_model(model, test_loader, configs, save_path):
    model.eval()

    mse_list, rmse_list, mae_list, mape_list, r2_list = [], [], [], [], []
    mse_speed_list, rmse_speed_list, mae_speed_list, mape_speed_list, r2_speed_list = [], [], [], [], []
    mse_speedY_list, rmse_speedY_list, mae_speedY_list, mape_speedY_list, r2_speedY_list = [], [], [], [], []

    ade_list, fde_list = [], []
    ade_speed_list, fde_speed_list = [], []
    ade_speedY_list, fde_speedY_list = [], []

    criterion = torch.nn.MSELoss()
    device = configs.device

    with torch.no_grad():
        for input_data, target_data, _, _ in test_loader:
            input_data, target_data = input_data.to(device), target_data.to(device)

            x_dec = torch.zeros(input_data.size(0), configs.pred_len, configs.dec_in).to(device)

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

                ade_samples = []
                fde_samples = []
                for sample_idx in range(output_np.shape[0]):
                    pred_seq = output_np[sample_idx, :, feature_idx]
                    true_seq = target_np[sample_idx, :, feature_idx]
                    ade, fde = compute_ADE_FDE(pred_seq, true_seq)
                    ade_samples.append(ade)
                    fde_samples.append(fde)
                ade_list.append(np.mean(ade_samples))
                fde_list.append(np.mean(fde_samples))

            for idx in [1]:
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

                ade_samples = []
                fde_samples = []
                for sample_idx in range(output_np.shape[0]):
                    pred_seq = output_np[sample_idx, :, idx]
                    true_seq = target_np[sample_idx, :, idx]
                    ade, fde = compute_ADE_FDE(pred_seq, true_seq)
                    ade_samples.append(ade)
                    fde_samples.append(fde)
                ade_speed_list.append(np.mean(ade_samples))
                fde_speed_list.append(np.mean(fde_samples))

            for idx in [0]:
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

                ade_samples = []
                fde_samples = []
                for sample_idx in range(output_np.shape[0]):
                    pred_seq = output_np[sample_idx, :, idx]
                    true_seq = target_np[sample_idx, :, idx]
                    ade, fde = compute_ADE_FDE(pred_seq, true_seq)
                    ade_samples.append(ade)
                    fde_samples.append(fde)
                ade_speedY_list.append(np.mean(ade_samples))
                fde_speedY_list.append(np.mean(fde_samples))

    avg_mse = np.mean(mse_list)
    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    avg_mape = np.mean(mape_list)
    avg_r2 = np.mean(r2_list)
    avg_ade = np.mean(ade_list)
    avg_fde = np.mean(fde_list)

    avg_mse_speed = np.mean(mse_speed_list)
    avg_rmse_speed = np.mean(rmse_speed_list)
    avg_mae_speed = np.mean(mae_speed_list)
    avg_mape_speed = np.mean(mape_speed_list)
    avg_r2_speed = np.mean(r2_speed_list)
    avg_ade_speed = np.mean(ade_speed_list)
    avg_fde_speed = np.mean(fde_speed_list)

    avg_mse_speedY = np.mean(mse_speedY_list)
    avg_rmse_speedY = np.mean(rmse_speedY_list)
    avg_mae_speedY = np.mean(mae_speedY_list)
    avg_mape_speedY = np.mean(mape_speedY_list)
    avg_r2_speedY = np.mean(r2_speedY_list)
    avg_ade_speedY = np.mean(ade_speedY_list)
    avg_fde_speedY = np.mean(fde_speedY_list)

    print(f"Evaluation Results:")
    print(f"  MSE: {avg_mse:.8f}")
    print(f"  RMSE: {avg_rmse:.8f}")
    print(f"  MAE: {avg_mae:.8f}")
    print(f"  MAPE: {avg_mape:.8f}")
    print(f"  R^2: {avg_r2:.8f}")
    print(f"  ADE: {avg_ade:.8f}")
    print(f"  FDE: {avg_fde:.8f}")

    print(f"\nSpeed Evaluation Results:")
    print(f"  MSE: {avg_mse_speed:.8f}")
    print(f"  RMSE: {avg_rmse_speed:.8f}")
    print(f"  MAE: {avg_mae_speed:.8f}")
    print(f"  MAPE: {avg_mape_speed:.8f}")
    print(f"  R^2: {avg_r2_speed:.8f}")
    print(f"  ADE: {avg_ade_speed:.8f}")
    print(f"  FDE: {avg_fde_speed:.8f}")

    print(f"\nSpeedY Evaluation Results:")
    print(f"  MSE: {avg_mse_speedY:.8f}")
    print(f"  RMSE: {avg_rmse_speedY:.8f}")
    print(f"  MAE: {avg_mae_speedY:.8f}")
    print(f"  MAPE: {avg_mape_speedY:.8f}")
    print(f"  R^2: {avg_r2_speedY:.8f}")
    print(f"  ADE: {avg_ade_speedY:.8f}")
    print(f"  FDE: {avg_fde_speedY:.8f}")

    eval_file = os.path.join(save_path, 'evaluation_results.txt')
    with open(eval_file, 'w') as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"MSE: {avg_mse:.8f}\n")
        f.write(f"RMSE: {avg_rmse:.8f}\n")
        f.write(f"MAE: {avg_mae:.8f}\n")
        f.write(f"MAPE: {avg_mape:.8f}\n")
        f.write(f"R^2: {avg_r2:.8f}\n")
        f.write(f"ADE: {avg_ade:.8f}\n")
        f.write(f"FDE: {avg_fde:.8f}\n")

        f.write(f"\nSpeed Evaluation Metrics:\n")
        f.write(f"MSE: {avg_mse_speed:.8f}\n")
        f.write(f"RMSE: {avg_rmse_speed:.8f}\n")
        f.write(f"MAE: {avg_mae_speed:.8f}\n")
        f.write(f"MAPE: {avg_mape_speed:.8f}\n")
        f.write(f"R^2: {avg_r2_speed:.8f}\n")
        f.write(f"ADE: {avg_ade_speed:.8f}\n")
        f.write(f"FDE: {avg_fde_speed:.8f}\n")

        f.write(f"\nSpeedY Evaluation Metrics:\n")
        f.write(f"MSE: {avg_mse_speedY:.8f}\n")
        f.write(f"RMSE: {avg_rmse_speedY:.8f}\n")
        f.write(f"MAE: {avg_mae_speedY:.8f}\n")
        f.write(f"MAPE: {avg_mape_speedY:.8f}\n")
        f.write(f"R^2: {avg_r2_speedY:.8f}\n")
        f.write(f"ADE: {avg_ade_speedY:.8f}\n")
        f.write(f"FDE: {avg_fde_speedY:.8f}\n")

    return (avg_mse, avg_rmse, avg_mae, avg_mape, avg_r2,
            avg_ade, avg_fde,
            avg_mse_speed, avg_rmse_speed, avg_mae_speed, avg_mape_speed, avg_r2_speed,
            avg_ade_speed, avg_fde_speed,
            avg_mse_speedY, avg_rmse_speedY, avg_mae_speedY, avg_mape_speedY, avg_r2_speedY,
            avg_ade_speedY, avg_fde_speedY)


def evaluate_and_visualize_single_trajectory(csv_file_path, model, configs, save_path):
    """
    对单条轨迹数据进行评估并绘制 Speed 和 SpeedY 的真实值与预测值对比图。

    :param csv_file_path: 轨迹 CSV 文件路径
    :param model: 已训练的模型
    :param configs: 模型配置
    :param save_path: 图像保存路径
    """
    model.eval()
    device = configs.device

    event_data = pd.read_csv(csv_file_path)
    event_data.fillna(-1, inplace=True)

    input_columns = [
        'carCenterXft', 'carCenterYft', 'speedY', 'speed', 'laneId', 'pressure',
        'leadCarCenterXft', 'leadCarCenterYft', 'leadSpeed',
        'leftLeadCarCenterXft', 'leftLeadCarCenterYft', 'leftLeadSpeed',
        'leftFollowCarCenterXft', 'leftFollowCarCenterYft', 'leftFollowSpeed'
    ]
    target_columns = [
         'speedY', 'speed'
    ]

    input_array = event_data[input_columns].to_numpy()
    target_array = event_data[target_columns].to_numpy()

    input_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    input_array = input_scaler.fit_transform(input_array)
    target_array = target_scaler.fit_transform(target_array)

    seq_length = configs.seq_length
    predictions = []

    with torch.no_grad():
        for i in range(len(target_array) - seq_length):
            input_seq = input_array[i:i + seq_length]
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)

            x_dec = torch.zeros(1, configs.pred_len, configs.dec_in).to(device)

            output = model(input_tensor, x_dec)
            predicted_point = output[:, 0, :].cpu().numpy().flatten()
            predictions.append(predicted_point)

    predictions = np.array(predictions)

    predictions_restored = target_scaler.inverse_transform(predictions)
    targets_restored = target_scaler.inverse_transform(target_array[seq_length:])

    speed_index = 1
    speedY_index = 0

    predictions_speed = predictions_restored[:, speed_index]
    predictions_speedY = predictions_restored[:, speedY_index]
    targets_speed = targets_restored[:, speed_index]
    targets_speedY = targets_restored[:, speedY_index]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(predictions_speed, label='Predicted Speed', linestyle='-')
    plt.plot(targets_speed, label='True Speed', linestyle=':')
    plt.xlabel('Time Step')
    plt.ylabel('Speed')
    plt.legend()
    plt.title('Speed Prediction')
    plt.subplot(1, 2, 2)
    plt.plot(predictions_speedY, label='Predicted SpeedY', linestyle='-')
    plt.plot(targets_speedY, label='True SpeedY', linestyle=':')
    plt.xlabel('Time Step')
    plt.ylabel('SpeedY')
    plt.legend()
    plt.title('SpeedY Prediction')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'single_trajectory_evaluation.png'))
    print(f"Saved single trajectory evaluation plot to {save_path}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'single_trajectory_evaluation.png'))
    print(f"Saved single trajectory evaluation plot to {save_path}")

    txt_file = os.path.join(save_path, "speed_speedY_values.txt")
    with open(txt_file, 'w') as f:
        f.write("Speed Prediction vs True Values:\n")
        for idx, (pred, true) in enumerate(zip(predictions_speed, targets_speed)):
            f.write(f"Time Step {idx}: Predicted Speed: {pred:.4f}, True Speed: {true:.4f}\n")
        f.write("\nSpeedY Prediction vs True Values:\n")
        for idx, (pred, true) in enumerate(zip(predictions_speedY, targets_speedY)):
            f.write(f"Time Step {idx}: Predicted SpeedY: {pred:.4f}, True SpeedY: {true:.4f}\n")
    print(f"Saved speed and speedY values to {txt_file}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the LSTM-only model and visualize results.")
    parser.add_argument('--data_folder', type=str, default="../data/111map2_data5",
                        help="Path to the folder containing the preprocessed data.")
    parser.add_argument('--config_name', type=str,
                        default="xr_LSTM_enc_in_15_d_model_128_d_ff_1024_ep_100_e_layers_6_d_layers_6_embed_64_lr_0.001map21024",
                        help="Name of the model configuration folder.")
    parser.add_argument('--specific_file', type=str, default="39_119_tracks.csv",
                        help="Name of the specific file for visualization.")
    args = parser.parse_args()


    class Config:
        def __init__(self):
            self.enc_in = 15
            self.d_model = 128
            self.embed = 64
            self.freq = 32
            self.dropout = 0.1
            self.dec_in =15
            self.c_out = 2
            self.pred_len = 10
            self.output_attention = False
            self.factor = 4
            self.n_heads = 8
            self.e_layers = 6
            self.d_layers = 6
            self.d_ff = 1024
            self.activation = 'gelu'
            self.batch_size = 1024
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.seq_length = 10


    configs = Config()

    from models.Model_LSTM import Model_LSTM_Only

    model = Model_LSTM_Only(configs)
    model_path = os.path.join("../checkpoints", args.config_name, "best_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(configs.device)

    from src.data_processing import load_and_split_data

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

    specific_file_path = os.path.join(args.data_folder, args.specific_file)
    print(f"Evaluating and visualizing single trajectory from {specific_file_path}...")
    evaluate_and_visualize_single_trajectory(specific_file_path, model, configs, save_path)
    print("Single trajectory evaluation and visualization completed.")
