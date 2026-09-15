import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse
from scipy.stats import norm
import glob


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


def plot_distribution_comparison(true_values, predicted_values, title, save_path, filename):
    """
    绘制真实值与预测值分布对比直方图，并拟合正态分布，打印拟合参数
    """
    plt.figure(figsize=(10, 6))
    # 拟合正态分布
    mu_true, std_true = norm.fit(true_values)
    mu_pred, std_pred = norm.fit(predicted_values)

    bins = 50
    plt.hist(true_values, bins=bins, alpha=0.6, label=f'True (μ={mu_true:.2f}, σ={std_true:.2f})', density=True)
    plt.hist(predicted_values, bins=bins, alpha=0.6, label=f'Predicted (μ={mu_pred:.2f}, σ={std_pred:.2f})',
             density=True)

    x = np.linspace(min(true_values.min(), predicted_values.min()),
                    max(true_values.max(), predicted_values.max()), 200)
    plt.plot(x, norm.pdf(x, mu_true, std_true), 'b--', linewidth=2)
    plt.plot(x, norm.pdf(x, mu_pred, std_pred), 'r--', linewidth=2)

    plt.title(f'Distribution Comparison: {title}')
    plt.xlabel(title)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file)
    print(f"Saved distribution plot to {save_file}")
    plt.close()

    print(f"{title} - True Distribution: μ = {mu_true:.4f}, σ = {std_true:.4f}")
    print(f"{title} - Predicted Distribution: μ = {mu_pred:.4f}, σ = {std_pred:.4f}")


fontsize = 20
plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


def configure_plot_histogram():
    """统一配置直方图绘图格式，包括字体、字号、图大小等"""
    plt.rcParams.update({
        'figure.figsize': (7, 5),
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'legend.fontsize': fontsize - 2,
        'xtick.labelsize': fontsize - 2,
        'ytick.labelsize': fontsize - 2,
    })


def plot_distribution_histogram_custom(predicted_values, true_values, x_label, save_path, filename):
    """
    绘制预测值与真实值的概率密度直方图，叠加正态分布拟合曲线，并打印拟合参数
    采用统一的风格配置（中文字体、颜色等）。
    """
    configure_plot_histogram()
    fig, ax = plt.subplots()

    min_val = min(predicted_values.min(), true_values.min())
    max_val = max(predicted_values.max(), true_values.max())
    common_bins = np.linspace(min_val, max_val, 40)

    pred_color = (104 / 255, 136 / 255, 188 / 255)
    true_color = (204 / 255, 110 / 255, 112 / 255)

    ax.hist(predicted_values, label='模型速度分布', bins=common_bins, alpha=0.5, density=True, color=pred_color)
    ax.hist(true_values, label='真实速度分布', bins=common_bins, alpha=0.5, density=True, color=true_color)

    pred_mean, pred_std = norm.fit(predicted_values)
    true_mean, true_std = norm.fit(true_values)

    x_values = np.linspace(min_val, max_val, 100)
    pdf_pred = norm.pdf(x_values, pred_mean, pred_std)
    pdf_true = norm.pdf(x_values, true_mean, true_std)

    ax.plot(x_values, pdf_pred, label='模型概率密度', color=pred_color, linewidth=2)
    ax.plot(x_values, pdf_true, label='真实概率密度', color=true_color, linewidth=2, linestyle='--')

    ax.set_xlabel(x_label)
    ax.set_ylabel('概率')
    ax.legend()
    ax.tick_params(direction='in')

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename), format='svg', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    print("预测分布拟合参数: 均值 = {:.5f}, 标准差 = {:.5f}".format(pred_mean, pred_std))
    print("真实分布拟合参数: 均值 = {:.5f}, 标准差 = {:.5f}".format(true_mean, true_std))


def evaluate_model(model, test_loader, configs, save_path, raw_target_csv=None):
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
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100  # 避免除零
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

    if raw_target_csv is not None:
        if os.path.isdir(raw_target_csv):
            csv_files = glob.glob(os.path.join(raw_target_csv, '*.csv'))
            data_list = []
            for f in csv_files:
                df = pd.read_csv(f)
                df.fillna(-1, inplace=True)
                target_columns = ['speedY', 'speed']

                data_list.append(df[target_columns])
            all_data = pd.concat(data_list, axis=0)
            target_array = all_data.to_numpy()
            print("Loaded {} CSV files from folder: {}".format(len(csv_files), raw_target_csv))
        else:
            df = pd.read_csv(raw_target_csv)
            df.fillna(-1, inplace=True)
            target_columns = [ 'speedY', 'speed']
            target_array = df[target_columns].to_numpy()
            print("Loaded single CSV file:", raw_target_csv)
        target_scaler = MinMaxScaler()
        target_scaler.fit(target_array)
        print("Fitted target_scaler using data from:", raw_target_csv)
    else:
        all_targets_for_scaler = []
        for _, target_data, _, _ in test_loader:
            all_targets_for_scaler.append(target_data.cpu().numpy())
        all_targets_for_scaler = np.concatenate(all_targets_for_scaler, axis=0)
        all_targets_for_scaler = all_targets_for_scaler.reshape(-1, all_targets_for_scaler.shape[-1])
        target_scaler = MinMaxScaler()
        target_scaler.fit(all_targets_for_scaler)
        print("Fitted target_scaler on test set targets.")

    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for input_data, target_data, _, _ in test_loader:
            input_data = input_data.to(device)
            x_dec = torch.zeros(input_data.size(0), configs.pred_len, configs.enc_in).to(device)
            output = model(input_data, x_dec)
            target_data = target_data[:, :configs.pred_len, :]

            all_outputs.append(output.cpu().numpy())
            all_targets.append(target_data.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    outputs_flat = all_outputs.reshape(-1, all_outputs.shape[-1])
    targets_flat = all_targets.reshape(-1, all_targets.shape[-1])

    outputs_denorm = target_scaler.inverse_transform(outputs_flat)
    targets_denorm = target_scaler.inverse_transform(targets_flat)

    pred_speed = outputs_denorm[:, 1]       #* 0.44704
    true_speed = targets_denorm[:, 1]      # * 0.44704
    pred_speedY = outputs_denorm[:, 0]      # * 0.44704
    true_speedY = targets_denorm[:, 0]        #* 0.44704

    plot_distribution_histogram_custom(true_speed, pred_speed, 'SpeedX/(m/s)', save_path,
                                       'speed_distribution_histogram.svg')
    plot_distribution_histogram_custom(true_speedY, pred_speedY, 'SpeedY/(m/s)', save_path,
                                       'speedY_distribution_histogram.svg')

    return (avg_mse, avg_rmse, avg_mae, avg_mape, avg_r2,
            avg_ade, avg_fde,
            avg_mse_speed, avg_rmse_speed, avg_mae_speed, avg_mape_speed, avg_r2_speed,
            avg_ade_speed, avg_fde_speed,
            avg_mse_speedY, avg_rmse_speedY, avg_mae_speedY, avg_mape_speedY, avg_r2_speedY,
            avg_ade_speedY, avg_fde_speedY)


def evaluate_and_visualize_single_trajectory(csv_file_path, model, configs, save_path):
    """
    对单条轨迹数据进行评估并绘制Speed和SpeedY的真实值与预测值对比图。
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
         'speedY', 'speed',
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
            x_dec = torch.zeros(1, configs.pred_len, configs.enc_in).to(device)
            output = model(input_tensor, x_dec)
            predicted_point = output[:, 0, :].cpu().numpy().flatten()
            predictions.append(predicted_point)

    predictions = np.array(predictions)
    predictions_restored = target_scaler.inverse_transform(predictions)
    targets_restored = target_scaler.inverse_transform(target_array[seq_length:])

    speed_index = 1
    speedY_index = 0

    predictions_speed = predictions_restored[:, speed_index]        #* 0.44704
    predictions_speedY = predictions_restored[:, speedY_index]      #* 0.44704
    targets_speed = targets_restored[:, speed_index]                #* 0.44704
    targets_speedY = targets_restored[:, speedY_index]              #* 0.44704

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(predictions_speed, label='Predicted Speed', linestyle='-')
    plt.plot(targets_speed, label='True Speed', linestyle=':')
    plt.xlabel('Time Step')
    plt.ylabel('Speed(m/s)')
    plt.legend()
    plt.title('Speed Prediction')
    plt.subplot(1, 2, 2)
    plt.plot(predictions_speedY, label='Predicted SpeedY', linestyle='-')
    plt.plot(targets_speedY, label='True SpeedY', linestyle=':')
    plt.xlabel('Time Step')
    plt.ylabel('SpeedY(m/s)')
    plt.legend()
    plt.title('SpeedY Prediction')

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
    parser = argparse.ArgumentParser(description="Evaluate the Transformer model and visualize results.")
    parser.add_argument('--data_folder', type=str, default="../data/111map2_data5",
                        help="Path to the folder containing the preprocessed data.")
    parser.add_argument('--config_name', type=str,
                        default="enc_in_15_d_model_128_d_ff_1024_ep_100_e_layers_6_d_layers_6_embed_64_lr_0.0001map2",
                        help="Name of the model configuration folder.")
    parser.add_argument('--specific_file', type=str, default="39_185_tracks.csv",
                        help="Name of the specific file for visualization.")
    args = parser.parse_args()


    class Config:
        def __init__(self):
            self.enc_in = 15
            self.d_model = 128
            self.embed = 64
            self.freq = 32
            self.dropout = 0.1
            self.dec_in = 15
            self.c_out = 2
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

    from models.MyTransformer import Model

    model = Model(configs)
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

    raw_target_csv = args.data_folder

    print("Starting evaluation...")
    evaluate_model(model, test_loader, configs, save_path, raw_target_csv=raw_target_csv)
    print("Evaluation completed.")

    print(f"Evaluating and visualizing single trajectory from {os.path.join(args.data_folder, args.specific_file)}...")
    evaluate_and_visualize_single_trajectory(os.path.join(args.data_folder, args.specific_file), model, configs,
                                             save_path)
    print("Single trajectory evaluation and visualization completed.")
