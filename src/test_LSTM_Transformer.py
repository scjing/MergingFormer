import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse


#################################################
# 计算 ADE 和 FDE 的辅助函数
#################################################
def compute_ADE_FDE(pred, actual):
    """
    计算 ADE 和 FDE
    :param pred: 预测值序列（list 或 array）
    :param actual: 实际值序列（list 或 array）
    :return: ade, fde
    """
    pred = np.array(pred)
    actual = np.array(actual)
    # 对齐序列长度
    min_len = min(len(pred), len(actual))
    pred = pred[:min_len]
    actual = actual[:min_len]

    ade = np.mean(np.abs(pred - actual))
    fde = np.abs(pred[-1] - actual[-1])
    return ade, fde


#################################################
# 模型评估函数（适用于 LSTM_Transformer 模型，包含 ADE 和 FDE）
#################################################
def evaluate_model(model, test_loader, configs, save_path):
    # 确保模型处于评估模式
    model.eval()

    # 准备评估指标的列表
    mse_list, rmse_list, mae_list, mape_list, r2_list = [], [], [], [], []
    mse_speed_list, rmse_speed_list, mae_speed_list, mape_speed_list, r2_speed_list = [], [], [], [], []
    mse_speedY_list, rmse_speedY_list, mae_speedY_list, mape_speedY_list, r2_speedY_list = [], [], [], [], []

    # 新增：ADE 和 FDE 指标列表
    ade_list, fde_list = [], []
    ade_speed_list, fde_speed_list = [], []
    ade_speedY_list, fde_speedY_list = [], []

    criterion = torch.nn.MSELoss()
    device = configs.device

    with torch.no_grad():
        for input_data, target_data, _, _ in test_loader:  # 忽略其他信息（如时间标记）
            input_data, target_data = input_data.to(device), target_data.to(device)

            # 构造解码器输入：LSTM_Transformer 模型需要 x_dec，形状为 [B, pred_len, enc_in]
            x_dec = torch.zeros(input_data.size(0), configs.pred_len, configs.enc_in).to(device)

            # 模型预测
            output = model(input_data, x_dec)  # 输出形状 [B, pred_len, c_out]

            # 截取目标数据前 pred_len 帧，保证形状一致
            target_data = target_data[:, :configs.pred_len, :]

            # 将数据转换为 numpy 格式
            output_np = output.cpu().numpy()
            target_np = target_data.cpu().numpy()

            # 逐特征计算评估指标及 ADE/FDE
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

                # 逐样本计算当前特征的 ADE 和 FDE
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

            # 针对 speed（第6列，索引 5）的评估
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

                # ADE/FDE for speed
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

            # 针对 speedY（第5列，索引 4）的评估
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

                # ADE/FDE for speedY
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

    # 计算整体平均指标
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

    # 打印评估结果
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

    # 保存评估结果到文件
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


#################################################
# 单轨迹评估与可视化函数
#################################################
def evaluate_and_visualize_single_trajectory(csv_file_path, model, configs, save_path):
    """
    对单条轨迹数据进行评估并绘制 Speed 和 SpeedY 的真实值与预测值对比图。

    :param csv_file_path: 轨迹 CSV 文件路径
    :param model: 已训练的模型
    :param configs: 模型配置
    :param save_path: 图像保存路径
    """
    model.eval()  # 确保模型在评估模式
    device = configs.device

    # 加载数据
    event_data = pd.read_csv(csv_file_path)
    event_data.fillna(-1, inplace=True)

    # 特征列和目标列
    input_columns = [
        'carCenterX', 'carCenterY', 'carCenterXft', 'carCenterYft', 'speedY', 'speed', 'laneId',
        'pressure', 'leadCarCenterX', 'leadCarCenterY', 'leadCarCenterXft', 'leadCarCenterYft',
        'leadSpeed', 'leftLeadCarCenterX', 'leftLeadCarCenterY', 'leftLeadCarCenterXft',
        'leftLeadCarCenterYft', 'leftLeadSpeed', 'leftFollowCarCenterX', 'leftFollowCarCenterY',
        'leftFollowCarCenterXft', 'leftFollowCarCenterYft', 'leftFollowSpeed'
    ]
    target_columns = ['carCenterX', 'carCenterY', 'carCenterXft', 'carCenterYft', 'speedY', 'speed', 'laneId']

    # 数据转换为 numpy 格式
    input_array = event_data[input_columns].to_numpy()
    target_array = event_data[target_columns].to_numpy()

    # 数据归一化
    input_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    input_array = input_scaler.fit_transform(input_array)
    target_array = target_scaler.fit_transform(target_array)

    # 逐步预测
    seq_length = configs.seq_length
    predictions = []

    with torch.no_grad():
        for i in range(len(target_array) - seq_length):
            # 每次从原始数据中取出一个长度为 seq_length 的输入序列
            input_seq = input_array[i:i + seq_length]
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)

            # 构造解码器输入，形状为 [1, pred_len, dec_in]
            x_dec = torch.zeros(1, configs.pred_len, configs.dec_in).to(device)

            # 模型预测
            output = model(input_tensor, x_dec)
            # 取第一个时间步的预测值
            predicted_point = output[:, 0, :].cpu().numpy().flatten()
            predictions.append(predicted_point)

    # 转换为 numpy 数组
    predictions = np.array(predictions)

    # 反归一化
    predictions_restored = target_scaler.inverse_transform(predictions)
    targets_restored = target_scaler.inverse_transform(target_array[seq_length:])

    # 提取速度数据
    speed_index = 5  # speed 为第6列
    speedY_index = 4  # speedY 为第5列

    predictions_speed = predictions_restored[:, speed_index]
    predictions_speedY = predictions_restored[:, speedY_index]
    targets_speed = targets_restored[:, speed_index]
    targets_speedY = targets_restored[:, speedY_index]

    # 绘图
    plt.figure(figsize=(12, 6))
    # 绘制 Speed
    plt.subplot(1, 2, 1)
    plt.plot(predictions_speed, label='Predicted Speed', linestyle='-')
    plt.plot(targets_speed, label='True Speed', linestyle=':')
    plt.xlabel('Time Step')
    plt.ylabel('Speed')
    plt.legend()
    plt.title('Speed Prediction')
    # 绘制 SpeedY
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
    plt.show()
    # 新增：将 speed 和 speedY 的真实值与预测值保存到 txt 文件
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


#################################################
# 主函数：解析参数，加载模型、数据，并执行评估与可视化
#################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the LSTM_Transformer model and visualize results.")
    parser.add_argument('--data_folder', type=str, default="../data/cleaned_test01",
                        help="Path to the folder containing the preprocessed data.")
    parser.add_argument('--config_name', type=str,
                        default="xr_LSTM_Transformer_enc_in_23_d_model_128_d_ff_1024_ep_100_e_layers_6_d_layers_6_embed_64_lr_0.0001",
                        help="Name of the model configuration folder.")
    parser.add_argument('--specific_file', type=str, default="car_75_data.csv",
                        help="Name of the specific file for visualization.")
    args = parser.parse_args()


    # 配置
    class Config:
        def __init__(self):
            self.enc_in = 23
            self.d_model = 128
            self.embed = 64
            self.freq = 100
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

    # 模型初始化：导入 LSTM_Transformer 模型
    from models.Model_LSTM_Transformer import Model_LSTM_Transformer

    model = Model_LSTM_Transformer(configs)
    model_path = os.path.join("../checkpoints", args.config_name, "best_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(configs.device)

    # 加载数据
    from src.data_processing import load_and_split_data

    train_loader, val_loader, test_loader = load_and_split_data(
        args.data_folder,
        seq_length=configs.seq_length,
        batch_size=configs.batch_size
    )

    # 创建保存路径
    save_path = os.path.join("../checkpoints", args.config_name)
    os.makedirs(save_path, exist_ok=True)

    # 评估模型
    print("Starting evaluation...")
    evaluate_model(model, test_loader, configs, save_path)
    print("Evaluation completed.")

    # 对单一轨迹进行评估和可视化
    specific_file_path = os.path.join(args.data_folder, args.specific_file)
    print(f"Evaluating and visualizing single trajectory from {specific_file_path}...")
    evaluate_and_visualize_single_trajectory(specific_file_path, model, configs, save_path)
    print("Single trajectory evaluation and visualization completed.")
