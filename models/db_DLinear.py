import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    移动平均模块，用于提取趋势部分
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    序列分解模块，将输入分解为趋势和季节性部分
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    def __init__(self, configs):
        """
        :param configs: 配置对象，与Transformer模型相同，包含以下字段：
            - seq_len: 输入序列长度
            - pred_len: 预测长度
            - enc_in: 输入特征数
            - c_out: 输出特征数
        """
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        self.decomp = SeriesDecomp(kernel_size=25)

        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)
        self.seasonal_linear = nn.Linear(self.seq_len, self.pred_len)

        self.projection = nn.Linear(self.enc_in, self.c_out)

    def forward(self, x_enc, x_dec=None, mask=None):
        """
        :param x_enc: [B, seq_length, enc_in] 编码器输入
        :param x_dec: 未使用，保持与Transformer接口一致
        :param mask: 未使用，保持接口一致
        :return: [B, pred_len, c_out] 预测结果
        """
        seasonal, trend = self.decomp(x_enc)

        trend = trend.permute(0, 2, 1)
        trend_pred = self.trend_linear(trend)
        trend_pred = trend_pred.permute(0, 2, 1)

        seasonal = seasonal.permute(0, 2, 1)
        seasonal_pred = self.seasonal_linear(seasonal)
        seasonal_pred = seasonal_pred.permute(0, 2, 1)

        pred = trend_pred + seasonal_pred

        pred = self.projection(pred)

        return pred
