import torch
import torch.nn as nn


class Model_LSTM_Only(nn.Module):
    def __init__(self, configs):
        """
        :param configs: 配置对象，应包含如下字段：
            - enc_in: 编码器输入特征数
            - dec_in: 解码器输入特征数（一般与 enc_in 相同）
            - d_model: LSTM 隐藏状态维度
            - c_out: 输出特征数
            - pred_len: 预测长度（解码器输出序列的时间步数）
        """
        super(Model_LSTM_Only, self).__init__()
        self.pred_len = configs.pred_len

        self.encoder = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=3,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=configs.dec_in,
            hidden_size=configs.d_model,
            num_layers=3,
            batch_first=True
        )

        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_dec, mask=None):
        """
        :param x_enc: [B, seq_length, enc_in] 编码器输入
        :param x_dec: [B, pred_len, dec_in] 解码器输入（例如全零张量）
        :param mask: 保留接口，与其他模型一致（本模型未使用）
        :return: 预测结果，形状 [B, pred_len, c_out]
        """
        batch_size = x_enc.size(0)
        device = x_enc.device
        num_layers = self.encoder.num_layers
        hidden_size = self.encoder.hidden_size

        h0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)

        _, (h, c) = self.encoder(x_enc, (h0, c0))

        decoder_out, _ = self.decoder(x_dec, (h, c))

        output = self.projection(decoder_out)
        return output
