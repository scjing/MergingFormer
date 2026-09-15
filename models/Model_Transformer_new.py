import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置编码模块，给输入加入位置信息
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [B, seq_len, d_model]
        :return: 加入位置编码后的 x
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Model_Transformer(nn.Module):
    """
    基于 Transformer 的序列到序列模型

    输入：
      - x_enc: [B, seq_length, enc_in]，历史数据
      - x_dec: [B, pred_len, dec_in]，解码器输入（通常为全零张量）
    输出：
      - 输出预测结果，形状为 [B, pred_len, c_out]
    """

    def __init__(self, configs):
        """
        :param configs: 配置对象，需包含如下字段：
            - enc_in: 编码器输入特征数
            - dec_in: 解码器输入特征数（一般与 enc_in 相同）
            - d_model: Transformer 隐藏状态维度
            - c_out: 输出特征数
            - pred_len: 预测长度（解码器输出序列的时间步数）
            - n_heads: 多头注意力的头数
            - e_layers: Transformer 编码器层数
            - d_layers: Transformer 解码器层数
            - d_ff: 前馈网络隐藏层维度
            - dropout: dropout 概率
        """
        super(Model_Transformer, self).__init__()
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        self.enc_in_proj = nn.Linear(configs.enc_in, configs.d_model)
        self.dec_in_proj = nn.Linear(configs.dec_in, configs.d_model)

        self.positional_encoding = PositionalEncoding(configs.d_model, dropout=configs.dropout)

        self.transformer = nn.Transformer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            num_encoder_layers=configs.e_layers,
            num_decoder_layers=configs.d_layers,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            batch_first=True
        )

        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_dec, mask=None):
        """
        :param x_enc: [B, seq_length, enc_in] 编码器输入
        :param x_dec: [B, pred_len, dec_in] 解码器输入（例如全零张量）
        :param mask: 保留接口（本模型未使用）
        :return: 预测结果，形状为 [B, pred_len, c_out]
        """
        src = self.enc_in_proj(x_enc)
        tgt = self.dec_in_proj(x_dec)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        transformer_out = self.transformer(src, tgt)

        output = self.projection(transformer_out)
        return output
