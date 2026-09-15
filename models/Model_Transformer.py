import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model_Transformer_Only(nn.Module):
    def __init__(self, configs):
        """
        :param configs: 配置对象，应包含以下字段：
            - enc_in: 编码器输入特征数
            - dec_in: 解码器输入特征数（通常与 enc_in 相同）
            - d_model: 模型内部特征维度
            - c_out: 输出特征数
            - pred_len: 预测长度（输出时序步数）
            - dropout: dropout 概率
            - factor: 用于 FullAttention 的因子（具体含义取决于实现）
            - n_heads: 注意力头数
            - e_layers: Encoder 层数
            - d_layers: Decoder 层数
            - d_ff: 前馈层维度
            - activation: 激活函数（例如 'gelu'）
            - output_attention: 是否输出注意力（可选）
        """
        super(Model_Transformer_Only, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, dropout=configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, dropout=configs.dropout)

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forecast(self, x_enc, x_dec):
        """
        预测函数：
          1. 对编码器输入进行嵌入后送入 Transformer Encoder 进行全局特征提取；
          2. 对解码器输入进行嵌入后，利用 Transformer Decoder 与 Encoder 输出交互进行预测。
        :param x_enc: [B, seq_length, enc_in] 编码器输入
        :param x_dec: [B, pred_len, dec_in] 解码器输入（通常为全零张量）
        :return: Transformer Decoder 输出，形状为 [B, L, c_out]
        """
        enc_emb = self.enc_embedding(x_enc)
        encoded_out, attns = self.encoder(enc_emb, attn_mask=None)
        dec_emb = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_emb, encoded_out, x_mask=None, cross_mask=None)
        return dec_out

    def forward(self, x_enc, x_dec, mask=None):
        """
        前向传播接口，与原模型保持一致，仅输出最后 pred_len 个时刻的预测结果
        :param x_enc: [B, seq_length, enc_in] 编码器输入
        :param x_dec: [B, pred_len, dec_in] 解码器输入
        :param mask: 保留接口（本模型未使用）
        :return: [B, pred_len, c_out] 预测结果
        """
        dec_out = self.forecast(x_enc, x_dec)
        return dec_out[:, -self.pred_len:, :]
