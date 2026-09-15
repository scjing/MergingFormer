import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer


class Model_Autoformer(nn.Module):
    def __init__(self, configs):
        """
        :param configs: 配置对象，包含以下字段（与Informer相同）：
            - enc_in: 编码器输入特征数
            - dec_in: 解码器输入特征数
            - d_model: 模型内部特征维度
            - c_out: 输出特征数
            - pred_len: 预测长度（输出时序步数）
            - dropout: dropout 概率
            - factor: 用于 ProbAttention 的采样因子
            - n_heads: 注意力头数
            - e_layers: Encoder 层数
            - d_layers: Decoder 层数
            - d_ff: 前馈层维度
            - activation: 激活函数（例如 'gelu'）
            - output_attention: 是否输出注意力（可选）
        """
        super(Model_Autoformer, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, dropout=configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
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
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
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
        预测函数，编码器处理长期趋势，解码器处理短期波动。
        :param x_enc: [B, seq_length, enc_in] 编码器输入
        :param x_dec: [B, pred_len, dec_in] 解码器输入（通常为全零张量）
        :return: Autoformer Decoder 输出，形状为 [B, L, c_out]
        """
        enc_emb = self.enc_embedding(x_enc)
        encoded_out, _ = self.encoder(enc_emb, attn_mask=None)
        dec_emb = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_emb, encoded_out, x_mask=None, cross_mask=None)
        return dec_out

    def forward(self, x_enc, x_dec, mask=None):
        """
        前向传播接口，与Transformer一致，仅输出最后 pred_len 个时刻的预测结果
        :param x_enc: [B, seq_length, enc_in] 编码器输入
        :param x_dec: [B, pred_len, dec_in] 解码器输入
        :param mask: 保留接口（未使用）
        :return: [B, pred_len, c_out] 预测结果
        """
        dec_out = self.forecast(x_enc, x_dec)
        return dec_out[:, -self.pred_len:, :]
