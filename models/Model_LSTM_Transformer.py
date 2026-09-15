import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cuda"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers,
            batch_first=True, bidirectional=False
        )

    def forward(self, x_enc):
        batch_size, seq_len = x_enc.shape[0], x_enc.shape[1]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        output, (h, c) = self.lstm(x_enc, (h_0, c_0))
        return output


class Model_LSTM_Transformer(nn.Module):
    def __init__(self, configs):
        """
        :param configs: 配置对象，需要包含以下字段：
            - pred_len: 预测步长
            - enc_in: 编码器输入特征数
            - dec_in: 解码器输入特征数（一般与 enc_in 相同）
            - d_model: 模型内部特征维度
            - c_out: 输出特征数
            - dropout: dropout 概率
            - factor, n_heads, e_layers, d_layers, d_ff, activation: Transformer 相关参数
            - batch_size: 批次大小
            - seq_length: 输入序列长度
        """
        super(Model_LSTM_Transformer, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, dropout=configs.dropout)

        self.lstm = LSTM(input_size=configs.enc_in, hidden_size=configs.d_model,
                         num_layers=3, batch_size=configs.batch_size)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
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
                        configs.d_model, configs.n_heads
                    ),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forecast(self, x_enc, x_dec):
        emb = self.enc_embedding(x_enc)
        lstm_out = self.lstm(x_enc)
        combined_features = emb + lstm_out
        encoded_out, attns = self.encoder(combined_features, attn_mask=None)
        dec_emb = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_emb, encoded_out, x_mask=None, cross_mask=None)
        return dec_out

    def forward(self, x_enc, x_dec, mask=None):
        dec_out = self.forecast(x_enc, x_dec)
        return dec_out[:, -self.pred_len:, :]
