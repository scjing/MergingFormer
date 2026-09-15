import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_attention(q, k, v):
    """
    采用 elu 激活作为随机特征映射：phi(x) = elu(x) + 1
    实现 Performer 中的线性注意力计算：
      y_i = (phi(q_i)^T * sum_j phi(k_j)*v_j) / (phi(q_i)^T * sum_j phi(k_j))
    """
    phi_q = F.elu(q) + 1
    phi_k = F.elu(k) + 1
    numerator = torch.einsum('bid,bjd,bjv->biv', phi_q, phi_k, v)
    k_sum = phi_k.sum(dim=1)
    denominator = torch.einsum('bid,bd->bi', phi_q, k_sum).unsqueeze(-1) + 1e-6
    out = numerator / denominator
    return out


class PerformerAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        多头 Performer 注意力模块
        :param d_model: 输入特征维度
        :param num_heads: 注意力头数
        """
        super(PerformerAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        """
        :param query: [B, L, d_model]
        :param key:   [B, S, d_model]
        :param value: [B, S, d_model]
        :return: [B, L, d_model]
        """
        B, L, _ = query.shape
        _, S, _ = key.shape
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        head_outputs = []
        for i in range(self.num_heads):
            q_i = q[:, i, :, :]
            k_i = k[:, i, :, :]
            v_i = v[:, i, :, :]
            out_i = linear_attention(q_i, k_i, v_i)
            head_outputs.append(out_i)
        out = torch.stack(head_outputs, dim=1)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        out = self.out_proj(out)
        return out


class PerformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Performer 编码器层，包括：
          - 多头自注意力模块
          - 前馈网络
          - 残差连接与层归一化
        """
        super(PerformerEncoderLayer, self).__init__()
        self.self_attn = PerformerAttention(d_model, num_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        """
        :param src: [B, L, d_model]
        :return: [B, L, d_model]
        """
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class PerformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Performer 解码器层，包括：
          - 解码器自注意力（可选择做因果 mask，但此处与原模型保持一致）
          - 编码器-解码器交叉注意力
          - 前馈网络
          - 残差连接与层归一化
        """
        super(PerformerDecoderLayer, self).__init__()
        self.self_attn = PerformerAttention(d_model, num_heads)
        self.cross_attn = PerformerAttention(d_model, num_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory):
        """
        :param tgt:    [B, L_t, d_model] 解码器输入
        :param memory: [B, L_s, d_model] 编码器输出
        :return: [B, L_t, d_model]
        """
        tgt2 = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class Model_Performer(nn.Module):
    """
    Performer 模型，保持数据维度与原有 Seq2Seq 模型一致：
      - x_enc: [B, seq_length, enc_in]
      - x_dec: [B, pred_len, dec_in]
      - 输出: [B, pred_len, c_out]
    """

    def __init__(self, configs):
        """
        :param configs: 配置对象，应包含如下字段：
            - enc_in: 编码器输入特征数
            - dec_in: 解码器输入特征数
            - d_model: Transformer 模型隐藏状态维度
            - c_out: 输出特征数
            - pred_len: 预测长度（解码器输出序列的时间步数）
            - num_encoder_layers: 编码器层数（可选，默认为 3）
            - num_decoder_layers: 解码器层数（可选，默认为 3）
            - num_heads: 多头注意力头数（可选，默认为 8）
            - dim_feedforward: 前馈网络维度（可选，默认为 2048）
            - dropout: dropout 概率（可选，默认为 0.1）
        """
        super(Model_Performer, self).__init__()
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.num_encoder_layers = getattr(configs, 'num_encoder_layers', 3)
        self.num_decoder_layers = getattr(configs, 'num_decoder_layers', 3)
        self.num_heads = getattr(configs, 'num_heads', 8)
        self.dim_feedforward = getattr(configs, 'dim_feedforward', 2048)
        self.dropout = getattr(configs, 'dropout', 0.1)

        self.enc_input_proj = nn.Linear(self.enc_in, self.d_model) if self.enc_in != self.d_model else nn.Identity()
        self.dec_input_proj = nn.Linear(self.dec_in, self.d_model) if self.dec_in != self.d_model else nn.Identity()

        self.enc_pos_embedding = nn.Parameter(torch.zeros(1, 500, self.d_model))
        self.dec_pos_embedding = nn.Parameter(torch.zeros(1, 500, self.d_model))

        self.encoder_layers = nn.ModuleList([
            PerformerEncoderLayer(self.d_model, self.num_heads, self.dim_feedforward, self.dropout)
            for _ in range(self.num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            PerformerDecoderLayer(self.d_model, self.num_heads, self.dim_feedforward, self.dropout)
            for _ in range(self.num_decoder_layers)
        ])

        self.projection = nn.Linear(self.d_model, configs.c_out)

    def forward(self, x_enc, x_dec, mask=None):
        """
        :param x_enc: [B, seq_length, enc_in] 编码器输入
        :param x_dec: [B, pred_len, dec_in] 解码器输入（例如全零张量）
        :param mask: 保留接口，与原模型一致（本模型未使用）
        :return: 预测结果，形状 [B, pred_len, c_out]
        """
        batch_size, seq_length, _ = x_enc.size()
        _, pred_len, _ = x_dec.size()
        enc_input = self.enc_input_proj(x_enc)
        dec_input = self.dec_input_proj(x_dec)
        enc_input = enc_input + self.enc_pos_embedding[:, :seq_length, :]
        dec_input = dec_input + self.dec_pos_embedding[:, :pred_len, :]
        memory = enc_input
        for layer in self.encoder_layers:
            memory = layer(memory)
        output = dec_input
        for layer in self.decoder_layers:
            output = layer(output, memory)
        output = self.projection(output)
        return output
