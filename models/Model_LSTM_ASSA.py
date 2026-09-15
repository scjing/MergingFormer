import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import repeat
from layers.Embed import DataEmbedding


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


class WindowAttention_sparse(nn.Module):
    def __init__(self, dim, win_size, num_heads=1, token_projection='linear', qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))
        coords_h = torch.arange(self.win_size[0])
        coords_w = torch.arange(self.win_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.win_size[0] - 1
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2
        else:
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0 * w1 + attn1 * w2
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cuda"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)

    def forward(self, x_enc):
        batch_size, seq_len = x_enc.shape[0], x_enc.shape[1]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        output, (h, c) = self.lstm(x_enc, (h_0, c_0))
        return output


class Model_LSTM_ASSA(nn.Module):
    def __init__(self, configs):
        """
        :param configs: 配置对象，需包含以下字段：
            - pred_len: 预测步长
            - enc_in: 输入特征维度
            - d_model: 模型内部维度
            - c_out: 输出特征维度
            - dropout: dropout 概率
            - batch_size: 批次大小
        """
        super(Model_LSTM_ASSA, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.c_out = configs.c_out

        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, dropout=configs.dropout)

        self.lstm = LSTM(input_size=self.enc_in, hidden_size=self.d_model,
                         num_layers=3, batch_size=configs.batch_size)

        self.window_attention_sparse1 = WindowAttention_sparse(self.d_model, (1, configs.pred_len))

        self.projection = nn.Linear(self.d_model, self.c_out)

    def forward(self, x):
        """
        :param x: 输入张量，形状为 [batch_size, seq_len, enc_in]
        :return: 预测结果，形状为 [batch_size, pred_len, c_out]
        """
        embed = self.enc_embedding(x)
        lstm_out = self.lstm(x)
        attn_out = self.window_attention_sparse1(lstm_out)
        combined = embed + attn_out
        forecast_features = combined[:, -self.pred_len:, :]
        forecast = self.projection(forecast_features)
        return forecast
