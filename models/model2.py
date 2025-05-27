import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# 读取 Excel 文件
import pandas as pd
import numpy as np


class DCLFormer(nn.Module):
    def __init__(self, input_size, output_size, cnn_kernel_size=3, 
                 lstm_hidden_size=128, num_layers=3, cnn_layers=4):
        super(DCLFormer, self).__init__()
        
        # 改进的CNN模块
        self.cnn_blocks = nn.Sequential(
            *[DilatedResBlock(input_size, 
                            dilation=2**i,  # 指数级膨胀系数
                            kernel_size=cnn_kernel_size) 
             for i in range(cnn_layers)]
        )
        
        # 时序位置编码
        self.position_enc = PositionalEncoding(input_size, max_len=100)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 多头注意力
        self.attention = MultiHeadAttention(
            embed_dim=lstm_hidden_size*2,  # 双向LSTM输出
            num_heads=8
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        # 位置编码
        x = self.position_enc(x) #b必须满足是偶数
        
        # CNN处理
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len)
        cnn_out = self.cnn_blocks(x)
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch_size, seq_len, features)
        
        # LSTM处理
        lstm_out, _ = self.lstm(cnn_out)  # (batch_size, seq_len, 2*hidden_size)
        
        # 注意力机制
        attn_out = self.attention(lstm_out)
        
        # 时序特征聚合
        context = torch.mean(attn_out, dim=1)  # (batch_size, 2*hidden_size)
        
        # 最终预测
        output = self.fc(context)
        return output  # (batch_size, output_size)

# 改进的残差块
class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation=1, kernel_size=3):
        super().__init__()
        padding = (kernel_size + (kernel_size-1)*(dilation-1) - 1) // 2
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels*2, kernel_size, 
                     padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels*2),
            nn.GELU(),
            nn.Conv1d(channels*2, channels, 1),  # 通道调整
            nn.BatchNorm1d(channels)
        )
        
        self.skip = nn.Conv1d(channels, channels, 3, padding=1) if dilation>1 else None
        
    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        if self.skip:
            residual = self.skip(residual)
        return F.gelu(x + residual)

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        
        # 计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 动态调整维度计算
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / d_model)
        )
        
        # 自动处理奇数维度
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])  # 自动适配奇数维度
        
        # 注册为缓冲区
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # 自动截断或填充位置编码
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x
    

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        self.query=nn.Linear(embed_dim, embed_dim)
        self.key=nn.Linear(embed_dim, embed_dim)
        self.value=nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        
    def forward(self,X):
        batch_size, seq_len, embed_dim = X.size()
         #线性映射
        q = self.query(X)
        k = self.key(X)
        v = self.value(X)
        # print("q1 shape:", q.shape)
        # print("k1 shape:", k.shape)
        #[batch_size, seq_len, embed_dim]变为[batch_size, seq_len, num_heads, head_dim]
        #transpose(1, 2) 调换了 seq_len 和 num_heads 的维度[batch_size, num_heads, seq_len, head_dim]
        q=q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # print("q shape:", q.shape)
        # print("k shape:", k.shape)

        
        #最核心的，计算点积
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        # 首先计算 k 的转置，k.transpose(-2, -1) 将 k 的最后两个维度调换，形状变为 [batch_size, num_heads，head_dim, seq_len]。
        # 然后计算 q 和转置后的 k 的点积，torch.matmul(q, k.transpose(-2, -1)) 结果形状为 [batch_size, num_heads, seq_len, seq_len]。
        # 最后除以 sqrt(head_dim) 进行缩放，这是为了稳定梯度，防止点积结果过大。
        attn_output = torch.matmul(attn_weights, v)
        #v: [batch_size, num_heads, seq_len, head_dim]结果
        # attn_output为：[batch_size, num_heads, seq_len, head_dim]。
#        这一步是加权求和，将每个位置的值向量 v 根据注意力权重进行加权求和。
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out(attn_output)
        return output