

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


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attn_weights = torch.softmax(self.attention(lstm_output).squeeze(2), dim=1)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context_vector, attn_weights
    
class Attentionv2(nn.Module):
    def __init__(self, embed_dim):
        super(Attentionv2, self).__init__()

        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)      
    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
 
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attn = attention_weights @ V

 
        return attn,attention_weights

class Res_Block(nn.Module):
    def __init__(self, input_size, cnn_kernel_size= 3):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=cnn_kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(128, 38, kernel_size=cnn_kernel_size, padding=1)
 
    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        x = x + res
        return x
    
class Res_Block2(nn.Module):
    def __init__(self, input_size, cnn_kernel_size= 3):
        super(Res_Block2, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=cnn_kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=cnn_kernel_size, padding=1)
 
    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return res
    

# 构建 LSTM 模型
# seq=27，是城市的数量，input_size=11是特征的数量
# 问题：现在是做的城市的融合，没有做时间上的融合，需要将输入数据整理为时序输入数据
# to do：
# 1.考虑CNN+LSTM+Attentino模型结构，在LSTM前先加卷积,在加attention
# 2.考虑位置编码
class DCLFormer(nn.Module):
    def __init__(self, input_size, output_size, cnn_kernel_size=3, 
                 lstm_hidden_size=128, num_layers=3):
        super(DCLFormer, self).__init__()
        # CNN 部分
        self.cnn1 = Res_Block(input_size, cnn_kernel_size=cnn_kernel_size)
        self.cnn2 = Res_Block2(input_size, cnn_kernel_size=cnn_kernel_size)
        
        # LSTM 部分
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # 注意力机制
        self.attention = MultiHeadAttention(lstm_hidden_size, num_heads=4)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN处理
        cnn_out = self.cnn1(x.permute(0, 2, 1))  # 转换为(batch_size, input_size, seq_len)
        cnn_out = cnn_out.permute(0, 2, 1)      # 恢复为(batch_size, new_seq_len, features)

        cnn_out2 = self.cnn2(cnn_out.permute(0, 2, 1))  # 转换为(batch_size, input_size, seq_len)
        cnn_out2 = cnn_out2.permute(0, 2, 1)      # 恢复为(batch_size, new_seq_len, features)
        
        # LSTM处理
        lstm_out, _ = self.lstm(cnn_out2)  # lstm_out shape: (batch_size, seq_len, hidden_size)
        
        # 注意力机制
        context = self.attention(lstm_out)
        
        # 全连接层
        output = self.fc(context)
        return output # 保持输出维度 (batch_size, 1, output_size)
    
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
