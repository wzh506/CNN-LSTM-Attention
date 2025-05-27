

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
        self.conv2 = nn.Conv1d(128, input_size, kernel_size=cnn_kernel_size, padding=1)
 
    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        x = x + res
        return x
    
class Res_Block2(nn.Module):
    def __init__(self, input_size, output_size=128,cnn_kernel_size= 3):
        super(Res_Block2, self).__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size=cnn_kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(output_size, output_size, kernel_size=cnn_kernel_size, padding=1)
 
    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return res
 
class SpatialAttention(nn.Module):
    """城市级空间注意力机制 (修复退化问题)"""
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0, "feature_dim必须能被num_heads整除"
        
        # 初始化QKV投影层
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1e-2)  # 小增益初始化
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1e-2)
        
        # 输出层 + 归一化
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)
        self.norm_in = nn.LayerNorm(feature_dim)
        
        self.norm_2 = nn.LayerNorm(feature_dim)  # 第二层归一化

    def forward(self, x,vis=False):
        batch_size, num_cities, _ = x.shape
        residual = x  # 保留残差连接
        
        # 投影QKV + Dropout
        x = self.norm_in(x)
        Q = self.norm_2(self.q_proj(x)) #torch.Size([3, 45, 256])
        K = self.norm_2(self.k_proj(x))
        V = self.v_proj(x)
        
        # 多头切分
        Q = Q.view(batch_size, num_cities, self.num_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, num_cities, self.num_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, num_cities, self.num_heads, self.head_dim).permute(0,2,1,3)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 加权聚合 + 残差
        context = torch.matmul(attn_weights, V)
        context = context.permute(0,2,1,3).contiguous().view(batch_size, num_cities, -1)
        attended = self.out_proj(context)
        return self.norm(attended + residual), attn_weights  # 必须保留残差！ 
    
    
    

# 构建 LSTM 模型
# seq=27，是城市的数量，input_size=11是特征的数量
# 问题：现在是做的城市的融合，没有做时间上的融合，需要将输入数据整理为时序输入数据
# to do：
# 1.考虑CNN+LSTM+Attentino模型结构，在LSTM前先加卷积,在加attention
# 2.考虑位置编码
class DCLFormer(nn.Module):
    def __init__(self, input_size, output_size, cnn_kernel_size=3, 
                 lstm_hidden_size=256, num_layers=3,num_cities=45,embed_dim=5,cnn_layers=2):  
        super(DCLFormer, self).__init__()

        # 城市编码嵌入层
        self.city_embed = nn.Embedding(
            num_embeddings=num_cities,  # 城市总数
            embedding_dim=embed_dim     # 推荐128-256维
        )
        
        # CNN 部分
        self.cnn1 = Res_Block(input_size + embed_dim -1,cnn_kernel_size=cnn_kernel_size)
        self.cnn2 = Res_Block2(input_size + embed_dim -1,lstm_hidden_size, cnn_kernel_size=cnn_kernel_size)
        
        # LSTM 部分
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size, 
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # 注意力机制
        self.spatial_attn = SpatialAttention(lstm_hidden_size, num_heads=8)
        self.temporal_attn = MultiHeadAttention(lstm_hidden_size, num_heads=8)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x,vis=False):
        batch_size = x.size(0)
        
        features = x[..., :-1]  # (batch, cities, input_features)
        city_ids = x[..., -1].long()  # (batch, cities)
        city_embeds = self.city_embed(city_ids)
        combined = torch.cat([features, city_embeds], dim=-1)
        cnn_input = combined.permute(0, 2, 1)
        
        # # CNN处理
        # cnn_out = self.cnn1(x.permute(0, 2, 1))  # 转换为(batch_size, input_size, seq_len)
        # cnn_out = cnn_out.permute(0, 2, 1)      # 恢复为(batch_size, new_seq_len, features)


        # cnn_out2 = self.cnn2(cnn_out.permute(0, 2, 1))  # 转换为(batch_size, input_size, seq_len)
        cnn_out2 = self.cnn2(cnn_input)  # 转换为(batch_size, input_size, seq_len)
        cnn_out2 = cnn_out2.permute(0, 2, 1)      # 恢复为(batch_size, new_seq_len, features)
        
        
        # 空间注意力 (城市间交互)
        spatial_out,spatial_attn_weights = self.spatial_attn(cnn_out2)  # [B, N, 128]
        
        # LSTM处理
        lstm_out, _ = self.lstm(spatial_out)  # lstm_out shape: (batch_size, seq_len, hidden_size)
        # 45个城市，21是特征
        # 时间注意力
        temporal_out,temporal_attn_weights = self.temporal_attn(lstm_out)  # [B, N, hidden_size]
        
        
        # 全连接层
        output = self.fc(temporal_out)
        if vis:
            return output,spatial_attn_weights,temporal_attn_weights
        else:
            return output # 保持输出维度 (batch_size, 1, output_size)
    def vis_weights(self,attn_weights,save_dir=None):
        import os
        import matplotlib.pyplot as plt

        # 可视化注意力权重
        # 不要计算单个head,而是进行汇总
        
        torch.set_printoptions(threshold=float('inf'))
        attn_matrix = attn_weights[2].detach().cpu().numpy()#有多个头
        attn_matrix = attn_matrix.sum(axis=0)
        
        
        
        plt.matshow(attn_matrix, cmap='viridis')  # 使用更清晰的配色

        # 添加颜色条和坐标轴标签
        cbar = plt.colorbar(shrink=0.8)
        cbar.set_label('Attention Strength', rotation=270, labelpad=15)
        plt.xlabel('Target City')
        plt.ylabel('Source City')

        # 标题和排版优化
        plt.title("Inter-city Attention Weights", fontsize=12, pad=20)

        # 保存设置
        if save_dir is None:
            save_dir = "./attention_visualizations"
            os.makedirs(save_dir, exist_ok=True) # 自动创建目录
        save_path = os.path.join(save_dir, "city_attn_weights.pdf")  # 矢量格式

        # 保存参数 (期刊级高清图)
        plt.savefig(
            "attn_weights.png",  # 保存文件名
            # format='pdf',          # 矢量格式便于论文使用
            dpi=300,               # 打印级分辨率
            bbox_inches='tight',   # 去除多余白边
            pad_inches=0.05        # 小幅边距
        )

        # 显示后关闭释放内存
        plt.show()
        plt.close()

        print(f"Attention weights saved to: {save_path}")
    
    
    
    
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
        
        self.norm1= nn.LayerNorm(embed_dim)  # 第一层归一化
        self.norm2= nn.LayerNorm(embed_dim)
        
        
    def forward(self,X):
        batch_size, seq_len, embed_dim = X.size()
         #线性映射
         
        #得有归一化
        X2 = self.norm1(X)
        
        
        
        q = self.norm2(self.query(X2))
        k = self.norm2(self.key(X2))
        v = self.norm2(self.value(X2))
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
        return output,attn_weights


class LSTM(nn.Module):
    def __init__(self, input_size, output_size,num_layers=3,hidden_dim=256,num_cities=45,embed_dim=1):
        super(LSTM, self).__init__()
        # self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True) #
        
        self.city_embed = nn.Embedding(
            num_embeddings=num_cities,  # 城市总数
            embedding_dim=embed_dim     # 推荐128-256维
        )
        
        self.lstm = nn.LSTM(
            input_size=input_size + embed_dim -1,
            # input_size=input_size, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        
        # features = x[..., :-1]  # (batch, cities, input_features)
        # city_ids = x[..., -1].long()  # (batch, cities)
        # city_embeds = self.city_embed(city_ids)
        # combined = torch.cat([features, city_embeds], dim=-1)

        combined=x
        
        out, _ = self.lstm(combined)  # out shape: [batch, seq_len, hidden_dim]
        out = self.fc(out)     # shape: [batch, seq_len, output_size]
        return out
    
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=256,num_cities=45,embed_dim=1):
        super(CNN_LSTM, self).__init__()

        self.city_embed = nn.Embedding(
            num_embeddings=num_cities,  # 城市总数
            embedding_dim=embed_dim     # 推荐128-256维
        )
        
        # CNN 模块
        self.cnn = nn.Sequential(
            nn.Conv1d(  # 一维卷积处理时间序列
                in_channels=input_size+embed_dim -1,  # 输入特征数
                out_channels=hidden_dim, 
                kernel_size=3,
                padding=1  # 保持序列长度不变
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=5,
                padding=2  # (5-1)/2=2
            ),
            nn.ReLU()
        )
        
        # LSTM 模块
        self.lstm = nn.LSTM(
            input_size=hidden_dim,  # 与 CNN 输出通道一致
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        """
        输入 x 形状: (batch_size, seq_len, input_size)
        输出形状: (batch_size, seq_len, output_size)
        """
        
        # features = x[..., :-1]  # (batch, cities, input_features)
        # city_ids = x[..., -1].long()  # (batch, cities)
        # city_embeds = self.city_embed(city_ids)
        # combined = torch.cat([features, city_embeds], dim=-1)
        # cnn_input = combined.permute(0, 2, 1)
        
        # 调整维度以适配 Conv1d
        x = x.permute(0, 2, 1)  # [batch, input_size, seq_len]
        
        # 通过 CNN 提取特征
        # cnn_out = self.cnn(cnn_input)  # [batch, hidden_dim, seq_len]
        cnn_out = self.cnn(x)
        
        # 恢复维度以适配 LSTM
        cnn_out = cnn_out.permute(0, 2, 1)  # [batch, seq_len, hidden_dim]
        
        # 通过 LSTM
        lstm_out, _ = self.lstm(cnn_out)  # [batch, seq_len, hidden_dim]
        
        # 全连接层输出
        output = self.fc(lstm_out)  # [batch, seq_len, output_size]
        
        return output