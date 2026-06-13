

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
    
# 模型第二版：输入为[batch,window,cities,features+1]，最后一维的最后一个元素为城市ID   
class TimeEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.fixed_pe = self._create_fixed_pe(d_model, max_len)
        self.learn_scale = nn.Parameter(torch.ones(1, 1, d_model))
        
    def _create_fixed_pe(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False) 
    
    def forward(self, x):
        fixed_part = self.fixed_pe[x] 
        return fixed_part * self.learn_scale
    
class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer based on "Attention is All You Need"
    
    Args:
        embed_dim: Input and output dimensionality (d_model)
        num_heads: Number of attention heads
        ff_dim: Inner dimensionality of the position-wise FFN (default=4*embed_dim)
        dropout: Dropout rate (applied to attention weights and FFN)
        activation: Activation function in FFN ('relu' or 'gelu')
        batch_first: If True, (batch, seq, feat) else (seq, batch, feat)
    """
    def __init__(self, embed_dim, num_heads, ff_dim=None, 
                 dropout=0.1, activation="relu", batch_first=True):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * embed_dim  # Default expansion ratio
        
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=batch_first
        )
        
        # Position-wise FFN
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        
        # Normalization Layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Input sequence [seq_len, N, E] or [N, seq_len, E] if batch_first=True
            src_mask: Attention mask [seq_len, seq_len]
            src_key_padding_mask: Padding mask [N, seq_len]
        """
        # Self-Attention Block
        attn_output, attn_weights = self._sa_block(
            self.norm1(src), src_mask, src_key_padding_mask
        )
        src = src + self.dropout1(attn_output)

        # FFN Block
        src = src + self.dropout2(self._ff_block(self.norm2(src)))
        
        return src, attn_weights

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x, attn = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        return self.dropout3(x), attn

    def _ff_block(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout3(x)
        return self.linear2(x) 


class DCLFormer2(nn.Module):
    def __init__(self, input_size, output_size, window_size=4, 
                 cnn_kernel_size=3, lstm_hidden_size=256, 
                 num_layers=3, num_cities=45, city_embed_dim=8,feature_embed_dim=64,
                 time_embed_dim=4,cnn_layers=2, attn_heads=8):
        super(DCLFormer2, self).__init__()
        self.window_size = window_size
        self.num_cities = num_cities

        # 城市编码嵌入层
        self.city_embed = nn.Embedding(
            num_embeddings=num_cities,
            embedding_dim=city_embed_dim
        )
        self.feature_embed = nn.Linear(input_size - 1, feature_embed_dim)  # 输入特征维度减去城市ID
        
        # 时间位置编码
        self.time_enc = TimeEncoding(d_model=time_embed_dim, max_len=window_size)
        
        # # 空间CNN模块1，这里实际上不仅融合了相邻城市间的特征，其实还对城市本身的特征进行了融合
        # self.spatial_cnn = nn.Sequential(
        #     nn.Conv1d(input_size + city_embed_dim, lstm_hidden_size, kernel_size=cnn_kernel_size, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(lstm_hidden_size, lstm_hidden_size, kernel_size=cnn_kernel_size, padding=1),
        #     nn.ReLU()
        # )
        # 空间CNN模块2，这里仅对城市本身的特征进行融合
        # self.spatial_cnn = nn.Sequential(
        #     nn.Conv1d(input_size + city_embed_dim-1, lstm_hidden_size, kernel_size=1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv1d(lstm_hidden_size, lstm_hidden_size, kernel_size=1, padding=0),
        #     nn.ReLU()
        # )
        self.spatial_cnn = nn.Sequential(
            nn.Conv1d(feature_embed_dim+city_embed_dim, lstm_hidden_size, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(lstm_hidden_size, lstm_hidden_size, kernel_size=1, padding=0),
            nn.ReLU()
        )
        
        # 时间注意力模块
        # 使用transformerEncoderLayer实现吧
        self.temporal_self_attn = TransformerEncoderLayer(
            embed_dim=lstm_hidden_size, 
            num_heads=attn_heads, 
            batch_first=True
        )
        # self.temporal_self_attn = MultiHeadAttention(
        #     feature_dim=lstm_hidden_size,
        #     num_heads=attn_heads
        # )
        
        # self.spatial_attn = SpatialAttention(lstm_hidden_size, num_heads=8)
        # self.temporal_attn = MultiHeadAttention(lstm_hidden_size, num_heads=8)
        
        # LSTM 部分 (按城市单独处理),这里提取的其实也是时间特征，LSTM不会改变特征维度，但是会融合seq_len的特征（也就是window那一维度）
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size, 
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # 时空注意力机制
        self.spatial_attn = TransformerEncoderLayer(
            embed_dim=lstm_hidden_size, 
            num_heads=attn_heads, 
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x, vis=False):
        """
        x shape: [batch, window, cities, features+1]
        最后一个特征为城市ID (int)
        """
        batch_size, window_size, num_cities, _ = x.size()
        
        # 拆分特征和城市ID
        features = x[..., :-1]  # [batch, window, cities, features]
        # assert x[:,0,:,-1]==x[:,1,:,-1] #实际中必须相等，不然就
        city_ids = x[..., -1].long()  # [batch, window, cities] -> 转为LongTensor
    
        # 城市嵌入 [batch, window, cities, embed_dim]
        city_embeds = self.city_embed(city_ids)
        
        feature_embeds = self.feature_embed(features)
        # 加入时间位置编码 (为每个时刻增加编码)
        # 这里明显有问题
        # time_embeds = torch.stack([self.time_enc(torch.zeros(batch_size, num_cities, city_embeds.size(-1))).to(x.device) for _ in range(window_size)], dim=1)
        # city_time_embeds = city_embeds + time_embeds
        
        # 连接特征和嵌入 [batch, window, cities, features + embed_dim]
        combined = torch.cat([feature_embeds, city_embeds], dim=-1)
        # combined = features
        
        # 重组维度: [batch * window, cities, features + embed_dim]
        spatial_input = combined.view(batch_size * window_size, num_cities, -1)
        
        # === 空间处理 (按时间窗口独立处理不同的特征维度融合) ===
        # CNN处理空间特征 [batch*win, cities, feat+emb] -> [batch*win, cities, hidden]
        cnn_in = spatial_input.permute(0, 2, 1)  # [batch*win, feat+emb, cities]
        spatial_out = self.spatial_cnn(cnn_in)    # [batch*win, hidden, cities]
        spatial_out = spatial_out.permute(0, 2, 1)  # [batch*win, cities, hidden]
        
        # === 时间注意力处理 ===
        # 重组回时间维度: [batch, window, cities, hidden]
        temporal_input = spatial_out.view(batch_size, window_size, num_cities, -1)
        
        # 时序注意力 (每个城市独立处理)
        # 转换维度: [batch, cities, window, hidden]
        temporal_input = temporal_input.permute(0, 2, 1, 3) 
        temporal_input = temporal_input.reshape(batch_size * num_cities, window_size, -1)
        # 这里务必注意，如何写如何做
        # 时间自注意力 [batch*cities, window, hidden],这个主要是融合Window的时间注意力
        temporal_out, temporal_attn = self.temporal_self_attn(
            temporal_input)  # transformerEncoderLayer返回的输出是[batch*cities, window, hidden]
        # 输出为： # [batch*cities, window, hidden]
        # === LSTM 时序建模 ===
        lstm_out, _ = self.lstm(temporal_out)  # [batch*cities, window, hidden]
        
        # lstm_out = torch.cat([lstm_out, city_embeds], dim=-1)  # [batch*cities, window, hidden+embed]
        # 取最后一个时间步作为输出 [batch*cities, hidden]
        lstm_out = lstm_out[:, -1, :].view(batch_size, num_cities, -1)
        
        # lstm_out = torch.cat([lstm_out, city_embeds], dim=-1)  # [batch*cities, window, hidden+embed]
        # === 空间注意力 (城市间交互) ===
        spatial_attn_out, spatial_attn = self.spatial_attn(
            lstm_out)  # [batch, cities, hidden]
        
        # 全连接输出层 [batch, cities, output_size]
        output = self.fc(spatial_attn_out)
        
        if vis:
            # 重组注意力权重用于可视化 [batch, cities, cities]
            spatial_attn = spatial_attn.view(batch_size, num_cities, num_cities)
            return output, spatial_attn, temporal_attn
        else:
            return output
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
    def __init__(self, input_size, output_size,num_layers=3,hidden_dim=256,num_cities=45,embed_dim=128):
        super(LSTM, self).__init__()
        # self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True) #
        
        self.city_embed = nn.Embedding(
            num_embeddings=num_cities,  # 城市总数
            embedding_dim=embed_dim     # 推荐128-256维
        )
        
        self.lstm = nn.LSTM(
            input_size=input_size -1,
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
        
        out, _ = self.lstm(combined[:,:,:-1])  # out shape: [batch, seq_len, hidden_dim]
        out = self.fc(out)     # shape: [batch, seq_len, output_size]
        return out
    
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=256,num_cities=45,embed_dim=128):
        super(CNN_LSTM, self).__init__()

        self.city_embed = nn.Embedding(
            num_embeddings=num_cities,  # 城市总数
            embedding_dim=embed_dim     # 推荐128-256维
        )
        
        # CNN 模块
        self.cnn = nn.Sequential(
            nn.Conv1d(  # 一维卷积处理时间序列
                in_channels=input_size-1,  # 输入特征数
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
        x2 = x[..., :-1]
        x2 = x2.permute(0, 2, 1)  # [batch, input_size, seq_len]
        
        
        # 通过 CNN 提取特征
        # cnn_out = self.cnn(cnn_input)  # [batch, hidden_dim, seq_len]
        cnn_out = self.cnn(x2)
        
        # 恢复维度以适配 LSTM
        cnn_out = cnn_out.permute(0, 2, 1)  # [batch, seq_len, hidden_dim]
        
        # 通过 LSTM
        lstm_out, _ = self.lstm(cnn_out)  # [batch, seq_len, hidden_dim]
        
        # 全连接层输出
        output = self.fc(lstm_out)  # [batch, seq_len, output_size]
        
        return output
    
if __name__ == "__main__":
    model = DCLFormer2(
        input_size=21, 
        output_size=1, 
        window_size=4,  # 新增参数：时间窗口大小
        num_cities=45
    )

    # 输入数据 [batch=8, window=10, cities=45, features=20 + 1]
    x = torch.ones(8, 4, 45, 21)

    # 前向传播
    output = model(x)
    print(output.shape)  # torch.Size([8, 45, 1])