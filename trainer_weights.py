import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from trainer import generate_data
import argparse
from tqdm import tqdm
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
import os
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        B, N, C = x.shape  # B: batch_size=19, N: seq_len=45, C: embed_dim=64
        
        # 3. 注意力计算
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # 分离QKV
        
        # 4. 转换为多头形状
        q = q.transpose(1, 2)  # [19, 4, 45, 16]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 5. 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [19, 4, 45, 45]
        
        # 6. 应用掩码（如果是自注意力）
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), -1e9)
        
        # 7. softmax
        attn = attn.softmax(dim=-1)  # [19, 4, 45, 45]
        
        # 8. 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [19, 45, 64]
        x = self.proj(x)
        return x, attn  # 返回注意力权重矩阵

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)[0]
        x = x + self.ffn(self.norm2(x))
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=17, output_dim=1, embed_dim=128, num_heads=4, num_layers=3,num_cities=45,city_embed_dim=4):
        super().__init__()
        # 输入投影层
        self.input_proj = nn.Linear(input_dim+city_embed_dim-1, embed_dim)
        self.output_proj = nn.Linear(output_dim, embed_dim)
        
        # 位置编码
        self.pos_enc = nn.Parameter(torch.randn(1, 45, embed_dim))  # [1, 45, 64]
        
        self.num_heads = num_heads
        # 编码器和解码器
        self.encoder = nn.ModuleList([EncoderLayer(embed_dim, self.num_heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([EncoderLayer(embed_dim, self.num_heads) for _ in range(num_layers)])
        
        # 输出层
        self.final_proj = nn.Linear(embed_dim, output_dim)
        
        # 城市编码嵌入层
        self.city_embed = nn.Embedding(
            num_embeddings=num_cities,  # 城市总数
            embedding_dim=city_embed_dim     # 推荐128-256维
        )
        # 注意力权重存储
        self.attn_weights = None

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: [19, 45, 17] - batch_size=19, seq_len=45, features=17
        # tgt: [19, 45, 1]  - batch_size=19, seq_len=45, features=1
        
        features = src[..., :-1]  # (batch, cities, input_features)
        city_ids = src[..., -1].long()  # (batch, cities)
        city_embeds = self.city_embed(city_ids)
        src_combined = torch.cat([features, city_embeds], dim=-1)
        
        # 1. 特征投影
        src_emb = self.input_proj(src_combined)  # [19, 45, 64]
        tgt_emb = self.output_proj(tgt)  # [19, 45, 64]
        
        # 2. 添加位置编码
        src_emb = src_emb + self.pos_enc  # [19, 45, 64]
        tgt_emb = tgt_emb + self.pos_enc  # [19, 45, 64]
        
        # 3. 编码器
        memory = src_emb
        for layer in self.encoder:
            memory = layer(memory, src_mask)  # [19, 45, 64]
        
        # 4. 解码器
        output = tgt_emb
        for i, layer in enumerate(self.decoder):
            if i == len(self.decoder) - 1: #取最后一层的作为结果
                output, self.attn_weights = layer.attn(layer.norm1(output), src_mask)
            else:
                output = layer(output, src_mask)
        
        # 5. 最终输出
        return self.final_proj(output)  # [19, 45, 1]

# def generate_data(seq_len=45, batch_size=19, input_dim=17, output_dim=1):
#     # 生成模拟数据 [batch_size, seq_len, features]
#     X = torch.randn(batch_size, seq_len, input_dim)
#     y = torch.randn(batch_size, seq_len, output_dim)
#     return X, y

def visualize_attention(attn_weights,current_cities,fpath=None):
    """可视化注意力权重"""
    
    for head in range(attn_weights.shape[1]):
        plt.figure(figsize=(12, 10))
        
        # 取第一个样本和指定的注意力头
        weights = attn_weights[-1].cpu().detach().numpy()[head]  # [45, 45] ,使用最后一年的
        
        # 创建时间标签
        time_steps = [f"t-{i}" for i in reversed(range(45))]
        cities = [current_cities[i] for i in range(len(current_cities))]
        # 绘制热力图
        sns.heatmap(weights, 
                    xticklabels=cities,
                    yticklabels=cities,
                    annot=True,
                    cmap="viridis",
                    cbar=True)
        
        plt.title(f"Attention Weights (Head {head})")
        plt.xlabel("Key Positions")
        plt.ylabel("Query Positions")
        plt.tight_layout()
        if fpath is None:
            fpath = "weights"
            if os.path.exists(fpath) is False:
                os.makedirs(fpath)
        plt.savefig(os.path.join(fpath, f"trainer_attention_weights_head{head}.png"))
        plt.show()
        print('注意力权重已保存为trainer_attention_weights.png')
    #最后绘制一张总的图形，希望能有帮助
    plt.figure(figsize=(12, 10))
    
    # 取第一个样本和指定的注意力头
    # weights = attn_weights[-1].cpu().detach().numpy()[head]  # [45, 45] ,使用最后一年的
    weights = attn_weights[-1].cpu().detach().numpy().mean(axis=0)  # [45, 45] ,使用最后一年的平均
    
    # 创建时间标签
    time_steps = [f"t-{i}" for i in reversed(range(45))]
    cities = [current_cities[i] for i in range(len(current_cities))]
    # 绘制热力图
    sns.heatmap(weights, 
                xticklabels=cities,
                yticklabels=cities,
                annot=True,
                cmap="viridis",
                cbar=True)
    
    plt.title(f"Attention Weights (Head mean)")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.tight_layout()
    if fpath is None:
        fpath = "weights"
        if os.path.exists(fpath) is False:
            os.makedirs(fpath)
    plt.savefig(os.path.join(fpath, "trainer_attention_weights_head_mean.png"))
    plt.show()
    print('注意力权重已保存为trainer_attention_weights.png')
    top3_values = np.partition(weights, -3, axis=1)[:, -3:]  # 每行最大的三个值（未排序）
    top3_indices = np.argpartition(weights, -3, axis=1)[:, -3:]  # 每行最大三个值的列索引（未排序）

    # 对每行的top3进行降序排序
    sorted_idx = np.argsort(-top3_values, axis=1)
    top3_values_sorted = np.take_along_axis(top3_values, sorted_idx, axis=1)
    top3_indices_sorted = np.take_along_axis(top3_indices, sorted_idx, axis=1)

    for i, (vals, idxs) in enumerate(zip(top3_values_sorted, top3_indices_sorted)):
        print(f"第{i}行最大3个值: {vals}，对应列号: {idxs}")
        print(f'对应城市：{current_cities[i]},最相关的城市分别为{current_cities[idxs[0]]},{current_cities[idxs[1]]},{current_cities[idxs[2]]}')
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size used in the training and validation loop.",
    )
    parser.add_argument(
        "--epochs", default=200000, type=int, help="Total number of epochs."
    )
    parser.add_argument(
        "--lr",
        default=0.0006,
        type=float,
        help="Base learning rate at the start of the training.",
    )
    parser.add_argument(
        "--ckpt", default=2000, type=int, help="Save model every ckpt epochs."
    )
    parser.add_argument(
        "--train_set_path", default="", type=str, help="Path to the training set."
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--dataset",
        default="huabei_1993to2017.xlsx",
        type=str,
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        type=str,
        help="Path to the save result.",
    )
    parser.add_argument(
        "--mod",
        default='DCLFormer',
        type=str,
        help="Model Type.",  
        choices=['DCLFormer', 'LSTM', 'CNN+LSTM'],
    )
    parser.add_argument(
        "--features",
        type=str,        # 指定每个元素的类型为字符串
        nargs='+',       # 接受一个或多个值,wins
        default=['prec', 'srad', 'Tmax', 'Tmin', 'wind', 'SPEI', 'VPD', 'RH'],
        help="Input features (space-separated strings). Example: --features prec srad Tmax"
    ) #可以这样写：--features prec srad Tmax，得到：['prec', 'srad', 'Tmax']
    parser.add_argument(
        "--sc",
        type=str,        # 指定每个元素的类型为字符串
        default=None,
        help="Input Scenes. Example: --sc 1.When sc is set,features will be ignored."
    ) #可以这样写：--features prec srad Tmax，得到：['prec', 'srad', 'Tmax']
    parser.add_argument(
        "--targets",
        type=str,        # 指定每个元素的类型为字符串
        nargs='+', 
        # default=['Wg','Wb'],
        default=['Wb'],
        help="The output targets.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="The window size.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["delete", "mean", "zero"],
        default="delete",
        help="The method to fix dataset.",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="0",
        help="The GPU ID to use.",
    )
    parser.add_argument(
        "--use_combined",
        action="store_true",  # 指定参数时变量值为 False
        help="Use previous targets as input features.",
    )
    parser.add_argument("--train", action="store_true", help="Whether to train the model.")
    parser.add_argument("--test", action="store_true", help="Whether to test the model.")
    parser.add_argument(
        "--task",
        default="wv3",
        type=str,
        choices=["wv3", "qb", "gf2"],
        help="Model to train (choices: wv3, qb, gf2).",
    )
    config = parser.parse_args()
    # train_model(config)
    
    current_cities, X_train, y_train, X_test, y_test, scaler, X,y = generate_data(config)


    # 创建模型
    model = TimeSeriesTransformer()
    
    model.cuda()
    X=X.cuda()
    y=y.cuda()
    # 生成示例数据
    # X, y = generate_data()
    X1 = X[0:-1]  # 去掉最后一列
    X2 = X[-1].unsqueeze(0)
    
    y1 = y[0:-1]  
    y2 = y[-1].unsqueeze(0)  # 保持最后一列的形状

    # 训练循环示例
    def train():
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # 生成因果掩码
        def generate_causal_mask(size):
            return torch.triu(torch.ones(size, size), diagonal=1).bool()
        
        # 训练步骤
        loss = 0
        for epoch in tqdm(range(2000)):
            optimizer.zero_grad()
            
            # 前向传播
            output = model(X1, y1)
            
            # 计算损失
            loss = criterion(output, y1)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # pbar.set_postfix({'loss': loss.item()})
            if epoch % 100 == 0:
                # print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 执行训练
    print("开始训练...")
    train()

    # 可视化注意力权重
    print("可视化注意力权重...")
    #进行一次前向传播，保存逻辑
    output2 = model(X2, y2)  # 使用最后一年的数据进行前向传播
    criterion = nn.MSELoss()
    loss = criterion(output2, y2)
    print(f'测试集上的Loss为: {loss.item():.4f}')
    visualize_attention(model.attn_weights,current_cities=current_cities)


