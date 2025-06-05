
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 自注意力层
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # 转换回来
        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # 适配 attn.shape = [B, H, N, N] 或 [B, H, T, S]
            attn = attn.masked_fill(mask, -1e9)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, attn # torch.Size([1, 2, 3, 3]) 【B,N,D,D]

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads)
        self.enc_dec_attn = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim)

        self.attn_weights = None  # 存储注意力权重

    def forward(self, tgt, memory, src_mask=None):
        # 自注意力（带因果掩码）
        def generate_causal_mask(size):
            # 生成一个上三角矩阵（不包括主对角线），用于自注意力的因果掩码
            return torch.triu(torch.ones(size, size), diagonal=1).bool()
        causal_mask = generate_causal_mask(tgt.size(1)).to(tgt.device)
        tgt2, _ = self.self_attn(self.norm1(tgt), mask=causal_mask)
        tgt = tgt + tgt2

        # 编码器-解码器注意力（带源序列掩码）
        tgt2, attn_weights = self.enc_dec_attn(self.norm2(tgt), mask=src_mask)
        tgt = tgt + tgt2
        self.attn_weights = attn_weights  # 保存注意力权重

        # 前馈网络
        tgt = tgt + self.ffn(self.norm3(tgt))
        return tgt

# 编码器
class Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ffn_dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 解码器
class Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, ffn_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory,src_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return self.norm(tgt)

# 整体模型
class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=64, num_heads=2, ffn_dim=128, num_layers=1):
        super().__init__()
        self.encoder = Encoder(num_layers, embed_dim, num_heads, ffn_dim)
        self.decoder = Decoder(num_layers, embed_dim, num_heads, ffn_dim)

        self.src_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, tgt_vocab_size)

        self.attn_weights = None

    def forward(self, src, tgt):
        # 编码器
        src_emb = self.src_embed(src)
        memory = self.encoder(src_emb)
        
        src_mask = (src == 0)  # [batch_size, src_seq_len]

        # 解码器
        tgt_emb = self.tgt_embed(tgt)
        tgt_emb = self.decoder(tgt_emb, memory,src_mask=src_mask)

        # 输出
        logits = self.output(tgt_emb)

        # 保存注意力权重（取最后一个解码层）
        self.attn_weights = self.decoder.layers[-1].attn_weights

        return logits




def visualize_attention(model, src_indices, tgt_indices):
    model.eval()
    with torch.no_grad():
        src = torch.tensor([src_indices])
        tgt = torch.tensor([tgt_indices])

        _ = model(src.cuda(), tgt.cuda())  # 前向传播触发注意力权重记录

        if model.attn_weights is not None:
            # 取第一个注意力头
            attn_weights = model.attn_weights[0].cpu().numpy()

            # 创建标签
            src_words = [vocab_en[i] for i in src_indices]
            tgt_words = [vocab_fr[i] for i in tgt_indices]

            # 绘制热力图
            plt.figure(figsize=(10, 8))
            # 可视化第一个
            sns.heatmap(attn_weights[1], 
                        xticklabels=src_words,
                        yticklabels=src_words,
                        annot=True,
                        fmt=".2f",
                        cmap="YlGnBu",
                        cbar=False)
            plt.title("attn_weights")
            plt.xlabel("src (English)")
            plt.ylabel("src_words (English)")
            plt.tight_layout()
            plt.savefig("manual_attention_weights.png")
            plt.show()
            print("注意力权重已保存为 manual_attention_weights.png")
        else:
            print("未能获取注意力权重！请检查模型是否正确保存。")
# 示例词典
vocab_en = {0: '<pad>', 1: 'hello', 2: 'world'}
vocab_fr = {0: '<pad>', 1: 'bonjour', 2: 'le', 3: 'monde'}

# 示例数据
data = [
    (torch.tensor([1, 2]), torch.tensor([0, 1, 2, 3])),  # hello world -> bonjour le monde
    (torch.tensor([2]), torch.tensor([0, 3]))            # world -> monde
]

# 模型初始化
model = EncoderDecoder(src_vocab_size=3, tgt_vocab_size=4).cuda()

# 损失函数与优化器
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
def train():
    model.train()
    for epoch in range(200):
        total_loss = 0
        for src, tgt in data:
            src = src.unsqueeze(0).cuda()
            tgt = tgt.unsqueeze(0).cuda()
            output = model(src, tgt[:, :-1])  # 解码器输入为 tgt[:-1]

            loss = criterion(output.view(-1, 4).cuda(), tgt[:, 1:].view(-1).cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data):.4f}")

# 执行训练
print("开始训练...")
train()

# 可视化注意力权重
print("可视化注意力权重...")
visualize_attention(model, [1, 2], [0, 1, 2, 3])