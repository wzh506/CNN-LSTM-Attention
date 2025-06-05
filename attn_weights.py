import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

# 配置可复现性
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 自定义数据集
class SequenceDataset(Dataset):
    def __init__(self, num_samples=2000, seq_length=30, num_classes=50, num_features=128):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.num_features = num_features
        self.data = torch.randn(num_samples, seq_length, num_features)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 简化版Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, num_features=128, num_heads=4, num_layers=3, num_classes=50):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(num_features, num_features)
        self.positional_encoding = self._create_positional_encoding(max_len=100, d_model=num_features)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features, 
            nhead=num_heads, 
            dim_feedforward=512,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.attention_weights = {}  # 存储注意力权重
        self.fc = nn.Linear(num_features, num_classes)
    
    def _create_positional_encoding(self, max_len=100, d_model=128):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # 为每一层注册钩子
        self.attention_weights = {}
        for i, layer in enumerate(self.encoder.layers):
            def hook(module, input, output, i=i):
                # 获取多头注意力的权重
                attn_weights = layer.self_attn(output[0][1])
                self.attention_weights[i] = attn_weights.detach()
            layer.self_attn.register_forward_hook(hook)
        
        x = self.encoder(x)
        x = x.mean(dim=1)  # 平均池化
        x = self.fc(x)
        return x

# 训练函数
def train_model(model, dataloader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {accuracy:.2f}%')
    
    return model

# 可视化注意力权重
def visualize_attention(model, data_loader, num_samples=3):
    device = next(model.parameters()).device
    model.eval()
    
    # 获取几个样本
    samples = []
    for i, (data, targets) in enumerate(data_loader):
        if i >= num_samples:
            break
        samples.append((data, targets))
    
    # 为每个样本创建可视化
    with torch.no_grad():
        for sample_idx, (data, targets) in enumerate(samples):
            data, targets = data.to(device), targets.to(device)
            
            # 前向传播（这会填充attention_weights字典）
            output = model(data)
            
            # 获取预测标签
            _, predicted = output.max(1)
            
            # 对每个Transformer层进行可视化
            num_layers = len(model.encoder.layers)
            fig, axes = plt.subplots(1, num_layers, figsize=(20, 5))
            
            if num_layers == 1:
                axes = [axes]
            
            fig.suptitle(f'Attention Visualization - Sample {sample_idx+1}\n' 
                         f'True Label: {targets[0].item()}, Predicted: {predicted[0].item()}', fontsize=16)
            
            for layer_idx in range(num_layers):
                # 获取该层的注意力权重
                if layer_idx in model.attention_weights:
                    attn_weights = model.attention_weights[layer_idx][0].cpu().numpy()
                    
                    # 绘制热力图
                    sns.heatmap(attn_weights, 
                                cmap='viridis',
                                annot=False,
                                fmt='.2f',
                                ax=axes[layer_idx],
                                cbar=(layer_idx == num_layers-1))
                    
                    axes[layer_idx].set_title(f'Layer {layer_idx+1}')
                    axes[layer_idx].set_xlabel('Key Positions')
                    axes[layer_idx].set_ylabel('Query Positions')
                else:
                    axes[layer_idx].text(0.5, 0.5, 'No Attention Weights',
                                        ha='center', va='center', fontsize=12)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f'attention_sample_{sample_idx+1}.png', dpi=300)
            plt.show()

# 主程序
def main():
    # 参数配置
    num_features = 128
    num_classes = 50
    seq_length = 20
    batch_size = 32
    num_heads = 8
    num_layers = 4
    epochs = 15

    # 创建数据集和数据加载器
    dataset = SequenceDataset(
        num_samples=2000,
        seq_length=seq_length,
        num_classes=num_classes,
        num_features=num_features
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = TransformerModel(
        num_features=num_features,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes
    )

    # 训练模型
    print("开始训练模型...")
    trained_model = train_model(model, data_loader, epochs=epochs)
    print("训练完成!")

    # 可视化注意力
    print("可视化注意力权重...")
    visualize_attention(trained_model, data_loader, num_samples=3)

if __name__ == "__main__":
    main()