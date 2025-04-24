import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# 读取 Excel 文件
df = pd.read_excel('merged_data_2.xlsx')

# 定义输入和输出列
input_cols = ['lrad', 'prec', 'srad', 'Tmax', 'Tmin', 'wind', 'SPEI', 'Kc', 'sa', 'VPD', 'RH']
output_cols = ['Wg', 'Wb']

# 分组并构造序列
X, y = [], []
for year, group in df.groupby('year'):
    if group.shape[0] < 2:
        continue
    X.append(group[input_cols].values)
    y.append(group[output_cols].values)

X = np.array(X)  # shape: [years, cities, features]
y = np.array(y)  # shape: [years, cities, targets]

# 将数据整理为 时序输入！



# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, len(input_cols))).reshape(X.shape)

# 转为 PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)#10:1的比例差不多

# 构建 LSTM 模型
# seq=27，是城市的数量，input_size=11是特征的数量
# 问题：现在是做的城市的融合，没有做时间上的融合，需要将输入数据整理为
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True) #
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # out shape: [batch, seq_len, hidden_dim]
        out = self.fc(out)     # shape: [batch, seq_len, output_dim]
        return out

# 初始化模型
model = LSTMModel(input_dim=len(input_cols), hidden_dim=64, output_dim=len(output_cols))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)

# 训练模型
for epoch in tqdm(range(20),desc='Training'):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model,'lstm.pth')

model2 = torch.load('lstm.pth', map_location=torch.device('cpu'))
# 测试模型
model.to('cpu')
model.eval()
with torch.no_grad():
    preds = model(X_test)
    test_loss = criterion(preds, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")


