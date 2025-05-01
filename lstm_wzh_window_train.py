import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# 读取 Excel 文件
df = pd.read_excel('final_data.xlsx')

window_size = 4 #选择3年作为窗口大小

features = ['lrad', 'prec', 'srad', 'Tmax', 'Tmin', 'wind', 'SPEI','SSD', 'VPD', 'RH']
targets = ['Wg', 'Wb']
combined_cols = features + targets

# 构建映射：year -> 城市顺序保持一致的数据
year_dict = {year: group.reset_index(drop=True) for year, group in df.groupby('year')}
years = sorted(year_dict.keys())

X_seq, y_seq = [], []

for i in range(window_size, len(years)):
    y_now = year_dict[years[i]]
    y_prev1 = year_dict[years[i - 1]]
    y_prev2 = year_dict[years[i - 2]]
    y_prev3 = year_dict[years[i - 3]]

    # 确保城市数一致（可加其他校验）
    if not (len(y_now) == len(y_prev1) == len(y_prev2) == len(y_prev3)):
        continue

    # 拼接顺序为：[prev3][prev2][prev1][curr_features]
    input_features = np.concatenate([
        y_prev3[combined_cols].values,
        y_prev2[combined_cols].values,
        y_prev1[combined_cols].values,
        y_now[features].values,
    ], axis=1)  # shape: [num_cities, 3*14 + 12 = 54]

    target_output = y_now[targets].values  # shape: [num_cities, 2]

    X_seq.append(input_features)
    y_seq.append(target_output)

# 将数据整理为 时序输入！

X = np.array(X_seq)  # shape: [samples, num_cities, input_dim]
y = np.array(y_seq)  # shape: [samples, num_cities, 2]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# 转为 PyTorch tensor
X_tensor = torch.tensor(X_scaled , dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.05, random_state=42)#10:1的比例差不多

# 构建 LSTM 模型
# seq=27，是城市的数量，input_size=11是特征的数量
# 问题：现在是做的城市的融合，没有做时间上的融合，需要将输入数据整理为时序输入数据
# to do：
# 1.考虑CNN+LSTM+Attentino模型结构，在LSTM前先加卷积,在加attention
# 2.考虑位置编码
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
model = LSTMModel(input_dim=X.shape[-1], hidden_dim=128, output_dim=y.shape[-1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)

# 训练模型
for epoch in tqdm(range(80000),desc='Training'):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    if (epoch + 1) % 1000 == 0:
        torch.save(model,f'lstm_{epoch}.pth')

torch.save(model,'lstm.pth')

model2 = torch.load('lstm.pth', map_location=torch.device('cpu'))
# 测试模型
model2.to('cpu')
model2.eval()
with torch.no_grad():
    preds = model2(X_test)
    test_loss = criterion(preds, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

print("\nRF模型评估结果：")
for i, target in enumerate(y_test):
    mse = mean_squared_error(y_test[i,:], preds[i,:])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test[i,:], preds[i,:]) #R2
    print(f"{target}: MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")


