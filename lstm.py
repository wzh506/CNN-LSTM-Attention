import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# =====================
# 1. 时间序列数据构建
# =====================
# 定义时间窗口参数
TIME_STEPS = 3  # 用过去3年预测当年

# 按省份生成序列数据
def create_sequences(data, time_steps):
    X, y = [], []
    for province in data['province'].unique():
        province_data = data[data['province']==province].sort_values('year')
        for i in range(len(province_data)-time_steps):
            X.append(province_data.iloc[i:i+time_steps, 2:11].values)  # 特征窗口
            y.append(province_data.iloc[i+time_steps, 11:14].values)   # 目标值
    return np.array(X), np.array(y)

# 生成序列数据集
X_seq, y_seq = create_sequences(df, TIME_STEPS)

# =====================
# 2. 数据预处理
# =====================
# 标准化处理
scaler_X = StandardScaler().fit(X_seq.reshape(-1, X_seq.shape[2]))
scaler_y = StandardScaler().fit(y_seq)
X_scaled = scaler_X.transform(X_seq.reshape(-1, X_seq.shape[2])).reshape(X_seq.shape)
y_scaled = scaler_y.transform(y_seq)

# 划分训练测试集（按时间顺序）
train_size = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# =====================
# 3. LSTM模型构建
# =====================
model = Sequential()
model.add(LSTM(64, input_shape=(TIME_STEPS, X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(3))  # 同时预测三个目标

model.compile(optimizer='adam', loss='mse')

# =====================
# 4. 模型训练与验证
# =====================
history = model.fit(X_train, y_train, 
                   epochs=50, 
                   batch_size=32,
                   validation_split=0.2,
                   verbose=1)

# =====================
# 5. 结果评估
# =====================
y_pred = model.predict(X_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

print("LSTM 模型评估：")
for i, target in enumerate(['Wg', 'Wb', 'WF']):
    rmse = np.sqrt(mean_squared_error(y_test_inv[:,i], y_pred_inv[:,i]))
    r2 = r2_score(y_test_inv[:,i], y_pred_inv[:,i])
    print(f"{target}: RMSE={rmse:.3f}, R²={r2:.3f}")