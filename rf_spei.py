# ========== RF模型 ==========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 设置随机种子保证可复现性
np.random.seed(42)

# 数据生成（与SVR完全相同）
provinces = [f'Province_{i}' for i in range(1,32)]
years = range(1980, 2014)
features = ['lrad','prec','srad','Tmax','Tmin','wind','SPEI','ET0','Yield']
targets = ['Wg','Wb','WF']

data = []
for province in provinces:
    features_data = np.random.normal(50, 20, (34, len(features)))
    Wg = 2.5 * features_data[:, 6] + np.random.normal(0, 5, 34)
    Wb = 1.8 * features_data[:, 6] + np.random.normal(0, 3, 34)
    WF = 3.2 * features_data[:, 6] + np.random.normal(0, 4, 34)
    for idx, year in enumerate(years):
        row = [province, year] + list(features_data[idx]) + [Wg[idx], Wb[idx], WF[idx]]
        data.append(row)

df = pd.DataFrame(data, columns=['province','year']+features+targets)

# 特征选择与预处理
X = df[['SPEI']].values
y = df[targets].values

scaler_X = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y)
X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# 模型训练
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# 预测评估
y_pred = rf_model.predict(X_test)
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)

# 结果输出
print("\nRF模型评估结果：")
for i, target in enumerate(targets):
    mse = mean_squared_error(y_test_inv[:,i], y_pred_inv[:,i])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv[:,i], y_pred_inv[:,i])
    print(f"{target}: MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")

# 可视化
plt.figure(figsize=(15,5))
for i, target in enumerate(targets):
    plt.subplot(1,3,i+1)
    plt.scatter(y_test_inv[:,i], y_pred_inv[:,i], alpha=0.6, color='orange')
    plt.plot([min(y_test_inv[:,i]), max(y_test_inv[:,i])], 
             [min(y_test_inv[:,i]), max(y_test_inv[:,i])], 'r--')
    plt.title(f'RF - {target}')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
plt.tight_layout()
plt.show()