
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

# =====================
# 1. 读取数据
# =====================
file_path = r'final_data.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# =====================
# 2. 数据预处理与特征工程
# =====================
# 添加降水滞后特征（以1年滞后为例）
#df['prec_lag1'] = df.groupby('city')['prec'].shift(1)
#df.fillna(method='bfill', inplace=True)  # 后向填充缺失值

# 添加时间相关特征（示例：年份的周期性编码）



df['year_sin'] = np.sin(2 * np.pi * (df['year'] - 2000) / 3)  # 23为周期参数
df['year_cos'] = np.cos(2 * np.pi * (df['year'] - 2000) / 3)

# 定义各目标的特征（示例配置，请根据实际数据调整）
features_Wb = ['VPD','RH',  'srad','year_sin', 'year_cos']

features_Wg = ['prec', 'Tmax','wind',  'year_sin', 'year_cos']

categorical_features = ['city']

# =====================
# 3. 数据预处理
# =====================
# 创建两个独立的预处理管道
preprocessor_Wg = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_Wg),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

preprocessor_Wb = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_Wb),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 划分数据集
train_idx = df[df['year'] < 2010].index
test_idx = df[df['year'] >= 2010].index

# 分别处理两个目标的特征
X_train_Wg = df.loc[train_idx, features_Wg + categorical_features]
X_test_Wg = df.loc[test_idx, features_Wg + categorical_features]

X_train_Wb = df.loc[train_idx, features_Wb + categorical_features]
X_test_Wb = df.loc[test_idx, features_Wb + categorical_features]

y = df[['Wg', 'Wb']]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

# 预处理数据
X_train_Wg_processed = preprocessor_Wg.fit_transform(X_train_Wg)
X_test_Wg_processed = preprocessor_Wg.transform(X_test_Wg)

X_train_Wb_processed = preprocessor_Wb.fit_transform(X_train_Wb)
X_test_Wb_processed = preprocessor_Wb.transform(X_test_Wb)

# 目标变量标准化
scaler_y = StandardScaler().fit(y_train)
y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# =====================
# 4. 模型训练与参数优化
# =====================
# 扩展参数网格
param_grid = {
    'kernel': ['rbf', 'poly'],  # 添加多项式核
    'C': [0.1, 1, 10, 100, 500, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
    'epsilon': [0.01, 0.05, 0.1, 0.2, 0.3],
    'degree': [2, 3]  # 多项式阶数
}

# 训练Wg模型
print('开始训练Wg模型，请稍等...')
grid_search_Wg = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5,
                              scoring='neg_mean_squared_error', n_jobs=-1)
print('正式开始训练')
grid_search_Wg.fit(X_train_Wg_processed, y_train_scaled[:, 0])
best_Wg = grid_search_Wg.best_estimator_
print('训练Wg模型结束！')

# 训练Wb模型
print('开始训练Wb模型，请稍等...')
grid_search_Wb = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5,
                              scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_Wb.fit(X_train_Wb_processed, y_train_scaled[:, 1])
print('训练Wb模型结束！')
best_Wb = grid_search_Wb.best_estimator_

# =====================
# 5. 模型评估
# =====================


def calculate_metrics(y_true, y_pred):
    """计算所有评估指标"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (y_true.max() - y_true.min())
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r_squared = r ** 2

    # 计算NSE
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1 - (numerator / denominator)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'NRMSE': nrmse,
        'R²': r_squared,
        'NSE': nse
    }


# 预测并计算指标
results = []

# Wg模型
y_pred_Wg = best_Wg.predict(X_test_Wg_processed) * scaler_y.scale_[0] + scaler_y.mean_[0]
y_true_Wg = y_test['Wg'].values
results.append({'Target': 'Wg', **calculate_metrics(y_true_Wg, y_pred_Wg)})

# Wb模型
y_pred_Wb = best_Wb.predict(X_test_Wb_processed) * scaler_y.scale_[1] + scaler_y.mean_[1]
y_true_Wb = y_test['Wb'].values
results.append({'Target': 'Wb', **calculate_metrics(y_true_Wb, y_pred_Wb)})

# 打印结果
print("\n模型评估结果：")
for res in results:
    print(f"{res['Target']}模型:")
    print(f"  MAE: {res['MAE']:.3f}")
    print(f"  RMSE: {res['RMSE']:.3f}")
    print(f"  NRMSE: {res['NRMSE']:.3f}")
    print(f"  R²: {res['R²']:.3f}")
    print(f"  NSE: {res['NSE']:.3f}\n")

# =====================
# 6. 结果可视化
# =====================
plt.figure(figsize=(12, 6))

# Wg结果
plt.subplot(1, 2, 1)
plt.scatter(y_true_Wg, y_pred_Wg, alpha=0.6)
plt.plot([min(y_true_Wg), max(y_true_Wg)], [min(y_true_Wg), max(y_true_Wg)], 'k--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Wg Prediction Performance')

# Wb结果
plt.subplot(1, 2, 2)
plt.scatter(y_true_Wb, y_pred_Wb, alpha=0.6)
plt.plot([min(y_true_Wb), max(y_true_Wb)], [min(y_true_Wb), max(y_true_Wb)], 'k--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Wb Prediction Performance')

plt.tight_layout()
plt.show()
# 开发时间:2025/4/2315:33
