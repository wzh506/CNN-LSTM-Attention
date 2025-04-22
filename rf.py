import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# =====================
# 1. 模拟数据生成（新增部分）
# =====================
provinces = [f'Province_{i}' for i in range(1,32)]  # 31个省市
years = range(1980, 2014)  # 34年
features = ['lrad','prec','srad','Tmax','Tmin','wind','SPEI','ET0','Yield']
targets = ['Wg','Wb','WF']

# 生成完整数据集
data = []
for province in provinces:
    features_data = np.random.normal(loc=50, scale=20, size=(34, len(features)))
    
    Wg = 0.3*features_data[:,0] + 0.2*features_data[:,1] + np.random.normal(0, 5, 34)
    Wb = 0.5*features_data[:,2] - 0.1*features_data[:,3] + np.random.normal(0, 3, 34)
    WF = 0.4*features_data[:,4] + 0.3*features_data[:,5] + np.random.normal(0, 4, 34)
    
    for idx, year in enumerate(years):
        row = [province, year] + list(features_data[idx]) + [Wg[idx], Wb[idx], WF[idx]]
        data.append(row)

df = pd.DataFrame(data, columns=['province','year'] + features + targets)

# =====================
# 2. 数据预处理（修正部分）
# =====================
X = df[features].values  # 直接使用特征列名称
y = df[targets].values

scaler_X = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y)
X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y)

# 按时间划分训练测试集（保留最后5年）
test_years = [2013, 2012, 2011, 2010, 2009]  # 修改为实际存在的年份
test_mask = df['year'].isin(test_years)

X_train, X_test = X_scaled[~test_mask], X_scaled[test_mask]
y_train, y_test = y_scaled[~test_mask], y_scaled[test_mask]

# =====================
# 3. 模型训练与评估（优化部分）
# =====================
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1  # 启用并行计算
)

# 多输出直接训练
rf.fit(X_train, y_train)

# 预测与反标准化
y_pred = rf.predict(X_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

# 结果展示（优化格式）
print("\n随机森林模型评估：")
metrics = []
for i, target in enumerate(targets):
    rmse = np.sqrt(mean_squared_error(y_test_inv[:,i], y_pred_inv[:,i]))
    r2 = r2_score(y_test_inv[:,i], y_pred_inv[:,i]))
    metrics.append(f"{target}: RMSE={rmse:.2f}, R²={r2:.3f}")

print('\n'.join(metrics))