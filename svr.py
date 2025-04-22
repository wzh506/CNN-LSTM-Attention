import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# =====================
# 1. 模拟数据生成
# =====================
provinces = [f'Province_{i}' for i in range(1,32)]  # 31个省市
years = range(1980, 2014)  # 34年
features = ['lrad','prec','srad','Tmax','Tmin','wind','SPEI','ET0','Yield']
targets = ['Wg','Wb','WF']

# 生成完整数据集
data = []
for province in provinces:
    # 生成特征数据（正态分布）
    features_data = np.random.normal(loc=50, scale=20, size=(34, len(features)))
    
    # 生成目标变量（基于特征的线性组合+噪声）
    Wg = 0.3*features_data[:,0] + 0.2*features_data[:,1] + np.random.normal(0, 5, 34)
    Wb = 0.5*features_data[:,2] - 0.1*features_data[:,3] + np.random.normal(0, 3, 34)
    WF = 0.4*features_data[:,4] + 0.3*features_data[:,5] + np.random.normal(0, 4, 34)
    
    # 组装数据块
    for idx, year in enumerate(years):
        row = [province, year] + list(features_data[idx]) + [Wg[idx], Wb[idx], WF[idx]]
        data.append(row)

# 创建DataFrame
df = pd.DataFrame(data, 
                 columns=['province','year'] + features + targets)

# =====================
# 2. 数据预处理
# =====================
# 划分特征和目标
X = df.iloc[:, 2:11]  # 第3-11列为特征
y = df.iloc[:, 11:14] # 第12-14列为目标

# 数据标准化
scaler_X = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y)
X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y)

# 划分训练测试集（保留最后5年作为测试集）
train_idx = [i for i in range(len(df)) if (df.iloc[i]['year'] < 2009)]
test_idx = [i for i in range(len(df)) if (df.iloc[i]['year'] >= 2009)]

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

# =====================
# 3. 模型训练与预测
# =====================
models = {
    'Wg': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'Wb': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'WF': SVR(kernel='rbf', C=1.0, epsilon=0.1)
}

results = []
for target_idx, target_name in enumerate(targets):
    # 训练模型
    models[target_name].fit(X_train, y_train[:, target_idx])
    
    # 预测结果
    y_pred = models[target_name].predict(X_test)
    
    # 评估指标
    rmse = np.sqrt(mean_squared_error(y_test[:, target_idx], y_pred))
    r2 = r2_score(y_test[:, target_idx], y_pred)
    
    # 存储结果
    results.append({
        'Target': target_name,
        'RMSE': rmse,
        'R2': r2,
        'Model': models[target_name]
    })

# =====================
# 4. 结果展示
# =====================
print("\n模型评估结果：")
for res in results:
    print(f"{res['Target']}模型: RMSE={res['RMSE']:.3f}, R²={res['R2']:.3f}")

# =====================
# 5. 结果验证示例
# =====================
# 随机选取一个测试样本
sample_idx = 10
sample_province = df.iloc[test_idx[sample_idx]]['province']
sample_year = df.iloc[test_idx[sample_idx]]['year']

print(f"\n验证样本：{sample_province} {sample_year}年")
print("原始值：", scaler_y.inverse_transform(y_test[sample_idx].reshape(1,-1))[0])
print("预测值：", scaler_y.inverse_transform(
    np.array([[m.predict(X_test[sample_idx].reshape(1,-1))[0] 
              for m in models.values()]]))[0])