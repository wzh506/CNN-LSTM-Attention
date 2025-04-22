import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import time
import joblib
import matplotlib.pyplot as plt
# 记录开始时间
start_time = time.time()

# ===================== 数据加载与预处理 =====================
# 读取Excel数据（使用原始字符串避免转义）
data = pd.read_excel(
    r"D:\40aTu\scratch_sheng\history\merged_data.xlsx",
    sheet_name='Sheet1'
)

# 删除不需要的列（省份和年份）
data = data.drop(columns=['province', 'year'])

# 处理缺失值
data = data.dropna()

# 确保目标变量为数值型
data['Yield'] = data['Yield'].astype(float)

# 设置随机种子
np.random.seed(123)

# ===================== 特征工程 =====================
# 定义特征和目标
features = data[['lrad', 'prec', 'srad', 'Tmax', 'Tmin', 'wind', 'ET0', 'SPEI']]
target = data['Yield']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.3,
    random_state=123,
    shuffle=True
)

# ===================== 特征标准化 =====================
# 初始化标准化器
scaler = StandardScaler()

# 只在训练集上拟合标准化器
X_train_scaled = scaler.fit_transform(X_train)

# 转换测试集（使用训练集的均值和标准差）
X_test_scaled = scaler.transform(X_test)
# 保存标准化器（在原有代码的标准化部分后添加）
joblib.dump(scaler, r"D:\40aTu\scratch_sheng\history\feature_scaler245.pkl")
# ===================== 线性回归基准 =====================
lm_model = LinearRegression()
lm_model.fit(X_train_scaled, y_train)

# 预测与评估
y_pred_lm = lm_model.predict(X_test_scaled)
lm_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lm))
lm_r2 = r2_score(y_test, y_pred_lm)
print(f"\n[线性模型] RMSE: {lm_rmse:.2f}, R²: {lm_r2:.2f}")

# ===================== XGBoost模型优化 =====================
# 优化后的参数网格
xgb_grid = {
    'n_estimators': [1200, 1500],       # 增大树的数量以增强表达能力
    'learning_rate': [0.005, 0.01],     # 配合更多树使用更低学习率
    'max_depth': [3, 4],                # 限制树深防止过拟合
    'subsample': [0.7, 0.8],            # 降低采样率增加随机性
    'reg_alpha': [0.1, 0.5],            # 新增L1正则化
    'reg_lambda': [0.1, 0.5],           # 增强L2正则化
    'gamma': [0.2, 0.3]                 # 提高分裂最小损失减少过拟合
}

# 初始化带早停的模型
xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=123,
    early_stopping_rounds=50,
    n_jobs=-1
)

# 交叉验证设置
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    n_jobs=-1
)

# 训练模型
grid_search.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

# ===================== 结果输出 =====================
# 最优参数
best_params = grid_search.best_params_
print("\n[最优参数]")
print(pd.Series(best_params))

# 模型保存
joblib.dump(grid_search.best_estimator_, r"D:\40aTu\scratch_sheng\history\best_xgb_model245.pkl")

# 最终评估
best_model = grid_search.best_estimator_
y_pred_xgb = best_model.predict(X_test_scaled)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_r2 = r2_score(y_test, y_pred_xgb)
print(f"\n[XGBoost最终性能] RMSE: {xgb_rmse:.2f}, R²: {xgb_r2:.2f}")

# 特征重要性分析（使用原始特征名称）
feature_importance = pd.DataFrame({
    'Feature': features.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n[特征重要性]")
print(feature_importance)

# ===================== 预测结果可视化 =====================
plt.figure(figsize=(12, 6))

# 创建样本索引（仅用于可视化）
sample_indices = np.arange(len(y_test))

# 绘制真实值和预测值曲线
plt.plot(sample_indices, y_test.values, label='true value', color='#2c7bb6', linewidth=2, marker='o', markersize=5)
plt.plot(sample_indices, y_pred_xgb, label='predicted value', color='#d7191c', linewidth=2, linestyle='--', marker='^', markersize=5)

# 图表美化
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Yield(kg/ha)', fontsize=12)
plt.title('XGBoost Model Test Set Prediction Performance Comparison', fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 保存和显示图表
plot_path = r"D:\40aTu\scratch_sheng\history\prediction_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n预测对比图已保存至：{plot_path}")
plt.show()

# ===================== 运行时间统计 =====================
total_time = time.time() - start_time
print(f"\n总运行时间：{total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.2f}s")

