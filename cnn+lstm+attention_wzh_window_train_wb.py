import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# 读取 Excel 文件
import os
import pandas as pd
import numpy as np
from models import DCLFormer
# 读取数据
if __name__ == "__main__":



    df = pd.read_excel('final_data.xlsx')
    window_size = 3
    features = ['prec', 'srad', 'Tmax', 'Tmin', 'wind', 'SPEI', 'VPD', 'RH']
    targets = ['Wb']
    combined_cols = features + targets

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 获取所有城市列表（假设有city列）
    all_cities = df['city'].unique().tolist()

    # 创建包含所有城市的空DataFrame模板
    full_city_df = pd.DataFrame({'city': all_cities})

    # 构建年份字典（带填充）
    year_dict = {year: group.reset_index(drop=True) for year, group in df.groupby('year')}
    # 在填充前创建存在性记录字典
    existence_dict = {}
    years = sorted(year_dict.keys())
    for year in years:
        # 获取该年实际存在的城市
        existing_cities = set(year_dict[year]['city'])
        # 创建存在性数组（1表示存在，0表示填充）
        existence = np.array([1 if c in existing_cities else 0 for c in all_cities])
        existence_dict[year] = existence

    # 转换为三维存在性张量（年份 × 城市 × 存在性）
    existence_tensor = np.stack([existence_dict[y] for y in sorted(years)])
    existence_tensor = existence_tensor[window_size:] #前面几年没有用上

    year_dict = {}
    city_features_mean = {}  # 存储每个城市的特征均值

    # 第一步：计算每个城市的跨年特征均值
    for city in all_cities:
        city_data = df[df['city'] == city][combined_cols]
        city_features_mean[city] = {
            'features': city_data[features].mean().to_dict(),
            'targets': city_data[targets].mean().to_dict()
        }




    # 第二步：填充每个年份的数据
    for year, group in df.groupby('year'):

        # 外连接合并确保包含所有城市
        merged = pd.merge(full_city_df, group, on='city', how='left', suffixes=('', '_y'))

        # 填充逻辑
        for city in all_cities:
            mask = merged['city'] == city
            if merged.loc[mask, features+targets].isnull().any().any():
                # 填充特征
                for f in features:
                    if pd.isna(merged.loc[mask, f]).any():
                        merged.loc[mask, f] = city_features_mean[city]['features'][f]
                # 填充目标
                for t in targets:
                    if pd.isna(merged.loc[mask, t]).any():
                        merged.loc[mask, t] = city_features_mean[city]['targets'][t]
        
        # 排序保持城市顺序一致
        year_dict[year] = merged.sort_values('city').reset_index(drop=True)

    # 构建时间序列数据集
    years = sorted(year_dict.keys())
    X_seq, y_seq = [], []



    for i in range(window_size, len(years)):
        # 当前预测年
        current_year = years[i]
        window_years = [years[i-k-1] for k in range(window_size)]
        
        # 收集窗口期数据（含3个历史年）
        window_data = []
        for y in window_years:
            df_year = year_dict[y][combined_cols]
            window_data.append(df_year.values)
        
        # 当前年特征
        current_features = year_dict[current_year][features].values
        
        # 拼接特征维度：[历史特征 + 当前特征]
        input_features = np.concatenate(window_data + [current_features], axis=1)
        
        # 获取目标值
        target_output = year_dict[current_year][targets].values
        
        # 添加到序列
        X_seq.append(input_features)
        y_seq.append(target_output)

    # 转换为数组
    X = np.array(X_seq)  # 形状: (样本数, 城市数, 输入维度)
    y = np.array(y_seq)  # 形状: (样本数, 城市数, 1)

    #检查是否有nan
    # for x in X:
    #     if np.isnan(x).any():
    #         print("nan !")
            


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # 转换为PyTorch Tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    existence_tensor = torch.tensor(existence_tensor, dtype=torch.float32)
    existence_tensor = existence_tensor.unsqueeze(-1)  # 添加一个维度以匹配目标输出的形状
    X_tensor = torch.cat([X_tensor,existence_tensor],dim=-1)  # 将存在性张量应用于输入数据,最后一个维度是当前年数据是否存在

    # 验证数据维度
    print(f"输入数据维度: {X_tensor.shape}")  # 应为 (样本数, 城市数, 输入特征数)
    print(f"目标数据维度: {y_tensor.shape}")  # 应为 (样本数, 城市数, 2)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)#10:1的比例差不多


    # 初始化模型
    model = DCLFormer(input_size=X.shape[-1], output_size=y.shape[-1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # 训练模型
    for epoch in tqdm(range(100000),desc='Training'):
        model.train()
        optimizer.zero_grad()
        inputs = X_train[:,:,:X_train.shape[-1]-1]  # 去掉最后一个维度
        existence = X_train[:,:,-1]  # 最后一个维度
        outputs = model(inputs)
        existence = existence.unsqueeze(-1)
        outputs = outputs * existence.repeat(1, 1, outputs.shape[-1])   # 乘以存在性张量
        y_train = y_train * existence.repeat(1, 1, y_train.shape[-1])  # 乘以存在性张量
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        if (epoch + 1) % 5000 == 0:
            torch.save(model.state_dict(),f'model/wb/DCLFormer_{epoch}.pth')

    torch.save(model.state_dict(),'model/wb/DCLFormer.pth')

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
    # model2 = torch.load('DCLFormer.pth', map_location=torch.device('cpu'))
    # # 测试模型
    # model2.to('cpu')
    # model2.eval()
    # with torch.no_grad():
    #     preds = model2(X_test)
    #     test_loss = criterion(preds, y_test)
    #     print(f"Test Loss: {test_loss.item():.4f}")

    # print("\nRF模型评估结果：")
    # for i, target in enumerate(y_test):
    #     mse = mean_squared_error(y_test[i,:], preds[i,:])
    #     rmse = np.sqrt(mse)
    #     r2 = r2_score(y_test[i,:], preds[i,:]) #R2
    #     print(f"{target}: MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")


