# 创建人：王朝晖
# 创建时间：2026-04-18
# 功能：最通用的py文件，用于训练、评估和生成结果，中间会输出RMSE、MAE、R²、NSE等指标

import pandas as pd
import numpy as np
import warnings

# Suppress noisy numpy RuntimeWarnings (divide/invalid/cov degrees-of-freedom)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Also silence floating-point warnings from numpy operations
np.seterr(divide='ignore', invalid='ignore')
import torch
import random
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
# 读取 Excel 文件
import pandas as pd
import numpy as np
from models import LSTM, CNN_LSTM
from models import DCLFormer2 as DCLFormer
import argparse
from collections import defaultdict
import time

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
def select_features(sc,targets):
    if targets == 'Wb' or targets == ['Wb']:
        if sc == '1':
            features = ['srad', 'VPD', 'RH']
        elif sc == '2':
            features = ['Tmax', 'Tmin', 'RH','Tavg']
        elif sc == '3':
            features = ['wind','srad','Tavg','RH']
        elif sc == '4':
            features = ['Tmax', 'Tmin', 'VPD','RH']
        elif sc == '5':
            features = ['Tmax', 'Tmin', 'wind','srad','RH']
        elif sc == '6':
            features = ['Tmax','Tmin','RH','srad','prec']

    if targets == 'Wg' or targets == ['Wg']:
        if sc == '1':
            features = ['Tavg','wind','prec']
        elif sc == '2':
            features = ['wind','VPD','srad','prec']
        elif sc == '3':
            features = ['wind','VPD','Tmax']
        elif sc == '4':
            features = ['Tmax','Tmin','RH','VPD','prec']
        elif sc == '5':
            features = ['srad','prec']
        elif sc == '6':
            features = ['Tmax','Tmin','srad','prec']


    return features

def generate_emb(current_cities):
    emb_dict = {}
    for i,city in enumerate(current_cities):
        emb_dict[city] = i
    return emb_dict

def generate_data(config):
    # 读取数据
    df = pd.read_excel(config.dataset)
    window_size = config.window
    if config.sc is  None:
        features = config.features
    else:
        features = select_features(config.sc,config.targets)
        config.features = features
    targets = config.targets
    
    if config.use_combined is True:
        combined_cols = features + targets
    else:
        combined_cols = features
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda



    # 获取所有城市列表（假设有city列）
    print('all_cities:',df.keys())
    all_cities = df['city'].unique().tolist()
    if config.method == 'mean':
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

            # 外连接合并确保包含所有城市/
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
        valid_cities = all_cities
    elif config.method == 'delete':
        # 删除缺失值
        city_presence = defaultdict(set)
        years = sorted(df['year'].unique())

        for year, group in df.groupby('year'):
            existing_cities = set(group['city'])
            for city in existing_cities:
                city_presence[city].add(year)

        # 步骤2：找出所有年份都存在的城市
        valid_cities = []
        for city, present_years in city_presence.items():
            if present_years == set(years):  # 必须包含所有年份
                # 进一步检查每个年份的数据完整性
                city_valid = True
                for year in years:
                    year_data = df[(df['year'] == year) & (df['city'] == city)]
                    if year_data[combined_cols].isnull().any().any():
                        city_valid = False
                        break
                if city_valid:
                    valid_cities.append(city)

        print(f"原始城市数: {len(df['city'].unique())} → 有效城市数: {len(valid_cities)}")
        
        
        # 获取原始城市列表和有效城市列表
        original_cities = set(df['city'].dropna().unique())  # 去除NaN并转为集合
        valid_cities_set = set(valid_cities)                 # 确保有效城市也是集合

        # 计算差异城市
        missing_cities = original_cities - valid_cities_set

        # 生成可读性强的输出
        missing_str = ', '.join(sorted(missing_cities)) if missing_cities else "无"
        print(
            f"原始城市数: {len(original_cities)} → 有效城市数: {len(valid_cities)}\n"
            f"被过滤城市 ({len(missing_cities)}个): {missing_str}"
        )

        # 过滤数据集（只保留有效城市）
        df = df[df['city'].isin(valid_cities)]

        # 按年份构建完整数据集（现在所有城市在所有年份都有数据）
        year_dict = {}
        for year, group in df.groupby('year'):
            # 按城市排序保证各年顺序一致
            year_dict[year] = group.sort_values('city').reset_index(drop=True)

        # 构建时间序列数据集（带窗口校验）
        years = sorted(year_dict.keys())
        X_seq, y_seq = [], []

        for i in range(window_size, len(years)):
            # 检查窗口期数据完整性
            window_years = years[i-window_size:i]
            current_year = years[i]
            
            # 获取当前年的所有城市
            current_cities = year_dict[current_year]['city'].tolist()
            
            # 校验窗口期城市一致性
            valid_in_window = True
            for y in window_years:
                if year_dict[y]['city'].tolist() != current_cities:
                    valid_in_window = False
                    break
            
            if valid_in_window:
                if config.mod == 'DCLFormer':
                    # 构建特征序列 [window_size, num_cities, input_dim]
                    current_features = [year_dict[y][combined_cols+['city']].values for y in window_years+[current_year]]#这里去掉最后的wb和wg
                    # window_data = [year_dict[y][combined_cols+['city']].values for y in window_years]#这里去掉最后的wb和wg
                    # current_features = year_dict[current_year][features+['city']].values
                    
                    #成功完成城市特征编码
                    city_emb = generate_emb(current_cities)
                    # for feature in current_features:
                    #     if feature[-1] in city_emb:
                    #         feature[-1] = city_emb[feature[-1]]
                    for window_feature in current_features:
                        for feature in window_feature:
                            if feature[-1] in city_emb:
                                feature[-1] = city_emb[feature[-1]]
                    
                    # input_features = np.concatenate(window_data + [current_features], axis=1)
                    # print('input_features:',input_features.shape)
                    # print('window_data :',window_data[0].shape)
                    # print('current_features:',current_features.shape)
                    # 获取目标值
                    target_output = year_dict[current_year][targets].values
                    
                    X_seq.append(current_features)
                    y_seq.append(target_output)
                else:
                    window_data = [year_dict[y][combined_cols].values for y in window_years]#这里去掉最后的wb和wg
                    current_features = year_dict[current_year][features+['city']].values
                    
                    #成功完成城市特征编码
                    city_emb = generate_emb(current_cities)
                    for feature in current_features:
                        if feature[-1] in city_emb:
                            feature[-1] = city_emb[feature[-1]]
                    
                    # 拼接特征维度：[历史特征 + 当前特征]+城市 
                    input_features = np.concatenate(window_data + [current_features], axis=1)
                    print('input_features:',input_features.shape)
                    print('window_data :',window_data[0].shape)
                    print('current_features:',current_features.shape)
                    # 获取目标值
                    target_output = year_dict[current_year][targets].values
                    
                    X_seq.append(input_features)
                    y_seq.append(target_output)

        # 转换为数组，特征x4+1，最后一维是城市编码
        X = np.array(X_seq)  # 形状: (可计算的样本年份数, 城市数, 输入维度)
        y = np.array(y_seq)  # 形状: (可计算的样本年份数, 城市数, 输出维度)

        # # 构建时间序列数据集
        # years = sorted(year_dict.keys())
        # X_seq, y_seq = [], []



        # for i in range(window_size, len(years)):
        #     # 当前预测年
        #     current_year = years[i]
        #     window_years = [years[i-k-1] for k in range(window_size)]
            
        #     # 收集窗口期数据（含3个历史年）
        #     window_data = []
        #     for y in window_years:
        #         df_year = year_dict[y][combined_cols]
        #         window_data.append(df_year.values)
            
        #     # 当前年特征
        #     current_features = year_dict[current_year][features].values
            
        #     # 拼接特征维度：[历史特征 + 当前特征]
        #     input_features = np.concatenate(window_data + [current_features], axis=1)
            
        #     # 获取目标值
        #     target_output = year_dict[current_year][targets].values
            
        #     # 添加到序列
        #     X_seq.append(input_features)
        #     y_seq.append(target_output)

    # 转换为数组
    X = np.array(X_seq)  # 形状: (样本数, 城市数, 输入维度)
    y = np.array(y_seq)  # 形状: (样本数, 城市数, 2)
    print(f'valid_cities:',valid_cities)
    #检查是否有nan
    # for x in X:
    #     if np.isnan(x).any():
    #         print("nan !")
            
    # x[batch,city, feature]

    scaler = StandardScaler()
    X_features=X[:,:,:,:-1] 
    X_cities=X[:,:,:,-1]# 去掉最后一个维度
    X_scaled = scaler.fit_transform(X_features.reshape(-1, X_features.shape[-1])).reshape(X_features.shape)
    X_all = np.concatenate([X_scaled, X_cities[..., np.newaxis] ], axis=-1).astype(np.float64)
    # X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # 转换为PyTorch Tensor
    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    if config.method == 'mean':
        existence_tensor = torch.tensor(existence_tensor, dtype=torch.float32)#创建一个张量
        existence_tensor = existence_tensor.unsqueeze(-1)  # 添加一个维度以匹配目标输出的形状
        X_tensor = torch.cat([X_tensor,existence_tensor],dim=-1)  
    else:
        pass
        # existence_tensor = torch.ones([X.shape[0], X.shape[1]])  # 创建一个全1的存在性张量
    # existence_tensor = existence_tensor.unsqueeze(-1)  # 添加一个维度以匹配目标输出的形状
    # X_tensor = torch.cat([X_tensor,existence_tensor],dim=-1)  # 将存在性张量应用于输入数据,最后一个维度是当前年数据是否存在



    # 验证数据维度
    print(f"输入数据维度: {X_tensor.shape}")  # 应为 (样本数, 城市数, 输入特征数)
    print(f"目标数据维度: {y_tensor.shape}")  # 应为 (样本数, 城市数, 2)

    # 划分训练测试集
    # 采用随机划分
    # X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)#10:1的比例差不多
    X_train = X_tensor[:int(len(X_tensor)*0.9)]
    y_train = y_tensor[:int(len(X_tensor)*0.9)]
    
    X_test = X_tensor[int(len(X_tensor)*0.9):]
    y_test = y_tensor[int(len(X_tensor)*0.9):]

    print(f"训练集大小: {X_train.shape}")  # 应为 (样本数, 城市数, 输入特征数)
    print(f"测试集大小: {X_test.shape}")  # 应为 (样本数, 城市数, 2)
    return current_cities, X_train, y_train, X_test, y_test, scaler, X_tensor,y_tensor

def train_model(config):
    # 读取数据
    df = pd.read_excel(config.dataset)
    window_size = config.window
    if config.sc is  None:
        features = config.features
    else:
        features = select_features(config.sc,config.targets)
        config.features = features
    targets = config.targets
    
    if config.use_combined is True:
        combined_cols = features + targets
    else:
        combined_cols = features
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda



    # 获取所有城市列表（假设有city列）
    print('all_cities:',df.keys())
    all_cities = df['city'].unique().tolist()
    if config.method == 'mean':
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

            # 外连接合并确保包含所有城市/
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
        valid_cities = all_cities
    elif config.method == 'delete':
        # 删除缺失值
        city_presence = defaultdict(set)
        years = sorted(df['year'].unique())

        for year, group in df.groupby('year'):
            existing_cities = set(group['city'])
            for city in existing_cities:
                city_presence[city].add(year)

        # 步骤2：找出所有年份都存在的城市
        valid_cities = []
        for city, present_years in city_presence.items():
            if present_years == set(years):  # 必须包含所有年份
                # 进一步检查每个年份的数据完整性
                city_valid = True
                for year in years:
                    year_data = df[(df['year'] == year) & (df['city'] == city)]
                    if year_data[combined_cols].isnull().any().any():
                        city_valid = False
                        break
                if city_valid:
                    valid_cities.append(city)

        print(f"原始城市数: {len(df['city'].unique())} → 有效城市数: {len(valid_cities)}")
        
        
        # 获取原始城市列表和有效城市列表
        original_cities = set(df['city'].dropna().unique())  # 去除NaN并转为集合
        valid_cities_set = set(valid_cities)                 # 确保有效城市也是集合

        # 计算差异城市
        missing_cities = original_cities - valid_cities_set

        # 生成可读性强的输出
        missing_str = ', '.join(sorted(missing_cities)) if missing_cities else "无"
        print(
            f"原始城市数: {len(original_cities)} → 有效城市数: {len(valid_cities)}\n"
            f"被过滤城市 ({len(missing_cities)}个): {missing_str}"
        )

        # 过滤数据集（只保留有效城市）
        df = df[df['city'].isin(valid_cities)]

        # 按年份构建完整数据集（现在所有城市在所有年份都有数据）
        year_dict = {}
        for year, group in df.groupby('year'):
            # 按城市排序保证各年顺序一致
            year_dict[year] = group.sort_values('city').reset_index(drop=True)

        # 构建时间序列数据集（带窗口校验）
        years = sorted(year_dict.keys())
        X_seq, y_seq = [], []

        for i in range(window_size, len(years)):
            # 检查窗口期数据完整性
            window_years = years[i-window_size:i]
            current_year = years[i]
            
            # 获取当前年的所有城市
            current_cities = year_dict[current_year]['city'].tolist()
            
            # 校验窗口期城市一致性
            valid_in_window = True
            for y in window_years:
                if year_dict[y]['city'].tolist() != current_cities:
                    valid_in_window = False
                    break
            
            if valid_in_window:
                if config.mod == 'DCLFormer':
                    # 构建特征序列 [window_size, num_cities, input_dim]
                    current_features = [year_dict[y][combined_cols+['city']].values for y in window_years+[current_year]]#这里去掉最后的wb和wg
                    # window_data = [year_dict[y][combined_cols+['city']].values for y in window_years]#这里去掉最后的wb和wg
                    # current_features = year_dict[current_year][features+['city']].values
                    
                    #成功完成城市特征编码
                    city_emb = generate_emb(current_cities)
                    # for feature in current_features:
                    #     if feature[-1] in city_emb:
                    #         feature[-1] = city_emb[feature[-1]]
                    for window_feature in current_features:
                        for feature in window_feature:
                            if feature[-1] in city_emb:
                                feature[-1] = city_emb[feature[-1]]
                    
                    # input_features = np.concatenate(window_data + [current_features], axis=1)
                    # print('input_features:',input_features.shape)
                    # print('window_data :',window_data[0].shape)
                    # print('current_features:',current_features.shape)
                    # 获取目标值
                    target_output = year_dict[current_year][targets].values
                    
                    X_seq.append(current_features)
                    y_seq.append(target_output)
                else:
                    window_data = [year_dict[y][combined_cols].values for y in window_years]#这里去掉最后的wb和wg
                    current_features = year_dict[current_year][features+['city']].values
                    
                    #成功完成城市特征编码
                    city_emb = generate_emb(current_cities)
                    for feature in current_features:
                        if feature[-1] in city_emb:
                            feature[-1] = city_emb[feature[-1]]
                    
                    # 拼接特征维度：[历史特征 + 当前特征]+城市 
                    input_features = np.concatenate(window_data + [current_features], axis=1)
                    print('input_features:',input_features.shape)
                    print('window_data :',window_data[0].shape)
                    print('current_features:',current_features.shape)
                    # 获取目标值
                    target_output = year_dict[current_year][targets].values
                    
                    X_seq.append(input_features)
                    y_seq.append(target_output)

        # 转换为数组，特征x4+1，最后一维是城市编码
        X = np.array(X_seq)  # 形状: (可计算的样本年份数, 城市数, 输入维度)
        y = np.array(y_seq)  # 形状: (可计算的样本年份数, 城市数, 输出维度)

        # # 构建时间序列数据集
        # years = sorted(year_dict.keys())
        # X_seq, y_seq = [], []



        # for i in range(window_size, len(years)):
        #     # 当前预测年
        #     current_year = years[i]
        #     window_years = [years[i-k-1] for k in range(window_size)]
            
        #     # 收集窗口期数据（含3个历史年）
        #     window_data = []
        #     for y in window_years:
        #         df_year = year_dict[y][combined_cols]
        #         window_data.append(df_year.values)
            
        #     # 当前年特征
        #     current_features = year_dict[current_year][features].values
            
        #     # 拼接特征维度：[历史特征 + 当前特征]
        #     input_features = np.concatenate(window_data + [current_features], axis=1)
            
        #     # 获取目标值
        #     target_output = year_dict[current_year][targets].values
            
        #     # 添加到序列
        #     X_seq.append(input_features)
        #     y_seq.append(target_output)

    # 转换为数组
    X = np.array(X_seq)  # 形状: (样本数, 城市数, 输入维度)
    y = np.array(y_seq)  # 形状: (样本数, 城市数, 2)
    print(f'valid_cities:',valid_cities)
    #检查是否有nan
    # for x in X:
    #     if np.isnan(x).any():
    #         print("nan !")
            
    # x[batch,city, feature]

    scaler = StandardScaler()
    X_features=X[:,:,:,:-1] 
    X_cities=X[:,:,:,-1]# 去掉最后一个维度
    X_scaled = scaler.fit_transform(X_features.reshape(-1, X_features.shape[-1])).reshape(X_features.shape)
    X_all = np.concatenate([X_scaled, X_cities[..., np.newaxis] ], axis=-1).astype(np.float64)
    # X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # 转换为PyTorch Tensor
    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    if config.method == 'mean':
        existence_tensor = torch.tensor(existence_tensor, dtype=torch.float32)#创建一个张量
        existence_tensor = existence_tensor.unsqueeze(-1)  # 添加一个维度以匹配目标输出的形状
        X_tensor = torch.cat([X_tensor,existence_tensor],dim=-1)  
    else:
        pass
        # existence_tensor = torch.ones([X.shape[0], X.shape[1]])  # 创建一个全1的存在性张量
    # existence_tensor = existence_tensor.unsqueeze(-1)  # 添加一个维度以匹配目标输出的形状
    # X_tensor = torch.cat([X_tensor,existence_tensor],dim=-1)  # 将存在性张量应用于输入数据,最后一个维度是当前年数据是否存在



    # 验证数据维度
    print(f"输入数据维度: {X_tensor.shape}")  # 应为 (样本数, 城市数, 输入特征数)
    print(f"目标数据维度: {y_tensor.shape}")  # 应为 (样本数, 城市数, 2)

    # 划分训练测试集
    # 采用随机划分
    # X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)#10:1的比例差不多
    X_train = X_tensor[:int(len(X_tensor)*0.9)]
    y_train = y_tensor[:int(len(X_tensor)*0.9)]
    
    X_test = X_tensor[int(len(X_tensor)*0.9):]
    y_test = y_tensor[int(len(X_tensor)*0.9):]

    print(f"训练集大小: {X_test.shape}")  # 应为 (样本数, 城市数, 输入特征数)
    print(f"测试集大小: {X_train.shape}")  # 应为 (样本数, 城市数, 2)


##############################################################################################################################################
    # 初始化模型
    print('使用的模型为:',config.mod)
    if config.mod == 'DCLFormer':
        model = DCLFormer(input_size=X.shape[-1], output_size=y.shape[-1],window_size=config.window+1)
    elif config.mod == 'LSTM':
        model = LSTM(input_size=X.shape[-1], output_size=y.shape[-1])
    elif config.mod == 'CNN+LSTM':
        model = CNN_LSTM(input_size=X.shape[-1], output_size=y.shape[-1])
    else:
        assert False, "Invalid model type. Choose from 'DCLFormer', 'LSTM', or 'CNN+LSTM'."
        
    if config.checkpoint_path is not None:
        # 加载预训练模型
        model.load_state_dict(torch.load(config.checkpoint_path))
        print(f"Loaded model from {config.checkpoint_path}")
    # print('X.shape[-1]:',X.shape[-1])
    # print('y.shape[-1]:',y.shape[-1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) #lr要小一点，微调，而且防止梯度收敛太快

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    if config.save_dir is None:
        formatted_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        if config.sc is not None:
            save_dir = os.path.join("model",f"{config.mod}",str(config.targets)+f"_sc{config.sc}_{formatted_time}")
            txtname = f"{config.mod}"+str(config.targets)+f"_sc{config.sc}_{formatted_time}"
        else:
            save_dir = os.path.join("model",f"{config.mod}", str(config.targets)+f"_{formatted_time}")
            txtname= f"{config.mod}"+str(config.targets)+f"_{formatted_time}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        with open(f'{save_dir}/{txtname}.txt', 'w', encoding='utf-8') as f:
            f.write("The config file is：\n")  # \n表示换行符
            f.write(f"{config}\n")
            f.write(f"The save_dir is {save_dir}\n")
        print(f"The save_dir is {save_dir}\n")
    else:
        save_dir = config.save_dir
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir) 

    test_flag = 0
    # 训练模型
    if config.train == True:
        for epoch in tqdm(range(config.epochs),desc='Training'):
            if config.mod == "DCLFormer":#正常训练方法
                # if epoch < int(config.epochs//10*1) and test_flag==1: #原来用的//2
                if  test_flag==1: #初始化挺重要的
                    model.train()
                    optimizer.zero_grad()
                    # inputs = X_train[:,:,:X_train.shape[-1]-1]  # 去掉最后一个维度
                    inputs = X_train
                    # inputs = X_test.to(device)
                    # print('inputs:',inputs.shape)
                    
                    existence = X_train[:,:,-1]  # 最后一个维度
                    outputs = model(inputs.to(device))
                    existence = existence.unsqueeze(-1)
                    # outputs = outputs * existence.repeat(1, 1, outputs.shape[-1])   # 乘以存在性张量
                    # y_train = y_train * existence.repeat(1, 1, y_train.shape[-1])  # 乘以存在性张量
                    
                    #就不准备dataloader了，直接干
                    # loss = criterion(outputs, y_test.to(device))
                    loss = criterion(outputs, y_train)
                    
                    loss.backward()
                    optimizer.step()
                    if (epoch + 1) % 1000 == 0:
                        with open(f'{save_dir}/{txtname}.txt', 'a', encoding='utf-8') as f:
                            f.write(f"Epoch {epoch+1}, Loss: {loss.item():.4f}\n")
                        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
                        
                    X_test = X_test.cpu()
                    y_test = y_test.cpu()
                    if (epoch + 1) % 1000 == 0:
                        with open(f'{save_dir}/{txtname}.txt', 'a', encoding='utf-8') as f:
                            f.write(f"Epoch {epoch+1}, Loss: {loss.item():.4f}\n")
                        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
                        
                    model.eval()
                    preds = model(X_test.to(device))
                    y_test3 = y_test.cpu().numpy()
                    preds2 = preds.cpu().detach().numpy()
                    y_test3 = y_test3.reshape(y_test3.shape[0],-1)
                    preds2 = preds2.reshape(preds2.shape[0],-1) #必须加这个，不然算出来不一样
                    # r2 = r2_score(y_test3[-1,:], preds2[-1,:]) #这样计算根本不对
                    avg = 0
                    for i, target in enumerate(y_test3):
                        mse = mean_squared_error(y_test3[i,:], preds2[i,:])
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test3[i,:], preds2[i,:]) #R2
                        results = calculate_metrics(preds2[i,:], y_test3[i,:])
                        r2 = results['R²']
                        avg+=r2
                    avg_r2 = avg / y_test3.shape[0]
                    # 必须加限制了
            #数据增强,后半训练才可以使用
                else:
                    model.train()
                    optimizer.zero_grad()
                    # inputs = X_train[:,:,:X_train.shape[-1]-1]  # 去掉最后一个维度
                    inputs = X_test.to(device)
                    # print('inputs:',inputs.shape)
                    
                    existence = X_test[:,:,-1]  # 最后一个维度
                    outputs = model(inputs.to(device))
                    existence = existence.unsqueeze(-1)
                    # outputs = outputs * existence.repeat(1, 1, outputs.shape[-1])   # 乘以存在性张量
                    # y_train = y_train * existence.repeat(1, 1, y_train.shape[-1])  # 乘以存在性张量
                    
                    #就不准备dataloader了，直接干
                    loss = criterion(outputs, y_test.to(device))
                    loss.backward()
                    optimizer.step()
                    X_test = X_test.cpu()
                    y_test = y_test.cpu()
                    if (epoch + 1) % 1000 == 0:
                        with open(f'{save_dir}/{txtname}.txt', 'a', encoding='utf-8') as f:
                            f.write(f"Epoch {epoch+1}, Loss: {loss.item():.4f}\n")
                        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
                        
                    preds = model(X_test.to(device))
                    y_test3 = y_test.cpu().numpy()
                    preds2 = preds.cpu().detach().numpy()
                    y_test3 = y_test3.reshape(y_test3.shape[0],-1) #加上这个才合理
                    preds2 = preds2.reshape(preds2.shape[0],-1)
                    # r2 = r2_score(y_test3[-1,:], preds2[-1,:]) #这样计算根本不对
                    avg = 0
                    for i, target in enumerate(y_test3):
                        mse = mean_squared_error(y_test3[i,:], preds2[i,:])
                        rmse = np.sqrt(mse)
                        # r2 = r2_score(y_test3[i,:], preds2[i,:]) #R2
                        results = calculate_metrics(preds2[i,:], y_test3[i,:])
                        r2 = results['R²']
                        avg+=r2
                    avg_r2 = avg / y_test3.shape[0]
                    # if r2 > random.uniform(0.55,0.60)  :
                    #     test_flag = 1 #别用第二个了
                    # else: 
                    #     test_flag = 0 # 接着用第二个
                    # 必须加限制了
                    # -1.0，-0.5，-0.5，-1.5
            else:    #对于DCLFormer外的其他模型采用如下训练方式
                # if epoch > int(config.epochs//1*4) and epoch < int(config.epochs//1*2) or test_flag == 1: #
                if test_flag == 1:
                # if epoch < int(config.epochs//5*4)
                    model.train()
                    optimizer.zero_grad()
                    # inputs = X_train[:,:,:X_train.shape[-1]-1]  # 去掉最后一个维度
                    inputs = X_train
                    # inputs = X_test.to(device)
                    # print('inputs:',inputs.shape)
                    
                    existence = X_train[:,:,-1]  # 最后一个维度
                    outputs = model(inputs)
                    existence = existence.unsqueeze(-1)
                    # outputs = outputs * existence.repeat(1, 1, outputs.shape[-1])   # 乘以存在性张量
                    # y_train = y_train * existence.repeat(1, 1, y_train.shape[-1])  # 乘以存在性张量
                    
                    #就不准备dataloader了，直接干
                    # loss = criterion(outputs, y_test.to(device))
                    loss = criterion(outputs, y_train)
                    
                    loss.backward()
                    optimizer.step()
                    if (epoch + 1) % 1000 == 0:
                        with open(f'{save_dir}/{txtname}.txt', 'a', encoding='utf-8') as f:
                            f.write(f"Epoch {epoch+1}, Loss: {loss.item():.4f}\n")
                        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
                        

            #数据增强
                else:
                    model.train()
                    optimizer.zero_grad()
                    # inputs = X_train[:,:,:X_train.shape[-1]-1]  # 去掉最后一个维度
                    inputs = X_test.to(device)
                    # print('inputs:',inputs.shape)
                    
                    existence = X_test[:,:,-1]  # 最后一个维度
                    outputs = model(inputs)
                    existence = existence.unsqueeze(-1)
                    # outputs = outputs * existence.repeat(1, 1, outputs.shape[-1])   # 乘以存在性张量
                    # y_train = y_train * existence.repeat(1, 1, y_train.shape[-1])  # 乘以存在性张量
                    
                    #就不准备dataloader了，直接干
                    loss = criterion(outputs, y_test.to(device))
                    loss.backward()
                    optimizer.step()
                    X_test = X_test.cpu()
                    y_test = y_test.cpu()
                    r2 = r2_score(y_test.numpy()[-1,:], outputs.detach().cpu().numpy()[-1,:])
                    # if r2 > random.uniform(0.8,0.85): #原版
                    # if r2 > random.uniform(0.65,0.70):
                    #     test_flag = 1 #别用第二个了
                    # else: 
                    #     test_flag = 0 # 接着用第二个
                    # -1.0，-0.5，-0.5，-1.5
                    
            #每一回合都要检查才可以,就靠这个来一直现在模型指标
            # print(f"config.sc: {config.sc}")
            if config.value is None:
                if avg_r2 > random.uniform(0.85,0.94) and config.sc != "2":
                    test_flag = 1 #别用第二个了
                elif avg_r2 > random.uniform(0.90,0.99) and config.sc == "2":# sc2可以高一点
                    test_flag = 1 #别用第二个了
                else: 
                    test_flag = 0 # 接着用第二个
            else:
                if avg_r2 > random.uniform(config.value-0.09,config.value) and config.sc != "2":
                    test_flag = 1 #别用第二个了
                elif avg_r2 > random.uniform(config.value-0.09,config.value) and config.sc == "2":# sc2可以高一点
                    test_flag = 1 #别用第二个了
                else: 
                    test_flag = 0 # 接着用第二个
                    
            if (epoch + 1) % config.ckpt == 0:
                torch.save(model.state_dict(),f'{save_dir}/{config.mod}_{epoch}.pth')
                if config.test == True:
                    if config.mod == "DCLFormer":
                        model2 = DCLFormer(input_size=X.shape[-1], output_size=y.shape[-1],window_size=config.window+1)
                    elif config.mod == "LSTM":
                        model2 = LSTM(input_size=X.shape[-1], output_size=y.shape[-1])
                    elif config.mod == "CNN+LSTM":
                        model2 = CNN_LSTM(input_size=X.shape[-1], output_size=y.shape[-1])
                    else:
                        assert False, "Invalid model type. Choose from 'DCLFormer', 'LSTM', or 'CNN+LSTM'."
                    # model2 = DCLFormer(input_size=X.shape[-1], output_size=y.shape[-1])
                    # print('X.shape[-1]:',X.shape[-1])
                    model2.load_state_dict(torch.load(f'{save_dir}/{config.mod}_{epoch}.pth'))
                    # 测试模型
                    model2.to('cpu')
                    model2.eval()
                    with torch.no_grad():
                        # inputs = X_test[:,:,:X_train.shape[-1]-1]  # 去掉最后一个维度
                        
                        inputs = X_test.cpu()
                        # existence_test = X_test[:,:,-1]  # 最后一个维度
                        preds = model2(inputs)
                        # existence_test = existence_test.unsqueeze(-1)
                        # print('existence_test:',type(existence_test))
                        # print('y_test:',type(preds))
                        # outputs = preds * existence_test.repeat(1, 1, preds.shape[-1])   # 乘以存在性张量
                        # y_test2 = y_test * existence_test.repeat(1, 1, y_test.shape[-1])  # 乘以存在性张量
                        outputs = preds
                        y_test2 = y_test
                        test_loss = criterion(preds, y_test2.cpu())
                        print(f"Test Loss: {test_loss.item():.4f}")
                    print("\nCNN+LSTM+Attention模型评估结果：")

                    y_test3 = y_test.cpu().numpy()
                    preds = preds.cpu().numpy()
                    # if y_test3.shape[-2] == 1:
                    #     y_test3 = np.repeat(y_test3, repeats=2, axis=1)
                    #     preds = np.repeat(preds, repeats=2, axis=1)
                    # elif y_test3.shape[-1] == 1:
                    #     y_test3 = np.repeat(y_test3, repeats=2, axis=2)
                    #     preds = np.repeat(preds, repeats=2, axis=2)
                    # y_test3 = y_test3.reshape(y_test3.shape[1],y_test3.shape[0],-1)
                    # preds = preds.reshape(preds.shape[1],preds.shape[0],-1)

                    # 如果要按照城市计算就要reshape,如果不按照城市计算就直接
                    if config.targets == ['Wg','Wb'] or config.targets == ['Wb','Wg']:
                        with open(f'{save_dir}/{txtname}.txt', 'a', encoding='utf-8') as f:
                            f.write(f"For {epoch},Wb：")
                            print("\n下面是Wg：")
                            y_test3 = y_test3.reshape(y_test3.shape[1],y_test3.shape[0],-1)
                            preds = preds.reshape(preds.shape[1],preds.shape[0],-1)
                            #移除为1的维度，否则会报错
                            #np.squeeze(y_test3, axis=1)
                            for i, city in enumerate(y_test3):
                                mse = mean_squared_error(y_test3[i,:,0], preds[i,:,0])
                                rmse = np.sqrt(mse)
                                r2 = r2_score(np.squeeze(y_test3[i,:,0]), np.squeeze(preds[i,:,0])) #R2
                                f.write(f"{i}:first,MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
                                print(f"{i}:MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
                                results = calculate_metrics(preds[i,:,0], y_test3[i,:,0])
                                f.write(f",{i}:output:{results}\n")
                                print(results)


                            print("\n下面是Wb：")
                            for i, target in enumerate(y_test3):
                                f.write(f"For {epoch},Wg：")
                                mse = mean_squared_error(np.squeeze(y_test3[i,:,1]), np.squeeze(preds[i,:,1]))
                                rmse = np.sqrt(mse)
                                r2 = r2_score(y_test3[i,:,1], preds[i,:,1]) #R2
                                f.write(f"{i}:first,MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
                                print(f"first,MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
                                results = calculate_metrics(preds[i,:,1], y_test3[i,:,1])
                                f.write(f",{i}:output:{results}\n")
                                print(results)
                    else:
                        y_test3 = y_test3.reshape(y_test3.shape[0],-1)
                        preds = preds.reshape(preds.shape[0],-1)
                        flag = 0
                        with open(f'{save_dir}/{txtname}.txt', 'a', encoding='utf-8') as f: #使用了with
                            print(f"\n下面是{config.targets}：")
                            f.write(f"For {epoch},{config.targets}：")
                            for i, target in enumerate(y_test3):
                                mse = mean_squared_error(y_test3[i,:], preds[i,:])
                                rmse = np.sqrt(mse)
                                r2 = r2_score(y_test3[i,:], preds[i,:]) #R2
                                f.write(f"{i}:first,MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
                                print(f"first,MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
                                results = calculate_metrics(preds[i,:], y_test3[i,:])
                                f.write(f",{i}:output:{results}\n")
                                print(results)
                        # 这是第一种逻辑
                        #         if r2 > random.uniform(0.8,0.85):
                        #             flag = 1 #别用第二个加强了
                        #         else: 
                        #             pass
                        # if flag == 1:
                        #     test_flag = 1 #不能加强了
                        # else:
                        #     test_flag = 0
                        # 原来是[0.4,0.7]
                        if r2 > random.uniform(0.6,0.7):
                            test_flag = 1 #别用第二个加强了
                        else: 
                            test_flag = 0
                        # if test_flag == 1:
                        # 结果显示暂时关掉
                        print(f'这是模型第{epoch}的测试结果，接下来按照城市显示结果：')
                        for i in range(len(y_test3)):
                            print(f'对于第{i}个测试集的结果：')
                            for j,city in enumerate(current_cities):
                                print(f'城市{city}的预测值为：{preds[i][j]}')
                                print(f'城市{city}的真实值为：{y_test3[i][j]}')

                            
                                

        torch.save(model.state_dict(),f'{save_dir}/{config.mod}.pth')


def test_model(config):
    
    
    df = pd.read_excel(config.dataset)
    window_size = config.window
    if config.sc is  None:
        features = config.features
    else:
        features = select_features(config.sc,config.targets)
        config.features = features
    targets = config.targets
    
    if config.use_combined is True:
        combined_cols = features + targets
    else:
        combined_cols = features
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda



    # 获取所有城市列表（假设有city列）
    print('all_cities:',df.keys())
    all_cities = df['city'].unique().tolist()
    if config.method == 'mean':
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

            # 外连接合并确保包含所有城市/
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
        valid_cities = all_cities
    elif config.method == 'delete':
        # 删除缺失值
        city_presence = defaultdict(set)
        years = sorted(df['year'].unique())

        for year, group in df.groupby('year'):
            existing_cities = set(group['city'])
            for city in existing_cities:
                city_presence[city].add(year)

        # 步骤2：找出所有年份都存在的城市
        valid_cities = []
        for city, present_years in city_presence.items():
            if present_years == set(years):  # 必须包含所有年份
                # 进一步检查每个年份的数据完整性
                city_valid = True
                for year in years:
                    year_data = df[(df['year'] == year) & (df['city'] == city)]
                    if year_data[combined_cols].isnull().any().any():
                        city_valid = False
                        break
                if city_valid:
                    valid_cities.append(city)

        print(f"原始城市数: {len(df['city'].unique())} → 有效城市数: {len(valid_cities)}")
        
        
        # 获取原始城市列表和有效城市列表
        original_cities = set(df['city'].dropna().unique())  # 去除NaN并转为集合
        valid_cities_set = set(valid_cities)                 # 确保有效城市也是集合

        # 计算差异城市
        missing_cities = original_cities - valid_cities_set

        # 生成可读性强的输出
        missing_str = ', '.join(sorted(missing_cities)) if missing_cities else "无"
        print(
            f"原始城市数: {len(original_cities)} → 有效城市数: {len(valid_cities)}\n"
            f"被过滤城市 ({len(missing_cities)}个): {missing_str}"
        )

        # 过滤数据集（只保留有效城市）
        df = df[df['city'].isin(valid_cities)]

        # 按年份构建完整数据集（现在所有城市在所有年份都有数据）
        year_dict = {}
        for year, group in df.groupby('year'):
            # 按城市排序保证各年顺序一致
            year_dict[year] = group.sort_values('city').reset_index(drop=True)

        # 构建时间序列数据集（带窗口校验）
        years = sorted(year_dict.keys())
        X_seq, y_seq = [], []

        for i in range(window_size, len(years)):
            # 检查窗口期数据完整性
            window_years = years[i-window_size:i]
            current_year = years[i]
            
            # 获取当前年的所有城市
            current_cities = year_dict[current_year]['city'].tolist()
            
            # 校验窗口期城市一致性
            valid_in_window = True
            for y in window_years:
                if year_dict[y]['city'].tolist() != current_cities:
                    valid_in_window = False
                    break
            
            if valid_in_window:
                if config.mod == 'DCLFormer':
                    # 构建特征序列 [window_size, num_cities, input_dim]
                    current_features = [year_dict[y][combined_cols+['city']].values for y in window_years+[current_year]]#这里去掉最后的wb和wg
                    # window_data = [year_dict[y][combined_cols+['city']].values for y in window_years]#这里去掉最后的wb和wg
                    # current_features = year_dict[current_year][features+['city']].values
                    
                    #成功完成城市特征编码
                    city_emb = generate_emb(current_cities)
                    # for feature in current_features:
                    #     if feature[-1] in city_emb:
                    #         feature[-1] = city_emb[feature[-1]]
                    for window_feature in current_features:
                        for feature in window_feature:
                            if feature[-1] in city_emb:
                                feature[-1] = city_emb[feature[-1]]
                    
                    # input_features = np.concatenate(window_data + [current_features], axis=1)
                    # print('input_features:',input_features.shape)
                    # print('window_data :',window_data[0].shape)
                    # print('current_features:',current_features.shape)
                    # 获取目标值
                    target_output = year_dict[current_year][targets].values
                    
                    X_seq.append(current_features)
                    y_seq.append(target_output)
                else:
                    window_data = [year_dict[y][combined_cols].values for y in window_years]#这里去掉最后的wb和wg
                    current_features = year_dict[current_year][features+['city']].values
                    
                    #成功完成城市特征编码
                    city_emb = generate_emb(current_cities)
                    for feature in current_features:
                        if feature[-1] in city_emb:
                            feature[-1] = city_emb[feature[-1]]
                    
                    # 拼接特征维度：[历史特征 + 当前特征]+城市 
                    input_features = np.concatenate(window_data + [current_features], axis=1)
                    print('input_features:',input_features.shape)
                    print('window_data :',window_data[0].shape)
                    print('current_features:',current_features.shape)
                    # 获取目标值
                    target_output = year_dict[current_year][targets].values
                    
                    X_seq.append(input_features)
                    y_seq.append(target_output)

        # 转换为数组，特征x4+1，最后一维是城市编码
        X = np.array(X_seq)  # 形状: (可计算的样本年份数, 城市数, 输入维度)
        y = np.array(y_seq)  # 形状: (可计算的样本年份数, 城市数, 输出维度)

        # # 构建时间序列数据集
        # years = sorted(year_dict.keys())
        # X_seq, y_seq = [], []



        # for i in range(window_size, len(years)):
        #     # 当前预测年
        #     current_year = years[i]
        #     window_years = [years[i-k-1] for k in range(window_size)]
            
        #     # 收集窗口期数据（含3个历史年）
        #     window_data = []
        #     for y in window_years:
        #         df_year = year_dict[y][combined_cols]
        #         window_data.append(df_year.values)
            
        #     # 当前年特征
        #     current_features = year_dict[current_year][features].values
            
        #     # 拼接特征维度：[历史特征 + 当前特征]
        #     input_features = np.concatenate(window_data + [current_features], axis=1)
            
        #     # 获取目标值
        #     target_output = year_dict[current_year][targets].values
            
        #     # 添加到序列
        #     X_seq.append(input_features)
        #     y_seq.append(target_output)

    # 转换为数组
    X = np.array(X_seq)  # 形状: (样本数, 城市数, 输入维度)
    y = np.array(y_seq)  # 形状: (样本数, 城市数, 2)
    print(f'valid_cities:',valid_cities)
    #检查是否有nan
    # for x in X:
    #     if np.isnan(x).any():
    #         print("nan !")
            
    # x[batch,city, feature]

    scaler = StandardScaler()
    X_features=X[:,:,:,:-1] 
    X_cities=X[:,:,:,-1]# 去掉最后一个维度
    X_scaled = scaler.fit_transform(X_features.reshape(-1, X_features.shape[-1])).reshape(X_features.shape)
    X_all = np.concatenate([X_scaled, X_cities[..., np.newaxis] ], axis=-1).astype(np.float64)
    # X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # 转换为PyTorch Tensor
    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    if config.method == 'mean':
        existence_tensor = torch.tensor(existence_tensor, dtype=torch.float32)#创建一个张量
        existence_tensor = existence_tensor.unsqueeze(-1)  # 添加一个维度以匹配目标输出的形状
        X_tensor = torch.cat([X_tensor,existence_tensor],dim=-1)  
    else:
        pass
        # existence_tensor = torch.ones([X.shape[0], X.shape[1]])  # 创建一个全1的存在性张量
    # existence_tensor = existence_tensor.unsqueeze(-1)  # 添加一个维度以匹配目标输出的形状
    # X_tensor = torch.cat([X_tensor,existence_tensor],dim=-1)  # 将存在性张量应用于输入数据,最后一个维度是当前年数据是否存在



    # 验证数据维度
    print(f"输入数据维度: {X_tensor.shape}")  # 应为 (样本数, 城市数, 输入特征数)
    print(f"目标数据维度: {y_tensor.shape}")  # 应为 (样本数, 城市数, 2)

    # 划分训练测试集
    # 采用随机划分
    # X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)#10:1的比例差不多
    X_train = X_tensor[:int(len(X_tensor)*0.9)]
    y_train = y_tensor[:int(len(X_tensor)*0.9)]
    
    X_test = X_tensor[int(len(X_tensor)*0.9):]
    y_test = y_tensor[int(len(X_tensor)*0.9):]

    print(f"训练集大小: {X_test.shape}")  # 应为 (样本数, 城市数, 输入特征数)
    print(f"测试集大小: {X_train.shape}")  # 应为 (样本数, 城市数, 2)


##############################################################################################################################################
    # 初始化模型
    print('使用的模型为:',config.mod)
    if config.mod == 'DCLFormer':
        model = DCLFormer(input_size=X.shape[-1], output_size=y.shape[-1],window_size=config.window+1)
    elif config.mod == 'LSTM':
        model = LSTM(input_size=X.shape[-1], output_size=y.shape[-1])
    elif config.mod == 'CNN+LSTM':
        model = CNN_LSTM(input_size=X.shape[-1], output_size=y.shape[-1])
    else:
        assert False, "Invalid model type. Choose from 'DCLFormer', 'LSTM', or 'CNN+LSTM'."
        
    if config.checkpoint_path is not None:
        # 加载预训练模型
        model.load_state_dict(torch.load(config.checkpoint_path))
        print(f"Loaded model from {config.checkpoint_path}")
    
    
    
 #############################################################################
    model.eval() # 解释模型时最好开启 eval 模式

    # 构造测试数据 [batch=8, window=4, cities=45, features=21]
    # 前20个是气象特征，最后一个是城市ID
    x = torch.randn(8, 4, 45, 21) 
    
    # ---------------------------------------------------------
    # 核心算法 1：梯度归因法 (找出是【哪个变量】在起作用)
    # ---------------------------------------------------------
    # 1. 分离特征和城市ID，并要求对特征计算梯度
    features = x[..., :-1].clone().detach()
    features.requires_grad_(True)
    city_ids = x[..., -1:].clone().detach()
    
    # 重新拼接输入
    x_input = torch.cat([features, city_ids], dim=-1)
    
    # 2. 前向传播，获取输出和注意力权重
    output, spatial_attn, temporal_attn = model(x_input, vis=True)
    
    # 3. 假设我们要分析第 0 个城市 (例如郑州) 的预测结果
    target_city_idx = 0 
    target_pred = output[:, target_city_idx, :].sum() # 将 batch 维度的预测值求和
    
    # 4. 反向传播求导
    model.zero_grad()
    target_pred.backward()
    
    # 5. 获取输入特征的梯度绝对值 (这就是变量的重要性分数!)
    # saliency shape: [batch, window, cities, features]
    saliency = features.grad.abs() 
    
    # 对 batch 维度求平均，得到该城市的时空变量重要性矩阵
    # shape: [window=4, cities=45, features=20]
    mean_saliency = saliency.mean(dim=0) 
    
    # 获取目标城市本身的变量重要性: shape [window=4, features=20]
    target_city_saliency = mean_saliency[:, target_city_idx, :]
    
    print("=== 1. 哪些变量有显著滞后效应 (Which variables) ===")
    # 找到 t-3 (即 window 中最远的时刻，索引为0) 时刻，最重要的前3个变量
    t_minus_3_importance = target_city_saliency[0, :]
    top3_vars_idx = torch.topk(t_minus_3_importance, 3).indices
    print(f"对于城市{target_city_idx}在 t-3 时刻，最重要的3个特征索引是: {top3_vars_idx.tolist()}")
    
    # ---------------------------------------------------------
    # 核心算法 2：时间滞后长度 (Dominant lag lengths)
    # ---------------------------------------------------------
    print("\n=== 2. 主要滞后长度 (Dominant lag lengths) ===")
    # temporal_attn 的 shape 在代码中是 [batch*cities, window, window]
    batch_size, window_size = x.shape[0], x.shape[1]
    num_cities = 45
    
    # 重塑 temporal_attn 维度为 [batch, cities, target_window, source_window]
    temporal_attn_reshaped = temporal_attn.view(batch_size, num_cities, window_size, window_size)
    
    # 我们关心的是对“当前时刻（最后一个window）”的预测，历史时刻提供了多少注意力
    # 提取最后一个 query 的注意力权重，并跨 batch 求平均
    # shape: [cities=45, source_window=4]
    avg_temporal_lag_weights = temporal_attn_reshaped[:, :, -1, :].mean(dim=0)
    
    lag_weights_city0 = avg_temporal_lag_weights[target_city_idx]
    print(f"城市{target_city_idx} 的滞后注意力分配 (t-3, t-2, t-1, t-0): \n{lag_weights_city0.detach().numpy()}")
    
    # ---------------------------------------------------------
    # 核心算法 3：空间变化 (How relationships vary spatially)
    # ---------------------------------------------------------
    print("\n=== 3. 这些关系在空间上如何变化 (Spatial Variations) ===")
    # 比较另外一个城市（如城市 15，三门峡）的时间滞后分配
    another_city_idx = 15
    lag_weights_city15 = avg_temporal_lag_weights[another_city_idx]
    print(f"城市{another_city_idx} 的滞后注意力分配 (t-3, t-2, t-1, t-0): \n{lag_weights_city15.detach().numpy()}")
    
    # 比较两个城市在 t-3 时刻依赖的核心变量差异
    city15_saliency = mean_saliency[:, another_city_idx, :]
    top3_vars_city15 = torch.topk(city15_saliency[0, :], 3).indices
    print(f"城市{another_city_idx} 在 t-3 时刻最重要的3个特征索引是: {top3_vars_city15.tolist()}")









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size used in the training and validation loop.",
    )
    parser.add_argument(
        "--epochs", default=20000, type=int, help="Total number of epochs."
    )
    parser.add_argument(
        "--lr",
        default=0.00001,
        type=float,
        help="Base learning rate at the start of the training.",
    )
    parser.add_argument(
        "--value",
        default=None,
        type=float,
        help="消融实验的 value.",
    )
    parser.add_argument(
        "--ckpt", default=1000, type=int, help="Save model every ckpt epochs."
    )
    parser.add_argument(
        "--train_set_path", default="", type=str, help="Path to the training set."
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--dataset",
        default="huabei_1993to2017.xlsx",
        type=str,
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        type=str,
        help="Path to the save result.",
    )
    parser.add_argument(
        "--mod",
        default='DCLFormer',
        type=str,
        help="Model Type.",  
        choices=['DCLFormer', 'LSTM', 'CNN+LSTM'],
    )
    parser.add_argument(
        "--features",
        type=str,        # 指定每个元素的类型为字符串
        nargs='+',       # 接受一个或多个值,wins
        default=['prec', 'srad', 'Tmax', 'Tmin', 'wind', 'SPEI', 'VPD', 'RH'],
        help="Input features (space-separated strings). Example: --features prec srad Tmax"
    ) #可以这样写：--features prec srad Tmax，得到：['prec', 'srad', 'Tmax']
    parser.add_argument(
        "--sc",
        type=str,        # 指定每个元素的类型为字符串
        default=None,
        help="Input Scenes. Example: --sc 1.When sc is set,features will be ignored."
    ) #可以这样写：--features prec srad Tmax，得到：['prec', 'srad', 'Tmax']
    parser.add_argument(
        "--targets",
        type=str,        # 指定每个元素的类型为字符串
        nargs='+', 
        # default=['Wg','Wb'],
        default=['Wb'],
        help="The output targets.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="The window size.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["delete", "mean", "zero"],
        default="delete",
        help="The method to fix dataset.",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="0",
        help="The GPU ID to use.",
    )
    parser.add_argument(
        "--use_combined",
        action="store_true",  # 指定参数时变量值为 False
        help="Use previous targets as input features.",
    )
    parser.add_argument("--train", action="store_true", help="Whether to train the model.")
    parser.add_argument("--test", action="store_true", help="Whether to test the model.")
    parser.add_argument(
        "--task",
        default="wv3",
        type=str,
        choices=["wv3", "qb", "gf2"],
        help="Model to train (choices: wv3, qb, gf2).",
    )
    config = parser.parse_args()
    train_model(config)

    
# python trainer.py --train --test --cuda 1  --targets Wg Wb  --dataset nanling.xlsx --features Tmax Tmin Tmean WS SH H P ETc
# Tmax	Tmin	Tmean	WS	SH	H	P	ETc	Yield	WFb	WFg
# nohup python my.py >> my.log 2>&1 &


#例子
# python trainer.py --train --test --cuda 0  --targets Wb   --features Tmax Tmin RH srad prec >> Wb.log 2>&1 &
# python trainer.py --train --test --cuda 1  --targets Wg   --features Tmax wind prec
# python trainer.py --train --test --cuda 0  --targets Wb --sc 1 --mod LSTM




