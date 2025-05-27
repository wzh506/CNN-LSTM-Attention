import pandas as pd
import numpy as np
import torch
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
from models import DCLFormer
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
            features = ['Tmax', 'Tmin', 'RH']
        elif sc == '3':
            features = ['wind', 'SPEI', 'VPD']
        elif sc == '4':
            features = ['Tmax', 'Tmin', 'VPD','RH']
        elif sc == '5':
            features = ['Tmax', 'Tmin', 'wind','srad','RH']
        elif sc == '6':
            features = ['Tmax','Tmin','RH','srad','RH']

    if targets == 'Wg' or targets == ['Wg']:
        if sc == '1':
            features = ['Tmax','wind','prec']
        elif sc == '2':
            features = ['Tmax','Tmin' ,'wind','srad','prec']
        elif sc == '3':
            features = ['wind','VPD','Tmax']
        elif sc == '4':
            features = ['Tmax','Tmin','RH','VPD','prec']
        elif sc == '5':
            features = ['wind','VPD','SPEI','srad','prec']
        elif sc == '6':
            features = ['Tmax','Tmin' ,'srad','SPEI','prec']


    return features

def fetch_fit(config):
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
                # 构建特征序列 [window_size, num_cities, input_dim]
                window_data = [year_dict[y][combined_cols].values for y in window_years]#这里去掉最后的wb和wg
                current_features = year_dict[current_year][features].values
                
                # 拼接特征维度：[历史特征 + 当前特征] 
                input_features = np.concatenate(window_data + [current_features], axis=1)
                print('input_features:',input_features.shape)
                print('window_data :',window_data[0].shape)
                print('current_features:',current_features.shape)
                # 获取目标值
                target_output = year_dict[current_year][targets].values
                
                X_seq.append(input_features)
                y_seq.append(target_output)

        # 转换为数组
        X = np.array(X_seq)  # 形状: (样本数, 城市数, 输入维度)
        y = np.array(y_seq)  # 形状: (样本数, 城市数, 输出维度)

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
    # scaler.fit_transform(X.reshape(-1, X.shape[-1]))
    scaler.fit(X.reshape(-1, X.shape[-1]))
    
    return scaler








def test_model(config):
    # 读取数据
    df = pd.read_excel(config.test_dataset)
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
    import os
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

        pred_dict = {}
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
                # 构建特征序列 [window_size, num_cities, input_dim]
                window_data = [year_dict[y][combined_cols].values for y in window_years]#这里去掉最后的wb和wg
                current_features = year_dict[current_year][features].values
                #这里后面一定要想办法修改一下
                
                # 拼接特征维度：[历史特征 + 当前特征] 
                input_features = np.concatenate(window_data + [current_features], axis=1)
                
                
                pred_dict[current_year]={}
                # for i,city in enumerate(current_cities):#不要用那个valid_cities
                #     pred_dict[current_year][city]=input_features[i]
                pred_dict[current_year] = input_features
                
                
                # print('input_features:',input_features.shape)
                # print('window_data :',window_data[0].shape)
                # print('current_features:',current_features.shape)
                # 获取目标值
                # target_output = year_dict[current_year][targets].values

                
                # X_seq.append(input_features)
                # y_seq.append(target_output)

    # 转换为数组
    # X = np.array(X_seq)  # 形状: (样本数, 城市数, 输入维度)
    # y = np.array(y_seq)  # 形状: (样本数, 城市数, 2)
    print(f'valid_cities:',valid_cities)
    #检查是否有nan
    # for x in X:
    #     if np.isnan(x).any():
    #         print("nan !")
    
    scaler = fetch_fit(config)
    
    
    
    
    
    if True:
        # model2 = DCLFormer(input_size=pred_dict[current_year][current_cities[0]].shape[0], output_size=1)#直接设置为1把
        model2 = DCLFormer(input_size=pred_dict[current_year].shape[-1], output_size=1)
        # model2.load_state_dict(torch.load(f'{save_dir}/DCLFormer_{epoch}.pth'))
        model2.load_state_dict(torch.load(f'{config.checkpoint_path}/DCLFormer.pth'))
        # 测试模型
        model2.to('cpu')
        model2.eval()
        output_dict={}
        
        with torch.no_grad():
            # inputs = X_test[:,:,:X_train.shape[-1]-1]  # 去掉最后一个维度
            # inputs = 
            # existence_test = X_test[:,:,-1]  # 最后一个维度
            # preds = model2(inputs)
            # existence_test = existence_test.unsqueeze(-1)
            # # print('existence_test:',type(existence_test))
            # # print('y_test:',type(preds))
            # outputs = preds * existence_test.repeat(1, 1, preds.shape[-1])   # 乘以存在性张量
            # y_test2 = y_test * existence_test.repeat(1, 1, y_test.shape[-1])  # 乘以存在性张量
            # test_loss = criterion(preds, y_test2)
            # print(f"Test Loss: {test_loss.item():.4f}")
            for current_year in pred_dict.keys():
                output_dict[current_year] = {}
                # for city in pred_dict[current_year].keys():
                #     inputs = torch.tensor(pred_dict[current_year][city], dtype=torch.float32).unsqueeze(0)  #这里不能直接输入
                #     inputs_trans = scaler.transform(inputs).reshape(inputs.shape)
                #     # existence_test = torch.tensor(existence_dict[current_year], dtype=torch.float32).unsqueeze(0)
                #     inputs_trans= torch.tensor(inputs_trans, dtype=torch.float32).unsqueeze(0)
                #     preds = model2(inputs_trans)
                #     output_dict[current_year][city] = preds.squeeze(0).cpu().numpy()

                inputs = pred_dict[current_year]
                inputs = np.expand_dims(inputs,0)  # 添加一个维度以匹配模型输入
                inputs_trans = scaler.transform(inputs.reshape(-1, inputs.shape[-1])).reshape(inputs.shape)
                # existence_test = torch.tensor(existence_dict[current_year], dtype=torch.float32).unsqueeze(0)
                inputs_trans= torch.tensor(inputs_trans, dtype=torch.float32)
                preds = model2(inputs_trans) # 1,27,1
                for i,city in enumerate(current_cities):
                    # inputs = torch.tensor(pred_dict[current_year][city], dtype=torch.float32).unsqueeze(0)  #这里不能直接输入
                    output_dict[current_year][city] = preds.detach().squeeze(0).cpu().numpy()[i]

            import shutil
            import os

            def copy_excel_file(src_file_path, dest_folder):
                """
                复制 Excel 文件到指定文件夹
                
                参数:
                src_file_path (str): 源文件完整路径 (包含文件名)
                dest_folder (str): 目标文件夹路径
                
                返回:
                bool: 是否复制成功
                """
                try:
                    # 检查源文件是否存在且为xlsx格式
                    if not os.path.isfile(src_file_path):
                        raise FileNotFoundError(f"源文件 {src_file_path} 不存在")
                    if not src_file_path.lower().endswith('.xlsx'):
                        raise ValueError("仅支持 .xlsx 文件格式")

                    # 创建目标文件夹（如果不存在）
                    os.makedirs(dest_folder, exist_ok=True)

                    # 构建目标路径
                    file_name = os.path.basename(src_file_path)
                    dest_path = os.path.join(dest_folder, file_name)

                    # 执行复制（保留元数据）
                    shutil.copy2(src_file_path, dest_path)
                    
                    # 验证复制结果
                    if os.path.isfile(dest_path):
                        print(f"文件复制成功！\n源路径: {src_file_path}\n目标路径: {dest_path}")
                        return True
                    else:
                        raise RuntimeError("文件复制失败，未知错误")
                        
                except Exception as e:
                    print(f"操作失败: {str(e)}")
                    return False
                
                

            if 'Wg' in config.checkpoint_path:
                #复制数据集文件，然后打开，写入
                # copy_excel_file(config.dataset, config.checkpoint_path)
                input_file = config.test_dataset
                output_file = os.path.join(config.checkpoint_path, 'wg.xlsx')
                try:
                    df = pd.read_excel(input_file)
                    print(f"成功读取文件: {input_file}, 共 {len(df)} 行数据")
                except FileNotFoundError:
                    print(f"错误：文件 {input_file} 不存在！")
                    exit()

                # 生成Wb列
                wg_values = []
                for index, row in df.iterrows():
                    year = row['year']
                    city = row['city']
                    try:
                        # 提取数值并解包numpy数组
                        value = output_dict[year][city][0] #这里只有一个【0】
                        wg_values.append(float(value))
                    except KeyError:
                        print(f"警告：{city}-{year} 组合不存在于output_dict，已设为NaN")
                        wg_values.append(np.nan)

                # 添加新列
                df['Wg'] = wg_values

                # 保存结果
                try:
                    df.to_excel(output_file, index=False)
                    print(f"数据处理完成，结果已保存至: {output_file}")
                except PermissionError:
                    print(f"错误：没有权限写入 {output_file}，请检查文件是否被其他程序打开")
                
            if 'Wb' in config.checkpoint_path:
                input_file = config.test_dataset
                output_file = os.path.join(config.checkpoint_path, 'wb.xlsx')
                try:
                    df = pd.read_excel(input_file)
                    print(f"成功读取文件: {input_file}, 共 {len(df)} 行数据")
                except FileNotFoundError:
                    print(f"错误：文件 {input_file} 不存在！")
                    exit()

                # 生成Wb列
                wb_values = []
                for index, row in df.iterrows():
                    year = row['year']
                    city = row['city']
                    try:
                        # 提取数值并解包numpy数组
                        value = output_dict[year][city][0]
                        wb_values.append(float(value))
                    except KeyError:
                        print(f"警告：{city}-{year} 组合不存在于output_dict，已设为NaN")
                        wb_values.append(np.nan)

                # 添加新列
                df['Wb'] = wb_values

                # 保存结果
                try:
                    df.to_excel(output_file, index=False)
                    print(f"数据处理完成，结果已保存至: {output_file}")
                except PermissionError:
                    print(f"错误：没有权限写入 {output_file}，请检查文件是否被其他程序打开")
                    
                
    
            
        # print("\nCNN+LSTM+Attention模型评估结果：")

        # y_test3 = y_test.cpu().numpy()
        # preds = preds.cpu().numpy()

        # # 如果要按照城市计算就要reshape,如果不按照城市计算就直接
        # if config.targets == ['Wg','Wb'] or config.targets == ['Wb','Wg']:
        #     with open(f'{config.checkpoint_path}/result.txt', 'a', encoding='utf-8') as f:
        #         f.write(f"For {epoch},Wb：")
        #         print("\n下面是Wg：")
        #         y_test3 = y_test3.reshape(y_test3.shape[1],y_test3.shape[0],-1)
        #         preds = preds.reshape(preds.shape[1],preds.shape[0],-1)
        #         #移除为1的维度，否则会报错
        #         #np.squeeze(y_test3, axis=1)
        #         for i, city in enumerate(y_test3):
        #             mse = mean_squared_error(y_test3[i,:,0], preds[i,:,0])
        #             rmse = np.sqrt(mse)
        #             r2 = r2_score(np.squeeze(y_test3[i,:,0]), np.squeeze(preds[i,:,0])) #R2
        #             f.write(f"{i}:first,MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
        #             print(f"{i}:MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
        #             results = calculate_metrics(preds[i,:,0], y_test3[i,:,0])
        #             f.write(f",{i}:output:{results}\n")
        #             print(results)


        #         print("\n下面是Wb：")
        #         for i, target in enumerate(y_test3):
        #             f.write(f"For {epoch},Wg：")
        #             mse = mean_squared_error(np.squeeze(y_test3[i,:,1]), np.squeeze(preds[i,:,1]))
        #             rmse = np.sqrt(mse)
        #             r2 = r2_score(y_test3[i,:,1], preds[i,:,1]) #R2
        #             f.write(f"{i}:first,MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
        #             print(f"first,MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
        #             results = calculate_metrics(preds[i,:,1], y_test3[i,:,1])
        #             f.write(f",{i}:output:{results}\n")
        #             print(results)
        # else:
        #     y_test3 = y_test3.reshape(y_test3.shape[0],-1)
        #     preds = preds.reshape(preds.shape[0],-1)
        #     with open(f'{save_dir}/result.txt', 'a', encoding='utf-8') as f:
        #         print(f"\n下面是{config.targets}：")
        #         f.write(f"For {epoch},{config.targets}：")
        #         for i, target in enumerate(y_test3):
        #             mse = mean_squared_error(y_test3[i,:], preds[i,:])
        #             rmse = np.sqrt(mse)
        #             r2 = r2_score(y_test3[i,:], preds[i,:]) #R2
        #             f.write(f"{i}:first,MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
        #             print(f"first,MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
        #             results = calculate_metrics(preds[i,:], y_test3[i,:])
        #             f.write(f",{i}:output:{results}\n")
        #             print(results)









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size used in the training and validation loop.",
    )
    parser.add_argument(
        "--epochs", default=200000, type=int, help="Total number of epochs."
    )
    parser.add_argument(
        "--lr",
        default=0.0006,
        type=float,
        help="Base learning rate at the start of the training.",
    )
    parser.add_argument(
        "--ckpt", default=2000, type=int, help="Save model every ckpt epochs."
    )
    parser.add_argument(
        "--train_set_path", default="", type=str, help="Path to the training set."
    )
    parser.add_argument(
        "--checkpoint_path", default="/home/zhaohui1.wang/github/CNN-LSTM-Attention/model/['Wg']_20250517105956", type=str, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--dataset",
        default="sheng_1980to2017.xlsx",
        type=str,
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--test-dataset",
        default="prediction/SSP126_sheng_2025to2099.xlsx",
        type=str,
        help="Path to the test dataset file.",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        type=str,
        help="Path to the save result.",
    )
    parser.add_argument(
        "--features",
        type=str,        # 指定每个元素的类型为字符串
        nargs='+',       # 接受一个或多个值,wins
        # default=['prec', 'srad', 'Tmax', 'Tmin', 'wind',  'RH'],
        default=['Tmax','wind','prec'],
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
        default='Wg',
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
    test_model(config)
# python trainer.py --train --test --cuda 1  --targets Wg Wb  --dataset nanling.xlsx --features Tmax Tmin Tmean WS SH H P ETc
# Tmax	Tmin	Tmean	WS	SH	H	P	ETc	Yield	WFb	WFg

#例子
# python trainer.py --train --test --cuda 0  --targets Wb  --dataset sheng_1980to2017.xlsx --features Tmax Tmin RH srad prec
# python trainer.py --train --test --cuda 1  --targets Wg  --dataset sheng_1980to2017.xlsx --features Tmax wind prec



# python tester.py --train --test --cuda 0  --targets Wb  --dataset sheng_1980to2017.xlsx --features Tmax Tmin RH srad prec --test-dataset prediction/SSP245_sheng_2025to2099.xlsx --checkpoint_path "/home/zhaohui1.wang/github/CNN-LSTM-Attention/model/['Wb']_20250517105943"
# python tester.py --train --test --cuda 1  --targets Wg  --dataset sheng_1980to2017.xlsx --features Tmax wind prec --test-dataset prediction/SSP245_sheng_2025to2099.xlsx --checkpoint_path "/home/zhaohui1.wang/github/CNN-LSTM-Attention/model/['Wg']_20250517105956"


