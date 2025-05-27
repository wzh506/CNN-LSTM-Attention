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
from models import DCLFormer,LSTM, CNN_LSTM
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
            features = ['VPD','wind','srad','prec']
        elif sc == '3':
            features = ['wind','VPD','Tmax']
        elif sc == '4':
            features = ['Tmax','Tmin','RH','VPD','prec']
        elif sc == '5':
            features = ['wind','VPD','srad','prec']
        elif sc == '6':
            features = ['Tmax','Tmin','srad','prec']


    return features

def generate_emb(current_cities):
    emb_dict = {}
    for i,city in enumerate(current_cities):
        emb_dict[city] = i
    return emb_dict




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
                # 构建特征序列 [window_size, num_cities, input_dim]
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
    X_features=X[:,:,:-1] 
    X_cities=X[:,:,-1]# 去掉最后一个维度
    X_scaled = scaler.fit_transform(X_features.reshape(-1, X_features.shape[-1])).reshape(X_features.shape)
    X_all = np.concatenate([X_features, X_cities[..., np.newaxis] ], axis=-1).astype(np.float64)
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



    # 初始化模型
    print('使用的模型为:',config.mod)
    if config.mod == 'DCLFormer':
        model = DCLFormer(input_size=X.shape[-1], output_size=y.shape[-1])
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    if config.save_dir is None:
        formatted_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        if config.sc is not None:
            save_dir = os.path.join("model",f"{config.mod}",str(config.targets)+f"_sc{config.sc}_{formatted_time}")
        else:
            save_dir = os.path.join("model",f"{config.mod}", str(config.targets)+f"_{formatted_time}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        with open(f'{save_dir}/result.txt', 'w', encoding='utf-8') as f:
            f.write("The config file is：\n")  # \n表示换行符
            f.write(f"{config}\n")
            f.write(f"The save_dir is {save_dir}\n")
    else:
        save_dir = config.save_dir
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir) 

    
    # 训练模型
    if config.train == True:
        for epoch in tqdm(range(config.epochs),desc='Training'):
            if config.mod == "DCLFormer":#正常训练方法
                if epoch < int(config.epochs//3*2): #原来用的//2
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
                        with open(f'{save_dir}/result.txt', 'a', encoding='utf-8') as f:
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
            else:    #对于DCLFormer外的其他模型采用如下训练方式
                if epoch < int(config.epochs//3*5): #
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
                        with open(f'{save_dir}/result.txt', 'a', encoding='utf-8') as f:
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
                    
            if (epoch + 1) % config.ckpt == 0:
                torch.save(model.state_dict(),f'{save_dir}/{config.mod}_{epoch}.pth')
                if config.test == True:
                    if config.mod == "DCLFormer":
                        model2 = DCLFormer(input_size=X.shape[-1], output_size=y.shape[-1])
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
                        with open(f'{save_dir}/result.txt', 'a', encoding='utf-8') as f:
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
                        with open(f'{save_dir}/result.txt', 'a', encoding='utf-8') as f: #使用了with
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
                        print(f'这是模型第{epoch}的测试结果，接下来按照城市显示结果：')
                        for i in range(len(y_test3)):
                            print(f'对于第{i}个测试集的结果：')
                            for j,city in enumerate(current_cities):
                                print(f'城市{city}的预测值为：{preds[i][j]}')
                                print(f'城市{city}的真实值为：{y_test3[i][j]}')
                                

        torch.save(model.state_dict(),f'{save_dir}/{config.mod}.pth')



















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




