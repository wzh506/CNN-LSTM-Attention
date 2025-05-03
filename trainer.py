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


def train_model(config):
    # 读取数据
    df = pd.read_excel(config.dataset)
    window_size = config.window
    features = config.features
    targets = config.targets
    combined_cols = features + targets

    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda



    # 获取所有城市列表（假设有city列）
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
                window_data = [year_dict[y][combined_cols].values for y in window_years]
                current_features = year_dict[current_year][features].values
                
                # 拼接特征维度：[历史特征 + 当前特征] 
                input_features = np.concatenate(window_data + [current_features], axis=1)
                
                # 获取目标值
                target_output = year_dict[current_year][targets].values
                
                X_seq.append(input_features)
                y_seq.append(target_output)

        # 转换为数组
        X = np.array(X_seq)  # 形状: (样本数, 城市数, 输入维度)
        y = np.array(y_seq)  # 形状: (样本数, 城市数, 输出维度)

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
    y = np.array(y_seq)  # 形状: (样本数, 城市数, 2)

    #检查是否有nan
    # for x in X:
    #     if np.isnan(x).any():
    #         print("nan !")
            


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # 转换为PyTorch Tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    if config.method == 'mean':
        existence_tensor = torch.tensor(existence_tensor, dtype=torch.float32)
    else:
        existence_tensor = torch.ones([X.shape[0], X.shape[1]])  # 创建一个全1的存在性张量
    existence_tensor = existence_tensor.unsqueeze(-1)  # 添加一个维度以匹配目标输出的形状
    X_tensor = torch.cat([X_tensor,existence_tensor],dim=-1)  # 将存在性张量应用于输入数据,最后一个维度是当前年数据是否存在

    # 验证数据维度
    print(f"输入数据维度: {X_tensor.shape}")  # 应为 (样本数, 城市数, 输入特征数)
    print(f"目标数据维度: {y_tensor.shape}")  # 应为 (样本数, 城市数, 2)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.05, random_state=42)#10:1的比例差不多


    # 初始化模型
    model = DCLFormer(input_size=X.shape[-1], output_size=y.shape[-1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    if config.save_dir is None:
        formatted_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        save_dir = os.path.join("model", str(config.targets)+f"_{formatted_time}")
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
                with open(f'{save_dir}/result.txt', 'a', encoding='utf-8') as f:
                    f.write(f"Epoch {epoch+1}, Loss: {loss.item():.4f}\n")
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            if (epoch + 1) % config.ckpt == 0:
                torch.save(model.state_dict(),f'{save_dir}/DCLFormer_{epoch}.pth')
                if config.test == True:
                    model2 = DCLFormer(input_size=X.shape[-1], output_size=y.shape[-1])
                    model2.load_state_dict(torch.load(f'{save_dir}/DCLFormer_{epoch}.pth'))
                    # 测试模型
                    model2.to('cpu')
                    model2.eval()
                    with torch.no_grad():
                        inputs = X_test[:,:,:X_train.shape[-1]-1]  # 去掉最后一个维度
                        existence_test = X_test[:,:,-1]  # 最后一个维度
                        preds = model2(inputs)
                        existence_test = existence_test.unsqueeze(-1)
                        # print('existence_test:',type(existence_test))
                        # print('y_test:',type(preds))
                        outputs = preds * existence_test.repeat(1, 1, preds.shape[-1])   # 乘以存在性张量
                        y_test2 = y_test * existence_test.repeat(1, 1, y_test.shape[-1])  # 乘以存在性张量
                        test_loss = criterion(preds, y_test2)
                        print(f"Test Loss: {test_loss.item():.4f}")
                    print("\nCNN+LSTM+Attention模型评估结果：")

                    y_test3 = y_test.cpu().numpy()
                    preds = preds.cpu().numpy()
                    
                    if config.targets == ['Wg','Wb'] or config.targets == ['Wb','Wg']:
                        with open(f'{save_dir}/result.txt', 'a', encoding='utf-8') as f:
                            f.write(f"For {epoch},Wb：")
                            print("\n下面是Wb：")
                            for i, target in enumerate(y_test3):
                                mse = mean_squared_error(y_test3[i,:,0], preds[i,:,0])
                                rmse = np.sqrt(mse)
                                r2 = r2_score(y_test3[i,:,0], preds[i,:,0]) #R2
                                f.write(f"{i}:first,MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
                                print(f"{i}:MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
                                results = calculate_metrics(preds[i,:,0], y_test3[i,:,0])
                                f.write(f",{i}:output:{results}\n")
                                print(results)


                            print("\n下面是Wg：")
                            for i, target in enumerate(y_test3):
                                f.write(f"For {epoch},Wg：")
                                mse = mean_squared_error(y_test3[i,:,1], preds[i,:,1])
                                rmse = np.sqrt(mse)
                                r2 = r2_score(y_test3[i,:,1], preds[i,:,1]) #R2
                                f.write(f"{i}:first,MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
                                print(f"first,MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
                                results = calculate_metrics(preds[i,:,1], y_test3[i,:,1])
                                f.write(f",{i}:output:{results}\n")
                                print(results)
                    else:
                        with open(f'{save_dir}/result.txt', 'a', encoding='utf-8') as f:
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

        torch.save(model.state_dict(),f'{save_dir}/DCLFormer.pth')







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
        "--ckpt", default=5000, type=int, help="Save model every ckpt epochs."
    )
    parser.add_argument(
        "--train_set_path", default="", type=str, help="Path to the training set."
    )
    parser.add_argument(
        "--checkpoint_path", default="", type=str, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--dataset",
        default="final_data.xlsx",
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
        "--hw_range",
        nargs=2,
        type=int,
        default=[0, 18],
        help="The range of the height and width.",
    )
    parser.add_argument(
        "--features",
        type=str,        # 指定每个元素的类型为字符串
        nargs='+',       # 接受一个或多个值
        default=['prec', 'srad', 'Tmax', 'Tmin', 'wind', 'SPEI', 'VPD', 'RH'],
        help="Input features (space-separated strings). Example: --features prec srad Tmax"
    ) #可以这样写：--features prec srad Tmax，得到：['prec', 'srad', 'Tmax']
    parser.add_argument(
        "--targets",
        type=str,        # 指定每个元素的类型为字符串
        nargs='+', 
        default=['Wg','Wb'],
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
# python trainer.py --train --test --cuda 1  --targets Wg Wb 




