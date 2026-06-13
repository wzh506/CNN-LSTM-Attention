

# dataset="huabei_1993to2017.xlsx"
echo "============================================"
dataset="huabei_1993to2021.xlsx"

cuda_device_0="0"
cuda_device_1="1"

targets_0="Wb"
targets_1="Wg"

# mod1='DCLFormer'
mod1='DCLFormer'
# mod1='CNN+LSTM'

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

# xiao=(0.99 0.98 0.975 0.96 )
xiao=(0.99 0.96 )
# xiao=(0.955 0.94 )
#对应关系
#xiao名称对应排序：noSAL,noTime, noTAL, noCITY 
#影响排序：noCity,noTAL,noTime, noSAL
# ssh -p 30611 zhaohui1.wang@10.251.18.148
epochs=30000
echo "============================================"
# 循环执行不同sc值
# for sc in {1..6}; do
for value in "${xiao[@]}"; do
#  for sc in {2..2}; do
    # 并行执行两个实验任务
    safe_mod1=$(echo "$mod1" | tr '+' '_' | tr ' ' '_')
    mkdir -p "logs/xiao"
    (
        # 任务1 - CUDA 0
        echo "============================================"
        echo "目前运行消融实验"
        echo "Starting experiment with parameters:"
        echo " - TARGETS: $targets_0"
        # echo " - Xiao: $value"
        echo " - CUDA DEVICE: $cuda_device_0"
        echo " - Model: $mod1"  
        echo "============================================"
        python -u trainer2.py \
            --train \
            --test \
            --cuda $cuda_device_0 \
            --targets $targets_0 \
            --dataset $dataset \
            --sc 2 \
            --value $value \
            --mod $mod1 \
            --epochs $epochs
    ) > "logs/xiao/${safe_mod1}_Wb_xiao_${value}.log" 2>&1 &  # 重定向输出到日志文件

    (
        # 任务2 - CUDA 1
        echo "============================================"
        echo "Starting experiment with parameters:"
        echo "目前运行消融实验"
        echo " - TARGETS: $targets_1"
        # echo " - Xiao: $value"
        echo " - CUDA DEVICE: $cuda_device_1"
        echo " - Model: $mod1"
        echo "============================================"
        python -u trainer2.py \
            --train \
            --test \
            --cuda $cuda_device_1 \
            --targets $targets_1 \
            --dataset $dataset \
            --sc 2 \
            --mod $mod1 \
            --value $value \
            --epochs $epochs
    ) > "logs/xiao/${safe_mod1}_Wg_xiao_${value}.log" 2>&1 &

    # 等待当前sc的两个并行任务完成
    wait
    
    # 添加间隔时间（可选）
    sleep 5
done