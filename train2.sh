

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

# ssh -p 30611 zhaohui1.wang@10.251.18.148
epochs=50000
echo "============================================"
# 循环执行不同sc值
# for sc in {1..6}; do
for sc in {1..6}; do
#  for sc in {2..2}; do
    # 并行执行两个实验任务
    safe_mod1=$(echo "$mod1" | tr '+' '_' | tr ' ' '_')
    mkdir -p "logs/$safe_mod1"
    (
        # 任务1 - CUDA 0
        echo "============================================"
        echo "Starting experiment with parameters:"
        echo " - TARGETS: $targets_0"
        echo " - SCALE (sc): $sc"
        echo " - CUDA DEVICE: $cuda_device_0"
        echo " - Model: $mod1"  
        echo "============================================"
        python trainer2.py \
            --train \
            --test \
            --cuda $cuda_device_0 \
            --targets $targets_0 \
            --dataset $dataset \
            --sc $sc \
            --mod $mod1 \
            --epochs $epochs
    ) > "logs/${safe_mod1}/${safe_mod1}_sc${sc}_Wb_v2.log" 2>&1 &  # 重定向输出到日志文件

    (
        # 任务2 - CUDA 1
        echo "============================================"
        echo "Starting experiment with parameters:"
        echo " - TARGETS: $targets_1"
        echo " - SCALE (sc): $sc"
        echo " - CUDA DEVICE: $cuda_device_1"
        echo " - Model: $mod1"
        echo "============================================"
        python trainer2.py \
            --train \
            --test \
            --cuda $cuda_device_1 \
            --targets $targets_1 \
            --dataset $dataset \
            --sc $sc \
            --mod $mod1 \
            --epochs $epochs
    ) > "logs/${safe_mod1}/${safe_mod1}_sc${sc}_Wg_v2.log" 2>&1 &

    # 等待当前sc的两个并行任务完成
    wait
    
    # 添加间隔时间（可选）
    sleep 5
done