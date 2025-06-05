

dataset="huabei_1993to2017.xlsx"

cuda_device_0="0"
cuda_device_1="1"

targets_0="Wb"
targets_1="Wg"

# mod1='DCLFormer'
mod1='LSTM'
# mod1='CNN+LSTM'

# ssh -p 30611 zhaohui1.wang@10.251.18.148
epochs=200000

# 循环执行不同sc值
# for sc in {1..6}; do
for sc in {1..6}; do
    # 并行执行两个实验任务
    (
        # 任务1 - CUDA 0
        echo "============================================"
        echo "Starting experiment with parameters:"
        echo " - TARGETS: $targets_0"
        echo " - SCALE (sc): $sc"
        echo " - CUDA DEVICE: $cuda_device_0"
        echo " - Model: $mod1"  
        echo "============================================"
        python trainer.py \
            --train \
            --test \
            --cuda $cuda_device_0 \
            --targets $targets_0 \
            --dataset $dataset \
            --sc $sc \
            --mod $mod1 \
            --epochs $epochs
    ) > "${mod1}_sc${sc}_Wb.log" 2>&1 &  # 重定向输出到日志文件

    (
        # 任务2 - CUDA 1
        echo "============================================"
        echo "Starting experiment with parameters:"
        echo " - TARGETS: $targets_1"
        echo " - SCALE (sc): $sc"
        echo " - CUDA DEVICE: $cuda_device_1"
        echo " - Model: $mod1"
        echo "============================================"
        python trainer.py \
            --train \
            --test \
            --cuda $cuda_device_1 \
            --targets $targets_1 \
            --dataset $dataset \
            --sc $sc \
            --mod $mod1 \
            --epochs $epochs
    ) > "${mod1}_sc${sc}_Wg.log" 2>&1 &

    # 等待当前sc的两个并行任务完成
    wait
    
    # 添加间隔时间（可选）
    sleep 5
done