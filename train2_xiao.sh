

dataset="huabei_1993to2017.xlsx"

cuda_device_0="0"
cuda_device_1="1"

targets_0="Wb"
targets_1="Wg"

# mod1='DCLFormer'
mod1='DCLFormer'
# mod1='CNN+LSTM'

# ssh -p 30611 zhaohui1.wang@10.251.18.148
epochs=30000

declare -a sc_values=(1 2 4 6)
# 循环执行不同sc值
# for sc in {1..6}; do
# for sc in {1}; do
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
        python trainer2.py \
            --train \
            --test \
            --cuda $cuda_device_0 \
            --targets $targets_0 \
            --dataset $dataset \
            --sc 2 \
            --mod $mod1 \
            --epochs $epochs \

    ) > "${mod1}_Wb_v2_xiao4.log" 2>&1 &  # 重定向输出到日志文件

    
#     # 添加间隔时间（可选）
#     sleep 5
# done