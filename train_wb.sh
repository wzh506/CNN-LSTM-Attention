

dataset="huabei_1993to2021.xlsx"

cuda_device_0="0"
cuda_device_1="1"

# targets_0="Wb"


# mod1='DCLFormer'
mod1='LSTM'
mod2='CNN+LSTM'
# mod1='CNN+LSTM'

# ssh -p 30611 zhaohui1.wang@10.251.18.148
epochs=20000

# 循环执行不同sc值
# for sc in {1..6}; do
for targets_0 in "Wb" "Wg"; do
    for sc in {1..6}; do
        # Prepare per-model log directories and safe names (replace '+' and spaces)
        safe_mod1=$(echo "$mod1" | tr '+' '_' | tr ' ' '_')
        safe_mod2=$(echo "$mod2" | tr '+' '_' | tr ' ' '_')
        mkdir -p "logs/$safe_mod1"
        mkdir -p "logs/$safe_mod2"
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
        ) > "logs/${safe_mod1}/${safe_mod1}_sc${sc}_${targets_0}.log" 2>&1 &  # 重定向输出到日志文件

        (
            # 任务2 - CUDA 1
            echo "============================================"
            echo "Starting experiment with parameters:"
            echo " - TARGETS: $targets_0"
            echo " - SCALE (sc): $sc"
            echo " - CUDA DEVICE: $cuda_device_1"
            echo " - Model: $mod2"
            echo "============================================"
            python trainer.py \
                --train \
                --test \
                --cuda $cuda_device_1 \
                --targets $targets_0 \
                --dataset $dataset \
                --sc $sc \
                --mod $mod2 \
                --epochs $epochs
        ) > "logs/${safe_mod2}/${safe_mod2}_sc${sc}_${targets_0}.log" 2>&1 &

        # 等待当前sc的两个并行任务完成
        wait
        
        # 添加间隔时间（可选）
        sleep 5
    done
done