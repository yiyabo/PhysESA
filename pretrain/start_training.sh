#!/bin/bash

# PhysESA 预训练启动脚本
# 用于避免命令行参数错误

echo "=========================================="
echo "PhysESA 预训练启动脚本"
echo "=========================================="

# 设置路径
DATA_DIR="./data_process"
OUTPUT_DIR="./training_output"
CONFIG_FILE="configs/pretrain_config_4090.yaml"

# 检查数据文件是否存在
echo "检查数据文件..."
if [ ! -f "$DATA_DIR/train_graphs.pkl" ]; then
    echo "错误: 找不到训练数据文件 $DATA_DIR/train_graphs.pkl"
    exit 1
fi

if [ ! -f "$DATA_DIR/val_graphs.pkl" ]; then
    echo "错误: 找不到验证数据文件 $DATA_DIR/val_graphs.pkl"
    exit 1
fi

echo "检查配置文件..."
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 找不到配置文件 $CONFIG_FILE"
    exit 1
fi

# 创建输出目录
echo "创建输出目录..."
mkdir -p "$OUTPUT_DIR"

# 显示配置信息
echo "=========================================="
echo "训练配置:"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "配置文件: $CONFIG_FILE"
echo "=========================================="

# 启动训练
echo "启动训练..."
python training/launch_pretraining.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG_FILE"

echo "训练脚本执行完成！"