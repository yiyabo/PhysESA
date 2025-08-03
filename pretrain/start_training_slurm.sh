#!/bin/bash
#SBATCH --gpus=2
#SBATCH -p vip_gpu_01
#SBATCH --job-name=physesa_pretrain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

echo "=========================================="
echo "PhysESA 预训练任务 - SLURM"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# 加载环境
module load anaconda/2024.10

# 使用正确的miniconda3路径初始化
source /home/dalhxwlyjsuo/criait_zhaozc/miniconda3/etc/profile.d/conda.sh

# 激活环境
conda activate esa

# 设置路径
DATA_DIR="./data_process"
OUTPUT_DIR="./training_output"
CONFIG_FILE="configs/pretrain_config_4090.yaml"

# 创建必要目录
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# 检查数据文件是否存在
echo "检查数据文件..."
if [ ! -f "$DATA_DIR/train_graphs.pkl" ]; then
    echo "错误: 找不到训练数据文件 $DATA_DIR/train_graphs.pkl"
    echo "请先运行数据预处理: sbatch process_data_slurm.sh"
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

# 显示系统信息
echo "=========================================="
echo "系统信息:"
echo "节点: $SLURM_NODELIST"
echo "GPU数量: $SLURM_GPUS"
echo "CPU核心: $SLURM_CPUS_PER_TASK"
echo "内存: ${SLURM_MEM_PER_NODE}MB"
echo "=========================================="

# 显示配置信息
echo "训练配置:"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "配置文件: $CONFIG_FILE"
echo "开始时间: $(date)"
echo "=========================================="

# 启动训练
echo "启动训练..."
python training/launch_pretraining.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG_FILE"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "训练完成！"
    echo "完成时间: $(date)"
    echo "=========================================="
    
    # 显示生成的模型文件
    echo "生成的模型文件:"
    ls -lh "$OUTPUT_DIR"/checkpoints/*.ckpt 2>/dev/null || echo "没有找到检查点文件"
    
    echo "✅ 预训练任务成功完成"
    
else
    echo "❌ 训练失败，退出码: $?"
    exit 1
fi