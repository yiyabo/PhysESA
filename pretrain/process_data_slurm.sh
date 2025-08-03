#!/bin/bash
#SBATCH --gpus=0
#SBATCH -p vip_gpu_01
#SBATCH --job-name=physesa_data_process
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/data_process_%j.out
#SBATCH --error=logs/data_process_%j.err

echo "=========================================="
echo "PhysESA 数据预处理任务 - SLURM"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# 加载环境
module load anaconda/2024.10

# 使用正确的miniconda3路径初始化
source /home/dalhxwlyjsuo/criait_zhaozc/miniconda3/etc/profile.d/conda.sh

# 激活环境
conda activate esa

# 设置路径
DATASET_ROOT="/home/dalhxwlyjsuo/criait_zhaozc/xinxiangwang"
OUTPUT_DIR="./data_process"

# 创建日志目录
mkdir -p logs

# 检查数据集是否存在
echo "检查数据集..."
if [ ! -d "$DATASET_ROOT/v2020-other-PL" ]; then
    echo "错误: 找不到数据集目录 $DATASET_ROOT/v2020-other-PL"
    exit 1
fi

# 显示数据集信息
echo "数据集路径: $DATASET_ROOT/v2020-other-PL"
COMPLEX_COUNT=$(ls -1 "$DATASET_ROOT/v2020-other-PL" | wc -l)
echo "发现复合物数量: $COMPLEX_COUNT"

# 创建输出目录
echo "创建输出目录..."
mkdir -p "$OUTPUT_DIR"

# 显示系统信息
echo "=========================================="
echo "系统信息:"
echo "节点: $SLURM_NODELIST"
echo "CPU核心: $SLURM_CPUS_PER_TASK"
echo "内存: ${SLURM_MEM_PER_NODE}MB"
echo "=========================================="

# 开始预处理
echo "开始数据预处理..."
echo "开始时间: $(date)"

# 运行预处理脚本
python preprocessing/v2020_data_processor.py \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --val_ratio 0.1 \
    --batch_size 100

# 检查处理结果
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "数据预处理完成！"
    echo "完成时间: $(date)"
    echo "=========================================="
    
    # 显示生成的文件
    echo "生成的文件:"
    ls -lh "$OUTPUT_DIR"/*.pkl 2>/dev/null
    
    # 显示文件大小统计
    echo "文件大小统计:"
    du -sh "$OUTPUT_DIR"
    
    echo "✅ 数据预处理任务成功完成"
    echo "现在可以提交训练任务"
    
else
    echo "❌ 数据预处理失败，退出码: $?"
    exit 1
fi