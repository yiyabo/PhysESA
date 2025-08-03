#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p vip_gpu_01
#SBATCH --job-name=physesa_data_fast
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/data_process_fast_%j.out
#SBATCH --error=logs/data_process_fast_%j.err

echo "=========================================="
echo "PhysESA 数据预处理任务 - 高速版"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# 加载环境
module load anaconda/2024.10
source /home/dalhxwlyjsuo/criait_zhaozc/miniconda3/etc/profile.d/conda.sh
conda activate esa

# 设置路径
DATASET_ROOT="/home/dalhxwlyjsuo/criait_zhaozc/xinxiangwang"
OUTPUT_DIR="./data_process"

# 创建目录
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# 检查数据集
echo "数据集路径: $DATASET_ROOT/v2020-other-PL"
COMPLEX_COUNT=$(ls -1 "$DATASET_ROOT/v2020-other-PL" | wc -l)
echo "复合物数量: $COMPLEX_COUNT"

echo "=========================================="
echo "高速配置:"
echo "CPU核心: $SLURM_CPUS_PER_TASK"
echo "内存: ${SLURM_MEM_PER_NODE}MB"
echo "预计时间: 2-3小时"
echo "=========================================="

echo "开始高速预处理..."
echo "开始时间: $(date)"

# 使用更大的batch_size和num_workers来提速
python preprocessing/v2020_data_processor.py \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --val_ratio 0.1 \
    --batch_size 200

echo "完成时间: $(date)"
echo "✅ 高速预处理完成"