#!/bin/bash
#SBATCH --gpus=4
#SBATCH -p vip_gpu_01
#SBATCH --job-name=physesa_train_fast
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --output=logs/training_fast_%j.out
#SBATCH --error=logs/training_fast_%j.err

echo "=========================================="
echo "PhysESA 预训练任务 - 高速版"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# 加载环境
module load anaconda/2024.10
source /home/dalhxwlyjsuo/criait_zhaozc/miniconda3/etc/profile.d/conda.sh
conda activate esa

# 设置路径
DATA_DIR="./data_process"
OUTPUT_DIR="./training_output_fast"
CONFIG_FILE="configs/pretrain_config_fast.yaml"

# 创建高速配置文件
cat > "$CONFIG_FILE" << 'EOF'
# PhysESA 预训练配置文件 - 高速版

experiment:
  name: "physesa_binding_site_pretrain_fast"
  description: "PhysESA预训练实验 - 高速版本"
  seed: 42

data:
  processed_dir: "./data_process"
  batch_size: 16  # 增大batch size利用更多GPU
  num_workers: 16  # 增加数据加载线程
  
model:
  esa_config:
    graph_dim: 256
    num_heads: 8
    num_inds: 16
    
    atomic_encoder_config:
      num_layers: 4
      dropout: 0.1
      use_bn: true
      linear_output_size: 256
    
    coarse_encoder_config:
      num_layers: 3
      dropout: 0.1
      use_bn: true
      linear_output_size: 256
  
  feature_dims:
    node_dim: 74
    edge_dim: 16
  
  learning_rate: 2e-4  # 增大学习率加快收敛
  weight_decay: 1e-5

training:
  max_epochs: 100  # 减少epochs
  gpus: 4  # 使用4张GPU
  strategy: "ddp"
  precision: 16
  
  val_check_interval: 1.0  # 每个epoch验证一次
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  
  early_stopping_patience: 15

logging:
  log_dir: "./logs"
  checkpoint_dir: "./checkpoints_fast"
  log_every_n_steps: 25

wandb:
  enabled: false
  project: "physesa-pretraining-fast"

hardware:
  memory: "256GB"
  gpus: "4x GPU"
  cpu: "32 cores"
  note: "高速训练配置"
EOF

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# 检查数据文件
echo "检查数据文件..."
if [ ! -f "$DATA_DIR/train_graphs.pkl" ]; then
    echo "错误: 找不到训练数据，请先运行数据预处理"
    exit 1
fi

echo "=========================================="
echo "高速训练配置:"
echo "GPU数量: $SLURM_GPUS"
echo "CPU核心: $SLURM_CPUS_PER_TASK"
echo "内存: ${SLURM_MEM_PER_NODE}MB"
echo "预计时间: 12-16小时"
echo "=========================================="

echo "开始高速训练..."
echo "开始时间: $(date)"

python training/launch_pretraining.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG_FILE"

echo "完成时间: $(date)"
echo "✅ 高速训练完成"