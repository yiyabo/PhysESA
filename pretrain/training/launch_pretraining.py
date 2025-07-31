#!/usr/bin/env python3
"""
预训练启动脚本
提供便捷的训练启动接口
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import yaml
import torch

def check_environment():
    """检查训练环境"""
    print("=" * 50)
    print("ENVIRONMENT CHECK")
    print("=" * 50)
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
    else:
        print("✗ CUDA not available")
        return False
    
    # 检查内存
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"✓ System memory: {mem.total // 1024**3}GB")
    except ImportError:
        print("? Memory info not available (psutil not installed)")
    
    # 检查PyTorch Lightning
    try:
        import pytorch_lightning as pl
        print(f"✓ PyTorch Lightning: {pl.__version__}")
    except ImportError:
        print("✗ PyTorch Lightning not available")
        return False
    
    # 检查PyTorch Geometric
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError:
        print("✗ PyTorch Geometric not available")
        return False
    
    print("=" * 50)
    return True

def setup_distributed_training():
    """设置分布式训练环境变量"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  # 使用所有8张GPU
    os.environ['NCCL_DEBUG'] = 'INFO'  # NCCL调试信息
    os.environ['PYTHONUNBUFFERED'] = '1'  # 实时输出

def update_config_paths(config_path: str, data_dir: str, output_dir: str):
    """更新配置文件中的路径"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新数据路径
    config['data']['processed_dir'] = data_dir
    
    # 更新输出路径
    config['logging']['log_dir'] = os.path.join(output_dir, 'logs')
    config['logging']['checkpoint_dir'] = os.path.join(output_dir, 'checkpoints')
    
    # 创建临时配置文件
    temp_config_path = config_path.replace('.yaml', '_temp.yaml')
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return temp_config_path

def main():
    parser = argparse.ArgumentParser(description="Launch PhysESA pretraining")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing processed training data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--config", type=str, 
                        default="configs/pretrain_config.yaml",
                        help="Configuration file path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only check environment, don't start training")
    parser.add_argument("--gpus", type=int, default=8,
                        help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    # 检查环境
    if not check_environment():
        print("Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    if args.dry_run:
        print("Dry run completed successfully.")
        sys.exit(0)
    
    # 检查数据目录
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist.")
        sys.exit(1)
    
    train_data = data_dir / "train_graphs.pkl"
    val_data = data_dir / "val_graphs.pkl"
    
    if not train_data.exists():
        print(f"Error: Training data {train_data} not found.")
        sys.exit(1)
    
    if not val_data.exists():
        print(f"Error: Validation data {val_data} not found.")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 更新配置文件
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / args.config
    
    if not config_path.exists():
        print(f"Error: Config file {config_path} not found.")
        sys.exit(1)
    
    temp_config_path = update_config_paths(str(config_path), str(data_dir), str(output_dir))
    
    # 设置分布式训练环境
    setup_distributed_training()
    
    # 构建训练命令
    trainer_script = script_dir / "pretrain_trainer.py"
    cmd = [
        sys.executable, str(trainer_script),
        "--config", temp_config_path
    ]
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    print("=" * 50)
    print("STARTING PRETRAINING")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Config file: {config_path}")
    print(f"GPUs: {args.gpus}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # 启动训练
        result = subprocess.run(cmd, check=True)
        print("Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        sys.exit(1)
    
    finally:
        # 清理临时配置文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

if __name__ == "__main__":
    main()