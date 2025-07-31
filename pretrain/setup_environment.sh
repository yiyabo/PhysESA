#!/bin/bash

# PhysESA 预训练环境设置脚本

echo "========================================"
echo "PhysESA 预训练环境设置"
echo "========================================"

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 设置环境名称
ENV_NAME="physesa_pretrain"

echo "正在创建conda环境: $ENV_NAME"

# 创建新环境
conda create -n $ENV_NAME python=3.11 -y

# 激活环境
echo "激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 安装PyTorch（根据你的CUDA版本调整）
echo "安装PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装PyTorch Geometric
echo "安装PyTorch Geometric..."
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 安装其他依赖
echo "安装其他依赖..."
pip install -r requirements_minimal.txt

# 安装RDKit（通过conda，更稳定）
echo "安装RDKit..."
conda install -c conda-forge rdkit -y

# 安装BioPython
echo "安装BioPython..."
pip install biopython

# 验证安装
echo "========================================"
echo "验证安装..."
echo "========================================"

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')

import torch_geometric
print(f'PyTorch Geometric: {torch_geometric.__version__}')

import numpy as np
print(f'NumPy: {np.__version__}')

import pandas as pd
print(f'Pandas: {pd.__version__}')

import pytorch_lightning as pl
print(f'PyTorch Lightning: {pl.__version__}')

from Bio import PDB
print('BioPython: OK')

from rdkit import Chem
print('RDKit: OK')

print('\\n✅ 所有依赖安装成功！')
"

echo "========================================"
echo "环境设置完成！"
echo "========================================"
echo "激活环境命令: conda activate $ENV_NAME"
echo "运行预处理: python pretrain/preprocessing/v2020_data_processor.py --help"