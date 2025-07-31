# v2020-other-PL 数据集结构分析

### 1. 总体概况
• **数据集规模**: 3,777个蛋白质-配体复合物
• **总大小**: 约2.5GB的原始数据
• **数据来源**: PDBbind v2020数据库中除CASF-2016之外的其他蛋白质-配体复合物

### 2. 目录结构
v2020-other-PL/
├── v2020-other-PL/          # 原始数据 (2.5GB, 3777个复合物)
│   ├── 11gs/
│   ├── 13gs/
│   ├── 16pk/
│   └── ... (每个PDB ID一个目录)
├── CASF-2016/               # 测试集数据 (11GB, 285个复合物)
├── processed/               # 预处理数据 (2.3MB)
│   ├── train.pkl           # 训练集 (12,246个复合物)
│   ├── valid.pkl           # 验证集 (1,360个复合物)
│   ├── test.pkl            # 测试集 (285个复合物，来自CASF-2016)
│   └── dataset_stats.txt   # 数据集统计信息
└── processed_get_format/    # GET模型格式数据 (198MB)
    ├── test.pkl
    └── models/             # 预训练模型检查点


### 3. 每个复合物的文件结构
每个复合物目录包含4个标准文件：
{pdb_id}/
├── {pdb_id}_protein.pdb    # 蛋白质结构文件
├── {pdb_id}_pocket.pdb     # 结合口袋结构文件
├── {pdb_id}_ligand.mol2    # 配体结构文件 (MOL2格式)
└── {pdb_id}_ligand.sdf     # 配体结构文件 (SDF格式)


### 4. 数据集划分
根据dataset_stats.txt的信息：
• **训练集**: 12,246个复合物 (90% of v2020-other-PL)
• **验证集**: 1,360个复合物 (10% of v2020-other-PL)  
• **测试集**: 285个复合物 (CASF-2016 coreset)
• **总计**: 13,891个复合物

### 5. 代码使用指南

#### 基本数据加载
python
import os
import pickle
from pathlib import Path

# 数据集根目录
dataset_root = "/path/to/v2020-other-PL"

# 加载预处理的数据
with open(f"{dataset_root}/processed/train.pkl", "rb") as f:
    train_data = pickle.load(f)
    
with open(f"{dataset_root}/processed/valid.pkl", "rb") as f:
    valid_data = pickle.load(f)
    
with open(f"{dataset_root}/processed/test.pkl", "rb") as f:
    test_data = pickle.load(f)


#### 原始文件访问
python
def load_complex_files(pdb_id, dataset_root):
    """加载单个复合物的所有文件"""
    complex_dir = Path(dataset_root) / "v2020-other-PL" / pdb_id
    
    files = {
        'protein': complex_dir / f"{pdb_id}_protein.pdb",
        'pocket': complex_dir / f"{pdb_id}_pocket.pdb", 
        'ligand_mol2': complex_dir / f"{pdb_id}_ligand.mol2",
        'ligand_sdf': complex_dir / f"{pdb_id}_ligand.sdf"
    }
    
    return files

# 示例使用
pdb_id = "2gph"
files = load_complex_files(pdb_id, dataset_root)


#### 批量处理
python
def get_all_complex_ids(dataset_root):
    """获取所有复合物的PDB ID"""
    v2020_dir = Path(dataset_root) / "v2020-other-PL"
    return [d.name for d in v2020_dir.iterdir() if d.is_dir()]

# 获取所有3777个复合物ID
all_ids = get_all_complex_ids(dataset_root)
print(f"Total complexes: {len(all_ids)}")


### 6. 与CASF-2016的关系
• CASF-2016作为标准测试集，包含285个高质量复合物
• v2020-other-PL提供了大规模训练数据，补充CASF-2016的有限规模
• 两者结合形成完整的训练-测试框架

### 7. 适用场景
• **分子对接算法训练**: 使用大规模训练集改进对接精度
• **评分函数开发**: 基于多样化的蛋白质-配体相互作用数据
• **深度学习模型**: 提供足够数据量支持复杂模型训练
• **基准测试**: 使用CASF-2016进行标准化评估

这个数据集为你的AI代码开发提供了完整的蛋白质-配体相互作用数据，支持从数据预处理到模型训练和评估的完整流程。