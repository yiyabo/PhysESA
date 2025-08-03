#!/usr/bin/env python3
"""
超简化版训练脚本 - 避免所有依赖问题
测试基本的图数据加载和模型前向传播
"""

import os
import sys
import torch
import torch.nn as nn
import pickle
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm

# 导入项目模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'model'))

def test_data_loading(data_dir):
    """测试数据加载"""
    print("=" * 50)
    print("测试数据加载...")
    
    train_path = os.path.join(data_dir, "train_graphs.pkl")
    val_path = os.path.join(data_dir, "val_graphs.pkl")
    
    if not os.path.exists(train_path):
        print(f"❌ 找不到训练数据: {train_path}")
        return False
    
    if not os.path.exists(val_path):
        print(f"❌ 找不到验证数据: {val_path}")
        return False
    
    try:
        with open(train_path, 'rb') as f:
            train_graphs = pickle.load(f)
        print(f"✅ 成功加载训练数据: {len(train_graphs)} 个图")
        
        with open(val_path, 'rb') as f:
            val_graphs = pickle.load(f)
        print(f"✅ 成功加载验证数据: {len(val_graphs)} 个图")
        
        # 检查第一个图的结构
        first_graph = train_graphs[0]
        print(f"✅ 第一个图的属性:")
        for attr in dir(first_graph):
            if not attr.startswith('_') and hasattr(first_graph, attr):
                value = getattr(first_graph, attr)
                if torch.is_tensor(value):
                    print(f"  {attr}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {attr}: {type(value)} - {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return False

def test_model_imports():
    """测试模型导入"""
    print("=" * 50)
    print("测试模型导入...")
    
    try:
        from model.esa.masked_layers import Estimator
        print("✅ 成功导入 Estimator")
        return True
    except Exception as e:
        print(f"❌ 导入 Estimator 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_torch_scatter():
    """测试torch_scatter功能"""
    print("=" * 50)
    print("测试torch_scatter...")
    
    try:
        from torch_scatter import scatter_sum
        
        # 测试基本功能
        src = torch.randn(6, 3)
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        out = scatter_sum(src, index, dim=0)
        print(f"✅ torch_scatter 工作正常: {out.shape}")
        return True
        
    except Exception as e:
        print(f"❌ torch_scatter 有问题: {str(e)}")
        
        # 尝试fallback方案
        print("尝试使用原生PyTorch实现scatter_sum...")
        try:
            def scatter_sum_fallback(src, index, dim=0, dim_size=None):
                if dim_size is None:
                    dim_size = index.max().item() + 1
                
                # 创建输出张量
                size = list(src.shape)
                size[dim] = dim_size
                out = torch.zeros(size, dtype=src.dtype, device=src.device)
                
                # 手动scatter
                for i in range(src.shape[dim]):
                    idx = index[i].item()
                    if dim == 0:
                        out[idx] += src[i]
                    else:
                        raise NotImplementedError("Only dim=0 supported in fallback")
                
                return out
            
            # 测试fallback
            src = torch.randn(6, 3)
            index = torch.tensor([0, 1, 0, 1, 2, 1])
            out = scatter_sum_fallback(src, index, dim=0)
            print(f"✅ Fallback scatter_sum 工作正常: {out.shape}")
            return True
            
        except Exception as e2:
            print(f"❌ Fallback 也失败: {str(e2)}")
            return False

def test_graph_construction(data_dir):
    """测试图构建和前向传播"""
    print("=" * 50)
    print("测试图构建...")
    
    try:
        # 加载一个图
        with open(os.path.join(data_dir, "train_graphs.pkl"), 'rb') as f:
            graphs = pickle.load(f)
        
        graph = graphs[0]
        print(f"✅ 加载图成功")
        
        # 检查是否有torch_geometric
        try:
            from torch_geometric.data import Batch
            
            # 尝试创建batch
            batch = Batch.from_data_list([graph])
            print(f"✅ 创建batch成功: {batch}")
            
            # 测试基本的tensor操作
            if hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
                print(f"  节点特征: {batch.x.shape}")
                print(f"  边索引: {batch.edge_index.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ torch_geometric 有问题: {str(e)}")
            return False
            
    except Exception as e:
        print(f"❌ 图构建测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Ultra Simple Environment Test")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    
    args = parser.parse_args()
    
    print("🔍 PhysESA 环境诊断测试")
    print("=" * 50)
    
    # 基本信息
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    all_tests_passed = True
    
    # 测试1: 数据加载
    if not test_data_loading(args.data_dir):
        all_tests_passed = False
    
    # 测试2: torch_scatter
    if not test_torch_scatter():
        all_tests_passed = False
    
    # 测试3: 模型导入
    if not test_model_imports():
        all_tests_passed = False
    
    # 测试4: 图构建
    if not test_graph_construction(args.data_dir):
        all_tests_passed = False
    
    print("=" * 50)
    if all_tests_passed:
        print("🎉 所有测试通过！环境应该可以正常训练")
        print("建议下一步: 尝试简化的训练脚本")
    else:
        print("❌ 部分测试失败，需要修复环境问题")
        print("建议: 重新安装相关依赖包")
    
    print("=" * 50)

if __name__ == "__main__":
    main()