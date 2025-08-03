#!/usr/bin/env python3
"""
简化版训练脚本 - 不使用PyTorch Lightning
直接使用原生PyTorch训练，避免依赖问题
"""

import os
import sys
import torch
import torch.nn as nn
import pickle
import yaml
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List
import argparse
import logging
from tqdm import tqdm
import time

# 导入项目模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'model'))
from pretrain.models.physesa_binding_site import PhysESABindingSite

class SimpleGraphDataset(Dataset):
    """简单图数据集"""
    
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            self.graphs = pickle.load(f)
        print(f"Loaded {len(self.graphs)} graphs from {data_path}")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

class SimpleTrainer:
    """简化版训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建模型
        self.model = self.create_model()
        self.model.to(self.device)
        
        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['model']['learning_rate'],
            weight_decay=config['model']['weight_decay']
        )
        
        # 创建学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 统计信息
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def create_model(self):
        """创建模型"""
        from pretrain.models.physesa_binding_site import PhysESABindingSite
        
        model_config = self.config['model']
        
        # 创建简化版本，不使用PyTorch Lightning
        class SimplePhysESA(nn.Module):
            def __init__(self, esa_config, feature_dims):
                super().__init__()
                # 导入原始模型的核心部分
                from model.esa.masked_layers import Estimator
                
                # 构建原子级编码器
                common_config = {k: v for k, v in esa_config.items() if k not in ['atomic_encoder_config', 'coarse_encoder_config']}
                atomic_specific_config = esa_config['atomic_encoder_config']
                atomic_config = {**common_config, **atomic_specific_config}
                atomic_config['num_features'] = feature_dims['node_dim']
                atomic_config['edge_dim'] = feature_dims['edge_dim']
                
                self.atomic_encoder = Estimator(**atomic_config)
                
                # 粗粒度编码器
                coarse_specific_config = esa_config['coarse_encoder_config']
                coarse_config = {**common_config, **coarse_specific_config}
                coarse_config['num_features'] = atomic_config['graph_dim']
                coarse_config['edge_dim'] = 0
                coarse_config['linear_output_size'] = atomic_config['graph_dim']
                
                self.coarse_encoder = Estimator(**coarse_config)
                
                # 结合位点预测头
                self.binding_site_head = nn.Sequential(
                    nn.Linear(atomic_config['graph_dim'], atomic_config['graph_dim'] // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(atomic_config['graph_dim'] // 2, 2)
                )
            
            def forward(self, batch):
                from torch_scatter import scatter_sum
                
                # 原子级编码
                atomic_edge_embeds, _ = self.atomic_encoder(batch, return_embeds=True)
                
                # 池化到粗粒度
                source_nodes = batch.edge_index[0]
                num_atomic_nodes = batch.num_nodes
                updated_atomic_node_features = scatter_sum(
                    atomic_edge_embeds, source_nodes, dim=0, dim_size=num_atomic_nodes
                )
                
                coarse_node_features = scatter_sum(
                    updated_atomic_node_features,
                    batch.atom_to_coarse_idx,
                    dim=0,
                    dim_size=batch.num_coarse_nodes
                )
                
                # 处理padding
                num_expected_coarse_nodes = batch.coarse_pos.shape[0]
                if coarse_node_features.shape[0] < num_expected_coarse_nodes:
                    pad_size = num_expected_coarse_nodes - coarse_node_features.shape[0]
                    padding = torch.zeros(pad_size, coarse_node_features.shape[1], device=coarse_node_features.device)
                    coarse_node_features = torch.cat([coarse_node_features, padding], dim=0)
                
                # 构建粗粒度图
                from torch_geometric.data import Batch as TorchBatch
                coarse_batch = TorchBatch(
                    x=coarse_node_features,
                    edge_index=batch.coarse_edge_index,
                    edge_attr=None,
                    batch=batch.coarse_batch,
                    pos=batch.coarse_pos,
                    max_edge_global=batch.coarse_max_edge_global,
                    max_node_global=batch.coarse_max_node_global
                )
                
                # 粗粒度编码
                coarse_embeddings, _ = self.coarse_encoder(coarse_batch, return_embeds=True)
                
                # 边嵌入池化回节点
                coarse_source_nodes = coarse_batch.edge_index[0]
                num_coarse_nodes = coarse_batch.num_nodes
                coarse_node_embeddings = scatter_sum(
                    coarse_embeddings, coarse_source_nodes, dim=0, dim_size=num_coarse_nodes
                )
                
                # 结合位点预测
                binding_site_logits = self.binding_site_head(coarse_node_embeddings)
                
                return binding_site_logits
        
        return SimplePhysESA(
            esa_config=model_config['esa_config'],
            feature_dims=model_config['feature_dims']
        )
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            logits = self.model(batch)
            labels = batch.binding_site_labels
            
            # 计算损失
            loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = batch.to(self.device)
                
                logits = self.model(batch)
                labels = batch.binding_site_labels
                
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(val_loader), correct / total
    
    def train(self, train_loader, val_loader, num_epochs):
        """完整训练流程"""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # 保存模型
                checkpoint_dir = Path("./checkpoints_simple")
                checkpoint_dir.mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                }, checkpoint_dir / f"best_model_epoch_{epoch+1}.pth")
                
                print(f"✅ New best model saved! Val Loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/15")
                
                if self.patience_counter >= 15:
                    print("Early stopping triggered!")
                    break

def main():
    parser = argparse.ArgumentParser(description="Simple PhysESA Training")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建数据加载器
    train_dataset = SimpleGraphDataset(os.path.join(args.data_dir, "train_graphs.pkl"))
    val_dataset = SimpleGraphDataset(os.path.join(args.data_dir, "val_graphs.pkl"))
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # 创建训练器
    trainer = SimpleTrainer(config)
    
    # 开始训练
    trainer.train(train_loader, val_loader, args.epochs)
    
    print("Training completed!")

if __name__ == "__main__":
    main()