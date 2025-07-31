"""
扩展PhysESA模型，支持结合位点预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_scatter import scatter_sum
from torchmetrics import Accuracy, Precision, Recall, F1Score
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.esa.masked_layers import Estimator

class PhysESABindingSite(pl.LightningModule):
    """
    PhysESA结合位点预测模型
    基于原有的多尺度PhysESA架构，添加结合位点预测功能
    """
    
    def __init__(self, 
                 esa_config: dict, 
                 feature_dims: dict, 
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 构建原子级编码器（共享表示学习）
        common_config = {k: v for k, v in esa_config.items() if k not in ['atomic_encoder_config', 'coarse_encoder_config']}
        atomic_specific_config = esa_config['atomic_encoder_config']
        atomic_config = {**common_config, **atomic_specific_config}
        atomic_config['num_features'] = feature_dims['node_dim']
        atomic_config['edge_dim'] = feature_dims['edge_dim']
        
        self.atomic_encoder = Estimator(**atomic_config)
        
        # 粗粒度编码器配置（用于结合位点预测）
        coarse_specific_config = esa_config['coarse_encoder_config']
        coarse_config = {**common_config, **coarse_specific_config}
        coarse_config['num_features'] = atomic_config['graph_dim']
        coarse_config['edge_dim'] = 0
        coarse_config['linear_output_size'] = atomic_config['graph_dim']  # 输出特征维度
        
        self.coarse_encoder = Estimator(**coarse_config)
        
        # 结合位点预测头
        self.binding_site_head = nn.Sequential(
            nn.Linear(atomic_config['graph_dim'], atomic_config['graph_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(atomic_config['graph_dim'] // 2, 2)  # 二分类
        )
        
        # 损失函数
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 评估指标
        self.train_accuracy = Accuracy(task='binary')
        self.train_precision = Precision(task='binary')
        self.train_recall = Recall(task='binary')
        self.train_f1 = F1Score(task='binary')
        
        self.val_accuracy = Accuracy(task='binary')
        self.val_precision = Precision(task='binary')
        self.val_recall = Recall(task='binary')
        self.val_f1 = F1Score(task='binary')
        
        self.test_accuracy = Accuracy(task='binary')
        self.test_precision = Precision(task='binary')
        self.test_recall = Recall(task='binary')
        self.test_f1 = F1Score(task='binary')
    
    def forward(self, batch: Batch):
        """前向传播，输出结合位点预测"""
        
        # 1. 原子级编码
        atomic_edge_embeds, _ = self.atomic_encoder(batch, return_embeds=True)
        
        # 2. 从原子到粗粒度的池化
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
            padding = torch.zeros(pad_size, coarse_node_features.shape[1], device=self.device)
            coarse_node_features = torch.cat([coarse_node_features, padding], dim=0)
        
        # 3. 构建粗粒度图
        coarse_batch = Batch(
            x=coarse_node_features,
            edge_index=batch.coarse_edge_index,
            edge_attr=None,
            batch=batch.coarse_batch,
            pos=batch.coarse_pos,
            max_edge_global=batch.coarse_max_edge_global,
            max_node_global=batch.coarse_max_node_global
        )
        
        # 4. 粗粒度编码（输出节点级特征而非图级特征）
        coarse_embeddings, _ = self.coarse_encoder(coarse_batch, return_embeds=True)
        
        # 将边嵌入池化回节点
        coarse_source_nodes = coarse_batch.edge_index[0]
        num_coarse_nodes = coarse_batch.num_nodes
        coarse_node_embeddings = scatter_sum(
            coarse_embeddings, coarse_source_nodes, dim=0, dim_size=num_coarse_nodes
        )
        
        # 5. 结合位点预测
        binding_site_logits = self.binding_site_head(coarse_node_embeddings)
        
        return binding_site_logits
    
    def _common_step(self, batch, batch_idx, stage):
        """通用的训练/验证/测试步骤"""
        # 前向传播
        logits = self(batch)
        labels = batch.binding_site_labels
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 获取预测结果
        preds = torch.argmax(logits, dim=1)
        
        # 更新指标
        if stage == 'train':
            self.train_accuracy(preds, labels)
            self.train_precision(preds, labels)
            self.train_recall(preds, labels)
            self.train_f1(preds, labels)
        elif stage == 'val':
            self.val_accuracy(preds, labels)
            self.val_precision(preds, labels)
            self.val_recall(preds, labels)
            self.val_f1(preds, labels)
        elif stage == 'test':
            self.test_accuracy(preds, labels)
            self.test_precision(preds, labels)
            self.test_recall(preds, labels)
            self.test_f1(preds, labels)
        
        return loss, preds, labels
    
    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx, 'train')
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx, 'val')
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx, 'test')
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log_dict({
            'train/accuracy': self.train_accuracy.compute(),
            'train/precision': self.train_precision.compute(),
            'train/recall': self.train_recall.compute(),
            'train/f1': self.train_f1.compute()
        })
        
        # 重置指标
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()
    
    def on_validation_epoch_end(self):
        self.log_dict({
            'val/accuracy': self.val_accuracy.compute(),
            'val/precision': self.val_precision.compute(),
            'val/recall': self.val_recall.compute(),
            'val/f1': self.val_f1.compute()
        })
        
        # 重置指标
        self.val_accuracy.reset()
        self.val_precision.reset()  
        self.val_recall.reset()
        self.val_f1.reset()
    
    def on_test_epoch_end(self):
        accuracy = self.test_accuracy.compute()
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        f1 = self.test_f1.compute()
        
        self.log_dict({
            'test/accuracy': accuracy,
            'test/precision': precision,
            'test/recall': recall,
            'test/f1': f1
        })
        
        print("\\n" + "="*50)
        print("     Binding Site Prediction Results     ")
        print("="*50)
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print("="*50)
        
        # 重置指标
        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def predict_binding_sites(self, batch: Batch, return_probabilities: bool = False):
        """预测结合位点（推理模式）"""
        self.eval()
        with torch.no_grad():
            logits = self(batch)
            
            if return_probabilities:
                probs = F.softmax(logits, dim=1)
                return probs
            else:
                preds = torch.argmax(logits, dim=1)
                return preds