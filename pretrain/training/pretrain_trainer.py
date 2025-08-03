"""
预训练训练脚本
在v2020-other-PL数据集上训练结合位点识别任务
"""

import os
import sys
import torch
import pickle
import yaml
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
# from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import logging

# 导入项目模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'model'))  # 添加model目录以支持utils导入
from pretrain.models.physesa_binding_site import PhysESABindingSite

class PretrainGraphDataset(Dataset):
    """预训练图数据集"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        
        with open(data_path, 'rb') as f:
            self.graphs = pickle.load(f)
        
        print(f"Loaded {len(self.graphs)} graphs from {data_path}")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

class PretrainDataModule(pl.LightningDataModule):
    """预训练数据模块"""
    
    def __init__(self, 
                 train_data_path: str,
                 val_data_path: str,
                 batch_size: int = 32,
                 num_workers: int = 8):
        super().__init__()
        
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = PretrainGraphDataset(self.train_data_path)
            self.val_dataset = PretrainGraphDataset(self.val_data_path)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

class PretrainTrainer:
    """预训练训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        
        # 计算类别权重（处理数据不平衡）
        self.class_weights = self.calculate_class_weights()
    
    def setup_logging(self):
        """设置日志"""
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_class_weights(self) -> torch.Tensor:
        """计算类别权重以处理数据不平衡"""
        # 从预处理统计中获取类别分布
        stats_path = Path(self.config['data']['processed_dir']) / "preprocessing_stats.pkl"
        
        if stats_path.exists():
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
            
            bs_stats = stats['binding_site_stats']
            total_residues = bs_stats['total_residues']
            binding_residues = bs_stats['binding_residues']
            non_binding_residues = bs_stats['non_binding_residues']
            
            # 计算权重（使用inverse frequency）
            weight_non_binding = total_residues / (2 * non_binding_residues)
            weight_binding = total_residues / (2 * binding_residues)
            
            weights = torch.tensor([weight_non_binding, weight_binding], dtype=torch.float32)
            
            self.logger.info(f"Class weights: non-binding={weight_non_binding:.3f}, binding={weight_binding:.3f}")
            return weights
        else:
            self.logger.warning("No preprocessing stats found, using balanced weights")
            return torch.tensor([1.0, 1.0], dtype=torch.float32)
    
    def create_model(self) -> PhysESABindingSite:
        """创建模型"""
        model_config = self.config['model']
        
        model = PhysESABindingSite(
            esa_config=model_config['esa_config'],
            feature_dims=model_config['feature_dims'],
            learning_rate=model_config['learning_rate'],
            weight_decay=model_config['weight_decay'],
            class_weights=self.class_weights
        )
        
        return model
    
    def create_data_module(self) -> PretrainDataModule:
        """创建数据模块"""
        data_config = self.config['data']
        
        train_data_path = Path(data_config['processed_dir']) / "train_graphs.pkl"
        val_data_path = Path(data_config['processed_dir']) / "val_graphs.pkl"
        
        return PretrainDataModule(
            train_data_path=str(train_data_path),
            val_data_path=str(val_data_path),
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers']
        )
    
    def create_callbacks(self) -> List[pl.Callback]:
        """创建训练回调"""
        callbacks = []
        
        # 模型检查点
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config['logging']['checkpoint_dir'],
            filename='{epoch:02d}-{val_loss:.4f}-{val/f1:.4f}',
            monitor='val/f1',
            mode='max',
            save_top_k=3,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # 早停
        early_stop_callback = EarlyStopping(
            monitor='val/f1',
            mode='max',
            patience=self.config['training']['early_stopping_patience'],
            verbose=True,
            min_delta=0.001
        )
        callbacks.append(early_stop_callback)
        
        # 学习率监控
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        return callbacks
    
    def create_loggers(self) -> List[pl.loggers.Logger]:
        """创建训练日志器"""
        loggers = []
        
        # TensorBoard日志
        tb_logger = TensorBoardLogger(
            save_dir=self.config['logging']['log_dir'],
            name='physesa_pretrain',
            version=self.config['experiment']['name']
        )
        loggers.append(tb_logger)
        
        # WandB日志（如果配置了）
        if self.config.get('wandb', {}).get('enabled', False):
            wandb_logger = WandbLogger(
                project=self.config['wandb']['project'],
                name=self.config['experiment']['name'],
                tags=self.config['wandb'].get('tags', [])
            )
            loggers.append(wandb_logger)
        
        return loggers
    
    def train(self):
        """开始训练"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING PHYSESA PRETRAINING")
        self.logger.info("=" * 60)
        self.logger.info(f"Experiment: {self.config['experiment']['name']}")
        self.logger.info(f"GPUs: {self.config['training']['gpus']}")
        self.logger.info(f"Epochs: {self.config['training']['max_epochs']}")
        self.logger.info(f"Batch size: {self.config['data']['batch_size']}")
        
        # 创建组件
        model = self.create_model()
        data_module = self.create_data_module()
        callbacks = self.create_callbacks()
        loggers = self.create_loggers()
        
        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=self.config['training']['max_epochs'],
            gpus=self.config['training']['gpus'],
            strategy=self.config['training']['strategy'],
            precision=self.config['training']['precision'],
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=self.config['logging']['log_every_n_steps'],
            val_check_interval=self.config['training']['val_check_interval'],
            gradient_clip_val=self.config['training']['gradient_clip_val'],
            accumulate_grad_batches=self.config['training']['accumulate_grad_batches'],
            deterministic=False,  # 允许cudnn自动调优以提高性能
            benchmark=True  # 启用cudnn benchmark
        )
        
        # 开始训练
        trainer.fit(model, data_module)
        
        # 保存最终模型
        final_model_path = Path(self.config['logging']['checkpoint_dir']) / "final_model.ckpt"
        trainer.save_checkpoint(final_model_path)
        
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
        self.logger.info(f"Final model saved at: {final_model_path}")

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建必要的目录
    for path_key in ['log_dir', 'checkpoint_dir']:
        path = Path(config['logging'][path_key])
        path.mkdir(parents=True, exist_ok=True)
    
    return config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PhysESA Pretraining")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建训练器并开始训练
    trainer = PretrainTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()