"""
CASF-2016评估脚本
测试预训练模型在CASF-2016数据集上的结合位点预测性能
"""

import os
import sys
import torch
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
import logging
import argparse
import json

# 导入项目模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'model'))  # 添加model目录以支持utils导入
from pretrain.models.physesa_binding_site import PhysESABindingSite
from pretrain.data_adapters.pretrain_data_adapter import PretrainDatasetManager
from pretrain.data_adapters.binding_site_extractor import add_binding_site_labels_to_graph

class CASFTestDataset(Dataset):
    """CASF-2016测试数据集"""
    
    def __init__(self, dataset_root: str, complex_ids: List[str]):
        self.dataset_root = dataset_root
        self.complex_ids = complex_ids
        self.dataset_manager = PretrainDatasetManager(dataset_root)
        
        # 预处理所有测试样本
        self.graphs = self._preprocess_test_data()
        
        print(f"CASF-2016 test dataset loaded: {len(self.graphs)} complexes")
    
    def _preprocess_test_data(self) -> List:
        """预处理测试数据"""
        graphs = []
        failed_complexes = []
        
        print("Preprocessing CASF-2016 test data...")
        
        for complex_id in self.complex_ids:
            try:
                # 构建分子图
                graph_data = self.dataset_manager.adapter.build_molecular_graph(
                    complex_id, self.dataset_root, "casf"
                )
                
                if graph_data is None:
                    failed_complexes.append(complex_id)
                    continue
                
                # 添加结合位点标签
                if hasattr(graph_data, 'pocket_pdb_path'):
                    graph_data = add_binding_site_labels_to_graph(
                        graph_data, graph_data.pocket_pdb_path
                    )
                
                graph_data.complex_id = complex_id
                graphs.append(graph_data)
                
            except Exception as e:
                print(f"Failed to process {complex_id}: {str(e)}")
                failed_complexes.append(complex_id)
        
        if failed_complexes:
            print(f"Failed to process {len(failed_complexes)} complexes: {failed_complexes}")
        
        return graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

class CASFEvaluator:
    """CASF-2016评估器"""
    
    def __init__(self, 
                 dataset_root: str,
                 checkpoint_path: str,
                 output_dir: str,
                 batch_size: int = 32):
        
        self.dataset_root = dataset_root
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 加载模型
        self.model = self.load_model()
        
        # 准备测试数据
        self.test_dataset = self.prepare_test_data()
        
        # 结果存储
        self.results = {
            'complex_predictions': {},
            'overall_metrics': {},
            'per_complex_metrics': {},
            'confusion_matrix': None
        }
    
    def setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / "casf_evaluation.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self) -> PhysESABindingSite:
        """加载预训练模型"""
        self.logger.info(f"Loading model from {self.checkpoint_path}")
        
        try:
            model = PhysESABindingSite.load_from_checkpoint(self.checkpoint_path)
            model.eval()
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            self.logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def prepare_test_data(self) -> CASFTestDataset:
        """准备测试数据"""
        # 获取CASF-2016所有复合物ID
        dataset_manager = PretrainDatasetManager(self.dataset_root)
        test_ids = dataset_manager.get_casf_test_ids()
        
        self.logger.info(f"Found {len(test_ids)} test complexes in CASF-2016")
        
        return CASFTestDataset(self.dataset_root, test_ids)
    
    def predict_batch(self, batch) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量预测"""
        with torch.no_grad():
            if torch.cuda.is_available():
                batch = batch.cuda()
            
            # 获取预测结果和概率
            predictions = self.model.predict_binding_sites(batch, return_probabilities=False)
            probabilities = self.model.predict_binding_sites(batch, return_probabilities=True)
            
            labels = batch.binding_site_labels
            
            return (
                predictions.cpu().numpy(),
                probabilities.cpu().numpy()[:, 1],  # 结合位点概率
                labels.cpu().numpy()
            )
    
    def evaluate_complex(self, complex_id: str, predictions: np.ndarray, 
                        probabilities: np.ndarray, labels: np.ndarray) -> Dict:
        """评估单个复合物的性能"""
        metrics = {
            'complex_id': complex_id,
            'num_residues': len(labels),
            'num_binding_sites': np.sum(labels),
            'num_predicted_sites': np.sum(predictions),
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1_score': f1_score(labels, predictions, zero_division=0),
            'mcc': matthews_corrcoef(labels, predictions)
        }
        
        # AUC（如果有正负样本）
        if len(np.unique(labels)) > 1:
            metrics['auc'] = roc_auc_score(labels, probabilities)
        else:
            metrics['auc'] = 0.0
        
        return metrics
    
    def run_evaluation(self):
        """运行完整评估"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING CASF-2016 EVALUATION")
        self.logger.info("=" * 60)
        self.logger.info(f"Test complexes: {len(self.test_dataset)}")
        self.logger.info(f"Model checkpoint: {self.checkpoint_path}")
        
        # 创建数据加载器
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_complex_ids = []
        
        # 批量预测
        for batch_idx, batch in enumerate(test_loader):
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
            
            predictions, probabilities, labels = self.predict_batch(batch)
            
            # 处理每个复合物
            batch_size = len(batch.complex_id)
            batch_ptr = batch.batch.max().item() + 1
            
            for i in range(batch_size):
                complex_id = batch.complex_id[i]
                
                # 获取当前复合物的节点掩码
                node_mask = (batch.batch == i)
                
                complex_preds = predictions[node_mask]
                complex_probs = probabilities[node_mask]
                complex_labels = labels[node_mask]
                
                # 计算单个复合物的指标
                complex_metrics = self.evaluate_complex(
                    complex_id, complex_preds, complex_probs, complex_labels
                )
                
                self.results['per_complex_metrics'][complex_id] = complex_metrics
                self.results['complex_predictions'][complex_id] = {
                    'predictions': complex_preds.tolist(),
                    'probabilities': complex_probs.tolist(),
                    'labels': complex_labels.tolist()
                }
                
                # 累积全局统计
                all_predictions.extend(complex_preds)
                all_probabilities.extend(complex_probs)
                all_labels.extend(complex_labels)
                all_complex_ids.extend([complex_id] * len(complex_labels))
        
        # 计算整体指标
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        self.results['overall_metrics'] = {
            'total_residues': len(all_labels),
            'total_binding_sites': np.sum(all_labels),
            'total_predicted_sites': np.sum(all_predictions),
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1_score': f1_score(all_labels, all_predictions, zero_division=0),
            'mcc': matthews_corrcoef(all_labels, all_predictions),
            'auc': roc_auc_score(all_labels, all_probabilities)
        }
        
        # 混淆矩阵
        self.results['confusion_matrix'] = confusion_matrix(
            all_labels, all_predictions
        ).tolist()
        
        # 保存结果
        self.save_results()
        
        # 打印评估报告
        self.print_evaluation_report()
    
    def save_results(self):
        """保存评估结果"""
        # 保存完整结果
        results_path = self.output_dir / "casf_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 保存指标汇总
        metrics_df = pd.DataFrame([
            self.results['per_complex_metrics'][cid] 
            for cid in self.results['per_complex_metrics']
        ])
        metrics_csv_path = self.output_dir / "per_complex_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        
        # 保存整体指标
        overall_df = pd.DataFrame([self.results['overall_metrics']])
        overall_csv_path = self.output_dir / "overall_metrics.csv"
        overall_df.to_csv(overall_csv_path, index=False)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def print_evaluation_report(self):
        """打印评估报告"""
        overall = self.results['overall_metrics']
        per_complex = list(self.results['per_complex_metrics'].values())
        
        self.logger.info("=" * 60)
        self.logger.info("CASF-2016 EVALUATION RESULTS")
        self.logger.info("=" * 60)
        
        # 整体性能
        self.logger.info("OVERALL PERFORMANCE:")
        self.logger.info(f"  Total residues: {overall['total_residues']}")
        self.logger.info(f"  Binding sites: {overall['total_binding_sites']}")
        self.logger.info(f"  Predicted sites: {overall['total_predicted_sites']}")
        self.logger.info(f"  Accuracy: {overall['accuracy']:.4f}")
        self.logger.info(f"  Precision: {overall['precision']:.4f}")
        self.logger.info(f"  Recall: {overall['recall']:.4f}")
        self.logger.info(f"  F1-Score: {overall['f1_score']:.4f}")
        self.logger.info(f"  MCC: {overall['mcc']:.4f}")
        self.logger.info(f"  AUC: {overall['auc']:.4f}")
        
        # 混淆矩阵
        cm = np.array(self.results['confusion_matrix'])
        self.logger.info("\nCONFUSION MATRIX:")
        self.logger.info(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        self.logger.info(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # 每个复合物的统计
        f1_scores = [m['f1_score'] for m in per_complex]
        accuracies = [m['accuracy'] for m in per_complex]
        
        self.logger.info("\nPER-COMPLEX STATISTICS:")
        self.logger.info(f"  Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        self.logger.info(f"  Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        self.logger.info(f"  Complexes evaluated: {len(per_complex)}")
        
        # 最佳和最差性能的复合物
        best_complex = max(per_complex, key=lambda x: x['f1_score'])
        worst_complex = min(per_complex, key=lambda x: x['f1_score'])
        
        self.logger.info(f"\nBEST PERFORMANCE: {best_complex['complex_id']} (F1: {best_complex['f1_score']:.4f})")
        self.logger.info(f"WORST PERFORMANCE: {worst_complex['complex_id']} (F1: {worst_complex['f1_score']:.4f})")
        
        self.logger.info("=" * 60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Evaluate PhysESA on CASF-2016")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory containing CASF-2016 dataset")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for evaluation results")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # 创建评估器并运行评估
    evaluator = CASFEvaluator(
        dataset_root=args.dataset_root,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()