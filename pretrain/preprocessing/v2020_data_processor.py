"""
v2020-other-PL数据集预处理脚本
生成13,891个带结合位点标签的图对象
"""

import os
import sys
import torch
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional
from torch_geometric.data import Data
from tqdm import tqdm
import traceback

# 导入项目模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'model'))  # 添加model目录以支持utils导入
from pretrain.data_adapters.pretrain_data_adapter import PretrainDatasetManager
from pretrain.data_adapters.binding_site_extractor import add_binding_site_labels_to_graph

class V2020DataProcessor:
    """v2020-other-PL数据集预处理器"""
    
    def __init__(self, 
                 dataset_root: str,
                 output_dir: str,
                 val_ratio: float = 0.1,
                 batch_size: int = 100):
        
        self.dataset_root = dataset_root
        self.output_dir = Path(output_dir)
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据管理器
        self.dataset_manager = PretrainDatasetManager(dataset_root)
        
        # 设置日志
        self.setup_logging()
        
        # 统计信息
        self.stats = {
            'total_complexes': 0,
            'successful_graphs': 0,
            'failed_complexes': [],
            'binding_site_stats': {
                'total_residues': 0,
                'binding_residues': 0,
                'non_binding_residues': 0
            }
        }
    
    def setup_logging(self):
        """设置日志系统"""
        log_file = self.output_dir / "preprocessing.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_single_complex(self, complex_id: str) -> Optional[Data]:
        """处理单个复合物，生成带结合位点标签的图"""
        try:
            # 构建分子图
            graph_data = self.dataset_manager.adapter.build_molecular_graph(
                complex_id, self.dataset_root, "v2020"
            )
            
            if graph_data is None:
                self.logger.warning(f"Failed to build graph for {complex_id}")
                return None
            
            # 添加结合位点标签
            if hasattr(graph_data, 'pocket_pdb_path'):
                graph_data = add_binding_site_labels_to_graph(
                    graph_data, graph_data.pocket_pdb_path
                )
                
                # 更新统计信息
                if hasattr(graph_data, 'binding_site_stats'):
                    stats = graph_data.binding_site_stats
                    self.stats['binding_site_stats']['total_residues'] += stats['total_residues']
                    self.stats['binding_site_stats']['binding_residues'] += stats['binding_site_residues']
                    self.stats['binding_site_stats']['non_binding_residues'] += stats['non_binding_residues']
            
            # 添加复合物ID
            graph_data.complex_id = complex_id
            
            return graph_data
            
        except Exception as e:
            self.logger.error(f"Error processing {complex_id}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def process_batch(self, complex_ids: List[str], batch_name: str) -> List[Data]:
        """批量处理复合物"""
        processed_graphs = []
        
        self.logger.info(f"Processing {batch_name}: {len(complex_ids)} complexes")
        
        for complex_id in tqdm(complex_ids, desc=f"Processing {batch_name}"):
            graph_data = self.process_single_complex(complex_id)
            
            if graph_data is not None:
                processed_graphs.append(graph_data)
                self.stats['successful_graphs'] += 1
            else:
                self.stats['failed_complexes'].append(complex_id)
        
        self.logger.info(f"{batch_name} completed: {len(processed_graphs)}/{len(complex_ids)} successful")
        return processed_graphs
    
    def save_processed_data(self, data: List[Data], filename: str):
        """保存处理后的数据"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.logger.info(f"Saved {len(data)} graphs to {output_path}")
        
        # 保存索引文件
        index_data = {
            'complex_ids': [graph.complex_id for graph in data],
            'num_graphs': len(data),
            'file_path': str(output_path)
        }
        
        index_path = output_path.with_suffix('.index')
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
    
    def save_statistics(self):
        """保存统计信息"""
        stats_path = self.output_dir / "preprocessing_stats.pkl"
        
        # 计算最终统计
        self.stats['success_rate'] = self.stats['successful_graphs'] / self.stats['total_complexes']
        
        if self.stats['binding_site_stats']['total_residues'] > 0:
            self.stats['binding_site_stats']['binding_ratio'] = (
                self.stats['binding_site_stats']['binding_residues'] / 
                self.stats['binding_site_stats']['total_residues']
            )
        
        with open(stats_path, 'wb') as f:
            pickle.dump(self.stats, f)
        
        # 打印统计摘要
        self.logger.info("=" * 60)
        self.logger.info("PREPROCESSING STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total complexes: {self.stats['total_complexes']}")
        self.logger.info(f"Successful graphs: {self.stats['successful_graphs']}")
        self.logger.info(f"Success rate: {self.stats['success_rate']:.2%}")
        self.logger.info(f"Failed complexes: {len(self.stats['failed_complexes'])}")
        
        bs_stats = self.stats['binding_site_stats']
        self.logger.info(f"Total residues: {bs_stats['total_residues']}")
        self.logger.info(f"Binding site residues: {bs_stats['binding_residues']}")
        self.logger.info(f"Binding site ratio: {bs_stats.get('binding_ratio', 0):.2%}")
        self.logger.info("=" * 60)
    
    def run_preprocessing(self):
        """运行完整的预处理流程"""
        self.logger.info("Starting v2020-other-PL dataset preprocessing")
        
        # 获取训练/验证分割
        train_ids, val_ids = self.dataset_manager.get_v2020_train_val_split(self.val_ratio)
        
        self.stats['total_complexes'] = len(train_ids) + len(val_ids)
        
        self.logger.info(f"Dataset split: {len(train_ids)} train, {len(val_ids)} validation")
        
        # 处理训练集
        if len(train_ids) > 0:
            train_graphs = []
            
            # 分批处理训练集
            for i in range(0, len(train_ids), self.batch_size):
                batch_ids = train_ids[i:i + self.batch_size]
                batch_name = f"train_batch_{i//self.batch_size + 1}"
                
                batch_graphs = self.process_batch(batch_ids, batch_name)
                train_graphs.extend(batch_graphs)
                
                # 定期保存中间结果
                if len(train_graphs) % (self.batch_size * 10) == 0:
                    temp_path = f"train_temp_{len(train_graphs)}.pkl"
                    self.save_processed_data(train_graphs, temp_path)
            
            # 保存完整训练集
            self.save_processed_data(train_graphs, "train_graphs.pkl")
        
        # 处理验证集
        if len(val_ids) > 0:
            val_graphs = self.process_batch(val_ids, "validation")
            self.save_processed_data(val_graphs, "val_graphs.pkl")
        
        # 保存统计信息
        self.save_statistics()
        
        self.logger.info("Preprocessing completed successfully!")
    
    def validate_processed_data(self, data_path: str):
        """验证处理后的数据"""
        self.logger.info(f"Validating processed data: {data_path}")
        
        with open(data_path, 'rb') as f:
            graphs = pickle.load(f)
        
        if not graphs:
            self.logger.error("No graphs found in processed data")
            return False
        
        # 验证第一个图的结构
        graph = graphs[0]
        required_attrs = ['x', 'edge_index', 'binding_site_labels', 'complex_id']
        
        for attr in required_attrs:
            if not hasattr(graph, attr):
                self.logger.error(f"Missing required attribute: {attr}")
                return False
        
        self.logger.info(f"Validation passed: {len(graphs)} graphs with all required attributes")
        return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process v2020-other-PL dataset")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory of datasets")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed data")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for processing")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate existing processed data")
    
    args = parser.parse_args()
    
    processor = V2020DataProcessor(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size
    )
    
    if args.validate_only:
        train_path = os.path.join(args.output_dir, "train_graphs.pkl")
        val_path = os.path.join(args.output_dir, "val_graphs.pkl")
        
        if os.path.exists(train_path):
            processor.validate_processed_data(train_path)
        if os.path.exists(val_path):
            processor.validate_processed_data(val_path)
    else:
        processor.run_preprocessing()

if __name__ == "__main__":
    main()