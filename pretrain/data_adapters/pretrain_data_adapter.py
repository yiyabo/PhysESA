"""
预训练数据适配器
统一处理v2020-other-PL和CASF-2016两种数据集格式
"""

import os
import pandas as pd
from typing import Optional, Tuple, Dict
from pathlib import Path
from Bio import PDB
from rdkit import Chem
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from molecular_graph import MultiScaleGraphBuilder

class PretrainDataAdapter:
    """预训练数据适配器，统一处理两种数据集格式"""
    
    def __init__(self, 
                 cutoff_radius: float = 5.0,
                 num_gaussians: int = 16,
                 use_knn: bool = True,
                 k: int = 16,
                 interface_cutoff: float = 8.0):
        
        self.graph_builder = MultiScaleGraphBuilder(
            cutoff_radius=cutoff_radius,
            num_gaussians=num_gaussians,
            use_knn=use_knn,
            k=k,
            interface_cutoff=interface_cutoff
        )
        self.pdb_parser = PDB.PDBParser(QUIET=True)
    
    def get_complex_files_v2020(self, complex_id: str, dataset_root: str) -> Dict[str, str]:
        """获取v2020-other-PL数据集中复合物的文件路径"""
        complex_dir = Path(dataset_root) / "v2020-other-PL" / complex_id
        
        return {
            'protein_pdb': str(complex_dir / f"{complex_id}_protein.pdb"),
            'ligand_sdf': str(complex_dir / f"{complex_id}_ligand.sdf"),
            'ligand_mol2': str(complex_dir / f"{complex_id}_ligand.mol2"),
            'pocket_pdb': str(complex_dir / f"{complex_id}_pocket.pdb")
        }
    
    def get_complex_files_casf(self, complex_id: str, dataset_root: str) -> Dict[str, str]:
        """获取CASF-2016数据集中复合物的文件路径"""
        complex_dir = Path(dataset_root) / "CASF-2016" / "coreset" / complex_id
        
        return {
            'protein_pdb': str(complex_dir / f"{complex_id}_protein.pdb"),
            'ligand_sdf': str(complex_dir / f"{complex_id}_ligand.sdf"),
            'ligand_mol2': str(complex_dir / f"{complex_id}_ligand.mol2"), 
            'pocket_pdb': str(complex_dir / f"{complex_id}_pocket.pdb")
        }
    
    def get_all_complex_ids(self, dataset_root: str, dataset_type: str) -> list:
        """获取数据集中所有复合物ID"""
        if dataset_type == "v2020":
            data_dir = Path(dataset_root) / "v2020-other-PL"
        elif dataset_type == "casf":
            data_dir = Path(dataset_root) / "CASF-2016" / "coreset"
        
        return [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    def parse_protein_structure(self, protein_pdb_path: str) -> Optional[pd.DataFrame]:
        """解析蛋白质结构文件"""
        return self.graph_builder.parse_protein_with_biopython(protein_pdb_path)
    
    def parse_ligand_structure(self, ligand_sdf_path: str) -> Tuple[Optional[pd.DataFrame], Optional[Chem.Mol]]:
        """解析配体结构文件"""
        return self.graph_builder.parse_sdf_ligand(ligand_sdf_path)
    
    def build_molecular_graph(self, complex_id: str, dataset_root: str, dataset_type: str):
        """构建分子图，统一接口"""
        
        # 获取文件路径
        if dataset_type == "v2020":
            files = self.get_complex_files_v2020(complex_id, dataset_root)
        elif dataset_type == "casf":
            files = self.get_complex_files_casf(complex_id, dataset_root)
        
        # 检查文件存在性
        for file_type, file_path in files.items():
            if not os.path.exists(file_path):
                print(f"Missing {file_type}: {file_path}")
                return None
        
        # 构建图
        graph_data = self.graph_builder.build_graph(
            complex_id=complex_id,
            pdb_file=files['protein_pdb'],
            sdf_file=files['ligand_sdf']
        )
        
        # 添加pocket文件路径信息
        if graph_data is not None:
            graph_data.pocket_pdb_path = files['pocket_pdb']
        
        return graph_data

class PretrainDatasetManager:
    """预训练数据集管理器"""
    
    def __init__(self, dataset_root: str):
        self.dataset_root = dataset_root
        self.adapter = PretrainDataAdapter()
    
    def get_v2020_train_val_split(self, val_ratio: float = 0.1) -> Tuple[list, list]:
        """获取v2020数据集的训练/验证分割"""
        all_ids = self.adapter.get_all_complex_ids(self.dataset_root, "v2020")
        
        # 简单按比例分割
        n_val = int(len(all_ids) * val_ratio)
        val_ids = all_ids[-n_val:]
        train_ids = all_ids[:-n_val]
        
        return train_ids, val_ids
    
    def get_casf_test_ids(self) -> list:
        """获取CASF-2016测试集ID"""
        return self.adapter.get_all_complex_ids(self.dataset_root, "casf")
    
    def process_dataset_batch(self, complex_ids: list, dataset_type: str, batch_name: str):
        """批量处理数据集"""
        processed_data = []
        
        print(f"Processing {batch_name}: {len(complex_ids)} complexes")
        
        for i, complex_id in enumerate(complex_ids):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(complex_ids)}")
            
            graph_data = self.adapter.build_molecular_graph(
                complex_id, self.dataset_root, dataset_type
            )
            
            if graph_data is not None:
                processed_data.append(graph_data)
        
        print(f"{batch_name} completed: {len(processed_data)} valid graphs")
        return processed_data