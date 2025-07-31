"""
结合位点标签提取器
从pocket.pdb文件中提取残基级别的结合位点标签
"""

import pandas as pd
import numpy as np
from Bio import PDB
from typing import Dict, List, Tuple
import torch

class BindingSiteExtractor:
    """结合位点标签提取器"""
    
    def __init__(self):
        self.pdb_parser = PDB.PDBParser(QUIET=True)
    
    def extract_residues_from_pdb(self, pdb_path: str) -> List[str]:
        """从PDB文件中提取残基ID列表"""
        structure = self.pdb_parser.get_structure("complex", pdb_path)
        residue_ids = []
        
        for residue in structure.get_residues():
            # 跳过水分子
            if residue.get_resname() == 'HOH':
                continue
            
            chain_id = residue.get_parent().id
            res_name = residue.get_resname()
            res_num = residue.get_id()[1]
            
            # 构建残基ID，与MultiScaleGraphBuilder保持一致
            res_id = f"{chain_id}_{res_name}_{res_num}"
            residue_ids.append(res_id)
        
        return residue_ids
    
    def extract_binding_site_labels(self, protein_pdb_path: str, pocket_pdb_path: str) -> Dict[str, int]:
        """提取结合位点标签
        
        Args:
            protein_pdb_path: 完整蛋白质PDB文件路径
            pocket_pdb_path: 结合口袋PDB文件路径
        
        Returns:
            Dict[str, int]: {residue_id: label} 其中label为0(非结合位点)或1(结合位点)
        """
        
        # 提取所有蛋白质残基
        all_residues = set(self.extract_residues_from_pdb(protein_pdb_path))
        
        # 提取结合口袋残基
        pocket_residues = set(self.extract_residues_from_pdb(pocket_pdb_path))
        
        # 生成标签字典
        labels = {}
        for res_id in all_residues:
            labels[res_id] = 1 if res_id in pocket_residues else 0
        
        return labels
    
    def create_residue_level_labels(self, graph_data, binding_site_labels: Dict[str, int]) -> torch.Tensor:
        """为粗粒度图创建残基级别的结合位点标签
        
        Args:
            graph_data: 图数据对象，包含coarse_node_id_map
            binding_site_labels: 结合位点标签字典
        
        Returns:
            torch.Tensor: 粗粒度节点的结合位点标签
        """
        
        # 获取粗粒度节点映射
        coarse_node_map = graph_data.coarse_node_id_map
        num_coarse_nodes = len(coarse_node_map)
        
        # 创建标签张量
        coarse_labels = torch.zeros(num_coarse_nodes, dtype=torch.long)
        
        for coarse_idx, node_id in coarse_node_map.items():
            if node_id.startswith("LIG_MOTIF"):
                # 配体官能团，标签设为0（非蛋白质结合位点）
                coarse_labels[coarse_idx] = 0
            else:
                # 蛋白质残基，查找对应标签
                if node_id in binding_site_labels:
                    coarse_labels[coarse_idx] = binding_site_labels[node_id]
                else:
                    coarse_labels[coarse_idx] = 0
        
        return coarse_labels
    
    def get_binding_site_statistics(self, labels: Dict[str, int]) -> Dict[str, int]:
        """获取结合位点统计信息"""
        stats = {
            'total_residues': len(labels),
            'binding_site_residues': sum(labels.values()),
            'non_binding_residues': len(labels) - sum(labels.values())
        }
        
        stats['binding_site_ratio'] = stats['binding_site_residues'] / stats['total_residues']
        
        return stats

def add_binding_site_labels_to_graph(graph_data, pocket_pdb_path: str):
    """为图数据添加结合位点标签
    
    Args:
        graph_data: 图数据对象
        pocket_pdb_path: 结合口袋PDB文件路径
    
    Returns:
        修改后的图数据对象，包含binding_site_labels字段
    """
    
    # 需要从graph_data中重建protein_pdb_path
    # 这里假设可以从complex_id和数据集路径重建
    # 实际使用时需要传入protein_pdb_path
    
    extractor = BindingSiteExtractor()
    
    # 临时方案：从现有原子信息中提取蛋白质残基
    # 实际使用时应该直接传入protein_pdb_path
    protein_residues = []
    ligand_atoms = []
    
    # 遍历原子，分离蛋白质和配体
    for i in range(graph_data.num_nodes):
        coarse_idx = graph_data.atom_to_coarse_idx[i].item()
        node_id = graph_data.coarse_node_id_map[coarse_idx]
        
        if not node_id.startswith("LIG_MOTIF"):
            protein_residues.append(node_id)
    
    # 从pocket.pdb提取结合位点残基
    pocket_residues = set(extractor.extract_residues_from_pdb(pocket_pdb_path))
    
    # 生成标签
    binding_site_labels = {}
    for res_id in set(protein_residues):
        binding_site_labels[res_id] = 1 if res_id in pocket_residues else 0
    
    # 创建粗粒度标签
    coarse_labels = extractor.create_residue_level_labels(graph_data, binding_site_labels)
    
    # 添加到图数据
    graph_data.binding_site_labels = coarse_labels
    graph_data.binding_site_stats = extractor.get_binding_site_statistics(binding_site_labels)
    
    return graph_data