#!/usr/bin/env python3
"""
计算ShapeNetSem模型的binvox物理属性
生成一个JSON文件供pipeline16使用
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.binvox_utils import get_binvox_properties

def compute_all_binvox_properties(shapenet_dir, output_file='binvox_properties.json'):
    """
    为所有ShapeNetSem模型计算binvox属性
    """
    # 目录路径
    binvox_dir = os.path.join(shapenet_dir, 'models-binvox')
    binvox_solid_dir = os.path.join(shapenet_dir, 'models-binvox-solid')
    
    if not os.path.exists(binvox_dir) or not os.path.exists(binvox_solid_dir):
        print("错误：找不到binvox目录")
        return False
    
    # 获取所有binvox文件
    binvox_files = [f for f in os.listdir(binvox_dir) if f.endswith('.binvox')]
    print(f"找到 {len(binvox_files)} 个binvox文件")
    
    # 加载metadata.csv获取类别信息
    metadata_path = os.path.join(shapenet_dir, 'metadata.csv')
    metadata_df = pd.read_csv(metadata_path)
    model_to_category = {}
    model_to_container = {}
    
    for _, row in metadata_df.iterrows():
        model_id = row['fullId'].replace('wss.', '')
        model_to_category[model_id] = row.get('category', 'unknown')
        # isContainerLike可能是字符串或布尔值
        container_val = row.get('isContainerLike', False)
        if isinstance(container_val, str):
            model_to_container[model_id] = container_val.lower() == 'true'
        else:
            model_to_container[model_id] = bool(container_val)
    
    # 计算所有模型的属性
    all_properties = {}
    
    for binvox_file in tqdm(binvox_files, desc="计算binvox属性"):
        model_id = binvox_file.replace('.binvox', '')
        
        # 获取binvox属性
        props = get_binvox_properties(model_id, binvox_dir, binvox_solid_dir)
        
        # 添加类别信息
        props['category'] = model_to_category.get(model_id, 'unknown')
        props['isContainerLike'] = model_to_container.get(model_id, False)
        
        # 保存
        all_properties[model_id] = props
    
    # 统计信息
    valid_solid = sum(1 for p in all_properties.values() if p['solidVolume'] is not None)
    valid_surface = sum(1 for p in all_properties.values() if p['surfaceVolume'] is not None)
    valid_support = sum(1 for p in all_properties.values() if p['supportSurfaceArea'] is not None)
    
    print(f"\n统计信息：")
    print(f"- 有效solidVolume: {valid_solid}/{len(all_properties)}")
    print(f"- 有效surfaceVolume: {valid_surface}/{len(all_properties)}")
    print(f"- 有效supportSurfaceArea: {valid_support}/{len(all_properties)}")
    
    # 保存到JSON文件
    output_path = os.path.join(shapenet_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(all_properties, f, indent=2)
    
    print(f"\n✅ 保存到: {output_path}")
    return True

def main():
    # ShapeNetSem数据目录
    shapenet_dir = os.path.abspath('../../output/ShapeNetSem_results')
    
    if not os.path.exists(shapenet_dir):
        print(f"错误：找不到ShapeNetSem目录: {shapenet_dir}")
        print(f"当前目录: {os.getcwd()}")
        # 尝试其他路径
        alternative_path = os.path.abspath('../output/ShapeNetSem_results')
        if os.path.exists(alternative_path):
            shapenet_dir = alternative_path
            print(f"使用替代路径: {shapenet_dir}")
        else:
            return
    
    compute_all_binvox_properties(shapenet_dir)

if __name__ == '__main__':
    main()