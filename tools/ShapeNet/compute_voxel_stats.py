#!/usr/bin/env python3
"""
计算ShapeNetSem所有模型的体素统计信息
用于确定归一化参数
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.binvox_utils import read_binvox

def compute_voxel_statistics(shapenet_dir):
    """
    计算所有模型的体素统计信息
    """
    binvox_dir = os.path.join(shapenet_dir, 'models-binvox')
    binvox_solid_dir = os.path.join(shapenet_dir, 'models-binvox-solid')
    
    if not os.path.exists(binvox_dir) or not os.path.exists(binvox_solid_dir):
        print("错误：找不到binvox目录")
        return None
    
    # 获取所有binvox文件
    binvox_files = [f for f in os.listdir(binvox_dir) if f.endswith('.binvox')]
    print(f"找到 {len(binvox_files)} 个binvox文件")
    
    # 统计数据
    stats = {
        'surface_voxels': [],
        'solid_voxels': [],
        'volume_ratios': [],
        'model_ids': []
    }
    
    # 收集所有模型的体素数据
    for binvox_file in tqdm(binvox_files[:1000], desc="分析体素数据"):  # 先分析前1000个
        model_id = binvox_file.replace('.binvox', '')
        
        surface_file = os.path.join(binvox_dir, binvox_file)
        solid_file = os.path.join(binvox_solid_dir, binvox_file)
        
        surface_voxels = 0
        solid_voxels = 0
        
        # 读取表面体素
        if os.path.exists(surface_file):
            try:
                result = read_binvox(surface_file)
                if result:
                    _, _, _, voxel_data = result
                    surface_voxels = np.sum(voxel_data)
            except:
                continue
        
        # 读取实体体素
        if os.path.exists(solid_file):
            try:
                result = read_binvox(solid_file)
                if result:
                    _, _, _, voxel_data = result
                    solid_voxels = np.sum(voxel_data)
            except:
                continue
        
        if surface_voxels > 0 and solid_voxels > 0:
            stats['surface_voxels'].append(surface_voxels)
            stats['solid_voxels'].append(solid_voxels)
            stats['volume_ratios'].append(solid_voxels / surface_voxels)
            stats['model_ids'].append(model_id)
    
    # 计算统计值
    if stats['solid_voxels']:
        # 实体体素统计
        solid_voxels = np.array(stats['solid_voxels'])
        print(f"\n实体体素统计 (n={len(solid_voxels)}):")
        print(f"  最小值: {np.min(solid_voxels)}")
        print(f"  最大值: {np.max(solid_voxels)}")
        print(f"  平均值: {np.mean(solid_voxels):.0f}")
        print(f"  中位数: {np.median(solid_voxels):.0f}")
        print(f"  标准差: {np.std(solid_voxels):.0f}")
        print(f"  95百分位: {np.percentile(solid_voxels, 95):.0f}")
        print(f"  99百分位: {np.percentile(solid_voxels, 99):.0f}")
        
        # 表面体素统计
        surface_voxels = np.array(stats['surface_voxels'])
        print(f"\n表面体素统计 (n={len(surface_voxels)}):")
        print(f"  最小值: {np.min(surface_voxels)}")
        print(f"  最大值: {np.max(surface_voxels)}")
        print(f"  平均值: {np.mean(surface_voxels):.0f}")
        print(f"  中位数: {np.median(surface_voxels):.0f}")
        
        # 体积比统计
        volume_ratios = np.array(stats['volume_ratios'])
        print(f"\n体积比统计 (solid/surface):")
        print(f"  最小值: {np.min(volume_ratios):.2f}")
        print(f"  最大值: {np.max(volume_ratios):.2f}")
        print(f"  平均值: {np.mean(volume_ratios):.2f}")
        print(f"  中位数: {np.median(volume_ratios):.2f}")
        
        # 找出最大体积的模型
        max_idx = np.argmax(solid_voxels)
        print(f"\n最大体积模型:")
        print(f"  ID: {stats['model_ids'][max_idx]}")
        print(f"  实体体素: {solid_voxels[max_idx]}")
        print(f"  表面体素: {stats['surface_voxels'][max_idx]}")
        
        # 保存统计结果
        output = {
            'max_solid_voxels': int(np.max(solid_voxels)),
            'max_surface_voxels': int(np.max(surface_voxels)),
            'percentile_95_solid': int(np.percentile(solid_voxels, 95)),
            'percentile_95_surface': int(np.percentile(surface_voxels, 95)),
            'percentile_99_solid': int(np.percentile(solid_voxels, 99)),
            'percentile_99_surface': int(np.percentile(surface_voxels, 99)),
            'median_solid': int(np.median(solid_voxels)),
            'median_surface': int(np.median(surface_voxels)),
            'n_samples': len(solid_voxels)
        }
        
        output_path = os.path.join(shapenet_dir, 'voxel_statistics.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ 保存统计结果到: {output_path}")
        return output
    
    return None

def main():
    # ShapeNetSem数据目录
    shapenet_dir = os.path.abspath('../../output/ShapeNetSem_results')
    
    if not os.path.exists(shapenet_dir):
        # 尝试另一个路径
        shapenet_dir = os.path.abspath('../output/ShapeNetSem_results')
        if not os.path.exists(shapenet_dir):
            print(f"错误：找不到ShapeNetSem目录")
            return
    
    compute_voxel_statistics(shapenet_dir)

if __name__ == '__main__':
    main()