#!/usr/bin/env python3
"""
计算binvox的相对物理属性（归一化到0-100）
避免单位问题，使用相对值
"""

import numpy as np
import os
from utils.binvox_utils import read_binvox

def calculate_relative_properties(model_id, binvox_dir, binvox_solid_dir):
    """
    计算模型的相对物理属性（0-100范围）
    返回: {
        'relativeVolume': float,  # 相对体积 (0-100)
        'relativeSurfaceArea': float,  # 相对表面积 (0-100)
        'relativeWeight': float,  # 相对重量 (0-100)
        'volumeRatio': float,  # 实体/表面体积比
    }
    """
    properties = {
        'relativeVolume': None,
        'relativeSurfaceArea': None,
        'relativeWeight': None,
        'volumeRatio': None
    }
    
    # 处理model_id前缀
    if model_id.startswith('wss.'):
        model_id = model_id[4:]
    
    # 读取binvox文件
    surface_file = os.path.join(binvox_dir, f"{model_id}.binvox")
    solid_file = os.path.join(binvox_solid_dir, f"{model_id}.binvox")
    
    surface_voxels = 0
    solid_voxels = 0
    
    # 读取表面体素
    if os.path.exists(surface_file):
        result = read_binvox(surface_file)
        if result:
            _, _, _, voxel_data = result
            surface_voxels = np.sum(voxel_data)
    
    # 读取实体体素
    if os.path.exists(solid_file):
        result = read_binvox(solid_file)
        if result:
            _, _, _, voxel_data = result
            solid_voxels = np.sum(voxel_data)
    
    # 计算相对值
    if solid_voxels > 0:
        # 相对体积：基于体素数量，使用对数缩放
        # 典型范围：1000-1000000体素 -> 0-100
        relative_volume = np.log10(solid_voxels + 1) / 6 * 100  # log10(1000000) = 6
        relative_volume = np.clip(relative_volume, 0, 100)
        properties['relativeVolume'] = relative_volume
        
        # 相对重量：体积的函数，但考虑密度差异
        # 这里简化为与体积相同，实际使用时会乘以材质密度比例
        properties['relativeWeight'] = relative_volume
    
    if surface_voxels > 0:
        # 相对表面积：基于表面体素数量
        relative_surface = np.log10(surface_voxels + 1) / 6 * 100
        relative_surface = np.clip(relative_surface, 0, 100)
        properties['relativeSurfaceArea'] = relative_surface
    
    # 体积比：实体/表面，用于判断物体是否空心
    if surface_voxels > 0 and solid_voxels > 0:
        properties['volumeRatio'] = solid_voxels / surface_voxels
    
    return properties


def normalize_to_range(value, min_val, max_val, target_min=0, target_max=100):
    """将值归一化到目标范围"""
    if max_val == min_val:
        return (target_min + target_max) / 2
    
    normalized = (value - min_val) / (max_val - min_val)
    return target_min + normalized * (target_max - target_min)


def compute_relative_properties_batch(model_ids, binvox_dir, binvox_solid_dir):
    """
    批量计算相对属性，使用整体统计进行归一化
    """
    all_properties = {}
    raw_values = {
        'solid_voxels': [],
        'surface_voxels': [],
        'volume_ratios': []
    }
    
    # 第一遍：收集原始数据
    for model_id in model_ids:
        if model_id.startswith('wss.'):
            model_id = model_id[4:]
            
        surface_file = os.path.join(binvox_dir, f"{model_id}.binvox")
        solid_file = os.path.join(binvox_solid_dir, f"{model_id}.binvox")
        
        surface_voxels = 0
        solid_voxels = 0
        
        if os.path.exists(surface_file):
            result = read_binvox(surface_file)
            if result:
                _, _, _, voxel_data = result
                surface_voxels = np.sum(voxel_data)
                raw_values['surface_voxels'].append(surface_voxels)
        
        if os.path.exists(solid_file):
            result = read_binvox(solid_file)
            if result:
                _, _, _, voxel_data = result
                solid_voxels = np.sum(voxel_data)
                raw_values['solid_voxels'].append(solid_voxels)
        
        if surface_voxels > 0 and solid_voxels > 0:
            raw_values['volume_ratios'].append(solid_voxels / surface_voxels)
        
        all_properties[model_id] = {
            'surface_voxels': surface_voxels,
            'solid_voxels': solid_voxels
        }
    
    # 计算统计值用于归一化
    if raw_values['solid_voxels']:
        solid_percentiles = np.percentile(raw_values['solid_voxels'], [5, 95])
        surface_percentiles = np.percentile(raw_values['surface_voxels'], [5, 95])
    else:
        solid_percentiles = [1, 1000000]
        surface_percentiles = [1, 100000]
    
    # 第二遍：归一化
    for model_id, props in all_properties.items():
        normalized_props = calculate_relative_properties(model_id, binvox_dir, binvox_solid_dir)
        
        # 使用百分位数进行更稳健的归一化
        if props['solid_voxels'] > 0:
            normalized_props['relativeVolume'] = normalize_to_range(
                props['solid_voxels'], 
                solid_percentiles[0], 
                solid_percentiles[1],
                1, 100
            )
            normalized_props['relativeWeight'] = normalized_props['relativeVolume']
        
        if props['surface_voxels'] > 0:
            normalized_props['relativeSurfaceArea'] = normalize_to_range(
                props['surface_voxels'],
                surface_percentiles[0],
                surface_percentiles[1],
                1, 100
            )
        
        all_properties[model_id] = normalized_props
    
    return all_properties