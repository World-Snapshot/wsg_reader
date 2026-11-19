#!/usr/bin/env python3
"""
Binvox工具函数，用于从binvox文件计算物理属性
"""

import numpy as np
import os

def read_binvox(filepath):
    """
    读取binvox文件
    返回: dims, translate, scale, voxel_data
    """
    with open(filepath, 'rb') as f:
        # 读取头部
        line = f.readline().decode('ascii').strip()
        if not line.startswith('#binvox'):
            raise ValueError('Not a binvox file')
        
        # 读取维度
        line = f.readline().decode('ascii').strip()
        dims = [int(i) for i in line.split(' ')[1:]]
        
        # 读取平移
        line = f.readline().decode('ascii').strip()
        translate = [float(i) for i in line.split(' ')[1:]]
        
        # 读取缩放
        line = f.readline().decode('ascii').strip()
        scale = float(line.split(' ')[1])
        
        # 跳过"data"行
        line = f.readline()
        
        # 读取RLE压缩的体素数据
        raw_data = f.read()
        
        # 解压RLE数据
        values = []
        for i in range(0, len(raw_data), 2):
            if i+1 < len(raw_data):
                value = raw_data[i]
                count = raw_data[i+1]
                values.extend([value] * count)
        
        # 将数据重塑为3D数组
        voxel_data = np.array(values, dtype=bool)
        if len(voxel_data) >= dims[0] * dims[1] * dims[2]:
            voxel_data = voxel_data[:dims[0] * dims[1] * dims[2]]
            voxel_data = voxel_data.reshape(dims)
        else:
            print(f"警告：体素数据大小不匹配")
            return None
            
    return dims, translate, scale, voxel_data

def calculate_volume_from_binvox(filepath):
    """
    从binvox文件计算体积（立方米）
    """
    try:
        result = read_binvox(filepath)
        if result is None:
            return None
            
        dims, translate, scale, voxel_data = result
        
        # 计算单个体素的体积
        voxel_size = scale / dims[0]  # 假设立方体体素
        voxel_volume = voxel_size ** 3
        
        # 计算占用的体素数
        occupied_voxels = np.sum(voxel_data)
        
        # 计算总体积
        total_volume = occupied_voxels * voxel_volume
        
        # 合理性检查和单位猜测
        # 大多数日常物品的体积应该在0.0001到1立方米之间
        if total_volume > 1000:
            # 可能是毫米单位
            total_volume = total_volume / (1000 ** 3)
        elif total_volume > 10:
            # 可能是厘米单位
            total_volume = total_volume / (100 ** 3)
        elif total_volume < 0.000001:
            # 太小了，可能scale单位有问题，尝试当作毫米处理
            total_volume = total_volume * (1000 ** 3)
        
        return total_volume
    except Exception as e:
        print(f"计算体积时出错: {e}")
        return None

def calculate_support_surface_area_from_binvox(filepath):
    """
    从binvox文件计算支撑表面积（底部表面积）
    """
    try:
        result = read_binvox(filepath)
        if result is None:
            return None
            
        dims, translate, scale, voxel_data = result
        
        # 计算单个体素的面积
        voxel_size = scale / dims[0]
        voxel_area = voxel_size ** 2
        
        # 找到底部层（Z=0）的占用体素
        bottom_layer = voxel_data[:, :, 0]
        occupied_bottom = np.sum(bottom_layer)
        
        # 计算底部表面积
        support_area = occupied_bottom * voxel_area
        
        return support_area
    except Exception as e:
        print(f"计算支撑表面积时出错: {e}")
        return None

def get_binvox_properties(model_id, binvox_dir, binvox_solid_dir):
    """
    获取模型的binvox计算属性
    返回: {
        'solidVolume': float,  # 实体体积 (m³)
        'surfaceVolume': float,  # 表面体积 (m³)
        'supportSurfaceArea': float  # 支撑表面积 (m²)
    }
    """
    properties = {
        'solidVolume': None,
        'surfaceVolume': None,
        'supportSurfaceArea': None
    }
    
    # 查找对应的binvox文件
    # ShapeNetSem的binvox文件名就是model_id（不带wss.前缀）
    # 如果model_id带有wss.前缀，去掉它
    if model_id.startswith('wss.'):
        model_id = model_id[4:]
    
    surface_file = os.path.join(binvox_dir, f"{model_id}.binvox")
    solid_file = os.path.join(binvox_solid_dir, f"{model_id}.binvox")
    
    # 计算表面体积
    if os.path.exists(surface_file):
        surface_volume = calculate_volume_from_binvox(surface_file)
        if surface_volume is not None:
            properties['surfaceVolume'] = surface_volume
            
            # 同时计算支撑表面积
            support_area = calculate_support_surface_area_from_binvox(surface_file)
            if support_area is not None:
                properties['supportSurfaceArea'] = support_area
    
    # 计算实体体积
    if os.path.exists(solid_file):
        solid_volume = calculate_volume_from_binvox(solid_file)
        if solid_volume is not None:
            properties['solidVolume'] = solid_volume
    
    return properties