#!/usr/bin/env python3
"""
PBR Material Renderer
简化的PBR材质渲染器，用于生成材质预览图
"""

import numpy as np
from PIL import Image, ImageFilter
import math

class SimplePBRRenderer:
    """简化的PBR渲染器"""
    
    def __init__(self, size=(300, 300)):
        self.width, self.height = size
        
        # 创建UV坐标
        u = np.linspace(0, 1, self.width)
        v = np.linspace(0, 1, self.height)
        self.uv_u, self.uv_v = np.meshgrid(u, v)
        
        # 默认光照设置
        self.light_dir = np.array([0.5, 0.5, 1.0])  # 光照方向
        self.light_dir = self.light_dir / np.linalg.norm(self.light_dir)
        self.ambient_light = 0.3
        self.light_intensity = 1.0
        
    def create_sphere_normals(self):
        """创建球体的法线贴图用于测试"""
        # 将UV映射到球面坐标
        theta = self.uv_u * 2 * np.pi
        phi = self.uv_v * np.pi
        
        # 计算球面法线
        nx = np.sin(phi) * np.cos(theta)
        ny = np.sin(phi) * np.sin(theta)
        nz = np.cos(phi)
        
        # 归一化并转换到0-1范围
        normals = np.stack([nx, ny, nz], axis=-1)
        normals = (normals + 1.0) / 2.0
        
        return normals
    
    def render_pbr_material(self, basecolor, normal, roughness, metallic, height=None):
        """
        渲染PBR材质
        
        参数:
        - basecolor: RGB基础颜色 (H, W, 3)
        - normal: 法线贴图 (H, W, 3)
        - roughness: 粗糙度贴图 (H, W)
        - metallic: 金属度贴图 (H, W)
        - height: 高度贴图 (H, W) 可选
        """
        # 确保输入尺寸正确
        if basecolor.shape[:2] != (self.height, self.width):
            basecolor_img = Image.fromarray((basecolor * 255).astype(np.uint8))
            basecolor_img = basecolor_img.resize((self.width, self.height), Image.LANCZOS)
            basecolor = np.array(basecolor_img) / 255.0
            
        if normal.shape[:2] != (self.height, self.width):
            normal_img = Image.fromarray((normal * 255).astype(np.uint8))
            normal_img = normal_img.resize((self.width, self.height), Image.LANCZOS)
            normal = np.array(normal_img) / 255.0
            
        if roughness.shape != (self.height, self.width):
            roughness_img = Image.fromarray((roughness * 255).astype(np.uint8), mode='L')
            roughness_img = roughness_img.resize((self.width, self.height), Image.LANCZOS)
            roughness = np.array(roughness_img) / 255.0
            
        if metallic.shape != (self.height, self.width):
            metallic_img = Image.fromarray((metallic * 255).astype(np.uint8), mode='L')
            metallic_img = metallic_img.resize((self.width, self.height), Image.LANCZOS)
            metallic = np.array(metallic_img) / 255.0
        
        # 解码法线贴图 (从0-1范围转回-1到1)
        normal_decoded = normal * 2.0 - 1.0
        
        # 归一化法线
        normal_length = np.sqrt(np.sum(normal_decoded**2, axis=-1, keepdims=True))
        normal_decoded = normal_decoded / (normal_length + 1e-8)
        
        # 如果没有法线贴图，使用平面法线
        if np.all(normal == 0.5):
            normal_decoded = np.zeros_like(normal_decoded)
            normal_decoded[:, :, 2] = 1.0
        
        # 计算光照
        # 漫反射光照 (Lambert)
        ndotl = np.maximum(0, np.sum(normal_decoded * self.light_dir, axis=-1))
        
        # 简化的镜面反射
        view_dir = np.array([0, 0, 1])  # 假设相机在正前方
        halfway = (self.light_dir + view_dir) / 2
        halfway = halfway / np.linalg.norm(halfway)
        
        ndoth = np.maximum(0, np.sum(normal_decoded * halfway, axis=-1))
        
        # 使用粗糙度计算镜面反射强度
        shininess = (1.0 - roughness) * 128.0
        specular = np.power(ndoth, shininess)
        
        # 组合光照
        # 金属材质：基础颜色影响镜面反射
        # 非金属材质：基础颜色只影响漫反射
        metallic_3d = np.expand_dims(metallic, axis=-1)
        
        # 漫反射颜色（非金属部分）
        diffuse_color = basecolor * (1.0 - metallic_3d)
        
        # 镜面反射颜色（金属使用基础颜色，非金属使用白色）
        specular_color = basecolor * metallic_3d + np.ones_like(basecolor) * (1.0 - metallic_3d) * 0.04
        
        # 最终颜色
        diffuse_term = diffuse_color * (self.ambient_light + ndotl[:, :, np.newaxis] * self.light_intensity)
        specular_term = specular_color * specular[:, :, np.newaxis] * self.light_intensity
        
        final_color = diffuse_term + specular_term * (1.0 - roughness[:, :, np.newaxis])
        
        # 应用高度贴图作为环境遮挡（如果提供）
        if height is not None:
            if height.shape != (self.height, self.width):
                height_img = Image.fromarray((height * 255).astype(np.uint8), mode='L')
                height_img = height_img.resize((self.width, self.height), Image.LANCZOS)
                height = np.array(height_img) / 255.0
                
            # 简单的环境遮挡模拟
            ao = 0.5 + 0.5 * height
            final_color *= ao[:, :, np.newaxis]
        
        # 色调映射和伽马校正
        final_color = final_color / (final_color + 1.0)  # Reinhard tone mapping
        final_color = np.power(final_color, 1.0/2.2)  # Gamma correction
        
        # 限制到0-1范围
        final_color = np.clip(final_color, 0, 1)
        
        return final_color
    
    def render_texture_preview(self, basecolor, normal=None, roughness=None, metallic=None, height=None):
        """
        渲染材质预览（平铺模式）
        如果某些贴图缺失，使用默认值
        """
        # 使用默认值填充缺失的贴图
        if normal is None:
            normal = np.ones((self.height, self.width, 3)) * 0.5
            normal[:, :, 2] = 1.0  # 默认指向Z轴
            normal = (normal + 1.0) / 2.0
            
        if roughness is None:
            roughness = np.ones((self.height, self.width)) * 0.5
            
        if metallic is None:
            metallic = np.zeros((self.height, self.width))
            
        return self.render_pbr_material(basecolor, normal, roughness, metallic, height)
    
    def systematic_pbr_variations(self, basecolor, normal, roughness, metallic, height=None, 
                                 specular=None, opacity=None):
        """
        系统地生成PBR属性变化，覆盖所有参数空间
        
        返回一个生成器，产生所有可能的参数组合
        """
        # 定义每个属性的变化值
        brightness_values = [0.5, 0.75, 1.0, 1.25, 1.5]  # 亮度变化 (5个值)
        roughness_values = [0.0, 0.25, 0.5, 0.75, 1.0]  # 粗糙度绝对值 (5个值)
        metallic_values = [0.0, 0.25, 0.5, 0.75, 1.0]   # 金属度绝对值 (5个值)
        
        # 如果有specular贴图，也添加变化
        if specular is not None:
            specular_values = [0.5, 0.75, 1.0, 1.25, 1.5]  # 镜面反射强度 (5个值)
        else:
            specular_values = [1.0]  # 不变化
            
        # 如果有opacity贴图，添加变化
        if opacity is not None:
            opacity_values = [0.2, 0.4, 0.6, 0.8, 1.0]  # 透明度值 (5个值)
        else:
            opacity_values = [1.0]  # 不变化
            
        # 生成所有组合
        total_variations = (len(brightness_values) * len(roughness_values) * 
                          len(metallic_values) * len(specular_values) * len(opacity_values))
        
        variation_index = 0
        
        # 计算实际需要的变化数量
        print(f"  总共可能的参数组合: {total_variations}")
        
        # 系统地遍历所有参数组合
        for brightness in brightness_values:
            for roughness_val in roughness_values:
                for metallic_val in metallic_values:
                    for specular_val in specular_values:
                        for opacity_val in opacity_values:
                                
                            # 创建变化后的贴图
                            aug_maps = {}
                            
                            # 基础颜色亮度调整
                            aug_maps['basecolor'] = np.clip(basecolor * brightness, 0, 1)
                            
                            # 粗糙度设置为固定值
                            aug_maps['roughness'] = np.ones_like(roughness) * roughness_val
                            
                            # 金属度设置为固定值
                            aug_maps['metallic'] = np.ones_like(metallic) * metallic_val
                            
                            # 法线贴图保持不变（或轻微调整）
                            aug_maps['normal'] = normal.copy()
                            
                            # 高度贴图保持不变
                            if height is not None:
                                aug_maps['height'] = height.copy()
                            
                            # 镜面反射调整
                            if specular is not None:
                                aug_maps['specular'] = np.clip(specular * specular_val, 0, 1)
                            
                            # 透明度设置
                            if opacity is not None:
                                aug_maps['opacity'] = np.ones_like(opacity) * opacity_val
                            
                            yield variation_index, aug_maps
                            variation_index += 1
    
    def augment_pbr_properties(self, basecolor, normal, roughness, metallic, height=None, 
                             variation_strength=0.2, seed=None):
        """
        保留原始的随机增强方法，用于兼容性
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 复制原始数据
        aug_basecolor = basecolor.copy()
        aug_normal = normal.copy()
        aug_roughness = roughness.copy()
        aug_metallic = metallic.copy()
        aug_height = height.copy() if height is not None else None
        
        # 1. 调整基础颜色
        # 亮度变化
        brightness_factor = 1.0 + np.random.uniform(-variation_strength, variation_strength)
        aug_basecolor = aug_basecolor * brightness_factor
        
        # 色相偏移（简化版）
        hue_shift = np.random.uniform(-variation_strength * 0.1, variation_strength * 0.1)
        r, g, b = aug_basecolor[:, :, 0], aug_basecolor[:, :, 1], aug_basecolor[:, :, 2]
        aug_basecolor[:, :, 0] = r * (1 - abs(hue_shift)) + g * max(0, hue_shift) + b * max(0, -hue_shift)
        aug_basecolor[:, :, 1] = g * (1 - abs(hue_shift)) + b * max(0, hue_shift) + r * max(0, -hue_shift)
        aug_basecolor[:, :, 2] = b * (1 - abs(hue_shift)) + r * max(0, hue_shift) + g * max(0, -hue_shift)
        
        # 2. 调整粗糙度
        roughness_shift = np.random.uniform(-variation_strength * 0.3, variation_strength * 0.3)
        aug_roughness = np.clip(aug_roughness + roughness_shift, 0, 1)
        
        # 3. 调整金属度（通常变化较小）
        if np.mean(metallic) > 0.5:  # 主要是金属材质
            metallic_shift = np.random.uniform(-variation_strength * 0.1, variation_strength * 0.1)
            aug_metallic = np.clip(aug_metallic + metallic_shift, 0.5, 1)
        else:  # 主要是非金属材质
            metallic_shift = np.random.uniform(0, variation_strength * 0.1)
            aug_metallic = np.clip(aug_metallic + metallic_shift, 0, 0.3)
        
        # 4. 轻微扰动法线（添加噪声）
        if variation_strength > 0:
            noise = np.random.normal(0, variation_strength * 0.05, normal.shape)
            aug_normal = aug_normal + noise
            # 重新归一化
            aug_normal = np.clip(aug_normal, 0, 1)
        
        # 5. 调整高度贴图（如果存在）
        if aug_height is not None:
            height_scale = 1.0 + np.random.uniform(-variation_strength * 0.2, variation_strength * 0.2)
            aug_height = np.clip(aug_height * height_scale, 0, 1)
        
        # 确保所有值在有效范围内
        aug_basecolor = np.clip(aug_basecolor, 0, 1)
        
        return {
            'basecolor': aug_basecolor,
            'normal': aug_normal,
            'roughness': aug_roughness,
            'metallic': aug_metallic,
            'height': aug_height
        }