#!/usr/bin/env python3
"""
WSG到PNG完整管道 - 支持RGB颜色读取
输出结构：
- glTF文件：output/test_dataset_results/MagicArticulate/gltf/对应id/
- Visualization files: output/test_dataset_results/MagicArticulate/visualization/对应id/
"""

import os
import sys
import subprocess
import argparse
import tempfile
import shutil
import glob
import json
import struct
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
from collections import deque


PALETTE_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "channel_visualization_palettes.json"


def load_visualization_specs():
    try:
        with PALETTE_CONFIG_PATH.open("r", encoding="utf-8") as f:
            return {int(channel): spec for channel, spec in json.load(f).items()}
    except (OSError, ValueError, TypeError):
        return {}


VISUALIZATION_SPECS = load_visualization_specs()
VISUALIZATION_CHANNEL_RANGES = {
    channel: tuple(map(float, spec["range"]))
    for channel, spec in VISUALIZATION_SPECS.items()
    if "range" in spec
}


def hex_to_rgb(value):
    value = str(value).lstrip("#")
    return np.array([int(value[index:index + 2], 16) for index in (0, 2, 4)], dtype=np.uint8)


def normalize_scalar_for_display(value_map, valid_mask, value_range=None):
    normalized = np.zeros_like(value_map, dtype=np.float32)
    if not np.any(valid_mask):
        return normalized, None
    if value_range is not None:
        vmin, vmax = map(float, value_range)
    else:
        if os.environ.get("WSG_READER_STRICT_GLOBAL", "0") in {"1", "true", "True"}:
            raise ValueError(
                "Strict WSG Reader rendering requires an explicit global range; "
                "per-image q1/q99 fallback is forbidden."
            )
        valid_values = value_map[valid_mask]
        vmin = float(np.quantile(valid_values, 0.01))
        vmax = float(np.quantile(valid_values, 0.99))
    if vmax > vmin:
        normalized[valid_mask] = np.clip(
            (value_map[valid_mask] - vmin) / (vmax - vmin), 0.0, 1.0
        )
    else:
        normalized[valid_mask] = 0.5
    return normalized, (vmin, vmax)


def apply_continuous_palette(normalized, channel, valid_mask=None):
    spec = VISUALIZATION_SPECS.get(int(channel), {})
    nodes = spec.get("nodes") or [
        [0.0, "#30123b"], [0.25, "#325a9b"], [0.5, "#27828e"],
        [0.75, "#6cbe64"], [1.0, "#f2dc54"],
    ]
    positions = np.asarray([float(node[0]) for node in nodes], dtype=np.float32)
    colors = np.asarray([hex_to_rgb(node[1]) for node in nodes], dtype=np.float32)
    values = np.clip(np.asarray(normalized, dtype=np.float32), 0.0, 1.0)
    rgb = np.stack(
        [np.interp(values, positions, colors[:, component]) for component in range(3)],
        axis=-1,
    )
    if valid_mask is not None:
        rgb[~np.asarray(valid_mask, dtype=bool)] = 0.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def render_transparency_brightness(value_map, valid_mask):
    """Render masked transparency: -1 invalid, 0 opaque, 1 fully transparent."""
    transparency = np.zeros_like(value_map, dtype=np.float32)
    transparency[valid_mask] = np.clip(value_map[valid_mask], 0.0, 1.0)
    brightness = np.zeros_like(value_map, dtype=np.float32)
    brightness[valid_mask] = 1.0 - transparency[valid_mask]
    return np.repeat(brightness[:, :, None], 3, axis=2)


def categorical_color(channel, value):
    value = int(value)
    class_spec = VISUALIZATION_SPECS.get(int(channel), {}).get("classes", {}).get(str(value))
    if class_spec and class_spec.get("color"):
        return hex_to_rgb(class_spec["color"])

    # Golden-angle hues assign each large-taxonomy class a fixed discrete color.
    hue = (value * 137.50776405003785) % 360.0
    sector = int(hue // 60.0) % 6
    fraction = hue / 60.0 - sector
    saturation = 0.62 + 0.16 * (value % 3) / 2.0
    brightness = 0.78 + 0.18 * ((value // 3) % 3) / 2.0
    chroma = brightness * saturation
    x = chroma * (1.0 - abs((hue / 60.0) % 2.0 - 1.0))
    rgb1 = ((chroma, x, 0), (x, chroma, 0), (0, chroma, x),
            (0, x, chroma), (x, 0, chroma), (chroma, 0, x))[sector]
    offset = brightness - chroma
    return np.rint((np.asarray(rgb1) + offset) * 255.0).astype(np.uint8)


def stable_category_color(value):
    return categorical_color(-1, value)


def refine_reader_valid_mask(value_map, valid_mask, channel):
    """Recover a black background when generated WSG lanes contain no -1 sentinel."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid_mask) or float(np.mean(valid_mask)) < 0.995:
        return valid_mask

    values = np.asarray(value_map, dtype=np.float32)
    border = np.concatenate((values[0], values[-1], values[:, 0], values[:, -1]))
    border = border[np.isfinite(border)]
    if border.size == 0:
        return valid_mask
    background = float(np.median(border))
    border_mad = float(np.median(np.abs(border - background)))
    value_range = VISUALIZATION_CHANNEL_RANGES.get(int(channel))
    span = float(value_range[1] - value_range[0]) if value_range else float(np.ptp(values[valid_mask]))
    tolerance = max(4.0 * border_mad, 0.015 * max(span, 1e-6), 1e-6)
    inferred = valid_mask & (np.abs(values - background) > tolerance)
    coverage = float(np.mean(inferred))
    return inferred if 0.001 <= coverage <= 0.90 else valid_mask


def reader_rgb_base(wsg_array):
    """Build an RGB overlay base with black pixels where RGB is absent."""
    rgb_values = np.asarray(wsg_array[:, :, :3], dtype=np.float32)
    valid_rgb = np.all(np.isfinite(rgb_values) & (rgb_values != -1), axis=2)
    rgb = np.zeros(rgb_values.shape, dtype=np.uint8)
    rgb[valid_rgb] = np.clip(rgb_values[valid_rgb], 0, 255).astype(np.uint8)
    return rgb


class BoneDataUnpacker:
    @staticmethod
    def unpack_weights(packed_float):
        try:
            packed = np.frombuffer(np.array([packed_float], dtype=np.float32), dtype=np.uint32)[0]
            w0 = (packed >> 24) & 0xFF
            w1 = (packed >> 16) & 0xFF
            w2 = (packed >> 8) & 0xFF
            w3 = packed & 0xFF
            weights = np.array([w0, w1, w2, w3], dtype=np.float32) / 255.0
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
            return weights
        except:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def unpack_indices(packed_float):
        try:
            packed = np.frombuffer(np.array([packed_float], dtype=np.float32), dtype=np.uint32)[0]
            indices = np.array([
                (packed >> 24) & 0xFF,
                (packed >> 16) & 0xFF,
                (packed >> 8) & 0xFF,
                packed & 0xFF
            ], dtype=np.int32)
            return indices
        except:
            return np.array([0, 0, 0, 0], dtype=np.int32)

    @staticmethod
    def unpack_skeleton_meta(packed_meta):
        try:
            packed_int = np.frombuffer(np.array([packed_meta], dtype=np.float32), dtype=np.uint32)[0]
            joint_count = (packed_int >> 22) & 0x3FF
            bone_count = (packed_int >> 12) & 0x3FF
            root_index = (packed_int >> 2) & 0x3FF
            return int(joint_count), int(bone_count), int(root_index)
        except:
            return 0, 0, 0


class WSGToPNGPipeline:
    """完整的WSG处理管道 - 支持RGB颜色"""

    def __init__(self):
        self.temp_dirs = []
        self.vertices = []
        self.normals = []
        self.uvs = []
        self.faces = []
        self.vertex_colors = []  # 🎨 新增颜色数组
        self.vertex_bone_weights = []
        self.vertex_bone_indices = []
        self.bone_unpacker = BoneDataUnpacker()
        self.joints = []
        self.bones = []
        self.joint_parent_mapping = {}
        self.joint_count = 0
        self.bone_count = 0
        self.root_index = 0
        self.metadata = None

    def __del__(self):
        self.cleanup_temp_dirs()

    def cleanup_temp_dirs(self):
        """清理临时目录"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception:
                pass
        self.temp_dirs.clear()
    
    def extract_rgb_from_wsg(self, wsg_array, output_path):
        """从WSG的前3或4个通道提取RGB/RGBA图像并保存"""
        try:
            # 检查WSG格式
            if len(wsg_array.shape) != 3 or wsg_array.shape[2] < 3:
                print("⚠️ WSG格式不正确，无法提取RGB")
                return False
            
            height, width = wsg_array.shape[:2]
            
            # 检查第4通道是否有alpha信息
            has_alpha = False
            if wsg_array.shape[2] >= 4:
                alpha_channel = wsg_array[:, :, 3]
                # 如果第4通道不全是-1，说明有alpha信息
                has_alpha = not np.all(alpha_channel == -1)
            
            if has_alpha:
                # 提取RGBA
                rgba_channels = wsg_array[:, :, 0:4]
                # WSG中-1表示无数据，对于RGB转换为255（白色），对于alpha转换为0（完全透明）
                rgba_image = np.where(rgba_channels[:, :, :3] == -1, 255, rgba_channels[:, :, :3])
                alpha_image = np.where(rgba_channels[:, :, 3:4] == -1, 0, rgba_channels[:, :, 3:4])
                rgba_image = np.concatenate([rgba_image, alpha_image], axis=2)
                # 确保值在0-255范围内
                rgba_image = np.clip(rgba_image, 0, 255).astype(np.uint8)
                
                # 保存为PNG
                from PIL import Image
                img = Image.fromarray(rgba_image, 'RGBA')
                img.save(output_path)
                print(f"   💾 保存RGBA图像: {output_path}")
            else:
                # 只提取RGB
                rgb_channels = wsg_array[:, :, 0:3]
                # WSG中-1表示无数据，需要转换为白色背景
                rgb_image = np.where(rgb_channels == -1, 255, rgb_channels)
                # 确保值在0-255范围内
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
                
                # 保存为PNG
                from PIL import Image
                img = Image.fromarray(rgb_image)
                img.save(output_path)
                print(f"   💾 保存RGB图像: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 提取RGB失败: {e}")
            return False
    
    def extract_physics_from_wsg(self, wsg_array, output_path):
        """从WSG通道126-136提取物理属性信息"""
        try:
            wsg_path = os.path.join(os.path.dirname(output_path), f"{Path(output_path).parent.name}.wsg")
            solid_stats = self.aggregate_continuous_channel(wsg_array, 126)
            surface_stats = self.aggregate_continuous_channel(wsg_array, 127)
            weight_stats = self.aggregate_continuous_channel(wsg_array, 128)
            support_stats = self.aggregate_continuous_channel(wsg_array, 129)
            friction_force_stats = self.aggregate_continuous_channel(wsg_array, 130)
            material_stats = self.aggregate_discrete_channel(wsg_array, 131)
            material_conf_stats = self.aggregate_continuous_channel(wsg_array, 132)
            density_stats = self.aggregate_continuous_channel(wsg_array, 133)
            friction_coeff_stats = self.aggregate_continuous_channel(wsg_array, 134)
            container_stats = self.aggregate_discrete_channel(wsg_array, 135)
            category_stats = self.aggregate_discrete_channel(wsg_array, 136)

            solidVolume = solid_stats["median"] if solid_stats else -1
            surfaceVolume = surface_stats["median"] if surface_stats else -1
            weight = weight_stats["median"] if weight_stats else -1
            supportSurfaceArea = support_stats["median"] if support_stats else -1
            staticFrictionForce = friction_force_stats["median"] if friction_force_stats else -1
            material_id = material_stats["mode"] if material_stats else -1
            material_confidence = material_conf_stats["median"] if material_conf_stats else -1
            density = density_stats["median"] if density_stats else -1
            staticFrictionCoeff = friction_coeff_stats["median"] if friction_coeff_stats else -1
            isContainerLike = container_stats["mode"] if container_stats else -1
            category_id = category_stats["mode"] if category_stats else -1
            
            # 加载材质和类别映射
            material_mapping = self.load_material_id_to_name(wsg_path)
            category_mapping = self.load_category_id_to_name(wsg_path)
            
            # 获取材质和类别名称
            material_name = material_mapping.get(material_id, f"unknown(ID={material_id})") if material_mapping else f"ID={material_id}"
            category_name = category_mapping.get(category_id, f"unknown(ID={category_id})") if category_mapping else f"ID={category_id}"
            
            # 保存物理属性信息
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("物理属性信息 (Physics Properties)\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("几何属性 (Geometry) - 相对值系统\n")
                f.write("-" * 30 + "\n")
                f.write(f"相对实体体积 (Relative Solid Volume): {solidVolume:.1f}/100\n")
                f.write(f"相对表面积 (Relative Surface Area): {surfaceVolume:.1f}/100\n")
                f.write(f"相对支撑面积 (Relative Support Area): {supportSurfaceArea:.1f}/100\n")
                f.write(f"是否为容器 (Is Container-like): {'是' if isContainerLike == 1 else '否'}\n")
                
                f.write("\n物理特性 (Physical Properties)\n")
                f.write("-" * 30 + "\n")
                f.write(f"相对重量 (Relative Weight): {weight:.1f} (相对体积 × 密度)\n")
                f.write(f"密度 (Density): {density:.2f} kg/L\n")
                f.write(f"相对摩擦力 (Relative Friction Force): {staticFrictionForce:.1f} (相对体积 × 摩擦系数)\n")
                f.write(f"静摩擦系数 (Static Friction Coefficient): {staticFrictionCoeff:.2f}\n")
                
                f.write("\n材质信息 (Material)\n")
                f.write("-" * 30 + "\n")
                f.write(f"材质 (Material): {material_name}\n")
                f.write(f"材质ID (Material ID): {material_id}\n")
                f.write(f"材质置信度 (Confidence): {material_confidence:.2%}\n")
                if material_stats and len(material_stats["unique_values"]) > 1:
                    f.write(f"可见材质ID集合 (Visible IDs): {material_stats['unique_values']}\n")
                
                f.write("\n类别信息 (Category)\n")
                f.write("-" * 30 + "\n")
                f.write(f"类别 (Category): {category_name}\n")
                f.write(f"类别ID (Category ID): {category_id}\n")
                if category_stats and len(category_stats["unique_values"]) > 1:
                    f.write(f"可见类别ID集合 (Visible IDs): {category_stats['unique_values']}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("注释:\n")
                f.write("- 相对值基于整个数据集的最大值进行归一化(0-100)\n")
                f.write("- 相对重量 = 相对体积 × 密度(kg/L)\n")
                f.write("- 相对摩擦力 = 相对体积 × 静摩擦系数\n")
                f.write("- 当前摘要值基于有效区域聚合（连续值取median，离散值取mode）\n")
                f.write("- 值为-1表示该属性未定义或不适用\n")
                
            print(f"   💾 保存物理属性: {output_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ 提取物理属性失败: {e}")
            return False
    
    def extract_material_from_wsg(self, wsg_array, output_path, wsg_path=None):
        """从WSG通道143-154提取材质区域和MTL属性信息"""
        try:
            region_mapping = self.load_channel143_region_mapping(wsg_path) if wsg_path else {}
            region_map = wsg_array[:, :, 143]
            valid_region_ids = sorted(int(v) for v in np.unique(region_map[region_map >= 0]))
            ka_stats = self.aggregate_triplet_channels(wsg_array, [144, 145, 146])
            kd_stats = self.aggregate_triplet_channels(wsg_array, [147, 148, 149])
            d_stats = self.aggregate_continuous_channel(wsg_array, 150)
            ks_stats = self.aggregate_triplet_channels(wsg_array, [151, 152, 153])
            ns_stats = self.aggregate_continuous_channel(wsg_array, 154)

            Ka_r, Ka_g, Ka_b = ka_stats["median"] if ka_stats else (-1, -1, -1)
            Kd_r, Kd_g, Kd_b = kd_stats["median"] if kd_stats else (-1, -1, -1)
            d = d_stats["median"] if d_stats else -1
            Ks_r, Ks_g, Ks_b = ks_stats["median"] if ks_stats else (-1, -1, -1)
            Ns = ns_stats["median"] if ns_stats else -1
            
            has_material = bool(valid_region_ids) or np.any(wsg_array[:, :, 144:155] != -1)

            def format_triplet(values):
                if np.any(np.asarray(values) == -1):
                    return "N/A"
                return f"[{values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f}]"

            def format_rgb_preview(values):
                if np.any(np.asarray(values) == -1):
                    return "N/A"
                ints = [int(np.clip(v, 0, 1) * 255) for v in values]
                return f"({ints[0]}, {ints[1]}, {ints[2]})"
            
            # 保存材质属性信息
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("材质区域与MTL属性信息 (Material Regions & Properties)\n")
                f.write("=" * 60 + "\n\n")
                
                if has_material:
                    f.write("材质区域信息 (Per-pixel Material Regions)\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"可见区域数量 (Visible Region Count): {len(valid_region_ids)}\n")
                    if valid_region_ids:
                        f.write(f"可见区域ID (Visible Region IDs): {valid_region_ids}\n")
                        f.write("\n区域ID到材质槽映射 (Region ID -> Material Slot)\n")
                        for region_id in valid_region_ids:
                            material_name = region_mapping.get(region_id, "unknown")
                            f.write(f"  - {region_id}: {material_name}\n")
                    else:
                        f.write("未检测到可见材质区域\n")
                    
                    f.write("\nMTL光照属性 (Lighting Properties)\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"环境光颜色 Ka (Ambient): {format_triplet([Ka_r, Ka_g, Ka_b])}\n")
                    f.write(f"漫反射颜色 Kd (Diffuse): {format_triplet([Kd_r, Kd_g, Kd_b])}\n")
                    f.write(f"镜面反射颜色 Ks (Specular): {format_triplet([Ks_r, Ks_g, Ks_b])}\n")
                    
                    f.write("\n表面属性 (Surface Properties)\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"不透明度 d (Opacity): {d:.3f}\n")
                    f.write(f"高光指数 Ns (Shininess): {Ns:.1f}\n")
                    
                    f.write("\n颜色预览 (Color Preview)\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"环境光 RGB: {format_rgb_preview([Ka_r, Ka_g, Ka_b])}\n")
                    f.write(f"漫反射 RGB: {format_rgb_preview([Kd_r, Kd_g, Kd_b])}\n")
                    f.write(f"镜面反射 RGB: {format_rgb_preview([Ks_r, Ks_g, Ks_b])}\n")
                else:
                    f.write("⚠️ 未找到材质区域或MTL材质数据\n")
                    f.write("此模型可能没有可见材质区域，或材质信息未被注入\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("注释:\n")
                f.write("- 143: 当前可见前景像素的材质区域ID，背景为-1\n")
                f.write("- Ka: 环境光反射系数 (0-1)\n")
                f.write("- Kd: 漫反射系数，决定物体的基本颜色 (0-1)\n")
                f.write("- Ks: 镜面反射系数，决定高光颜色 (0-1)\n")
                f.write("- d: 不透明度，1.0=完全不透明，0.0=完全透明\n")
                f.write("- Ns: 高光指数，值越大高光越集中 (0-1000)\n")
                
            print(f"   💾 保存材质属性: {output_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ 提取材质属性失败: {e}")
            return False

    def find_channel143_mapping_path(self, wsg_path):
        """查找143号通道区域映射文件"""
        wsg_path = Path(wsg_path)
        dataset_root = wsg_path.parent.parent.parent if len(wsg_path.parents) >= 3 else None
        candidates = []
        if dataset_root is not None:
            candidates.append(dataset_root / "material_channel_143_region_mapping.json")
        candidates.extend([
            Path(__file__).resolve().parent.parent / "output" / "ShapeNetSem_results" / "material_channel_143_region_mapping.json",
            Path("./output/ShapeNetSem_results/material_channel_143_region_mapping.json").resolve(),
        ])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def load_channel143_region_mapping(self, wsg_path):
        """读取单个物体的143号通道区域ID映射"""
        mapping_path = self.find_channel143_mapping_path(wsg_path)
        if mapping_path is None:
            return {}
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)
            uuid = Path(wsg_path).stem
            obj = mapping_data.get("objects", {}).get(uuid, {})
            return {int(k): str(v) for k, v in obj.get("region_id_to_material_name", {}).items()}
        except Exception:
            return {}

    def find_dataset_root(self, wsg_path):
        wsg_path = Path(wsg_path)
        if len(wsg_path.parents) >= 3:
            return wsg_path.parent.parent.parent
        return None

    def load_material_id_to_name(self, wsg_path):
        dataset_root = self.find_dataset_root(wsg_path)
        candidates = []
        if dataset_root is not None:
            candidates.append(dataset_root / "material_id_mapping.json")
        candidates.extend([
            Path(__file__).resolve().parent.parent / "output" / "ShapeNetSem_results" / "material_id_mapping.json",
            Path("./output/ShapeNetSem_results/material_id_mapping.json").resolve(),
        ])
        for candidate in candidates:
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return {int(k): str(v) for k, v in data.get("id_to_material", {}).items()}
                except Exception:
                    pass
        return {}

    def load_category_id_to_name(self, wsg_path):
        dataset_root = self.find_dataset_root(wsg_path)
        candidates = []
        if dataset_root is not None:
            candidates.append(dataset_root / "category_id_mapping.json")
        candidates.extend([
            Path(__file__).resolve().parent.parent / "output" / "ShapeNetSem_results" / "category_id_mapping.json",
            Path("./output/ShapeNetSem_results/category_id_mapping.json").resolve(),
        ])
        for candidate in candidates:
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return {int(v): str(k) for k, v in data.get("category_to_id", {}).items()}
                except Exception:
                    pass
        return {}

    def aggregate_continuous_channel(self, wsg_array, channel):
        values = wsg_array[:, :, channel]
        valid = values[values != -1]
        if valid.size == 0:
            return None
        return {
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "count": int(valid.size),
        }

    def aggregate_triplet_channels(self, wsg_array, channels):
        stacked = wsg_array[:, :, channels]
        valid_mask = np.all(stacked != -1, axis=2)
        if not np.any(valid_mask):
            return None
        valid = stacked[valid_mask]
        return {
            "mean": np.mean(valid, axis=0).astype(np.float32),
            "median": np.median(valid, axis=0).astype(np.float32),
            "min": np.min(valid, axis=0).astype(np.float32),
            "max": np.max(valid, axis=0).astype(np.float32),
            "count": int(valid.shape[0]),
        }

    def aggregate_discrete_channel(self, wsg_array, channel):
        values = wsg_array[:, :, channel]
        valid = values[values != -1]
        if valid.size == 0:
            return None
        ints = valid.astype(np.int64)
        unique, counts = np.unique(ints, return_counts=True)
        order = np.argsort(-counts)
        unique = unique[order]
        counts = counts[order]
        return {
            "mode": int(unique[0]),
            "count": int(valid.size),
            "unique_values": [int(v) for v in unique.tolist()],
            "counts": [int(c) for c in counts.tolist()],
        }

    def make_region_palette(self, size):
        base_colors = np.asarray(
            [
                [230, 57, 70],
                [69, 123, 157],
                [42, 157, 143],
                [244, 162, 97],
                [233, 196, 106],
                [141, 153, 174],
                [239, 71, 111],
                [6, 214, 160],
                [17, 138, 178],
                [131, 56, 236],
                [255, 107, 107],
                [77, 144, 142],
                [188, 108, 37],
                [58, 134, 255],
                [255, 183, 3],
            ],
            dtype=np.uint8,
        )
        if size <= len(base_colors):
            return base_colors[:size]
        repeat = int(np.ceil(size / len(base_colors)))
        return np.tile(base_colors, (repeat, 1))[:size]

    def blend_overlay(self, base_rgb, color_rgb, mask):
        overlay = base_rgb.copy()
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color_rgb[mask]).astype(np.uint8)
        return overlay

    def scalar_to_heatmap(self, normalized):
        normalized = np.clip(normalized, 0.0, 1.0)
        r = np.clip(1.5 * normalized - 0.1, 0, 1)
        g = np.clip(1.5 - 2.0 * np.abs(normalized - 0.5), 0, 1)
        b = np.clip(1.2 - 1.4 * normalized, 0, 1)
        return np.stack([r, g, b], axis=-1)

    def find_component_centers(self, mask):
        """为可视化找连通区域中心点"""
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        centers = []
        min_pixels = 80

        for y, x in np.argwhere(mask):
            if visited[y, x]:
                continue
            queue = deque([(int(y), int(x))])
            visited[y, x] = True
            coords = []
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if len(coords) < min_pixels:
                continue
            coords_arr = np.asarray(coords, dtype=np.int32)
            center_y = int(np.mean(coords_arr[:, 0]))
            center_x = int(np.mean(coords_arr[:, 1]))
            centers.append((center_x, center_y))

        return centers

    def export_channel143_region_artifacts(self, wsg_array, wsg_path, output_dir):
        """导出143号通道的区域可视化和单通道NPZ"""
        try:
            from PIL import Image, ImageDraw, ImageFont

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            uuid = Path(wsg_path).stem
            region_map = wsg_array[:, :, 143].astype(np.int32)
            rgb = reader_rgb_base(wsg_array)
            foreground = region_map >= 0
            if not np.any(foreground):
                return False
            region_ids = sorted(int(v) for v in np.unique(region_map[foreground]))
            color_by_region = {region_id: stable_category_color(region_id) for region_id in region_ids}

            region_rgb = np.zeros_like(rgb)
            overlay = rgb.copy()
            for region_id in region_ids:
                mask = region_map == region_id
                color = color_by_region[region_id]
                region_rgb[mask] = color
                overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)

            labeled = Image.fromarray(region_rgb.copy())
            draw = ImageDraw.Draw(labeled)
            font = ImageFont.load_default()
            for region_id in region_ids:
                mask = region_map == region_id
                for cx, cy in self.find_component_centers(mask):
                    text = str(region_id)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        draw.text((cx + dx, cy + dy), text, font=font, fill=(0, 0, 0))
                    draw.text((cx, cy), text, font=font, fill=(255, 255, 255))

            Image.fromarray(overlay).save(output_dir / f"{uuid}_channel143_overlay.png")
            Image.fromarray(region_rgb).save(output_dir / f"{uuid}_channel143_region_ids.png")
            labeled.save(output_dir / f"{uuid}_channel143_region_ids_labeled.png")

            np.savez_compressed(
                output_dir / f"{uuid}_channel143_map.npz",
                channel143_map=region_map[:, :, None].astype(np.float32),
            )
            print(f"   💾 保存143号通道可视化和NPZ: {output_dir}")
            return True
        except Exception as e:
            print(f"   ❌ 导出143号通道可视化失败: {e}")
            return False

    def export_discrete_channel_artifacts(
        self,
        wsg_array,
        wsg_path,
        output_dir,
        channel,
        slug,
        id_to_label=None,
    ):
        try:
            from PIL import Image, ImageDraw, ImageFont

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            uuid = Path(wsg_path).stem
            raw_value_map = wsg_array[:, :, channel].astype(np.float32)
            # Discrete IDs start at zero. Predictions below the midpoint between
            # the absent sentinel (-1) and class 0 are background, even before
            # rounding to an integer category.
            valid_mask = np.isfinite(raw_value_map) & (raw_value_map >= -0.5)
            valid_mask = refine_reader_valid_mask(raw_value_map, valid_mask, channel)
            if not np.any(valid_mask):
                return False
            value_map = np.full(raw_value_map.shape, -1, dtype=np.int32)
            value_map[valid_mask] = np.rint(raw_value_map[valid_mask]).astype(np.int32)

            rgb = reader_rgb_base(wsg_array)
            values = sorted(int(v) for v in np.unique(value_map[valid_mask]))
            color_by_value = {value: categorical_color(channel, value) for value in values}

            value_rgb = np.zeros_like(rgb)
            for value in values:
                mask = value_map == value
                value_rgb[mask] = color_by_value[value]
            overlay = self.blend_overlay(rgb, value_rgb, valid_mask)

            labeled_overlay = Image.fromarray(overlay.copy())
            draw = ImageDraw.Draw(labeled_overlay)
            font = ImageFont.load_default()
            for value in values:
                mask = value_map == value
                label_text = str(value)
                if id_to_label and value in id_to_label:
                    label_text = f"{value}:{id_to_label[value]}"
                for cx, cy in self.find_component_centers(mask):
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        draw.text((cx + dx, cy + dy), label_text, font=font, fill=(0, 0, 0))
                    draw.text((cx, cy), label_text, font=font, fill=(255, 255, 255))

            prefix = output_dir / f"{uuid}_channel{channel}_{slug}"
            Image.fromarray(overlay).save(f"{prefix}_overlay.png")
            Image.fromarray(value_rgb).save(f"{prefix}_value_ids.png")
            labeled_overlay.save(f"{prefix}_overlay_labeled.png")
            np.savez_compressed(
                f"{prefix}_map.npz",
                channel_map=value_map[:, :, None].astype(np.float32),
            )
            with open(f"{prefix}_stats.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "channel": channel,
                        "slug": slug,
                        "unique_values": values,
                        "labels": {str(v): id_to_label[v] for v in values if id_to_label and v in id_to_label},
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            return True
        except Exception as e:
            print(f"   ❌ 导出离散通道{channel}失败: {e}")
            return False

    def export_triplet_channel_artifacts(self, wsg_array, wsg_path, output_dir, channels, slug):
        try:
            from PIL import Image

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            uuid = Path(wsg_path).stem
            triplet = wsg_array[:, :, channels].astype(np.float32)
            valid_mask = np.all(triplet != -1, axis=2)
            if not np.any(valid_mask):
                return False

            rgb_base = reader_rgb_base(wsg_array)
            triplet_rgb = np.zeros_like(rgb_base)
            triplet_rgb[valid_mask] = np.clip(triplet[valid_mask], 0, 1) * 255
            triplet_rgb = triplet_rgb.astype(np.uint8)
            overlay = self.blend_overlay(rgb_base, triplet_rgb, valid_mask)

            prefix = output_dir / f"{uuid}_channel{channels[0]}_{channels[-1]}_{slug}"
            Image.fromarray(triplet_rgb).save(f"{prefix}_rgb.png")
            Image.fromarray(overlay).save(f"{prefix}_overlay.png")
            np.savez_compressed(
                f"{prefix}_map.npz",
                channel_map=triplet.astype(np.float32),
            )
            with open(f"{prefix}_stats.json", "w", encoding="utf-8") as f:
                stats = self.aggregate_triplet_channels(wsg_array, channels)
                json.dump(
                    {
                        "channels": channels,
                        "slug": slug,
                        "count": stats["count"] if stats else 0,
                        "mean": stats["mean"].tolist() if stats else None,
                        "median": stats["median"].tolist() if stats else None,
                        "min": stats["min"].tolist() if stats else None,
                        "max": stats["max"].tolist() if stats else None,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            return True
        except Exception as e:
            print(f"   ❌ 导出三通道{channels}失败: {e}")
            return False

    def export_scalar_channel_artifacts(self, wsg_array, wsg_path, output_dir, channel, slug):
        try:
            from PIL import Image

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            uuid = Path(wsg_path).stem
            value_map = wsg_array[:, :, channel].astype(np.float32)
            valid_mask = np.isfinite(value_map) & (value_map != -1)
            valid_mask = refine_reader_valid_mask(value_map, valid_mask, channel)
            if not np.any(valid_mask):
                return False

            valid_values = value_map[valid_mask]
            vmin = float(np.min(valid_values))
            vmax = float(np.max(valid_values))
            if int(channel) == 150:
                heatmap = np.rint(
                    render_transparency_brightness(value_map, valid_mask) * 255.0
                ).astype(np.uint8)
                display_range = (0.0, 1.0)
            else:
                normalized, display_range = normalize_scalar_for_display(
                    value_map,
                    valid_mask,
                    VISUALIZATION_CHANNEL_RANGES.get(int(channel)),
                )
                heatmap = apply_continuous_palette(normalized, channel, valid_mask)

            rgb_base = reader_rgb_base(wsg_array)
            overlay = self.blend_overlay(rgb_base, heatmap, valid_mask)

            prefix = output_dir / f"{uuid}_channel{channel}_{slug}"
            Image.fromarray(heatmap).save(f"{prefix}_map.png")
            Image.fromarray(overlay).save(f"{prefix}_overlay.png")
            np.savez_compressed(
                f"{prefix}_map.npz",
                channel_map=value_map[:, :, None].astype(np.float32),
            )
            with open(f"{prefix}_stats.json", "w", encoding="utf-8") as f:
                stats = self.aggregate_continuous_channel(wsg_array, channel)
                json.dump(
                    {
                        "channel": channel,
                        "slug": slug,
                        **(stats or {}),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            return True
        except Exception as e:
            print(f"   ❌ 导出单通道{channel}失败: {e}")
            return False

    def export_expanded_property_maps(self, wsg_array, wsg_path, output_dir):
        """导出扩散后的属性map可视化和NPZ"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        material_labels = self.load_material_id_to_name(wsg_path)
        category_labels = self.load_category_id_to_name(wsg_path)
        region_labels = self.load_channel143_region_mapping(wsg_path)

        discrete_specs = [
            (131, "material_id", material_labels),
            (135, "is_container_like", {0: "false", 1: "true"}),
            (136, "category_id", category_labels),
            (143, "material_region_id", region_labels),
        ]
        scalar_specs = [
            (126, "relative_solid_volume"),
            (127, "relative_surface_volume"),
            (128, "relative_weight"),
            (129, "support_surface_area"),
            (130, "relative_static_friction_force"),
            (132, "material_confidence"),
            (133, "density"),
            (134, "static_friction_coeff"),
            (150, "opacity"),
            (154, "shininess"),
        ]
        triplet_specs = [
            ([144, 145, 146], "ambient_rgb"),
            ([147, 148, 149], "diffuse_rgb"),
            ([151, 152, 153], "specular_rgb"),
        ]

        for channel, slug, labels in discrete_specs:
            if channel == 143:
                self.export_channel143_region_artifacts(wsg_array, wsg_path, output_dir)
            else:
                self.export_discrete_channel_artifacts(
                    wsg_array=wsg_array,
                    wsg_path=wsg_path,
                    output_dir=output_dir,
                    channel=channel,
                    slug=slug,
                    id_to_label=labels,
                )

        for channel, slug in scalar_specs:
            self.export_scalar_channel_artifacts(
                wsg_array=wsg_array,
                wsg_path=wsg_path,
                output_dir=output_dir,
                channel=channel,
                slug=slug,
            )

        for channels, slug in triplet_specs:
            self.export_triplet_channel_artifacts(
                wsg_array=wsg_array,
                wsg_path=wsg_path,
                output_dir=output_dir,
                channels=channels,
                slug=slug,
            )
    
    def create_comprehensive_visualization(self, wsg_array, output_dir):
        """创建综合可视化图表"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            
            plt.style.use('default')
            fig = plt.figure(figsize=(20, 16), facecolor='white')
            
            # 创建网格布局
            gs = gridspec.GridSpec(3, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1, 1],
                                 wspace=0.3, hspace=0.3)
            
            # 1. WSG热图 (左上角，占2x2)
            ax1 = fig.add_subplot(gs[0:2, 0])
            self.plot_wsg_heatmap(ax1, wsg_array)
            
            # 2. 物理属性可视化 (右上)
            ax2 = fig.add_subplot(gs[0, 1])
            self.plot_physics_properties(ax2, wsg_array)
            
            # 3. 材质属性可视化 (右中)
            ax3 = fig.add_subplot(gs[0, 2])
            self.plot_material_properties(ax3, wsg_array)
            
            # 4. 通道使用情况 (中右)
            ax4 = fig.add_subplot(gs[1, 1])
            self.plot_channel_usage(ax4, wsg_array)
            
            # 5. 标签分布 (中最右)
            ax5 = fig.add_subplot(gs[1, 2])
            self.plot_label_distribution(ax5, wsg_array)
            
            # 6. 骨骼信息 (底部左)
            ax6 = fig.add_subplot(gs[2, 0])
            self.plot_skeleton_info(ax6, wsg_array)
            
            # 7. RGB颜色分析 (底部中)
            ax7 = fig.add_subplot(gs[2, 1])
            self.plot_color_analysis(ax7, wsg_array)
            
            # 8. 统计摘要 (底部右)
            ax8 = fig.add_subplot(gs[2, 2])
            self.plot_statistics_summary(ax8, wsg_array)
            
            # 添加总标题
            fig.suptitle('WSG Comprehensive Data Visualization', fontsize=20, y=0.98)
            
            # 保存图片
            viz_path = os.path.join(output_dir, "wsg_comprehensive_visualization.png")
            plt.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"   📊 保存综合可视化: {viz_path}")
            return viz_path
            
        except Exception as e:
            print(f"   ❌ 创建可视化失败: {e}")
            return None
    
    def plot_wsg_heatmap(self, ax, wsg_array):
        """绘制WSG数据热图"""
        # 选择有意义的通道进行可视化
        channels_to_show = [
            (0, "Red"), (1, "Green"), (2, "Blue"),
            (111, "Object ID"), (112, "Class Value"), 
            (126, "Solid Volume"), (127, "Surface Area"),
            (128, "Weight"), (131, "Material ID"),
            (143, "Material Region ID")
        ]
        
        # 创建热图数据
        heatmap_data = []
        labels = []
        
        for ch_idx, ch_name in channels_to_show:
            if ch_idx < wsg_array.shape[2]:
                channel_data = wsg_array[:, :, ch_idx]
                # 取中心区域的平均值
                center_value = np.mean(channel_data[300:420, 300:420])
                heatmap_data.append(center_value)
                labels.append(f"Ch{ch_idx}: {ch_name}")
        
        # 绘制
        im = ax.imshow([heatmap_data], cmap='viridis', aspect='auto')
        ax.set_yticks([0])
        ax.set_yticklabels(['Channel Values'])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title('WSG Channel Heatmap', fontsize=14)
        
        # 添加数值标签
        for i, val in enumerate(heatmap_data):
            ax.text(i, 0, f'{val:.1f}', ha='center', va='center', color='white' if val < 0.5 else 'black')
    
    def plot_physics_properties(self, ax, wsg_array):
        """绘制物理属性"""
        properties = {
            'Solid Volume': wsg_array[0, 0, 126],
            'Surface Area': wsg_array[0, 0, 127],
            'Weight': wsg_array[0, 0, 128],
            'Support Area': wsg_array[0, 0, 129],
            'Friction Force': wsg_array[0, 0, 130],
            'Density': wsg_array[0, 0, 133],
            'Friction Coeff': wsg_array[0, 0, 134]
        }
        
        # 过滤有效值
        valid_props = {k: v for k, v in properties.items() if v != -1}
        
        if valid_props:
            bars = ax.bar(range(len(valid_props)), list(valid_props.values()))
            ax.set_xticks(range(len(valid_props)))
            ax.set_xticklabels(list(valid_props.keys()), rotation=45, ha='right')
            ax.set_ylabel('Value')
            ax.set_title('Physical Properties', fontsize=14)
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, valid_props.values())):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{val:.1f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No Physical Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Physical Properties', fontsize=14)
    
    def plot_material_properties(self, ax, wsg_array):
        """绘制材质属性"""
        # 提取材质颜色
        Ka = wsg_array[0, 0, 144:147]
        Kd = wsg_array[0, 0, 147:150]
        Ks = wsg_array[0, 0, 151:154]
        
        colors = ['Ambient (Ka)', 'Diffuse (Kd)', 'Specular (Ks)']
        color_values = [Ka, Kd, Ks]
        
        has_material = any(np.any(cv != -1) for cv in color_values)
        
        if has_material:
            # 创建颜色条 - 使用barh而不是Rectangle
            for i, (name, rgb) in enumerate(zip(colors, color_values)):
                if np.all(rgb == -1):
                    rgb = [0.5, 0.5, 0.5]  # 灰色表示无数据
                else:
                    rgb = np.clip(rgb, 0, 1)
                
                # 使用横向条形图代替Rectangle
                ax.barh(i, 1, height=0.8, color=rgb, edgecolor='black', linewidth=0.5)
                ax.text(1.1, i, f'{name}', va='center', fontsize=10)
                ax.text(-0.1, i, f'[{rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f}]', 
                       ha='right', va='center', fontsize=8)
            
            ax.set_xlim(-0.5, 2)
            ax.set_ylim(-0.5, len(colors) - 0.5)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.axis('off')
            ax.set_title('Material Colors', fontsize=14)
        else:
            ax.text(0.5, 0.5, 'No Material Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            ax.set_title('Material Colors', fontsize=14)
    
    def plot_channel_usage(self, ax, wsg_array):
        """绘制通道使用情况"""
        # 统计每个通道的非-1值数量
        channel_usage = []
        for i in range(0, 256, 5):  # 每5个通道采样一次
            if i < wsg_array.shape[2]:
                non_empty = np.sum(wsg_array[:, :, i] != -1)
                channel_usage.append(non_empty)
        
        ax.plot(range(0, len(channel_usage)*5, 5), channel_usage, 'b-', linewidth=2)
        ax.fill_between(range(0, len(channel_usage)*5, 5), channel_usage, alpha=0.3)
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Non-empty Pixels')
        ax.set_title('Channel Usage Pattern', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 255)
    
    def plot_label_distribution(self, ax, wsg_array):
        """绘制标签分布"""
        # 统计object_id分布
        object_ids = wsg_array[:, :, 111]
        unique_ids = np.unique(object_ids[object_ids != -1])
        
        if len(unique_ids) > 0:
            counts = [np.sum(object_ids == uid) for uid in unique_ids]
            ax.pie(counts, labels=[f'ID {int(uid)}' for uid in unique_ids[:5]], autopct='%1.1f%%')
            ax.set_title('Object ID Distribution', fontsize=14)
        else:
            ax.text(0.5, 0.5, 'No Label Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Object ID Distribution', fontsize=14)
    
    def plot_skeleton_info(self, ax, wsg_array):
        """绘制骨骼信息"""
        # 统计骨骼数据
        joint_count = np.sum(wsg_array[:, :, 112] == 3)  # Class Value = 3 for joints
        bone_count = np.sum(wsg_array[:, :, 112] == 4)   # Class Value = 4 for bones
        
        data = {
            'Joints': joint_count,
            'Bones': bone_count,
            'Total Points': joint_count + bone_count
        }
        
        bars = ax.bar(data.keys(), data.values())
        ax.set_ylabel('Count')
        ax.set_title('Skeleton Information', fontsize=14)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{int(height)}', ha='center', va='bottom')
    
    def plot_color_analysis(self, ax, wsg_array):
        """分析RGB颜色通道"""
        # 提取RGB值
        r_channel = wsg_array[:, :, 0]
        g_channel = wsg_array[:, :, 1]
        b_channel = wsg_array[:, :, 2]
        
        # 计算有效区域的平均颜色
        valid_mask = (r_channel != -1) & (g_channel != -1) & (b_channel != -1)
        
        if np.any(valid_mask):
            avg_r = np.mean(r_channel[valid_mask])
            avg_g = np.mean(g_channel[valid_mask])
            avg_b = np.mean(b_channel[valid_mask])
            
            # 绘制颜色直方图
            colors = ['red', 'green', 'blue']
            values = [avg_r, avg_g, avg_b]
            bars = ax.bar(colors, values, color=colors, alpha=0.7)
            
            ax.set_ylim(0, 1)
            ax.set_ylabel('Average Intensity')
            ax.set_title('RGB Color Analysis', fontsize=14)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No Color Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('RGB Color Analysis', fontsize=14)
    
    def plot_statistics_summary(self, ax, wsg_array):
        """绘制统计摘要"""
        stats_text = []
        
        # 基本统计
        total_pixels = wsg_array.shape[0] * wsg_array.shape[1]
        non_empty_pixels = np.sum(wsg_array[:, :, 111] != -1)  # 使用object_id通道
        
        stats_text.append(f"Image Size: {wsg_array.shape[0]}×{wsg_array.shape[1]}")
        stats_text.append(f"Total Channels: {wsg_array.shape[2]}")
        stats_text.append(f"Non-empty Pixels: {non_empty_pixels:,} ({non_empty_pixels/total_pixels*100:.1f}%)")
        
        # 数据范围
        valid_data = wsg_array[wsg_array != -1]
        if len(valid_data) > 0:
            stats_text.append(f"Data Range: [{np.min(valid_data):.2f}, {np.max(valid_data):.2f}]")
            stats_text.append(f"Mean Value: {np.mean(valid_data):.2f}")
            stats_text.append(f"Std Dev: {np.std(valid_data):.2f}")
        
        # 显示统计文本
        y_pos = 0.9
        for text in stats_text:
            ax.text(0.1, y_pos, text, transform=ax.transAxes, fontsize=11, 
                   family='monospace')
            y_pos -= 0.12
        
        ax.axis('off')
        ax.set_title('Statistics Summary', fontsize=14)
    
    def create_property_progress_visualization(self, wsg_array, output_dir):
        """创建物理和材质属性的进度条可视化"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # 物理属性进度条
            self.plot_physics_progress_bars(ax1, wsg_array)
            
            # 材质属性进度条
            self.plot_material_progress_bars(ax2, wsg_array)
            
            plt.tight_layout()
            
            # 保存图片
            progress_path = os.path.join(output_dir, "property_progress_bars.png")
            plt.savefig(progress_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"   [DONE] Saved property progress bars: {progress_path}")
            return progress_path
            
        except Exception as e:
            print(f"   ❌ 创建属性进度条失败: {e}")
            return None
    
    def plot_physics_progress_bars(self, ax, wsg_array):
        """绘制物理属性的进度条"""
        # 定义物理属性及其范围 - 使用ASCII标签
        physics_properties = [
            ('[VOL] Relative Volume', wsg_array[0, 0, 126], 0, 100, 'Vol'),
            ('[AREA] Relative Surface Area', wsg_array[0, 0, 127], 0, 100, 'Area'),
            ('[WEIGHT] Relative Weight', wsg_array[0, 0, 128], 0, 280, 'Wgt'),  # 最大可能值: 100 * 2.8
            ('[SUP] Relative Support Area', wsg_array[0, 0, 129], 0, 100, 'Sup'),
            ('[FRIC] Relative Static Friction', wsg_array[0, 0, 130], 0, 70, 'Fric'),  # 最大可能值: 100 * 0.7
            ('[DENS] Density (kg/L)', wsg_array[0, 0, 133], 0.192, 2.8, 'Dens'),
            ('[MU] Static Friction Coeff', wsg_array[0, 0, 134], 0.2, 0.7, 'Mu'),
        ]
        
        # 过滤有效值
        valid_properties = [(name, val, min_val, max_val, icon) 
                           for name, val, min_val, max_val, icon in physics_properties 
                           if val != -1]
        
        if valid_properties:
            y_positions = range(len(valid_properties))
            
            # 绘制背景进度条
            for i, (name, val, min_val, max_val, icon) in enumerate(valid_properties):
                # 背景条
                ax.barh(i, 1, height=0.6, color='lightgray', edgecolor='gray')
                
                # 实际值进度条
                if max_val > min_val:
                    progress = (val - min_val) / (max_val - min_val)
                    color = self.get_color_by_progress(progress)
                    ax.barh(i, progress, height=0.6, color=color, edgecolor='darkgray')
                    
                    # 标签 - 不使用icon，直接使用name
                    ax.text(-0.02, i, f'{name}:', ha='right', va='center', fontsize=11)
                    
                    # 值标签
                    ax.text(1.05, i, f'{val:.1f}', ha='left', va='center', fontsize=10, fontweight='bold')
                    
                    # 范围标签
                    ax.text(0, i-0.35, f'{min_val:.1f}', ha='center', va='center', fontsize=8, color='gray')
                    ax.text(1, i-0.35, f'{max_val:.1f}', ha='center', va='center', fontsize=8, color='gray')
                    
                    # 百分比标签
                    percentage = progress * 100
                    ax.text(0.5, i, f'{percentage:.1f}%', ha='center', va='center', 
                           fontsize=9, color='white' if progress > 0.5 else 'black')
            
            ax.set_xlim(-0.5, 1.2)
            ax.set_ylim(-0.5, len(valid_properties) - 0.5)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title('Physical Properties - Relative Position', fontsize=16, pad=20)
            
            # 添加说明文字
            ax.text(0.5, -0.8, 'Progress bars show the relative position of each property within its valid range', 
                   ha='center', va='center', transform=ax.transData, fontsize=10, style='italic')
        else:
            ax.text(0.5, 0.5, 'No Physical Properties Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            ax.set_title('Physical Properties', fontsize=16)
    
    def plot_material_progress_bars(self, ax, wsg_array):
        """绘制材质属性的进度条"""
        # 提取材质数据
        material_properties = []
        
        # 颜色通道
        Ka = wsg_array[0, 0, 144:147]
        Kd = wsg_array[0, 0, 147:150]
        Ks = wsg_array[0, 0, 151:154]
        
        # 环境光强度
        if np.any(Ka != -1):
            ka_intensity = np.mean(Ka[Ka != -1])
            material_properties.append(('[Ka] Ambient Intensity', ka_intensity, 0, 1, 'Ka'))
        
        # 漫反射强度
        if np.any(Kd != -1):
            kd_intensity = np.mean(Kd[Kd != -1])
            material_properties.append(('[Kd] Diffuse Intensity', kd_intensity, 0, 1, 'Kd'))
        
        # 镜面反射强度
        if np.any(Ks != -1):
            ks_intensity = np.mean(Ks[Ks != -1])
            material_properties.append(('[Ks] Specular Intensity', ks_intensity, 0, 1, 'Ks'))
        
        # 不透明度
        opacity = wsg_array[0, 0, 150]
        if opacity != -1:
            material_properties.append(('[d] Opacity', opacity, 0, 1, 'd'))
        
        # 高光指数
        shininess = wsg_array[0, 0, 154]
        if shininess != -1:
            material_properties.append(('[Ns] Shininess', shininess, 0, 1000, 'Ns'))
        
        if material_properties:
            y_positions = range(len(material_properties))
            
            # 绘制进度条
            for i, (name, val, min_val, max_val, icon) in enumerate(material_properties):
                # 背景条
                ax.barh(i, 1, height=0.6, color='lightgray', edgecolor='gray')
                
                # 实际值进度条
                if max_val > min_val:
                    progress = (val - min_val) / (max_val - min_val)
                    color = self.get_color_by_progress(progress)
                    ax.barh(i, progress, height=0.6, color=color, edgecolor='darkgray')
                    
                    # 标签 - 不使用icon，直接使用name
                    ax.text(-0.02, i, f'{name}:', ha='right', va='center', fontsize=11)
                    
                    # 值标签
                    if max_val > 100:
                        ax.text(1.05, i, f'{val:.0f}', ha='left', va='center', fontsize=10, fontweight='bold')
                    else:
                        ax.text(1.05, i, f'{val:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
                    
                    # 范围标签
                    ax.text(0, i-0.35, f'{min_val}', ha='center', va='center', fontsize=8, color='gray')
                    ax.text(1, i-0.35, f'{max_val}', ha='center', va='center', fontsize=8, color='gray')
                    
                    # 百分比标签
                    percentage = progress * 100
                    ax.text(0.5, i, f'{percentage:.1f}%', ha='center', va='center', 
                           fontsize=9, color='white' if progress > 0.5 else 'black')
            
            # 绘制颜色预览
            if np.any(Ka != -1) or np.any(Kd != -1) or np.any(Ks != -1):
                y_offset = len(material_properties) + 0.5
                colors_to_show = [
                    ('Ka', Ka, 'Ambient'),
                    ('Kd', Kd, 'Diffuse'),
                    ('Ks', Ks, 'Specular')
                ]
                
                for j, (label, color, desc) in enumerate(colors_to_show):
                    if np.any(color != -1):
                        rgb = np.clip(color, 0, 1) if np.any(color != -1) else [0.5, 0.5, 0.5]
                        ax.barh(y_offset + j*0.3, 0.2, height=0.25, 
                               color=rgb, edgecolor='black', linewidth=0.5)
                        ax.text(0.25, y_offset + j*0.3, f'{desc}: RGB({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})', 
                               va='center', fontsize=9)
                
                ax.set_ylim(-0.5, y_offset + 1)
            else:
                ax.set_ylim(-0.5, len(material_properties) - 0.5)
            
            ax.set_xlim(-0.5, 1.2)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title('Material Properties - Relative Position', fontsize=16, pad=20)
        else:
            ax.text(0.5, 0.5, 'No Material Properties Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            ax.set_title('Material Properties', fontsize=16)
    
    def get_color_by_progress(self, progress):
        """根据进度值返回颜色"""
        if progress < 0.25:
            return '#2ecc71'  # 绿色
        elif progress < 0.5:
            return '#f39c12'  # 橙色
        elif progress < 0.75:
            return '#e74c3c'  # 红色
        else:
            return '#9b59b6'  # 紫色
    
    def create_property_progress_with_references(self, wsg_array, output_dir):
        """创建带参考物的属性进度条可视化"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            
            # 物理属性进度条（带参考物）
            self.plot_physics_progress_bars_with_refs(ax1, wsg_array)
            
            # 材质属性进度条（带参考物）
            self.plot_material_progress_bars_with_refs(ax2, wsg_array)
            
            plt.tight_layout()
            
            # 保存图片
            progress_path = os.path.join(output_dir, "property_progress_bars_with_references.png")
            plt.savefig(progress_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"   [DONE] Saved property progress bars with references: {progress_path}")
            return progress_path
            
        except Exception as e:
            print(f"   ❌ 创建带参考物的进度条失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_physics_progress_bars_with_refs(self, ax, wsg_array):
        """绘制带参考物的物理属性进度条"""
        # 定义物理属性及其范围和参考物
        physics_properties = [
            {
                'name': '[VOL] Relative Volume',
                'value': float(wsg_array[0, 0, 126]),
                'range': [0, 100],
                'references': [
                    (5, 'Mouse'),
                    (15, 'Phone'),
                    (30, 'Skateboard'),
                    (50, 'Chair'),
                    (70, 'Desk'),
                    (90, 'Sofa')
                ]
            },
            {
                'name': '[AREA] Relative Surface Area',
                'value': float(wsg_array[0, 0, 127]),
                'range': [0, 100],
                'references': [
                    (10, 'Cup'),
                    (30, 'Plate'),
                    (50, 'Table'),
                    (80, 'Carpet')
                ]
            },
            {
                'name': '[WEIGHT] Relative Weight',
                'value': float(wsg_array[0, 0, 128]),
                'range': [0, 280],
                'references': [
                    (1, 'Pen'),
                    (10, 'Book'),
                    (50, 'Chair'),
                    (100, 'Desk'),
                    (200, 'Refrigerator')
                ]
            },
            {
                'name': '[DENS] Density (kg/L)',
                'value': float(wsg_array[0, 0, 133]),
                'range': [0.192, 2.8],
                'references': [
                    (0.5, 'Wood'),
                    (1.0, 'Water'),
                    (1.5, 'Glass'),
                    (2.7, 'Aluminum')
                ]
            },
            {
                'name': '[MU] Static Friction Coeff',
                'value': float(wsg_array[0, 0, 134]),
                'range': [0.2, 0.7],
                'references': [
                    (0.2, 'Ice'),
                    (0.3, 'Metal'),
                    (0.5, 'Wood'),
                    (0.7, 'Rubber')
                ]
            }
        ]
        
        # 过滤有效值
        valid_properties = [prop for prop in physics_properties if prop['value'] != -1]
        
        if valid_properties:
            bar_height = 0.7
            y_spacing = 1.5
            
            for i, prop in enumerate(valid_properties):
                y_pos = i * y_spacing
                value = prop['value']
                min_val, max_val = prop['range']
                
                # 计算进度
                if max_val > min_val:
                    progress = (value - min_val) / (max_val - min_val)
                else:
                    progress = 0
                
                # 绘制背景条
                ax.barh(y_pos, 1, height=bar_height, color='lightgray', edgecolor='black', linewidth=1)
                
                # 绘制进度条
                color = self.get_color_by_progress(progress)
                ax.barh(y_pos, progress, height=bar_height, color=color, alpha=0.8, edgecolor='black', linewidth=1)
                
                # 属性名称
                ax.text(-0.02, y_pos, f"{prop['name']}:", ha='right', va='center', fontsize=12, fontweight='bold')
                
                # 当前值
                value_text = f"{value:.2f}"
                if progress > 0.15:
                    ax.text(progress/2, y_pos, value_text, ha='center', va='center', 
                           fontsize=11, color='white', fontweight='bold')
                else:
                    ax.text(progress + 0.02, y_pos, value_text, ha='left', va='center',
                           fontsize=11, color='black', fontweight='bold')
                
                # 百分比
                percentage = progress * 100
                ax.text(1.02, y_pos, f"{percentage:.1f}%", ha='left', va='center', 
                       fontsize=10, color='darkgreen', fontweight='bold')
                
                # 范围
                ax.text(0, y_pos - bar_height/2 - 0.1, f"{min_val:.1f}", ha='center', va='top', fontsize=9, color='gray')
                ax.text(1, y_pos - bar_height/2 - 0.1, f"{max_val:.1f}", ha='center', va='top', fontsize=9, color='gray')
                
                # 添加参考物标记
                for ref_val, ref_name in prop.get('references', []):
                    if min_val <= ref_val <= max_val:
                        ref_progress = (ref_val - min_val) / (max_val - min_val)
                        # 添加垂直线标记
                        y_bottom = (y_pos - bar_height/2) / (len(valid_properties) * y_spacing)
                        y_top = (y_pos + bar_height/2) / (len(valid_properties) * y_spacing)
                        ax.axvline(ref_progress, ymin=y_bottom, ymax=y_top, 
                                  color='red', linestyle='--', alpha=0.5, linewidth=1)
                        # 添加参考物标签
                        ax.text(ref_progress, y_pos + bar_height/2 + 0.15, ref_name, 
                               ha='center', va='bottom', fontsize=8, color='darkred',
                               rotation=45)
            
            # 设置轴
            ax.set_xlim(-0.6, 1.15)
            ax.set_ylim(-0.5, len(valid_properties) * y_spacing)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title('Physical Properties with Reference Objects', fontsize=16, fontweight='bold', pad=20)
            
            # 添加说明
            ax.text(0.5, -0.4, 'Red markers show reference objects for comparison', 
                   ha='center', va='center', transform=ax.transData, fontsize=10, 
                   style='italic', color='darkred')
            
            # 隐藏边框
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            ax.text(0.5, 0.5, 'No Physical Properties Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.axis('off')
    
    def plot_material_progress_bars_with_refs(self, ax, wsg_array):
        """绘制带参考物的材质属性进度条"""
        # 提取材质数据
        material_properties = []
        
        # 材质属性及其参考物
        Ka = wsg_array[0, 0, 144:147]
        Kd = wsg_array[0, 0, 147:150]
        Ks = wsg_array[0, 0, 151:154]
        
        # 环境光强度
        if np.any(Ka != -1):
            ka_intensity = np.mean(Ka[Ka != -1])
            material_properties.append({
                'name': '[Ka] Ambient Intensity',
                'value': ka_intensity,
                'range': [0, 1],
                'references': [
                    (0.1, 'Dark Room'),
                    (0.3, 'Indoor'),
                    (0.7, 'Daylight'),
                    (0.9, 'Bright Studio')
                ]
            })
        
        # 漫反射强度
        if np.any(Kd != -1):
            kd_intensity = np.mean(Kd[Kd != -1])
            material_properties.append({
                'name': '[Kd] Diffuse Intensity',
                'value': kd_intensity,
                'range': [0, 1],
                'references': [
                    (0.2, 'Velvet'),
                    (0.5, 'Matte Paint'),
                    (0.8, 'Paper'),
                    (0.95, 'White Plaster')
                ]
            })
        
        # 镜面反射强度
        if np.any(Ks != -1):
            ks_intensity = np.mean(Ks[Ks != -1])
            material_properties.append({
                'name': '[Ks] Specular Intensity',
                'value': ks_intensity,
                'range': [0, 1],
                'references': [
                    (0.1, 'Matte'),
                    (0.3, 'Satin'),
                    (0.6, 'Glossy'),
                    (0.9, 'Mirror')
                ]
            })
        
        # 不透明度
        opacity = wsg_array[0, 0, 150]
        if opacity != -1:
            material_properties.append({
                'name': '[d] Opacity',
                'value': opacity,
                'range': [0, 1],
                'references': [
                    (0.1, 'Clear Glass'),
                    (0.5, 'Frosted Glass'),
                    (0.8, 'Translucent'),
                    (1.0, 'Opaque')
                ]
            })
        
        # 高光指数
        shininess = wsg_array[0, 0, 154]
        if shininess != -1:
            material_properties.append({
                'name': '[Ns] Shininess',
                'value': shininess,
                'range': [0, 1000],
                'references': [
                    (10, 'Rough'),
                    (100, 'Satin'),
                    (500, 'Polished'),
                    (900, 'Chrome')
                ]
            })
        
        if material_properties:
            bar_height = 0.7
            y_spacing = 1.5
            
            for i, prop in enumerate(material_properties):
                y_pos = i * y_spacing
                value = prop['value']
                min_val, max_val = prop['range']
                
                # 计算进度
                if max_val > min_val:
                    progress = (value - min_val) / (max_val - min_val)
                else:
                    progress = 0
                
                # 绘制背景条
                ax.barh(y_pos, 1, height=bar_height, color='lightgray', edgecolor='black', linewidth=1)
                
                # 绘制进度条
                color = self.get_color_by_progress(progress)
                ax.barh(y_pos, progress, height=bar_height, color=color, alpha=0.8, edgecolor='black', linewidth=1)
                
                # 属性名称
                ax.text(-0.02, y_pos, f"{prop['name']}:", ha='right', va='center', fontsize=12, fontweight='bold')
                
                # 当前值
                if max_val > 100:
                    value_text = f"{value:.0f}"
                else:
                    value_text = f"{value:.3f}"
                    
                if progress > 0.15:
                    ax.text(progress/2, y_pos, value_text, ha='center', va='center', 
                           fontsize=11, color='white', fontweight='bold')
                else:
                    ax.text(progress + 0.02, y_pos, value_text, ha='left', va='center',
                           fontsize=11, color='black', fontweight='bold')
                
                # 百分比
                percentage = progress * 100
                ax.text(1.02, y_pos, f"{percentage:.1f}%", ha='left', va='center', 
                       fontsize=10, color='darkgreen', fontweight='bold')
                
                # 范围
                ax.text(0, y_pos - bar_height/2 - 0.1, f"{min_val}", ha='center', va='top', fontsize=9, color='gray')
                ax.text(1, y_pos - bar_height/2 - 0.1, f"{max_val}", ha='center', va='top', fontsize=9, color='gray')
                
                # 添加参考物标记
                for ref_val, ref_name in prop.get('references', []):
                    if min_val <= ref_val <= max_val:
                        ref_progress = (ref_val - min_val) / (max_val - min_val)
                        # 添加垂直线标记
                        y_bottom = (y_pos - bar_height/2) / (len(material_properties) * y_spacing)
                        y_top = (y_pos + bar_height/2) / (len(material_properties) * y_spacing)
                        ax.axvline(ref_progress, ymin=y_bottom, ymax=y_top, 
                                  color='red', linestyle='--', alpha=0.5, linewidth=1)
                        # 添加参考物标签
                        ax.text(ref_progress, y_pos + bar_height/2 + 0.15, ref_name, 
                               ha='center', va='bottom', fontsize=8, color='darkred',
                               rotation=45)
            
            # 设置轴
            ax.set_xlim(-0.6, 1.15)
            ax.set_ylim(-0.5, len(material_properties) * y_spacing)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title('Material Properties with Reference Materials', fontsize=16, fontweight='bold', pad=20)
            
            # 隐藏边框
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            ax.text(0.5, 0.5, 'No Material Properties Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.axis('off')
    
    def extract_label_from_wsg(self, wsg_array, output_path):
        """从WSG通道111提取object_id并保存为标签文件"""
        try:
            import numpy as np
            
            height, width = wsg_array.shape[:2]
            
            # 快速检查通道111是否有有效数据 - 只需检查第一个像素
            first_pixel_value = wsg_array[0, 0, 111]
            print(f"   🔍 检查通道111第一个像素: {first_pixel_value}")
            if first_pixel_value == -1 or first_pixel_value <= 0:
                print("   ⚠️ 通道111无有效标签数据，跳过标签提取")
                return False
            
            # 收集所有唯一的object_id
            object_ids = set()
            class_counts = {1: 0, 2: 0, 3: 0, 4: 0}  # 顶点、面、关节、骨骼
            
            # 使用numpy向量化操作
            print("   📊 开始提取标签信息...")
            print(f"   🔍 检查通道112...")
            valid_mask = wsg_array[:, :, 112] > 0
            print(f"   📊 有效像素数: {np.sum(valid_mask)}")
            if np.any(valid_mask):
                valid_pixels = wsg_array[valid_mask]
                object_ids = set(valid_pixels[:, 111].astype(int))
                
                # 统计各类型数量
                for class_type in [1, 2, 3, 4]:
                    class_counts[class_type] = np.sum(valid_pixels[:, 112] == class_type)
            
            # 加载类别映射
            # 从 export_obj_from_wsg 向上找到 Articulation_results 目录
            result_dir = output_path
            while os.path.basename(result_dir) != "Articulation_results" and result_dir != "/":
                result_dir = os.path.dirname(result_dir)
            
            category_mapping_path = os.path.join(result_dir, "category_id_mapping.json")
            
            # 如果找不到，尝试从tools目录查找
            if not os.path.exists(category_mapping_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                tools_path = os.path.join(script_dir, "tools", "Objaverse", "category_id_mapping.json")
                if os.path.exists(tools_path):
                    category_mapping_path = tools_path
            
            id_to_category = {}
            if os.path.exists(category_mapping_path):
                with open(category_mapping_path, 'r') as f:
                    mapping_data = json.load(f)
                    # 反转映射
                    id_to_category = {int(v): k for k, v in mapping_data['category_to_id'].items()}
            else:
                # ShapeNetSem数据集可能没有类别映射文件
                print("   ⚠️ 未找到类别映射文件")
                return False
            
            # 写入标签文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=== WSG Label Information ===\n")
                f.write(f"WSG Shape: {wsg_array.shape}\n")
                f.write(f"\nObject IDs found: {sorted(object_ids)}\n")
                
                for obj_id in sorted(object_ids):
                    category = id_to_category.get(obj_id, "unknown")
                    f.write(f"\nObject ID: {obj_id}\n")
                    f.write(f"Category: {category}\n")
                
                f.write(f"\nData Statistics:\n")
                f.write(f"  Vertices (class=1): {class_counts.get(1, 0)} pixels\n")
                f.write(f"  Faces (class=2): {class_counts.get(2, 0)} pixels\n")
                f.write(f"  Joints (class=3): {class_counts.get(3, 0)} pixels\n")
                f.write(f"  Bones (class=4): {class_counts.get(4, 0)} pixels\n")
                
            print(f"   📝 保存标签信息: {output_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ 提取标签失败: {e}")
            return False
    
    def create_skeleton_visualization_3d(self, wsg_array, output_dir, num_views=12):
        """创建骨骼的3D可视化（多个视角）"""
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib
            matplotlib.use('Agg')  # 无GUI后端
            
            # 提取所有关节的3D位置
            joints_3d = {}
            bones = []
            
            height, width = wsg_array.shape[:2]
            
            # 使用numpy向量化操作提取关节和骨骼
            # 提取关节位置
            joint_mask = wsg_array[:, :, 112] == 3
            if np.any(joint_mask):
                joint_pixels = wsg_array[joint_mask]
                for i in range(len(joint_pixels)):
                    joint_id = int(joint_pixels[i, 113])
                    joint_x = float(joint_pixels[i, 114])
                    joint_y = float(joint_pixels[i, 115])
                    joint_z = float(joint_pixels[i, 116])
                    joints_3d[joint_id] = (joint_x, joint_y, joint_z)
            
            # 提取骨骼连接
            bone_mask = wsg_array[:, :, 112] == 4
            if np.any(bone_mask):
                bone_pixels = wsg_array[bone_mask]
                for i in range(len(bone_pixels)):
                    start_joint = int(bone_pixels[i, 114])
                    end_joint = int(bone_pixels[i, 115])
                    if start_joint in joints_3d and end_joint in joints_3d:
                        bones.append((start_joint, end_joint))
            
            if not joints_3d:
                print("   ⚠️ 未找到关节数据")
                return False
            
            # 计算边界
            joint_coords = np.array(list(joints_3d.values()))
            min_coords = joint_coords.min(axis=0)
            max_coords = joint_coords.max(axis=0)
            center = (min_coords + max_coords) / 2
            scale = max(max_coords - min_coords) * 0.6
            
            # 生成多个视角
            for i in range(num_views):
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # 计算相机角度以匹配Blender
                if i == 0:
                    # 第一个视角：front view（相机在Y轴正方向看向原点）
                    # 在matplotlib中，azim=-90使Y轴指向观察者
                    azim, elev = -90, 0
                else:
                    # 其他视角：使用球面采样（类似Blender的randomize_camera）
                    # 生成随机角度
                    np.random.seed(i)  # 确保可重复性
                    theta = np.random.uniform(0, 2 * np.pi)
                    phi = np.random.uniform(-np.pi/2, np.pi/2)
                    
                    # 转换为matplotlib的azim和elev
                    azim = np.degrees(theta)
                    elev = np.degrees(phi)
                
                ax.view_init(elev=elev, azim=azim)
                
                # 绘制骨骼（交换Y和Z轴以匹配Blender的坐标系）
                for start_id, end_id in bones:
                    start_pos = joints_3d[start_id]
                    end_pos = joints_3d[end_id]
                    # 在matplotlib中: X保持不变, Y变成-Z, Z变成Y
                    ax.plot([start_pos[0], end_pos[0]], 
                           [-start_pos[2], -end_pos[2]], 
                           [start_pos[1], end_pos[1]], 
                           'b-', linewidth=3)
                
                # 绘制关节（同样交换坐标）
                for joint_id, pos in joints_3d.items():
                    ax.scatter(pos[0], -pos[2], pos[1], c='red', s=100, marker='o')
                    ax.text(pos[0], -pos[2], pos[1], str(joint_id), fontsize=8)
                
                # 设置坐标轴（调整为新的坐标映射）
                ax.set_xlim(center[0] - scale, center[0] + scale)
                ax.set_ylim(-center[2] - scale, -center[2] + scale)
                ax.set_zlim(center[1] - scale, center[1] + scale)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Z (negated)')
                ax.set_zlabel('Y')
                ax.set_title(f'Skeleton View {i:04d}')
                
                # 保存图像（使用与Blender相同的命名格式）
                output_path = os.path.join(output_dir, f'{i:03d}.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                
            print(f"   🦴 生成了 {num_views} 个骨架视图")
            return True
            
        except Exception as e:
            print(f"   ❌ 骨骼3D可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            

    def find_components(self):
        """查找必要的组件"""
        # 首先检查是否有bpy模块
        has_bpy = False
        try:
            import bpy
            has_bpy = True
            print("✅ 检测到bpy模块")
        except ImportError:
            pass
            
        # 查找Blender
        blender_paths = ["blender", "/usr/bin/blender", "/usr/local/bin/blender"]
        blender_path = None
        if has_bpy:
            blender_path = "bpy"  # 使用bpy模块标记
        else:
            for path in blender_paths:
                try:
                    result = subprocess.run(["which", path], capture_output=True, text=True)
                    if result.returncode == 0:
                        blender_path = path
                        break
                except Exception:
                    continue

        # 查找Blender脚本
        script_paths = [
            "utils/blender_script_shapenet.py",  # 优先使用ShapeNet专用脚本
            "utils/blender_script.py",
            "main/utils/blender_script_shapenet.py",
            "main/utils/blender_script.py",
            "../main/utils/blender_script_shapenet.py",
            "../main/utils/blender_script.py",
            "blender_script_shapenet.py",
            "blender_script.py"
        ]
        blender_script = None
        for path in script_paths:
            if os.path.exists(path):
                blender_script = path
                break

        return blender_path, blender_script

    def load_wsg_file(self, wsg_path):
        """加载WSG文件"""
        try:
            if wsg_path.endswith('.wsg'):
                data = np.load(wsg_path, allow_pickle=True)
                if 'wsg_data' in data.files:
                    return data['wsg_data']
            return None
        except Exception as e:
            print(f"加载WSG文件失败: {e}")
            return None

    def load_txt_metadata(self, txt_path):
        """加载对应的TXT元数据文件"""
        try:
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                return metadata
            else:
                return None
        except Exception as e:
            return None

    def load_complete_model(self, wsg_path):
        """
        加载WSG文件和对应的TXT文件，合并还原出完整的模型数据
        实现 WSG + TXT = NPZ 的完整还原
        """
        print(f"📖 读取WSG文件: {os.path.basename(wsg_path)}")

        # 加载WSG文件
        wsg_data = self.load_wsg_file(wsg_path)
        if wsg_data is None:
            print("❌ WSG文件加载失败")
            return False

        # 加载对应的TXT文件（同一文件夹）
        txt_path = wsg_path.replace('.wsg', '.txt')
        self.metadata = self.load_txt_metadata(txt_path)
        if self.metadata:
            print(f"   加载了对应的TXT元数据")

        # 从WSG中提取几何数据
        success = self.extract_complete_data_from_wsg(wsg_data)

        if success and self.metadata:
            # 合并TXT中的元数据
            self._merge_metadata()

        return success

    def _merge_metadata(self):
        """将TXT文件中的元数据合并到当前模型数据中"""
        if not self.metadata:
            return

        # 更新关节名称
        if 'joint_names' in self.metadata and len(self.metadata['joint_names']) > 0:
            self.joint_names = self.metadata['joint_names']

        # 更新根关节索引
        if 'root_index' in self.metadata:
            self.root_index = self.metadata['root_index']

    def extract_complete_data_from_wsg(self, wsg_array):
        """从WSG数组中提取完整的模型数据，包括RGB颜色"""
        # 检查WSG格式 - 应该是HxWx256的图像格式
        if len(wsg_array.shape) != 3:
            print(f"❌ WSG数组格式错误: {wsg_array.shape}")
            return False
            
        height, width, channels = wsg_array.shape
        if channels != 256:
            print(f"⚠️ 非标准WSG通道数: {channels}，期望: 256")
        
        print(f"   📐 WSG尺寸: {height}x{width}x{channels}")

        # 重置所有数据
        self.vertices = []
        self.normals = []
        self.uvs = []
        self.faces = []
        self.vertex_colors = []  # 🎨 重置颜色数组
        self.vertex_bone_weights = []
        self.vertex_bone_indices = []
        self.joints = []
        self.bones = []
        self.joint_parent_mapping = {}
        self.joint_count = 0
        self.bone_count = 0
        self.root_index = 0

        vertex_records = []
        face_records = []
        joint_records = []
        bone_records = []

        # 使用numpy向量化操作快速扫描
        print(f"   🔍 快速扫描数据...")
        start_scan = time.time()
        
        # 找到所有有效数据的位置
        object_mask = wsg_array[:, :, 111] > 0
        if not np.any(object_mask):
            print("❌ 没有找到有效数据")
            return False
            
        # 获取有效数据的坐标
        valid_coords = np.argwhere(object_mask)
        record_types = wsg_array[valid_coords[:, 0], valid_coords[:, 1], 112]
        
        # 按类型分组
        for idx, (y, x) in enumerate(valid_coords):
            record_type = record_types[idx]
            if record_type == 1:  # Vertex Data
                vertex_records.append((y, x))
            elif record_type == 2:  # Face Data
                face_records.append((y, x))
            elif record_type == 3:  # Joint Data
                joint_records.append((y, x))
            elif record_type == 4:  # Bone Data
                bone_records.append((y, x))
        
        scan_time = time.time() - start_scan
        print(f"   ✅ 扫描完成: {scan_time:.2f}秒，找到 {len(vertex_records)} 顶点，{len(face_records)} 面")

        if len(vertex_records) == 0:
            print("❌ 没有找到有效的顶点数据")
            return False

        # 提取骨架元数据
        if len(joint_records) > 0:
            first_joint_y, first_joint_x = joint_records[0]
            packed_meta = wsg_array[first_joint_y, first_joint_x, 117]  # 新位置：通道117
            if packed_meta != -1.0:
                self.joint_count, self.bone_count, self.root_index = self.bone_unpacker.unpack_skeleton_meta(
                    packed_meta)

        # 提取关节数据
        if len(joint_records) > 0:
            for y, x in joint_records:
                joint_id = int(wsg_array[y, x, 113])
                joint_pos = [
                    float(wsg_array[y, x, 114]),
                    float(wsg_array[y, x, 115]),
                    float(wsg_array[y, x, 116])
                ]
                parent_id = int(wsg_array[y, x, 118])  # 新位置：通道118
                if parent_id == -1:
                    parent_id = None

                self.joints.append(joint_pos)
                self.joint_parent_mapping[joint_id] = parent_id

        # 提取骨骼数据
        if len(bone_records) > 0:
            for y, x in bone_records:
                bone_id = int(wsg_array[y, x, 113])
                start_joint = int(wsg_array[y, x, 114])
                end_joint = int(wsg_array[y, x, 115])
                bone_length = float(wsg_array[y, x, 116])

                bone_info = {
                    'id': bone_id,
                    'start_joint': start_joint,
                    'end_joint': end_joint,
                    'length': bone_length
                }
                self.bones.append(bone_info)

        # 提取顶点数据（包括RGB颜色）
        vertex_id_mapping = {}
        colors_found = 0
        print(f"   📊 开始提取 {len(vertex_records)} 个顶点...")
        start_vertex = time.time()

        for new_idx, (y, x) in enumerate(vertex_records):
            if new_idx % 10000 == 0 and new_idx > 0:
                elapsed = time.time() - start_vertex
                print(f"      进度: {new_idx}/{len(vertex_records)} ({new_idx/len(vertex_records)*100:.1f}%), 耗时: {elapsed:.1f}s")
            original_vertex_id = int(wsg_array[y, x, 113])

            # 🎨 提取RGB颜色（通道 0, 1, 2）
            color_r = float(wsg_array[y, x, 0])
            color_g = float(wsg_array[y, x, 1])
            color_b = float(wsg_array[y, x, 2])

            # 检查是否有有效的颜色数据
            if color_r != -1.0 or color_g != -1.0 or color_b != -1.0:
                colors_found += 1
                # 确保颜色值在[0,1]范围内
                color_r = max(0.0, min(1.0, color_r))
                color_g = max(0.0, min(1.0, color_g))
                color_b = max(0.0, min(1.0, color_b))
                self.vertex_colors.append([color_r, color_g, color_b])
            else:
                # 使用默认浅灰色
                self.vertex_colors.append([0.8, 0.8, 0.8])

            # 位置
            pos_x = float(wsg_array[y, x, 114])
            pos_y = float(wsg_array[y, x, 115])
            pos_z = float(wsg_array[y, x, 116])

            # 法线
            normal_x = float(wsg_array[y, x, 36])
            normal_y = float(wsg_array[y, x, 37])
            normal_z = float(wsg_array[y, x, 38])

            # 骨骼权重和索引
            packed_weights = wsg_array[y, x, 117]
            packed_indices = wsg_array[y, x, 118]

            if packed_weights != -1.0 and packed_indices != -1.0:
                weights = self.bone_unpacker.unpack_weights(packed_weights)
                indices = self.bone_unpacker.unpack_indices(packed_indices)
            else:
                weights = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                indices = np.array([0, 0, 0, 0], dtype=np.int32)

            # UV坐标
            uv_u = float(wsg_array[y, x, 120]) if wsg_array[y, x, 120] != -1.0 else 0.0
            uv_v = float(wsg_array[y, x, 121]) if wsg_array[y, x, 121] != -1.0 else 0.0

            self.vertices.append([pos_x, pos_y, pos_z])
            self.normals.append([normal_x, normal_y, normal_z])
            self.uvs.append([uv_u, uv_v])
            self.vertex_bone_weights.append(weights)
            self.vertex_bone_indices.append(indices)

            vertex_id_mapping[original_vertex_id] = new_idx

        # 提取面数据
        print(f"   📊 开始提取 {len(face_records)} 个面...")
        start_face = time.time()
        for face_idx, (y, x) in enumerate(face_records):
            if face_idx % 10000 == 0 and face_idx > 0:
                elapsed = time.time() - start_face
                print(f"      进度: {face_idx}/{len(face_records)} ({face_idx/len(face_records)*100:.1f}%), 耗时: {elapsed:.1f}s")
            # 获取面ID（通道113）
            face_id = int(wsg_array[y, x, 113])
            # 获取三个顶点索引（通道114-116）
            vertex_id_0 = int(wsg_array[y, x, 114])
            vertex_id_1 = int(wsg_array[y, x, 115])
            vertex_id_2 = int(wsg_array[y, x, 116])

            if (vertex_id_0 in vertex_id_mapping and
                    vertex_id_1 in vertex_id_mapping and
                    vertex_id_2 in vertex_id_mapping):
                new_face = [
                    vertex_id_mapping[vertex_id_0],
                    vertex_id_mapping[vertex_id_1],
                    vertex_id_mapping[vertex_id_2]
                ]
                self.faces.append(new_face)

        # 🎨 颜色统计
        color_info = ""
        if colors_found > 0:
            color_info = f", {colors_found}/{len(self.vertices)} 顶点有颜色"
        else:
            color_info = ", 使用默认颜色"

        print(
            f"   提取数据: {len(self.vertices)} 顶点{color_info}, {len(self.faces)} 面, {len(self.joints)} 关节, {len(self.bones)} 骨骼")
        return len(self.vertices) > 0

    def create_skeleton_visualization_data(self):
        """创建骨骼和关节的可视化数据"""
        skeleton_vertices = []
        skeleton_colors = []
        skeleton_indices = []
        
        # 添加关节球体（用简单的点表示）
        for joint_pos in self.joints:
            skeleton_vertices.append(joint_pos)
            # 关节用红色表示
            skeleton_colors.append([1.0, 0.0, 0.0])
        
        # 添加骨骼线段
        base_vertex_idx = len(skeleton_vertices)
        for bone in self.bones:
            if bone['start_joint'] < len(self.joints) and bone['end_joint'] < len(self.joints):
                start_pos = self.joints[bone['start_joint']]
                end_pos = self.joints[bone['end_joint']]
                
                # 添加骨骼的两个端点
                skeleton_vertices.append(start_pos)
                skeleton_vertices.append(end_pos)
                
                # 骨骼用蓝色表示
                skeleton_colors.append([0.0, 0.0, 1.0])
                skeleton_colors.append([0.0, 0.0, 1.0])
                
                # 添加线段索引
                skeleton_indices.append([base_vertex_idx, base_vertex_idx + 1])
                base_vertex_idx += 2
        
        return skeleton_vertices, skeleton_colors, skeleton_indices

    def create_gltf(self, model_name="WSG_Model"):
        """创建glTF格式，支持顶点颜色和骨骼可视化"""
        print("🔧 创建glTF文件（包含顶点颜色和骨骼）...")

        vertices_flat = np.array(self.vertices, dtype=np.float32).flatten()
        normals_flat = np.array(self.normals, dtype=np.float32).flatten()
        uvs_flat = np.array(self.uvs, dtype=np.float32).flatten()

        # 🎨 添加顶点颜色
        colors_flat = np.array(self.vertex_colors, dtype=np.float32).flatten()

        if len(self.faces) > 0:
            indices_flat = np.array(self.faces, dtype=np.uint32).flatten()
        else:
            indices_flat = np.arange(len(self.vertices), dtype=np.uint32)

        vertices_array = np.array(self.vertices)
        min_pos = vertices_array.min(axis=0).tolist()
        max_pos = vertices_array.max(axis=0).tolist()

        binary_data = b''
        accessors = []
        buffer_views = []

        def add_accessor(data, component_type, type_name, target=None):
            nonlocal binary_data

            if component_type == 5126:  # FLOAT
                data = data.astype(np.float32)
            elif component_type == 5123:  # UNSIGNED_SHORT
                data = data.astype(np.uint16)
            elif component_type == 5125:  # UNSIGNED_INT
                data = data.astype(np.uint32)

            data_bytes = data.tobytes()
            padding = (4 - len(data_bytes) % 4) % 4
            data_bytes += b'\x00' * padding

            buffer_view = {
                "buffer": 0,
                "byteOffset": len(binary_data),
                "byteLength": len(data_bytes)
            }
            if target is not None:
                buffer_view["target"] = target

            buffer_views.append(buffer_view)

            element_size = {
                "SCALAR": 1,
                "VEC2": 2,
                "VEC3": 3,
                "VEC4": 4,
                "MAT4": 16
            }

            count = len(data) // element_size[type_name]

            accessor = {
                "bufferView": len(buffer_views) - 1,
                "componentType": component_type,
                "count": count,
                "type": type_name
            }

            binary_data += data_bytes
            accessors.append(accessor)
            return len(accessors) - 1

        # 添加访问器
        position_accessor = add_accessor(vertices_flat, 5126, "VEC3", 34962)
        accessors[position_accessor]["min"] = min_pos
        accessors[position_accessor]["max"] = max_pos

        normal_accessor = add_accessor(normals_flat, 5126, "VEC3", 34962)
        uv_accessor = add_accessor(uvs_flat, 5126, "VEC2", 34962)

        # 🎨 添加颜色访问器
        color_accessor = add_accessor(colors_flat, 5126, "VEC3", 34962)

        if len(self.vertices) < 65536:
            indices_flat = indices_flat.astype(np.uint16)
            indices_accessor = add_accessor(indices_flat, 5123, "SCALAR", 34963)
        else:
            indices_flat = indices_flat.astype(np.uint32)
            indices_accessor = add_accessor(indices_flat, 5125, "SCALAR", 34963)

        # 🎨 使用顶点颜色的材质
        material = {
            "name": f"{model_name}_Material",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],  # 白色基础，让顶点颜色显示
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8
            },
            "doubleSided": True
        }

        # 🎨 网格包含顶点颜色
        primitive = {
            "attributes": {
                "POSITION": position_accessor,
                "NORMAL": normal_accessor,
                "TEXCOORD_0": uv_accessor,
                "COLOR_0": color_accessor  # 🎨 添加顶点颜色属性
            },
            "indices": indices_accessor,
            "material": 0,
            "mode": 4  # TRIANGLES
        }

        mesh = {
            "name": f"{model_name}_Mesh",
            "primitives": [primitive]
        }

        # 节点和场景
        node = {
            "name": f"{model_name}_Node",
            "mesh": 0
        }

        scene = {
            "name": "Scene",
            "nodes": [0]
        }

        buffer = {
            "byteLength": len(binary_data)
        }

        # 组装glTF JSON
        gltf_json = {
            "asset": {
                "version": "2.0",
                "generator": "WSG-to-glTF Converter with Vertex Colors",
                "copyright": "Generated from WSG format with RGB data"
            },
            "scene": 0,
            "scenes": [scene],
            "nodes": [node],
            "meshes": [mesh],
            "materials": [material],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [buffer]
        }

        return gltf_json, binary_data

    def save_gltf_file(self, gltf_json, binary_data, output_path):
        """保存glTF文件"""
        try:
            base_path = output_path.replace('.gltf', '').replace('.glb', '')
            gltf_path = base_path + '.gltf'
            bin_path = base_path + '.bin'

            # 设置相对路径
            gltf_json['buffers'][0]['uri'] = os.path.basename(bin_path)

            # 确保目录存在
            os.makedirs(os.path.dirname(gltf_path), exist_ok=True)

            # 保存文件
            with open(gltf_path, 'w', encoding='utf-8') as f:
                json.dump(gltf_json, f, indent=2, ensure_ascii=False)

            with open(bin_path, 'wb') as f:
                f.write(binary_data)

            print(f"✅ glTF文件保存成功: {os.path.basename(gltf_path)}")
            return gltf_path
        except Exception as e:
            print(f"❌ glTF文件保存失败: {e}")
            return None

    def wsg_to_gltf(self, wsg_path: str, output_dir: str) -> Optional[str]:
        """WSG到glTF转换，保存到指定目录"""
        print(f"🔄 WSG转glTF: {os.path.basename(wsg_path)}")

        try:
            # 加载WSG文件
            if not self.load_complete_model(wsg_path):
                print("❌ WSG文件读取失败")
                return None

            # 转换为glTF
            model_name = Path(wsg_path).stem
            gltf_json, binary_data = self.create_gltf(model_name)

            gltf_path = os.path.join(output_dir, f"{model_name}.gltf")
            saved_path = self.save_gltf_file(gltf_json, binary_data, gltf_path)

            if saved_path:
                print(f"✅ WSG转换成功: {os.path.basename(saved_path)}")
                return saved_path
            else:
                print("❌ glTF保存失败")
                return None

        except Exception as e:
            print(f"❌ WSG转换异常: {e}")
            return None

    def render_gltf(self, gltf_path: str, output_dir: str, blender_path: str,
                    blender_script: str, num_renders: int = 12, resolution: int = 1024) -> bool:
        """使用Blender渲染glTF文件"""
        print(f"🎨 渲染glTF文件: {os.path.basename(gltf_path)}")
        print(f"   分辨率: {resolution}x{resolution}")
        print(f"   视角数: {num_renders}")
        print(f"📁 输出目录: {output_dir}")
        print(f"🖼️  预期生成 {num_renders} 个PNG文件")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 如果使用bpy模块，直接运行Python脚本
        if blender_path == "bpy":
            print("🐍 使用bpy模块渲染...")
            cmd = [
                sys.executable,  # 使用当前Python解释器
                blender_script,
                "--object_path", gltf_path,
                "--output_dir", output_dir,
                "--engine", "CYCLES",
                "--num_renders", str(num_renders),
                "--resolution", str(resolution)
                # 不指定fixed_camera，让脚本使用多个视角
            ]
        else:
            # 使用外部Blender程序
            cmd = [
                blender_path,
                "--background",
                "--python", blender_script,
                "--",
                "--object_path", gltf_path,
                "--output_dir", output_dir,
                "--engine", "CYCLES",
                "--num_renders", str(num_renders),
                "--resolution", str(resolution)
                # 不指定fixed_camera，让脚本使用多个视角
            ]

        print("\n🚀 开始Blender渲染...")
        print(f"📝 命令: {' '.join(cmd)}")
        print("⏳ 渲染中，请耐心等待...")
        print("=" * 50)

        start_time = time.time()

        try:
            # 使用最简单可靠的方法 - subprocess.call
            return_code = subprocess.call(cmd, timeout=1800)

            elapsed_time = time.time() - start_time
            print("=" * 50)

            if return_code == 0:
                print(f"✅ Blender进程完成 (耗时: {elapsed_time:.1f}秒)")

                # 检查生成的文件
                png_files = sorted(list(Path(output_dir).glob("*.png")))
                npy_files = sorted(list(Path(output_dir).glob("*.npy")))

                print(f"\n📊 检查生成文件:")
                print(f"   PNG文件: {len(png_files)} 个")
                print(f"   NPY文件: {len(npy_files)} 个")

                if png_files:
                    print(f"\n📸 生成的文件:")
                    for i, png_file in enumerate(png_files[:3]):  # 显示前3个
                        file_size = png_file.stat().st_size / 1024
                        print(f"   - {png_file.name} ({file_size:.1f} KB)")
                    if len(png_files) > 3:
                        print(f"   ... 还有 {len(png_files) - 3} 个文件")

                if len(png_files) >= num_renders:
                    print(f"\n🎉 渲染成功！生成了 {len(png_files)} 个图片")
                    print(f"📁 文件保存在: {output_dir}")
                    return True
                else:
                    print(f"\n⚠️  文件数量不足: 期望{num_renders}，实际{len(png_files)}")
                    return False
            else:
                print(f"❌ Blender进程失败，返回码: {return_code}")
                return False

        except subprocess.TimeoutExpired:
            print(f"\n❌ 渲染超时 (30分钟)")
            return False
        except KeyboardInterrupt:
            print(f"\n❌ 用户中断渲染")
            return False
        except Exception as e:
            print(f"\n❌ 渲染异常: {e}")
            return False

    def process_wsg_file(self, wsg_path: str, png_output_dir: str, gltf_output_dir: str,
                         num_renders: int = 12, resolution: int = 1024,
                         keep_gltf: bool = True) -> bool:
        """处理单个WSG文件的完整流程"""
        print(f"\n🚀 处理WSG文件: {os.path.basename(wsg_path)}")
        print("=" * 60)

        try:
            # 阶段0: 提取WSG的RGB通道和骨骼可视化
            print("\n🖼️ 阶段0: 提取WSG RGB通道和骨骼可视化")
            # 加载WSG数据用于RGB提取
            wsg_array = self.load_wsg_file(wsg_path)
            if wsg_array is not None:
                # 创建vis_rgb_from_wsg目录（与visualization和gltf同级）
                rgb_output_dir = os.path.dirname(png_output_dir)  # export_obj_from_wsg目录
                rgb_dir = os.path.join(rgb_output_dir, "vis_rgb_from_wsg")
                os.makedirs(rgb_dir, exist_ok=True)
                
                # 保存RGB图像
                rgb_filename = f"{Path(wsg_path).stem}_rgb.png"
                rgb_output_path = os.path.join(rgb_dir, rgb_filename)
                self.extract_rgb_from_wsg(wsg_array, rgb_output_path)
                print(f"   📁 RGB目录: {rgb_dir}")
                
                # 创建骨骼可视化目录
                rig_vis_dir = os.path.join(os.path.dirname(png_output_dir), "visualization_with_rig")
                os.makedirs(rig_vis_dir, exist_ok=True)
                
                # 创建骨骼3D可视化（12个视角）
                if not self.create_skeleton_visualization_3d(wsg_array, rig_vis_dir, num_views=num_renders):
                    print(f"   ⚠️ 跳过骨骼可视化（无骨骼数据）")
                else:
                    print(f"   📁 骨骼可视化目录: {rig_vis_dir}")
                
                # 提取并保存标签信息
                label_filename = "vis_label_from_wsg.txt"
                label_output_path = os.path.join(rgb_output_dir, label_filename)
                if not self.extract_label_from_wsg(wsg_array, label_output_path):
                    print(f"   ⚠️ 跳过标签提取（无标签数据或类别映射）")
                
                # 提取并保存物理信息
                physics_filename = "physics_properties.txt"
                # 在WSG文件所在目录创建物理信息文件
                wsg_dir = os.path.dirname(wsg_path)
                physics_output_path = os.path.join(wsg_dir, physics_filename)
                self.extract_physics_from_wsg(wsg_array, physics_output_path)
                
                # 提取并保存材质信息
                material_filename = "material_properties.txt"
                material_output_path = os.path.join(wsg_dir, material_filename)
                self.extract_material_from_wsg(wsg_array, material_output_path, wsg_path=wsg_path)
                
                # 创建综合可视化 (已禁用，只创建进度条可视化)
                # print("\n📈 阶段1: 创建综合数据可视化")
                viz_output_dir = os.path.join(os.path.dirname(wsg_path), "visualization_analysis")
                os.makedirs(viz_output_dir, exist_ok=True)
                # self.create_comprehensive_visualization(wsg_array, viz_output_dir)
                print(f"   📁 可视化目录: {viz_output_dir}")
                channel_maps_dir = os.path.join(viz_output_dir, "channel_maps")
                os.makedirs(channel_maps_dir, exist_ok=True)
                self.export_expanded_property_maps(wsg_array, wsg_path, channel_maps_dir)
                
                # 创建物理和材质属性的进度条可视化
                print("\n📊 阶段2: 创建属性进度条可视化")
                self.create_property_progress_visualization(wsg_array, viz_output_dir)
                
                # 创建带参考物的进度条可视化
                print("\n📊 阶段3: 创建带参考物的属性进度条可视化")
                self.create_property_progress_with_references(wsg_array, viz_output_dir)
            
            # 查找必要组件
            blender_path, blender_script = self.find_components()
            if not blender_path or not blender_script:
                print("\n⚠️ 缺少Blender组件，跳过3D渲染:")
                if not blender_path:
                    print("   - Blender可执行文件")
                if not blender_script:
                    print("   - Blender渲染脚本 (应在main/utils/blender_script.py)")
                print("\n✅ 但是骨骼可视化已完成!")
                return True

            print(f"✅ Blender: {blender_path}")
            print(f"✅ 渲染脚本: {blender_script}")
            
            # 阶段1: WSG → glTF，保存到指定的glTF目录
            print("\n🔄 阶段1: WSG → glTF")
            gltf_path = self.wsg_to_gltf(wsg_path, gltf_output_dir)
            if not gltf_path:
                return False

            # 阶段2: glTF → PNG，保存到指定的PNG目录
            print(f"\n🎨 阶段2: glTF → PNG渲染")
            os.makedirs(png_output_dir, exist_ok=True)

            render_success = self.render_gltf(
                gltf_path, png_output_dir, blender_path, blender_script,
                num_renders, resolution
            )

            if not render_success:
                return False

            print(f"\n🎉 WSG处理完成!")
            print(f"   glTF文件: {gltf_path}")
            print(f"   Visualization directory: {png_output_dir}")
            return True

        except Exception as e:
            print(f"\n💥 处理异常: {e}")
            return False


def find_wsg_files_in_output():
    """在output/test_dataset_results/MagicArticulate/wsg+txt/目录中查找WSG文件"""
    wsg_files = []

    # 搜索指定的wsg+txt目录
    wsg_txt_dir = "./output/test_dataset_results/MagicArticulate/wsg+txt"

    if not os.path.exists(wsg_txt_dir):
        print(f"📁 WSG+TXT目录不存在: {wsg_txt_dir}")
        print("   请确保WSG文件已保存在正确的目录结构中")
        return []

    print(f"🔍 在WSG+TXT目录中搜索WSG文件: {wsg_txt_dir}")

    # 在wsg+txt目录及其子目录中搜索WSG文件
    for root, dirs, files in os.walk(wsg_txt_dir):
        for file in files:
            if file.endswith('.wsg'):
                full_path = os.path.join(root, file)
                wsg_files.append(full_path)

    # 按序号排序（从路径中提取序号）
    def extract_sequence_number(path):
        """从路径中提取序号"""
        try:
            # 从路径中找到序号目录，如 0001, 0002 等
            path_parts = Path(path).parts
            for part in path_parts:
                if part.isdigit():
                    return int(part)
            # 如果没找到数字目录，尝试从文件名提取
            filename = Path(path).stem
            if filename.isdigit():
                return int(filename)
            return 0
        except:
            return 0

    if wsg_files:
        wsg_files.sort(key=extract_sequence_number)

    print(f"   找到 {len(wsg_files)} 个WSG文件")

    # 显示找到的文件和它们的序号
    if wsg_files:
        print("   文件列表:")
        for i, wsg_file in enumerate(wsg_files[:5]):  # 显示前5个
            seq_num = extract_sequence_number(wsg_file)
            rel_path = os.path.relpath(wsg_file)
            print(f"     {i + 1}. 序号{seq_num:04d}: {rel_path}")
        if len(wsg_files) > 5:
            print(f"     ... 还有 {len(wsg_files) - 5} 个文件")

    return wsg_files


def batch_process_wsg_files(output_dir="./output", max_files=None, num_renders=12, resolution=1024, keep_gltf=True):
    """批量处理WSG文件 - 从wsg+txt目录读取，按指定目录结构保存"""
    print("🚀 开始批量处理WSG文件...")
    print("📁 读取目录: output/test_dataset_results/MagicArticulate/wsg+txt/对应序号/")
    print("📁 输出目录结构:")
    print("   - glTF文件: output/test_dataset_results/MagicArticulate/gltf/对应序号/")
    print("   - Visualization files: output/test_dataset_results/MagicArticulate/visualization/对应序号/")

    # 查找所有WSG文件
    wsg_files = find_wsg_files_in_output()

    if not wsg_files:
        print("❌ 未找到WSG文件")
        print("   请确保WSG文件位于: output/test_dataset_results/MagicArticulate/wsg+txt/序号/ 目录中")
        return False

    # 限制文件数量
    if max_files and len(wsg_files) > max_files:
        wsg_files = wsg_files[:max_files]
        print(f"📦 限制处理文件数: {max_files}")

    print(f"📦 找到 {len(wsg_files)} 个WSG文件准备处理")

    # 创建输出目录结构: output/test_dataset_results/MagicArticulate/
    test_results_dir = os.path.join(output_dir, "test_dataset_results")
    magic_base_dir = os.path.join(test_results_dir, "MagicArticulate")
    gltf_base_dir = os.path.join(magic_base_dir, "gltf")
    png_base_dir = os.path.join(magic_base_dir, "png")

    os.makedirs(gltf_base_dir, exist_ok=True)
    os.makedirs(png_base_dir, exist_ok=True)

    print(f"📁 glTF基目录: {gltf_base_dir}")
    print(f"📁 PNG基目录: {png_base_dir}")

    # 创建管道实例
    pipeline = WSGToPNGPipeline()

    success_count = 0
    failed_count = 0

    def extract_sequence_number(path):
        """从路径中提取序号"""
        try:
            # 从路径中找到序号目录，如 0001, 0002 等
            path_parts = Path(path).parts
            for part in path_parts:
                if part.isdigit():
                    return part  # 返回字符串格式的序号，保持前导零
            # 如果没找到数字目录，尝试从文件名提取
            filename = Path(path).stem
            if filename.isdigit():
                return f"{int(filename):04d}"  # 格式化为4位数字
            return "0000"
        except:
            return "0000"

    try:
        for i, wsg_path in enumerate(wsg_files, 1):
            print(f"\n{'=' * 60}")
            print(f"📁 [{i}/{len(wsg_files)}] 处理: {os.path.relpath(wsg_path)}")
            print(f"{'=' * 60}")

            try:
                # 从路径中提取序号作为目录名
                sequence_id = extract_sequence_number(wsg_path)

                print(f"🔢 提取序号: {sequence_id}")

                # Visualization files saved to: output/test_dataset_results/MagicArticulate/visualization/序号/
                png_output_dir = os.path.join(png_base_dir, sequence_id)

                # glTF文件保存到: output/test_dataset_results/MagicArticulate/gltf/序号/
                gltf_output_dir = os.path.join(gltf_base_dir, sequence_id)

                print(f"📁 Visualization output directory: {png_output_dir}")
                print(f"📁 glTF输出目录: {gltf_output_dir}")

                # 处理单个文件
                success = pipeline.process_wsg_file(
                    wsg_path,
                    png_output_dir,
                    gltf_output_dir,
                    num_renders=num_renders,
                    resolution=resolution,
                    keep_gltf=keep_gltf
                )

                if success:
                    success_count += 1
                    print(f"✅ [{i}/{len(wsg_files)}] 成功: 序号{sequence_id}")
                    print(f"   🎨 Visualization directory: {png_output_dir}")
                    print(f"   📄 glTF目录: {gltf_output_dir}")
                else:
                    failed_count += 1
                    print(f"❌ [{i}/{len(wsg_files)}] 失败: 序号{sequence_id}")

            except Exception as e:
                failed_count += 1
                sequence_id = extract_sequence_number(wsg_path)
                print(f"❌ [{i}/{len(wsg_files)}] 异常: 序号{sequence_id} - {e}")
                continue

    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断批量处理")
    finally:
        # 清理临时文件
        pipeline.cleanup_temp_dirs()

    # 输出统计信息
    print(f"\n{'=' * 60}")
    print(f"🎉 批量处理完成!")
    print(f"{'=' * 60}")
    print(f"📊 处理统计:")
    print(f"   总文件数: {len(wsg_files)}")
    print(f"   成功: {success_count}")
    print(f"   失败: {failed_count}")
    print(f"   成功率: {success_count / len(wsg_files) * 100:.1f}%" if len(wsg_files) > 0 else "0%")

    print(f"\n📁 输出目录结构:")
    print(f"   glTF基目录: {gltf_base_dir}")
    print(f"   PNG基目录: {png_base_dir}")

    # 列出生成的目录
    if success_count > 0:
        print(f"\n🎨 生成的文件:")

        # 统计glTF目录
        gltf_dirs = sorted([d for d in Path(gltf_base_dir).iterdir() if d.is_dir()])
        total_gltf_files = 0
        for gltf_dir in gltf_dirs:
            gltf_files_in_dir = list(gltf_dir.glob("*.gltf"))
            total_gltf_files += len(gltf_files_in_dir)

        print(f"   glTF目录: {len(gltf_dirs)} 个，总计 {total_gltf_files} 个glTF文件")
        for gltf_dir in gltf_dirs[:3]:
            gltf_count = len(list(gltf_dir.glob("*.gltf")))
            print(f"     - {gltf_dir.name}/ ({gltf_count} 个glTF文件)")
        if len(gltf_dirs) > 3:
            print(f"     ... 还有 {len(gltf_dirs) - 3} 个目录")

        # 统计PNG目录
        png_dirs = sorted([d for d in Path(png_base_dir).iterdir() if d.is_dir()])
        total_png_files = 0
        for png_dir in png_dirs:
            png_files_in_dir = list(png_dir.glob("*.png"))
            total_png_files += len(png_files_in_dir)

        print(f"   PNG目录: {len(png_dirs)} 个，总计 {total_png_files} 个PNG文件")
        for png_dir in png_dirs[:3]:
            png_count = len(list(png_dir.glob("*.png")))
            print(f"     - {png_dir.name}/ ({png_count} 张图片)")
        if len(png_dirs) > 3:
            print(f"     ... 还有 {len(png_dirs) - 3} 个目录")

    return success_count > 0


def extract_sequence_number(path):
    """从路径中提取序号"""
    try:
        path_parts = Path(path).parts
        for part in path_parts:
            if part.isdigit():
                return part
        filename = Path(path).stem
        if filename.isdigit():
            return f"{int(filename):04d}"
        return "0000"
    except:
        return "0000"

def main():
    # ==================== 配置区域 ====================
    # 输入路径配置 - 可以是文件或文件夹
    # input_path = "../output/Articulation_results/wsg_3d_objects"  # 处理整个文件夹
    # input_path = "../output/Articulation_results/wsg_3d_objects/0003"  # 处理单个模型文件夹
    # input_path = "../output/Articulation_results/articulation_xlv2_test/1cd5b75e-5348-55a5-9deb-3e081250b2fc/1cd5b75e-5348-55a5-9deb-3e081250b2fc_articulation_xlv2_test_0014.wsg"  # 单个文件示例
    
    # ShapeNetSem路径
    input_path = "../output/ShapeNetSem_results/wsg"  # ShapeNetSem WSG文件夹
    
    # 渲染配置
    num_renders = 12      # 渲染视角数
    resolution = 1024     # 渲染分辨率
    keep_gltf = True      # 是否保留glTF文件
    test_mode = False      # 测试模式(仅转换,不渲染) - 设为False进行完整渲染
    max_files = None      # 最大处理文件数 (None表示处理所有)
    
    # 渲染模式
    fast_mode = False     # 快速模式: 512分辨率, 6视角
    high_quality = False  # 高质量模式: 2048分辨率, 24视角
    
    # ==================== 应用模式设置 ====================
    # 快速模式设置
    if fast_mode:
        resolution = 512
        num_renders = 6
        print("🚄 快速模式: 512分辨率, 6视角")
    
    # 高质量模式设置
    if high_quality:
        resolution = 2048
        num_renders = 24
        print("💎 高质量模式: 2048分辨率, 24视角")
    
    # ==================== 开始处理 ====================
    print("=" * 50)
    print("3D对象WSG到PNG渲染器")
    print("=" * 50)
    print(f"输入路径: {input_path}")
    print(f"分辨率: {resolution}")
    print(f"视角数: {num_renders}")
    print(f"保留glTF: {keep_gltf}")
    print(f"测试模式: {test_mode}")
    print("=" * 50)
    
    # 创建管道
    pipeline = WSGToPNGPipeline()
    
    try:
        # 判断输入是文件还是文件夹
        if os.path.isfile(input_path) and input_path.endswith('.wsg'):
            # 单文件处理模式
            if not os.path.exists(input_path):
                print(f"❌ WSG文件不存在: {input_path}")
                return 1

            sequence_id = extract_sequence_number(input_path)

            # 基于输入路径构建输出目录
            base_dir = os.path.dirname(os.path.dirname(input_path))  # wsg_3d_objects的父目录
            dataset_name = os.path.basename(os.path.dirname(base_dir))  # 例如 Articulation_results
            output_base = os.path.join("../output", dataset_name)
            # 修改输出目录到 export_obj_from_wsg 子目录
            # 获取模型目录 (如 0003)
            model_dir = os.path.dirname(input_path)  # wsg_3d_objects/0003
            export_base = os.path.join(model_dir, "export_obj_from_wsg")
            png_output_dir = os.path.join(export_base, "visualization")
            gltf_output_dir = os.path.join(export_base, "gltf")

            if test_mode:
                # 测试模式：仅测试WSG读取和glTF转换
                print("🧪 测试模式: 仅WSG→glTF转换")
                print(f"🔢 序号: {sequence_id}")

                gltf_path = pipeline.wsg_to_gltf(input_path, gltf_output_dir)
                if gltf_path:
                    print(f"✅ 测试成功! glTF文件: {gltf_path}")
                    return 0
                else:
                    print("❌ 测试失败")
                    return 1
            else:
                # 完整单文件模式
                print(f"🔢 序号: {sequence_id}")
                print(f"📁 Visualization output directory: {png_output_dir}")
                print(f"📁 glTF输出目录: {gltf_output_dir}")
                print(f"📐 渲染设置: {resolution}x{resolution}, {num_renders}视角")
                success = pipeline.process_wsg_file(
                    input_path,
                    png_output_dir,
                    gltf_output_dir,
                    num_renders=num_renders,
                    resolution=resolution,
                    keep_gltf=keep_gltf
                )
                return 0 if success else 1

        elif os.path.isdir(input_path):
            # 文件夹批量处理模式
            print(f"\n📂 批量处理文件夹: {input_path}")
            
            # 收集所有WSG文件
            wsg_files = []
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if file.endswith('.wsg'):
                        wsg_files.append(os.path.join(root, file))
            
            if not wsg_files:
                print("❌ 未找到任何WSG文件")
                return 1
                
            print(f"🔍 找到 {len(wsg_files)} 个WSG文件")
            
            # 限制处理数量
            if max_files and len(wsg_files) > max_files:
                wsg_files = wsg_files[:max_files]
                print(f"⚡ 限制处理前 {max_files} 个文件")
            
            # 处理每个文件
            success_count = 0
            for i, wsg_file in enumerate(wsg_files, 1):
                print(f"\n[{i}/{len(wsg_files)}] 处理: {os.path.basename(wsg_file)}")
                
                sequence_id = extract_sequence_number(wsg_file)
                
                # 构建输出路径
                base_dir = os.path.dirname(os.path.dirname(wsg_file))  
                dataset_name = os.path.basename(os.path.dirname(base_dir))
                output_base = os.path.join("../output", dataset_name)
                # 修改输出目录到 export_obj_from_wsg 子目录
                model_dir = os.path.dirname(wsg_file)  # wsg_3d_objects/0003
                export_base = os.path.join(model_dir, "export_obj_from_wsg")
                png_output_dir = os.path.join(export_base, "visualization")
                gltf_output_dir = os.path.join(export_base, "gltf")
                
                success = pipeline.process_wsg_file(
                    wsg_file,
                    png_output_dir,
                    gltf_output_dir,
                    num_renders=num_renders,
                    resolution=resolution,
                    keep_gltf=keep_gltf
                )
                
                if success:
                    success_count += 1
            
            print(f"\n✅ 成功处理: {success_count}/{len(wsg_files)}")
            return 0 if success_count > 0 else 1
            
        else:
            print(f"❌ 无效的输入路径: {input_path}")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        return 1
    except Exception as e:
        print(f"\n💥 程序错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        pipeline.cleanup_temp_dirs()


if __name__ == "__main__":
    sys.exit(main())
