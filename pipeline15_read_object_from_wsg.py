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
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

from categorical_visualization import (
    CHANNEL_111_ID_TO_CATEGORY,
    categorical_ids_to_rgb,
    dense_channel111_category_map,
    rounded_class_distribution,
)


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
        """从WSG的前3个通道提取RGB图像并保存"""
        try:
            # 检查WSG格式
            if len(wsg_array.shape) != 3 or wsg_array.shape[2] < 3:
                print("⚠️ WSG格式不正确，无法提取RGB")
                return False
            
            height, width = wsg_array.shape[:2]
            
            # 提取前3个通道 (RGB)
            rgb_channels = wsg_array[:, :, 0:3]
            
            # 检查是否有有效的RGB数据
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
    
    def extract_label_from_wsg(self, wsg_array, output_path):
        """从WSG通道111提取object_id并保存为标签文件"""
        try:
            import numpy as np
            
            height, width = wsg_array.shape[:2]
            
            # 收集所有唯一的object_id
            object_ids = set()
            class_counts = {1: 0, 2: 0, 3: 0, 4: 0}  # 顶点、面、关节、骨骼
            
            for y in range(height):
                for x in range(width):
                    class_value = wsg_array[y, x, 112]
                    if class_value > 0:  # 有数据
                        object_id = wsg_array[y, x, 111]
                        object_ids.add(int(object_id))
                        class_counts[int(class_value)] = class_counts.get(int(class_value), 0) + 1
            
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

    def export_category_id_map(self, wsg_array, output_dir, base_name="wsg"):
        """Export channel 111 as a dense, discrete category-color mask."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        category_map = dense_channel111_category_map(wsg_array, require_alpha=False)
        category_rgb = categorical_ids_to_rgb(category_map)
        mask_path = output_dir / f"{base_name}_channel111_category_mask.png"
        Image.fromarray(category_rgb, mode="RGB").save(mask_path)

        distribution = rounded_class_distribution(
            category_map, tuple(CHANNEL_111_ID_TO_CATEGORY)
        )
        summary_path = output_dir / f"{base_name}_channel111_categories.txt"
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write("Channel 111 dense category map\n")
            handle.write("Invalid background: -1 (rendered black)\n")
            for category_id, count in sorted(distribution.items()):
                name = CHANNEL_111_ID_TO_CATEGORY.get(category_id, "unknown")
                handle.write(f"{category_id}: {name} ({count} pixels)\n")
        print(f"   Saved channel 111 category mask: {mask_path}")
        return [str(mask_path), str(summary_path)]
    
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
            
            # 提取关节位置
            for y in range(height):
                for x in range(width):
                    if wsg_array[y, x, 112] == 3:  # 关节数据
                        joint_id = int(wsg_array[y, x, 113])
                        joint_x = float(wsg_array[y, x, 114])
                        joint_y = float(wsg_array[y, x, 115])
                        joint_z = float(wsg_array[y, x, 116])
                        joints_3d[joint_id] = (joint_x, joint_y, joint_z)
            
            # 提取骨骼连接
            for y in range(height):
                for x in range(width):
                    if wsg_array[y, x, 112] == 4:  # 骨骼数据
                        start_joint = int(wsg_array[y, x, 114])
                        end_joint = int(wsg_array[y, x, 115])
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
            "utils/blender_script.py",
            "main/utils/blender_script.py",
            "../main/utils/blender_script.py",
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

        # 扫描整个图像寻找数据
        for y in range(height):
            for x in range(width):
                # 检查该像素是否包含数据（通道111: Object ID）
                object_id = wsg_array[y, x, 111]
                if object_id > 0:  # 有效数据
                    record_type = wsg_array[y, x, 112]  # Class Value
                    
                    if record_type == 1:  # Vertex Data
                        vertex_records.append((y, x))
                    elif record_type == 2:  # Face Data
                        face_records.append((y, x))
                    elif record_type == 3:  # Joint Data
                        joint_records.append((y, x))
                    elif record_type == 4:  # Bone Data
                        bone_records.append((y, x))

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

        for new_idx, (y, x) in enumerate(vertex_records):
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
        for y, x in face_records:
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
                self.create_skeleton_visualization_3d(wsg_array, rig_vis_dir, num_views=num_renders)
                print(f"   📁 骨骼可视化目录: {rig_vis_dir}")
                
                # 提取并保存标签信息
                label_filename = "vis_label_from_wsg.txt"
                label_output_path = os.path.join(rgb_output_dir, label_filename)
                self.extract_label_from_wsg(wsg_array, label_output_path)
            
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
    input_path = "../output/Articulation_results/articulation_xlv2_test/1cd5b75e-5348-55a5-9deb-3e081250b2fc/1cd5b75e-5348-55a5-9deb-3e081250b2fc_articulation_xlv2_test_0014.wsg"  # 单个文件示例
    
    # 渲染配置
    num_renders = 12      # 渲染视角数
    resolution = 1024     # 渲染分辨率
    keep_gltf = True      # 是否保留glTF文件
    test_mode = False      # 测试模式(仅转换,不渲染) - 设为True因为没有Blender
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
