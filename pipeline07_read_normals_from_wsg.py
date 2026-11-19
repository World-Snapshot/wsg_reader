import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import cv2
from normalized_wsg_loader import load_preferred_wsg_data


def normalize_normal_directions(normal_map):
    """Normalize valid normal vectors for direction-color visualization only."""
    normal_map = np.asarray(normal_map, dtype=np.float32)
    lengths = np.linalg.norm(normal_map, axis=2, keepdims=True)
    # Exact WSG padding is -1 in all three lanes; VAE-decoded padding can
    # soften slightly above -1. A real unit normal cannot have all XYZ
    # components below -0.90 simultaneously, so this remains unambiguous.
    sentinel = np.all(normal_map <= -0.90, axis=2, keepdims=True)
    invalid = sentinel | ~np.isfinite(lengths) | (lengths <= 1e-8)
    valid = ~invalid
    normalized = np.divide(
        normal_map,
        lengths,
        where=valid,
        out=np.zeros_like(normal_map),
    )
    normalized[invalid.repeat(3, axis=2)] = -1.0
    return normalized


def process_single_wsg_file(args):
    """处理单个WSG文件"""
    wsg_path, output_dir, modes = args

    try:
        base_name = os.path.splitext(os.path.basename(wsg_path))[0]

        wsg_array, _used_normalized = load_preferred_wsg_data(
            wsg_path, [0, 1, 2, 36, 37, 38]
        )

        if wsg_array is None:
            print(f"  跳过 {base_name}: WSG数据加载失败")
            return False

        # 提取Normal数据 (channels 36-38)
        normal_data = wsg_array[:, :, 36:39].astype(np.float32, copy=False)

        success_count = 0

        # 准备normal数据（共用）
        normal_map = normal_data

        # 确保normal在[-1, 1]范围（共用转换）
        normal_for_processing = normal_map
        normal_for_direction_vis = normalize_normal_directions(normal_for_processing)

        # 方法1: NormalCrafter风格的Normal Map可视化
        if 'vis_normalcrafter' in modes or 'vis_rgb' in modes or 'vis' in modes:
            try:
                # NormalCrafter style: unit XYZ direction mapped linearly to RGB.
                normal_rgb = np.clip(normal_for_direction_vis, -1.0, 1.0)
                normal_rgb = normal_rgb * 0.5 + 0.5  # [-1,1] -> [0,1]

                # 转换为8位颜色
                normal_rgb_uint8 = (normal_rgb * 255).astype(np.uint8)
                normal_color_rgb = cv2.cvtColor(normal_rgb_uint8, cv2.COLOR_RGB2BGR)

                vis_path_rgb = os.path.join(output_dir, f"export_normal_from_wsg_{base_name}_vis_normalcrafter.png")
                cv2.imwrite(vis_path_rgb, normal_color_rgb)
                print(f"    保存Normal可视化（NormalCrafter风格，XYZ→RGB直接映射）: {vis_path_rgb}")
                success_count += 1
            except Exception as e:
                print(f"    vis_normalcrafter failed: {str(e)}")

        # 方法2: Magnitude + INFERNO colormap（显示法线变化强度）
        if 'vis_magnitude' in modes or 'vis' in modes:
            try:
                # 计算法线向量长度
                normal_magnitude = np.linalg.norm(normal_for_processing, axis=2)
                # 使用百分位归一化拉开分布
                finite_magnitude = normal_magnitude[np.isfinite(normal_magnitude)]
                p5, p95 = np.percentile(finite_magnitude, [5, 95])
                if p95 > p5:
                    mag_norm = (normal_magnitude - p5) / (p95 - p5)
                else:
                    mag_norm = np.full_like(normal_magnitude, 0.5, dtype=np.float32)
                mag_norm = np.clip(mag_norm, 0, 1)
                mag_vis = (mag_norm * 255).astype(np.uint8)
                # INFERNO: 暖色系，高对比度（黑→紫→红→橙→黄）
                mag_color = cv2.applyColorMap(mag_vis, cv2.COLORMAP_INFERNO)
                vis_path_mag = os.path.join(output_dir, f"export_normal_from_wsg_{base_name}_vis.png")
                cv2.imwrite(vis_path_mag, mag_color)
                print(f"    保存Normal可视化（INFERNO magnitude, P5-P95）: {vis_path_mag}")
                success_count += 1
            except Exception as e:
                print(f"    vis_magnitude failed: {str(e)}")

        # 方法3: 标准法线贴图可视化（图形学标准，蓝紫色基调）
        if 'vis_blue' in modes or 'vis' in modes:
            try:
                # 使用P5-P95自适应归一化每个通道，增强细节
                normal_standard = np.zeros_like(normal_for_processing, dtype=np.float32)
                for c in range(3):
                    channel = normal_for_direction_vis[:, :, c]
                    finite_channel = channel[np.isfinite(channel)]
                    p5, p95 = np.percentile(finite_channel, [5, 95])
                    # 拉伸到 [-1, 1]
                    if p95 > p5:
                        ch_norm = (channel - p5) / (p95 - p5) * 2.0 - 1.0
                    else:
                        ch_norm = np.zeros_like(channel, dtype=np.float32)
                    ch_norm = np.clip(ch_norm, -1, 1)
                    normal_standard[:, :, c] = ch_norm

                # 标准映射: [-1,1] → [0,1] → [0,255]
                normal_rgb_standard = (normal_standard * 0.5 + 0.5)
                normal_rgb_standard = np.clip(normal_rgb_standard, 0, 1)
                normal_rgb_standard = (normal_rgb_standard * 255).astype(np.uint8)
                normal_color_standard = cv2.cvtColor(normal_rgb_standard, cv2.COLOR_RGB2BGR)

                vis_path_standard = os.path.join(output_dir, f"export_normal_from_wsg_{base_name}_vis_blue.png")
                cv2.imwrite(vis_path_standard, normal_color_standard)

                # 打印每个通道的拉伸范围
                print(f"    保存Normal可视化（标准法线贴图, XYZ→RGB, P5-P95拉伸）:")
                for c, name in enumerate(['X/Red', 'Y/Green', 'Z/Blue']):
                    finite_channel = normal_for_direction_vis[:, :, c]
                    finite_channel = finite_channel[np.isfinite(finite_channel)]
                    p5, p95 = np.percentile(finite_channel, [5, 95])
                    print(f"      {name}: [{p5:.3f}, {p95:.3f}]")
                print(f"    {vis_path_standard}")
                success_count += 1
            except Exception as e:
                print(f"    vis_blue failed: {str(e)}")

        # 方法4: Blender风格Normal渲染（将Normal映射为RGB颜色，然后光照渲染）
        # 类似物理pipeline：Normal XYZ → RGB顶点颜色 → 光照渲染
        if 'vis_blender' in modes or 'vis_color' in modes or 'vis' in modes:
            try:
                # 归一化normal向量到单位长度
                normal_lengths = np.linalg.norm(normal_for_processing, axis=2, keepdims=True)
                normal_normalized = normal_for_direction_vis

                # 将normal的XYZ映射到RGB作为基础材质颜色
                # 这样每个像素的颜色由其法线方向决定
                base_color_from_normal = (normal_normalized * 0.5 + 0.5)  # [-1,1] -> [0,1]
                base_color_from_normal = np.clip(base_color_from_normal, 0, 1)

                # 调试：检查normal数据
                valid_normals = np.sum(normal_lengths > 1e-8)
                total_pixels = normal_lengths.size
                print(f"    DEBUG: 有效normal像素: {valid_normals}/{total_pixels} ({valid_normals/total_pixels*100:.1f}%)")
                if valid_normals > 0:
                    print(f"    DEBUG: Normal范围: [{normal_for_processing.min():.3f}, {normal_for_processing.max():.3f}]")
                    print(f"    DEBUG: Normal长度范围: [{normal_lengths[normal_lengths>1e-8].min():.3f}, {normal_lengths[normal_lengths>1e-8].max():.3f}]")

                # 三点照明设置（增强对比度版本）
                # 降低整体光照，增加明暗对比

                # 主光源 (Key Light) - 从右上侧方照射（更侧面以增加对比）
                key_light_dir = np.array([0.8, 0.3, 0.5])  # 更侧面的角度
                key_light_dir = key_light_dir / np.linalg.norm(key_light_dir)
                key_light_color = np.array([1.0, 0.98, 0.95])  # 暖白色
                key_light_intensity = 1.2  # 提高强度以增加对比

                # 补光 (Fill Light) - 从对侧较弱
                fill_light_dir = np.array([-0.6, 0.1, 0.4])
                fill_light_dir = fill_light_dir / np.linalg.norm(fill_light_dir)
                fill_light_color = np.array([0.6, 0.65, 0.8])  # 冷色调
                fill_light_intensity = 0.2  # 降低补光

                # 轮廓光 (Rim Light) - 从后上方
                rim_light_dir = np.array([0.0, 0.9, -0.4])
                rim_light_dir = rim_light_dir / np.linalg.norm(rim_light_dir)
                rim_light_intensity = 0.6

                # 计算各光源的漫反射
                key_diffuse = np.maximum(0, np.sum(normal_normalized * key_light_dir, axis=2))
                fill_diffuse = np.maximum(0, np.sum(normal_normalized * fill_light_dir, axis=2))
                rim_diffuse = np.maximum(0, np.sum(normal_normalized * rim_light_dir, axis=2))

                # 非线性调整以增加对比度（类似gamma correction但只对diffuse）
                # 使用power函数压暗中间值，保留高光
                key_diffuse = np.power(key_diffuse, 1.5)  # 压暗中间调
                fill_diffuse = np.power(fill_diffuse, 1.2)
                rim_diffuse = np.power(rim_diffuse, 1.3)

                # 调试：检查光照分布
                print(f"    DEBUG: Key diffuse 范围: [{key_diffuse.min():.3f}, {key_diffuse.max():.3f}], 平均: {key_diffuse.mean():.3f}")
                print(f"    DEBUG: Fill diffuse 范围: [{fill_diffuse.min():.3f}, {fill_diffuse.max():.3f}], 平均: {fill_diffuse.mean():.3f}")
                print(f"    DEBUG: Rim diffuse 范围: [{rim_diffuse.min():.3f}, {rim_diffuse.max():.3f}], 平均: {rim_diffuse.mean():.3f}")

                # 添加高光（Blinn-Phong）
                view_dir = np.array([0, 0, 1])  # 观察方向
                half_vec_key = (key_light_dir + view_dir) / np.linalg.norm(key_light_dir + view_dir)
                specular = np.maximum(0, np.sum(normal_normalized * half_vec_key, axis=2))
                specular = np.power(specular, 80)  # 更锐利的高光

                # 环境光（提高到0.3以保持颜色可见性）
                ambient = 0.3

                # 计算光照对normal颜色材质的影响
                # 主光源贡献
                key_contribution = (key_light_intensity * key_diffuse)[:, :, np.newaxis] * key_light_color * base_color_from_normal

                # 补光贡献
                fill_contribution = (fill_light_intensity * fill_diffuse)[:, :, np.newaxis] * fill_light_color * base_color_from_normal

                # 轮廓光贡献
                rim_contribution = (rim_light_intensity * rim_diffuse)[:, :, np.newaxis] * base_color_from_normal

                # 环境光贡献（保留基础颜色）
                ambient_contribution = ambient * base_color_from_normal

                # 高光（白色）
                specular_contribution = 0.3 * specular[:, :, np.newaxis] * np.array([1.0, 1.0, 1.0])

                # 综合所有光照
                final_color = ambient_contribution + key_contribution + fill_contribution + rim_contribution + specular_contribution

                # Tone mapping和Gamma correction
                final_color = np.clip(final_color, 0, 1.3)
                final_color = np.power(final_color, 0.9)  # Gamma correction
                final_color = np.clip(final_color, 0, 1)

                final_color_uint8 = (final_color * 255).astype(np.uint8)
                final_color_bgr = cv2.cvtColor(final_color_uint8, cv2.COLOR_RGB2BGR)

                vis_path_blender = os.path.join(output_dir, f"export_normal_from_wsg_{base_name}_vis_blender.png")
                cv2.imwrite(vis_path_blender, final_color_bgr)
                print(f"    保存Normal可视化（Blender风格渐变彩色渲染）: {vis_path_blender}")
                success_count += 1
            except Exception as e:
                print(f"    vis_blender failed: {str(e)}")
        
        # 生成叠加可视化图像
        if 'vis_overlay' in modes:
            try:
                # 获取原始RGB图像（前3个通道）
                original_rgb = wsg_array[:, :, :3].astype(np.uint8)
                original_pil = Image.fromarray(original_rgb)
                original_cv2 = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
                normal_map = normal_data

                # Normalize to [0, 1] for visualization
                normal_vis = (normal_for_direction_vis + 1.0) / 2.0
                normal_vis = np.clip(normal_vis, 0, 1)
                normal_vis = (normal_vis * 255).astype(np.uint8)
                normal_color = cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR)
                normal_overlay = cv2.addWeighted(original_cv2, 0.5, normal_color, 0.5, 0)
                overlay_path = os.path.join(output_dir, f"export_normal_from_wsg_{base_name}_overlay.png")
                cv2.imwrite(overlay_path, normal_overlay)
                print(f"    保存叠加可视化: {overlay_path}")
                success_count += 1
            except Exception as e:
                print(f"    overlay visualize failed: {str(e)}")
        
        # 保存NPZ数据
        if 'save' in modes:
            try:
                npz_path = os.path.join(output_dir, f"export_primedepth_from_wsg_{base_name}.npz")
                np.savez_compressed(npz_path, normal_data=normal_data)
                print(f"    保存NPZ数据: {npz_path}")
                success_count += 1
                
            except Exception as e:
                print(f"    NPZ数据保存失败: {str(e)}")
        
        return success_count > 0
        
    except Exception as e:
        print(f"  处理文件失败 {wsg_path}: {str(e)}")
        return False


def process_wsg_input(input_path, output_path, modes, max_workers=4):
    """处理WSG输入（文件或文件夹）"""
    
    if os.path.isfile(input_path) and input_path.endswith('.wsg'):
        # 单个文件处理
        if output_path is None:
            output_path = os.path.dirname(input_path)
        
        os.makedirs(output_path, exist_ok=True)
        print(f"处理单个WSG文件: {input_path}")
        print(f"输出目录: {output_path}")
        
        success = process_single_wsg_file((input_path, output_path, modes))
        print(f"处理完成! 成功: {'是' if success else '否'}")
        
    elif os.path.isdir(input_path):
        # 文件夹处理
        if output_path is None:
            parent_dir = os.path.dirname(input_path.rstrip('/'))
            folder_name = os.path.basename(input_path.rstrip('/'))
            output_path = os.path.join(parent_dir, f"export_normal_from_{folder_name}")
        
        os.makedirs(output_path, exist_ok=True)
        
        # 获取所有WSG文件
        wsg_files = glob.glob(os.path.join(input_path, "*.wsg"))
        wsg_files.sort()
        
        if not wsg_files:
            print(f"在目录 {input_path} 中没有找到WSG文件")
            return
        
        print(f"处理WSG文件夹: {input_path}")
        print(f"输出目录: {output_path}")
        print(f"找到 {len(wsg_files)} 个WSG文件")
        
        # 准备任务
        tasks = [(wsg_path, output_path, modes) for wsg_path in wsg_files]
        
        # 并行处理
        success_count = 0
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_single_wsg_file, task) for task in tasks]
                
                for future in tqdm(futures, desc="提取Depth"):
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as e:
                        print(f"任务失败: {str(e)}")
        else:
            # 单线程处理
            for task in tqdm(tasks, desc="提取Depth"):
                if process_single_wsg_file(task):
                    success_count += 1
        
        print(f"处理完成! 成功: {success_count}/{len(wsg_files)}")
    
    else:
        print(f"无效的输入路径: {input_path}")


def main():
    """主函数"""
    # ==================== 配置区域 ====================
    # 输入配置
    input_path = "../output/test_dataset_results/human_signer_480/wsg"  # WSG文件或文件夹路径
    # input_path = "../output/test_dataset_results/human_mutiperson/wsg/00000004.wsg"  # 单个文件示例
    
    # 输出配置
    output_path = None  # None表示使用默认输出路径
    
    # 模式配置
    # 可选模式:
    # - 'vis': 生成所有可视化（包含下面4种）
    # - 'vis_normalcrafter': NormalCrafter风格直接映射
    # - 'vis_magnitude': INFERNO magnitude热图
    # - 'vis_blue': 标准法线贴图（P5-P95拉伸）
    # - 'vis_blender': Blender风格RGB材质+光照渲染
    # - 'vis_overlay': 叠加可视化
    # - 'save': 保存NPZ数据
    modes = ['vis', 'vis_overlay', 'save']  # 默认：所有可视化
    # modes = ['vis_normalcrafter', 'vis_blender']  # 只生成NormalCrafter和Blender风格
    # modes = ['vis_magnitude']  # 只生成magnitude热图
    
    # 处理配置
    max_workers = mp.cpu_count() 
    
    print("=" * 50)
    print("WSG Normal提取和可视化工具")
    print("=" * 50)
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path if output_path else '自动生成'}")
    print(f"处理模式: {', '.join(modes)}")
    print(f"工作器数量: {max_workers}")
    print("=" * 50)
    
    try:
        process_wsg_input(input_path, output_path, modes, max_workers)
        print("\n✅ 所有提取任务完成!")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
