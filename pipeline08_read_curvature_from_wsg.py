import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import cv2
import matplotlib.cm as mpl_cm


def _create_depth_colormap(name='magma'):
    """Construct a perceptually-uniform depth colormap using matplotlib."""
    cmap = mpl_cm.get_cmap(name, 256)
    rgb = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    bgr = rgb[:, ::-1]
    return bgr.reshape(256, 1, 3)


CURVATURE_COLORMAP = _create_depth_colormap()


def normalize_curvature_map(curvature_map, percentile=99.0):
    """对曲率图进行对称归一化并增强局部对比度"""
    curvature = curvature_map.astype(np.float32)
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

    valid = curvature[np.isfinite(curvature)]
    if valid.size == 0:
        return np.zeros_like(curvature, dtype=np.uint8)

    max_val = np.percentile(np.abs(valid), percentile)
    if max_val <= 1e-6:
        return np.zeros_like(curvature, dtype=np.uint8)

    clipped = np.clip(curvature, -max_val, max_val)
    normalized = (clipped / (2 * max_val) + 0.5) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    if np.any(normalized):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(normalized)

    return normalized


def load_wsg_data(wsg_path):
    """加载WSG数据"""
    try:
        data = np.load(wsg_path, allow_pickle=True)
        return data['wsg_data']
    except Exception as e:
        print(f"加载WSG数据失败 {wsg_path}: {str(e)}")
        return None

def process_single_wsg_file(args):
    """处理单个WSG文件"""
    wsg_path, output_dir, modes = args
    
    try:
        base_name = os.path.splitext(os.path.basename(wsg_path))[0]
        
        # 加载WSG数据
        wsg_array = load_wsg_data(wsg_path)
        if wsg_array is None:
            print(f"  跳过 {base_name}: WSG数据加载失败")
            return False
        
        # 提取Depth数据
        curvature_data = wsg_array[:, :, 39]

        success_count = 0
        
        # 生成depth可视化图像
        if 'vis' in modes:
            try:
                curvature_vis = normalize_curvature_map(curvature_data)
                curvature_vis_color = cv2.applyColorMap(curvature_vis, CURVATURE_COLORMAP)

                vis_path = os.path.join(output_dir, f"export_curvature_from_wsg_{base_name}_vis.png")
                cv2.imwrite(vis_path, curvature_vis_color)
                print(f"    保存纯Curvature可视化: {vis_path}")
                success_count += 1
            except Exception as e:
                print(f"    visualize failed: {str(e)}")
        
        # 生成叠加可视化图像
        if 'vis_overlay' in modes:
            try:
                # 获取原始RGB图像（前3个通道）
                original_rgb = wsg_array[:, :, :3].astype(np.uint8)
                original_pil = Image.fromarray(original_rgb)
                original_cv2 = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
                curvature_vis = normalize_curvature_map(curvature_data)
                curvature_vis_color = cv2.applyColorMap(curvature_vis, CURVATURE_COLORMAP)
                curvature_overlay = cv2.addWeighted(original_cv2, 0.6, curvature_vis_color, 0.4, 0)
                overlay_path = os.path.join(output_dir, f"export_curvature_from_wsg_{base_name}_overlay.png")
                cv2.imwrite(overlay_path, curvature_overlay)
                print(f"    保存叠加可视化: {overlay_path}")
                success_count += 1
            except Exception as e:
                print(f"    overlay visualize failed: {str(e)}")
        
        # 保存NPZ数据
        if 'save' in modes:
            try:
                npz_path = os.path.join(output_dir, f"export_curvature_from_wsg_{base_name}.npz")
                np.savez_compressed(npz_path, curvature_data=curvature_data)
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
            output_path = os.path.join(parent_dir, f"export_curvature_from_{folder_name}")
        
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
    #input_path = "../output/AVIS_test_results/-wgEIkAongk/wsg"

    # 输出配置
    output_path = None  # None表示使用默认输出路径
    
    # 模式配置
    modes = ['vis', 'vis_overlay', 'save']  # 可选: 'vis', 'vis_overlay', 'save'
    # modes = ['vis']  # 只生成骨骼图
    # modes = ['vis_overlay']  # 只生成叠加图
    # modes = ['save']  # 只保存NPZ
    
    # 处理配置
    max_workers = mp.cpu_count()
    
    print("=" * 50)
    print("WSG Curvature提取和可视化工具")
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
