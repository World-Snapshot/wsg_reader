import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import cv2


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
        depth_data = wsg_array[:, :, 32].astype(np.float32, copy=False)
        valid_depth = np.isfinite(depth_data) & (depth_data != -1.0)
        if not valid_depth.any():
            print(f"  跳过 {base_name}: channel 32 has no valid PrimeDepth data")
            return False

        valid_values = depth_data[valid_depth]
        depth_min = float(valid_values.min())
        depth_max = float(valid_values.max())
        if depth_max > depth_min:
            depth_norm = (depth_data - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.full_like(depth_data, 0.5, dtype=np.float32)
        depth_norm = np.where(valid_depth, np.clip(depth_norm, 0.0, 1.0), 0.0)
        
        success_count = 0
        
        # 生成纯深度可视化图像（使用PrimeDepth官方Spectral颜色映射）
        if 'vis' in modes:
            try:
                # 使用matplotlib的Spectral颜色映射（PrimeDepth官方）
                import matplotlib.cm as cm
                
                # 应用Spectral颜色映射
                cmap = cm.get_cmap('Spectral')
                depth_colored = cmap(depth_norm)
                
                # 转换为BGR格式供OpenCV使用
                depth_bgr = (depth_colored[:, :, :3] * 255).astype(np.uint8)
                depth_bgr = cv2.cvtColor(depth_bgr, cv2.COLOR_RGB2BGR)
                
                vis_path = os.path.join(output_dir, f"export_primedepth_from_wsg_{base_name}_vis.png")
                cv2.imwrite(vis_path, depth_bgr)
                print(f"    保存纯Depth可视化: {vis_path}")
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
                # 使用Spectral颜色映射（PrimeDepth官方）
                import matplotlib.cm as cm
                
                # 应用Spectral颜色映射
                cmap = cm.get_cmap('Spectral')
                depth_colored = cmap(depth_norm)
                depth_color = (depth_colored[:, :, :3] * 255).astype(np.uint8)
                depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
                depth_overlay = cv2.addWeighted(original_cv2, 0.5, np.array(depth_color), 0.5, 0)
                overlay_path = os.path.join(output_dir, f"export_primedepth_from_wsg_{base_name}_overlay.png")
                cv2.imwrite(overlay_path, depth_overlay)
                print(f"    保存叠加可视化: {overlay_path}")
                success_count += 1
            except Exception as e:
                print(f"    overlay visualize failed: {str(e)}")
        
        # 保存NPZ数据
        if 'save' in modes:
            try:
                npz_path = os.path.join(output_dir, f"export_primedepth_from_wsg_{base_name}.npz")
                np.savez_compressed(npz_path, depth_data=depth_data)
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
            output_path = os.path.join(parent_dir, f"export_primedepth_from_{folder_name}")
        
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
    modes = ['vis', 'vis_overlay', 'save']  # 可选: 'vis', 'vis_overlay', 'save'
    # modes = ['vis']  # 只生成骨骼图
    # modes = ['vis_overlay']  # 只生成叠加图
    # modes = ['save']  # 只保存NPZ
    
    # 处理配置
    max_workers = mp.cpu_count() // 2  # 并行工作器数量
    
    print("=" * 50)
    print("WSG Relative Depth提取和可视化工具")
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
