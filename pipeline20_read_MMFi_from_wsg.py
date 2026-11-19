
import numpy as np
import os
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import argparse
import cv2
from PIL import Image

def load_wsg_data(wsg_path):
    """加载WSG数据"""
    try:
        data = np.load(wsg_path, allow_pickle=True)
        return data['wsg_data']
    except Exception as e:
        print(f"加载WSG数据失败 {wsg_path}: {str(e)}")
        return None

def process_single_wsg_file(args):
    """处理单个WSG文件，提取MMFi数据"""
    wsg_path, output_dir, extract_modes = args
    
    try:
        # Load WSG data
        wsg_array = load_wsg_data(wsg_path)
        if wsg_array is None:
            return False
        
        frame_name = os.path.basename(wsg_path).replace('.wsg', '')
        success_count = 0
        
        # Extract and save LiDAR data
        if 'lidar' in extract_modes:
            lidar_data = wsg_array[:, :, 207:210]
            # Get valid points (where all channels are not -1)
            mask = np.all(lidar_data != -1, axis=-1)
            valid_lidar = lidar_data[mask]
            
            if len(valid_lidar) > 0:
                lidar_dir = os.path.join(output_dir, 'lidar')
                os.makedirs(lidar_dir, exist_ok=True)
                lidar_path = os.path.join(lidar_dir, f'{frame_name}.npy')
                np.save(lidar_path, valid_lidar)
                success_count += 1
        
        # Extract and save radar data
        if 'radar' in extract_modes:
            radar_data = wsg_array[:, :, 211:216]
            # Get valid points
            mask = np.all(radar_data != -1, axis=-1)
            valid_radar = radar_data[mask]
            
            if len(valid_radar) > 0:
                radar_dir = os.path.join(output_dir, 'mmwave')
                os.makedirs(radar_dir, exist_ok=True)
                radar_path = os.path.join(radar_dir, f'{frame_name}.npy')
                np.save(radar_path, valid_radar)
                success_count += 1
        
        # Extract and save WiFi features
        if 'wifi' in extract_modes:
            # WiFi features are in channels 217:227
            wifi_features = wsg_array[:, :, 217:227]
            # Get presence map (first channel of WiFi features)
            presence_map = wifi_features[:, :, 0]
            
            # Save only points where presence > 0
            if np.any(presence_map > 0):
                wifi_dir = os.path.join(output_dir, 'wifi_features')
                os.makedirs(wifi_dir, exist_ok=True)
                wifi_path = os.path.join(wifi_dir, f'{frame_name}.npy')
                np.save(wifi_path, wifi_features)
                success_count += 1
        
        # Extract and save RGB image
        if 'rgb' in extract_modes:
            rgb_data = wsg_array[:, :, :3].astype(np.uint8)
            rgb_dir = os.path.join(output_dir, 'rgb_extracted')
            os.makedirs(rgb_dir, exist_ok=True)
            rgb_path = os.path.join(rgb_dir, f'{frame_name}.png')
            Image.fromarray(rgb_data).save(rgb_path)
            success_count += 1
        
        # Generate visualizations
        if 'vis' in extract_modes:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Visualize LiDAR depth
            lidar_z = wsg_array[:, :, 209]  # Z channel
            valid_mask = lidar_z != -1
            valid_count = np.sum(valid_mask)
            print(f"    LiDAR Z轴有效像素数: {valid_count}")
            
            if np.any(valid_mask):
                # 创建可视化图像
                lidar_vis = np.zeros((lidar_z.shape[0], lidar_z.shape[1], 3), dtype=np.uint8)
                
                # 获取有效点的位置
                valid_points = np.where(valid_mask)
                depths = lidar_z[valid_mask]
                
                # 归一化深度值
                min_val = np.min(depths)
                max_val = np.max(depths)
                if max_val > min_val:
                    depths_norm = (depths - min_val) / (max_val - min_val)
                else:
                    depths_norm = depths * 0
                
                # 为每个点绘制一个大圆点
                colors = cv2.applyColorMap((depths_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
                colors = colors.reshape(-1, 3)
                
                for i, (y, x) in enumerate(zip(valid_points[0], valid_points[1])):
                    color = colors[i].tolist()
                    color = [int(color[0]), int(color[1]), int(color[2])]  # BGR format
                    cv2.circle(lidar_vis, (x, y), radius=3, color=color, thickness=-1)  # 填充的圆，半径3像素
                
                lidar_path = os.path.join(vis_dir, f'{frame_name}_lidar.png')
                cv2.imwrite(lidar_path, lidar_vis)
                print(f"    保存LiDAR可视化: {lidar_path}")
            else:
                print(f"    跳过LiDAR可视化: 没有有效数据")
            
            # Visualize radar intensity
            radar_intensity = wsg_array[:, :, 214]  # Intensity channel
            valid_mask = radar_intensity != -1
            valid_count = np.sum(valid_mask)
            print(f"    mmWave Radar强度有效像素数: {valid_count}")
            
            if np.any(valid_mask):
                # 创建可视化图像
                radar_vis = np.zeros((radar_intensity.shape[0], radar_intensity.shape[1], 3), dtype=np.uint8)
                
                # 获取有效点的位置
                valid_points = np.where(valid_mask)
                intensities = radar_intensity[valid_mask]
                
                # 归一化强度值
                min_val = np.min(intensities)
                max_val = np.max(intensities)
                if max_val > min_val:
                    intensities_norm = (intensities - min_val) / (max_val - min_val)
                else:
                    intensities_norm = intensities * 0
                
                # 为每个点绘制一个大圆点
                colors = cv2.applyColorMap((intensities_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_HOT)
                colors = colors.reshape(-1, 3)
                
                for i, (y, x) in enumerate(zip(valid_points[0], valid_points[1])):
                    color = colors[i].tolist()
                    color = [int(color[0]), int(color[1]), int(color[2])]  # BGR format
                    cv2.circle(radar_vis, (x, y), radius=5, color=color, thickness=-1)  # 更大的圆，半径5像素
                
                radar_path = os.path.join(vis_dir, f'{frame_name}_mmwave_radar.png')
                cv2.imwrite(radar_path, radar_vis)
                print(f"    保存mmWave Radar可视化: {radar_path}")
            else:
                print(f"    跳过mmWave Radar可视化: 没有有效数据")
            
            # 添加RGB和传感器叠加可视化
            rgb_data = wsg_array[:, :, :3].astype(np.uint8)
            
            # LiDAR叠加
            if np.any(lidar_z != -1):
                # 创建彩色图像用于叠加
                lidar_color_full = np.zeros((lidar_z.shape[0], lidar_z.shape[1], 3), dtype=np.uint8)
                valid_points = np.where(lidar_z != -1)
                depths = lidar_z[lidar_z != -1]
                
                # 归一化并着色
                if len(depths) > 0:
                    min_val = np.min(depths)
                    max_val = np.max(depths)
                    if max_val > min_val:
                        depths_norm = (depths - min_val) / (max_val - min_val)
                    else:
                        depths_norm = depths * 0
                    
                    colors = cv2.applyColorMap((depths_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
                    colors = colors.reshape(-1, 3)
                    
                    for i, (y, x) in enumerate(zip(valid_points[0], valid_points[1])):
                        color = colors[i].tolist()
                        color = [int(color[0]), int(color[1]), int(color[2])]
                        cv2.circle(lidar_color_full, (x, y), radius=3, color=color, thickness=-1)
                
                lidar_overlay = cv2.addWeighted(cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR), 0.7, lidar_color_full, 0.3, 0)
                overlay_path = os.path.join(vis_dir, f'{frame_name}_lidar_overlay.png')
                cv2.imwrite(overlay_path, lidar_overlay)
                print(f"    保存LiDAR叠加图: {overlay_path}")
            
            # mmWave Radar叠加
            if np.any(radar_intensity != -1):
                # 需要重新创建彩色图像用于叠加
                radar_color_full = np.zeros((radar_intensity.shape[0], radar_intensity.shape[1], 3), dtype=np.uint8)
                radar_valid_points = np.where(radar_intensity != -1)
                radar_intensities = radar_intensity[radar_intensity != -1]
                
                # 归一化并着色
                if len(radar_intensities) > 0:
                    min_val = np.min(radar_intensities)
                    max_val = np.max(radar_intensities)
                    if max_val > min_val:
                        intensities_norm = (radar_intensities - min_val) / (max_val - min_val)
                    else:
                        intensities_norm = radar_intensities * 0
                    
                    radar_colors = cv2.applyColorMap((intensities_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_HOT)
                    radar_colors = radar_colors.reshape(-1, 3)
                    
                    for i, (y, x) in enumerate(zip(radar_valid_points[0], radar_valid_points[1])):
                        color = radar_colors[i].tolist()
                        color = [int(color[0]), int(color[1]), int(color[2])]
                        cv2.circle(radar_color_full, (x, y), radius=5, color=color, thickness=-1)
                
                radar_overlay = cv2.addWeighted(cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR), 0.7, radar_color_full, 0.3, 0)
                overlay_path = os.path.join(vis_dir, f'{frame_name}_mmwave_radar_overlay.png')
                cv2.imwrite(overlay_path, radar_overlay)
                print(f"    保存mmWave Radar叠加图: {overlay_path}")
            
            # Visualize WiFi CSI heatmap
            # 现在WiFi数据是10个天线的完整图像，每个通道是一个天线
            wifi_data = wsg_array[:, :, 217:227]  # (H, W, 10) - 10个天线
            
            # 方案1：显示所有天线的平均强度
            wifi_mean = np.mean(wifi_data, axis=2)  # 平均所有天线
            
            # 归一化
            min_val = np.min(wifi_mean)
            max_val = np.max(wifi_mean)
            if max_val > min_val:
                wifi_norm = (wifi_mean - min_val) / (max_val - min_val)
            else:
                wifi_norm = wifi_mean
            
            # 应用热力图颜色
            wifi_color = cv2.applyColorMap((wifi_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            
            wifi_path = os.path.join(vis_dir, f'{frame_name}_wifi_mean.png')
            cv2.imwrite(wifi_path, wifi_color)
            print(f"    保存WiFi CSI平均热力图: {wifi_path}")
            
            # WiFi叠加图
            wifi_overlay = cv2.addWeighted(cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR), 0.6, wifi_color, 0.4, 0)
            overlay_path = os.path.join(vis_dir, f'{frame_name}_wifi_overlay.png')
            cv2.imwrite(overlay_path, wifi_overlay)
            print(f"    保存WiFi CSI叠加图: {overlay_path}")
            
            # 方案2：显示单个天线的热力图（可以看到时频结构）
            for ant_idx in range(min(3, wifi_data.shape[2])):  # 显示前3个天线
                wifi_single = wifi_data[:, :, ant_idx]
                wifi_single_norm = (wifi_single - np.min(wifi_single)) / (np.max(wifi_single) - np.min(wifi_single) + 1e-6)
                wifi_single_color = cv2.applyColorMap((wifi_single_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
                single_path = os.path.join(vis_dir, f'{frame_name}_wifi_antenna{ant_idx}.png')
                cv2.imwrite(single_path, wifi_single_color)
                print(f"    保存WiFi天线{ant_idx}热力图: {single_path}")
            
            success_count += 1
        
        return success_count > 0
        
    except Exception as e:
        print(f"  处理文件失败 {wsg_path}: {str(e)}")
        return False

def process_experiment_folder(experiment_dir, output_base_dir, extract_modes, max_workers):
    """处理单个实验文件夹"""
    experiment_name = os.path.basename(experiment_dir)
    wsg_dir = os.path.join(experiment_dir, 'wsg')
    
    if not os.path.exists(wsg_dir):
        print(f"  WSG目录不存在: {wsg_dir}")
        return False
    
    # Get all WSG files
    wsg_files = glob.glob(os.path.join(wsg_dir, '*.wsg'))
    wsg_files.sort()
    
    if not wsg_files:
        print(f"  没有找到WSG文件")
        return False
    
    print(f"  找到 {len(wsg_files)} 个WSG文件")
    
    # Create output directory in the same experiment folder
    output_dir = os.path.join(experiment_dir, 'export_mmfi_from_wsg')
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare tasks
    tasks = [(wsg_path, output_dir, extract_modes) for wsg_path in wsg_files]
    
    # Process in parallel
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_wsg_file, task) for task in tasks]
        
        for future in tqdm(futures, desc=f"  提取{experiment_name}"):
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                print(f"  任务失败: {str(e)}")
    
    print(f"  完成! 成功: {success_count}/{len(wsg_files)}")
    return success_count > 0

def process(input_path, output_path=None, extract_modes=['lidar', 'radar'], max_workers=None):
    """处理WSG文件或文件夹"""
    if max_workers is None:
        max_workers = mp.cpu_count() // 2
    
    if os.path.isfile(input_path) and input_path.endswith('.wsg'):
        # Single file processing
        if output_path is None:
            # 获取WSG文件所在的实验目录
            wsg_dir = os.path.dirname(input_path)
            experiment_dir = os.path.dirname(wsg_dir)
            output_dir = os.path.join(experiment_dir, 'export_mmfi_from_wsg')
        else:
            output_dir = output_path if not output_path.endswith('.wsg') else os.path.dirname(output_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        success = process_single_wsg_file((input_path, output_dir, extract_modes))
        print(f"处理完成! 成功: {'是' if success else '否'}")
        
    elif os.path.isdir(input_path):
        # Check if this is a WSG results directory with experiment folders
        subdirs = [d for d in os.listdir(input_path) 
                   if os.path.isdir(os.path.join(input_path, d)) and d.startswith('E')]
        
        if subdirs:  # This is a dataset directory with multiple experiments
            print(f"处理数据集目录: {input_path}")
            print(f"找到 {len(subdirs)} 个实验文件夹")
            
            success_count = 0
            for experiment_folder in tqdm(subdirs, desc="处理实验"):
                experiment_dir = os.path.join(input_path, experiment_folder)
                print(f"\n处理实验: {experiment_folder}")
                
                if process_experiment_folder(experiment_dir, None, extract_modes, max_workers):
                    success_count += 1
            
            print(f"\n数据集处理完成! 成功: {success_count}/{len(subdirs)}")
            
        else:  # This might be a single WSG directory
            wsg_files = glob.glob(os.path.join(input_path, '*.wsg'))
            if wsg_files:
                print(f"处理WSG目录: {input_path}")
                print(f"找到 {len(wsg_files)} 个WSG文件")
                
                # 如果是wsg目录，输出到上一级的export_mmfi_from_wsg
                if os.path.basename(input_path) == 'wsg':
                    experiment_dir = os.path.dirname(input_path)
                    output_dir = os.path.join(experiment_dir, 'export_mmfi_from_wsg')
                else:
                    output_dir = output_path if output_path else os.path.join(input_path, 'export_mmfi_from_wsg')
                
                os.makedirs(output_dir, exist_ok=True)
                
                tasks = [(wsg_path, output_dir, extract_modes) for wsg_path in wsg_files]
                
                success_count = 0
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_single_wsg_file, task) for task in tasks]
                    
                    for future in tqdm(futures, desc="提取MMFi数据"):
                        try:
                            if future.result():
                                success_count += 1
                        except Exception as e:
                            print(f"任务失败: {str(e)}")
                
                print(f"处理完成! 成功: {success_count}/{len(wsg_files)}")
            else:
                print(f"在目录 {input_path} 中没有找到WSG文件")

def main():
    """主函数"""
    # ==================== 配置区域 ====================
    # 输入配置 - WSG文件所在的目录
    #input_path = "../output/MMFi_Dataset_results"  # 处理整个数据集
    # input_path = "../output/MMFi_Dataset_results/E01S01A01/wsg"  # 处理单个实验的WSG文件夹
    input_path = "../output/MMFi_Dataset_results/E01S01A01/wsg/frame020.wsg"  # 处理单个文件示例
    
    # 输出配置
    output_path = None  # None表示使用默认输出路径（在输入路径旁边创建MMFi_extracted文件夹）
    # output_path = "../output/MMFi_extracted_data"  # 指定输出路径示例
    
    # 模式配置
    modes = ['lidar', 'radar', 'wifi', 'rgb', 'vis']  # 提取所有数据和生成可视化
    # modes = ['lidar', 'radar']  # 只提取LiDAR和雷达数据
    # modes = ['vis']  # 只生成可视化
    # modes = ['rgb']  # 只提取RGB图像
    
    # 处理配置
    max_workers = mp.cpu_count() // 2  # 并行工作器数量
    # max_workers = 8  # 可以手动设置工作器数量
    
    # ==================== 命令行参数（可选）====================
    parser = argparse.ArgumentParser(description='从WSG提取MMFi数据')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='覆盖默认的WSG文件或文件夹路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='覆盖默认的输出目录')
    parser.add_argument('--modes', '-m', nargs='+', default=None,
                        choices=['lidar', 'radar', 'wifi', 'rgb', 'vis'],
                        help='覆盖默认的提取模式')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='覆盖默认的并行工作器数量')
    
    args = parser.parse_args()
    
    # 如果提供了命令行参数，覆盖默认值
    if args.input:
        input_path = args.input
    if args.output is not None:  # 注意：这里要检查是否为None，因为空字符串也是有效输入
        output_path = args.output
    if args.modes:
        modes = args.modes
    if args.workers:
        max_workers = args.workers
    
    print("=" * 50)
    print("WSG MMFi数据提取工具")
    print("=" * 50)
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path if output_path else '自动生成'}")
    print(f"提取模式: {', '.join(modes)}")
    print(f"工作器数量: {max_workers}")
    print("=" * 50)
    
    process(input_path, output_path, modes, max_workers)


if __name__ == '__main__':
    main()