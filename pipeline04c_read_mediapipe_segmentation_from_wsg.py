import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import cv2

from categorical_visualization import categorical_colormap_soft, rounded_class_distribution


def load_wsg_data(wsg_path):
    """加载WSG数据"""
    try:
        data = np.load(wsg_path, allow_pickle=True)
        return data['wsg_data']
    except Exception as e:
        print(f"加载WSG数据失败 {wsg_path}: {str(e)}")
        return None

# MediaPipe分割类别定义
SEGMENTATION_CLASSES = {
    0: "background",
    1: "hair", 
    2: "body-skin",
    3: "face-skin",
    4: "clothes",
    5: "others"  # accessories
}

# 颜色映射 (BGR格式)
SEGMENTATION_COLORS = {
    0: (0, 0, 0),       # 背景 - 黑色
    1: (255, 0, 0),     # 头发 - 蓝色
    2: (0, 255, 255),   # 身体皮肤 - 黄色
    3: (255, 255, 0),   # 脸部皮肤 - 青色
    4: (0, 255, 0),     # 衣服 - 绿色
    5: (255, 0, 255),   # 配饰 - 洋红色
}

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
        
        # 提取MediaPipe Segmentation数据 (channel 103)
        segmentation_data = wsg_array[:, :, 103]
        
        # 检查是否有有效的分割数据
        if np.all(segmentation_data == -1):
            print(f"  跳过 {base_name}: 通道103没有分割数据")
            return False
        
        success_count = 0
        
        # 生成纯分割可视化图像
        if 'vis' in modes:
            try:
                colored_segmentation = categorical_colormap_soft(segmentation_data, SEGMENTATION_COLORS)
                
                vis_path = os.path.join(output_dir, f"export_mediapipe_segmentation_from_wsg_{base_name}_vis.png")
                cv2.imwrite(vis_path, colored_segmentation)
                print(f"    保存纯分割可视化: {vis_path}")
                
                # 统计各类别像素数
                class_dist = rounded_class_distribution(segmentation_data, tuple(SEGMENTATION_COLORS))
                print(f"    分割类别分布(按最近类别量化): {class_dist}")
                
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
                
                # 创建彩色分割图。VAE decode 后类别值可能是连续数，
                # 因此按类别色锚做插值，而不是只匹配精确整数。
                colored_segmentation = categorical_colormap_soft(segmentation_data, SEGMENTATION_COLORS)
                
                # 叠加到原图
                segmentation_overlay = cv2.addWeighted(original_cv2, 0.5, colored_segmentation, 0.7, 0)
                overlay_path = os.path.join(output_dir, f"export_mediapipe_segmentation_from_wsg_{base_name}_overlay.png")
                cv2.imwrite(overlay_path, segmentation_overlay)
                print(f"    保存叠加可视化: {overlay_path}")
                success_count += 1
            except Exception as e:
                print(f"    overlay visualize failed: {str(e)}")
        
        # 保存NPZ数据
        if 'save' in modes:
            try:
                npz_path = os.path.join(output_dir, f"export_mediapipe_segmentation_from_wsg_{base_name}.npz")
                np.savez_compressed(npz_path, segmentation_mask=segmentation_data)
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
            # 获取视频文件夹路径（wsg的上级目录）
            output_path = os.path.dirname(os.path.dirname(input_path))
        
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
            output_path = os.path.join(parent_dir, f"export_mediapipe_segmentation_from_{folder_name}")
        
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
                
                for future in tqdm(futures, desc="提取MediaPipe Segmentation"):
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as e:
                        print(f"任务失败: {str(e)}")
        else:
            # 单线程处理
            for task in tqdm(tasks, desc="提取MediaPipe Segmentation"):
                if process_single_wsg_file(task):
                    success_count += 1
        
        print(f"处理完成! 成功: {success_count}/{len(wsg_files)}")
    
    else:
        print(f"无效的输入路径: {input_path}")


def main():
    """主函数"""
    # ==================== 配置区域 ====================
    # 输入配置
    #input_path = "../output/test_dataset_results/human_signer_480/wsg"  # WSG文件或文件夹路径
    # input_path = "../output/test_dataset_results/human_mutiperson/wsg/00000004.wsg"  # 单个文件示例
    input_path = "../output/AVIS_val_results/00000105/wsg/00000002.wsg"
    
    # 输出配置
    output_path = None  # None表示使用默认输出路径
    
    # 模式配置
    modes = ['vis', 'vis_overlay', 'save']  # 可选: 'vis', 'vis_overlay', 'save'
    # modes = ['vis']  # 只生成分割图
    # modes = ['vis_overlay']  # 只生成叠加图
    # modes = ['save']  # 只保存NPZ
    
    # 处理配置
    max_workers = mp.cpu_count() // 2  # 并行工作器数量
    
    print("=" * 50)
    print("WSG MediaPipe Segmentation提取和可视化工具")
    print("=" * 50)
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path if output_path else '自动生成'}")
    print(f"处理模式: {', '.join(modes)}")
    print(f"目标通道: 103")
    print(f"工作器数量: {max_workers}")
    print("=" * 50)
    print("MediaPipe分割类别:")
    for class_id, class_name in SEGMENTATION_CLASSES.items():
        color = SEGMENTATION_COLORS[class_id]
        print(f"  {class_id}: {class_name} - RGB{color[::-1]}")
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
