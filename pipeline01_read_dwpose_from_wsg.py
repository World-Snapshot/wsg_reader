import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from utils.draw_dw_lib import draw_pose, process_pose_data

STORED_VISUALIZATION_CHANNELS = slice(67, 70)


def load_wsg_data(wsg_path):
    """加载WSG数据"""
    try:
        data = np.load(wsg_path, allow_pickle=True)
        return data['wsg_data']
    except Exception as e:
        print(f"加载WSG数据失败 {wsg_path}: {str(e)}")
        return None


def get_dwpose_keypoint_mapping():
    """获取DWPose关键点映射（与注入器相同的逆向映射）"""
    reverse_mapping = {}
    current_id = 0
    
    # 身体关键点 (0-17)
    for i in range(18):
        reverse_mapping[current_id] = ('body', i)
        current_id += 1
    
    # 面部关键点 (18-85)
    for i in range(68):
        reverse_mapping[current_id] = ('face', i)
        current_id += 1
    
    # 左手关键点 (86-106)
    for i in range(21):
        reverse_mapping[current_id] = ('left_hand', i)
        current_id += 1
    
    # 右手关键点 (107-127)
    for i in range(21):
        reverse_mapping[current_id] = ('right_hand', i)
        current_id += 1
    
    return reverse_mapping


def extract_dwpose_from_wsg(wsg_array):
    """
    从WSG数组中提取DWPose数据，还原为原始格式
    
    返回格式：与您的DWPose代码输出格式相同
    {
        'num_persons': 2,
        'person_0': {
            'body_keypoints': np.array([[x, y], ...]),  # (18, 2)
            'body_scores': np.array([score, ...]),      # (18,)
            'face_keypoints': np.array([[x, y], ...]),  # (68, 2)
            'face_scores': np.array([score, ...]),      # (68,)
            'left_hand_keypoints': np.array([[x, y], ...]),  # (21, 2)
            'left_hand_scores': np.array([score, ...]),      # (21,)
            'right_hand_keypoints': np.array([[x, y], ...]), # (21, 2)
            'right_hand_scores': np.array([score, ...]),     # (21,)
        },
        ...
    }
    """
    if wsg_array is None:
        return None
    
    height, width = wsg_array.shape[:2]
    keypoint_mapping = get_dwpose_keypoint_mapping()
    
    # 提取所有有效的关键点像素
    person_channel = wsg_array[:, :, 64]  # Person ID
    keypoint_channel = wsg_array[:, :, 65]  # Keypoint ID
    score_channel = wsg_array[:, :, 66]  # Score
    
    # 找到所有非-1的像素（有效关键点）
    valid_mask = (person_channel != -1) & (keypoint_channel != -1) & (score_channel != -1)
    valid_coords = np.where(valid_mask)
    
    if len(valid_coords[0]) == 0:
        return None
    
    # 初始化人员数据结构
    persons_raw_data = {}
    
    # 提取原始关键点数据
    for i in range(len(valid_coords[0])):
        y, x = valid_coords[0][i], valid_coords[1][i]
        person_id = int(person_channel[y, x])
        keypoint_id = int(keypoint_channel[y, x])
        score = float(score_channel[y, x])
        
        # 确保person_id有效
        if person_id < 0 or keypoint_id not in keypoint_mapping:
            continue
            
        if person_id not in persons_raw_data:
            persons_raw_data[person_id] = {
                'body': {},      # {keypoint_idx: (x, y, score)}
                'face': {},
                'left_hand': {},
                'right_hand': {}
            }
        
        # 根据keypoint_id分类存储
        keypoint_type, keypoint_idx = keypoint_mapping[keypoint_id]
        # 将像素坐标转换回归一化坐标
        normalized_x = x / width
        normalized_y = y / height
        persons_raw_data[person_id][keypoint_type][keypoint_idx] = (normalized_x, normalized_y, score)
    
    if not persons_raw_data:
        return None
    
    # 转换为标准DWPose格式
    dwpose_data = {
        'num_persons': len(persons_raw_data)
    }
    
    for contiguous_person_id, person_id in enumerate(sorted(persons_raw_data)):
        raw_data = persons_raw_data[person_id]
        person_key = f'person_{contiguous_person_id}'
        
        # 初始化各部分的数组
        body_keypoints = np.full((18, 2), -1.0, dtype=np.float32)
        body_scores = np.full(18, -1.0, dtype=np.float32)
        
        face_keypoints = np.full((68, 2), -1.0, dtype=np.float32)
        face_scores = np.full(68, -1.0, dtype=np.float32)
        
        left_hand_keypoints = np.full((21, 2), -1.0, dtype=np.float32)
        left_hand_scores = np.full(21, -1.0, dtype=np.float32)
        
        right_hand_keypoints = np.full((21, 2), -1.0, dtype=np.float32)
        right_hand_scores = np.full(21, -1.0, dtype=np.float32)
        
        # 填充身体关键点
        for keypoint_idx, (x, y, score) in raw_data['body'].items():
            if 0 <= keypoint_idx < 18:
                body_keypoints[keypoint_idx] = [x, y]
                body_scores[keypoint_idx] = score
        
        # 填充面部关键点
        for keypoint_idx, (x, y, score) in raw_data['face'].items():
            if 0 <= keypoint_idx < 68:
                face_keypoints[keypoint_idx] = [x, y]
                face_scores[keypoint_idx] = score
        
        # 填充左手关键点
        for keypoint_idx, (x, y, score) in raw_data['left_hand'].items():
            if 0 <= keypoint_idx < 21:
                left_hand_keypoints[keypoint_idx] = [x, y]
                left_hand_scores[keypoint_idx] = score
        
        # 填充右手关键点
        for keypoint_idx, (x, y, score) in raw_data['right_hand'].items():
            if 0 <= keypoint_idx < 21:
                right_hand_keypoints[keypoint_idx] = [x, y]
                right_hand_scores[keypoint_idx] = score
        
        dwpose_data[person_key] = {
            'source_person_id': np.int32(person_id),
            'body_keypoints': body_keypoints,
            'body_scores': body_scores,
            'face_keypoints': face_keypoints,
            'face_scores': face_scores,
            'left_hand_keypoints': left_hand_keypoints,
            'left_hand_scores': left_hand_scores,
            'right_hand_keypoints': right_hand_keypoints,
            'right_hand_scores': right_hand_scores,
        }
    
    return dwpose_data


def extract_stored_dwpose_visualization(wsg_array):
    """Extract the dense black-background DWPose visualization from channels 67-69."""
    if wsg_array is None or wsg_array.shape[-1] < STORED_VISUALIZATION_CHANNELS.stop:
        return None
    stored = np.asarray(wsg_array[:, :, STORED_VISUALIZATION_CHANNELS], dtype=np.float32)
    if stored.shape[-1] != 3 or np.all(stored == -1):
        return None
    stored = np.where(stored == -1, 0.0, stored)
    finite = stored[np.isfinite(stored)]
    if finite.size == 0:
        return None
    if float(finite.max()) <= 1.0:
        stored = stored * 255.0
    return np.clip(stored, 0.0, 255.0).astype(np.uint8)


def process_single_wsg_file(args):
    """处理单个WSG文件"""
    wsg_path, output_dir, modes = args
    
    try:
        base_name = os.path.splitext(os.path.basename(wsg_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载WSG数据
        wsg_array = load_wsg_data(wsg_path)
        if wsg_array is None:
            print(f"  跳过 {base_name}: WSG数据加载失败")
            return False
        
        # 提取DWPose数据
        dwpose_data = extract_dwpose_from_wsg(wsg_array)
        stored_visualization = extract_stored_dwpose_visualization(wsg_array)
        if (
            (dwpose_data is None or dwpose_data['num_persons'] == 0)
            and stored_visualization is None
        ):
            print(f"  跳过 {base_name}: 未找到DWPose数据")
            return False
        
        height, width = wsg_array.shape[:2]
        if dwpose_data is not None:
            print(f"  {base_name}: 提取到 {dwpose_data['num_persons']} 个人的姿态数据")
        
        # 获取原始RGB图像（前3个通道）
        original_rgb = wsg_array[:, :, 0:3].astype(np.uint8)
        original_pil = Image.fromarray(original_rgb)
        
        success_count = 0

        if 'vis' in modes and stored_visualization is not None:
            stored_path = os.path.join(
                output_dir,
                f"export_dwpose_from_wsg_{base_name}_stored.png",
            )
            Image.fromarray(stored_visualization).save(stored_path)
            print(f"    保存通道67-69原始姿势可视化: {stored_path}")
            success_count += 1
        
        # 生成纯骨骼可视化图像
        if 'vis' in modes and dwpose_data is not None:
            try:
                vis_img = draw_pose(
                    pose=dwpose_data,
                    H=height,
                    W=width,
                    include_body=True,
                    include_hand=True,
                    include_face=True,
                    line_scale=3.0,
                    point_scale=2.0
                )
                
                # 处理通道顺序
                if isinstance(vis_img, np.ndarray):
                    if vis_img.shape[0] == 3 and len(vis_img.shape) == 3:
                        vis_img = np.transpose(vis_img, (1, 2, 0))
                    
                    if vis_img.dtype != np.uint8:
                        vis_img = (vis_img * 255).astype(np.uint8)
                    
                    pred_img = Image.fromarray(vis_img)
                    vis_path = os.path.join(output_dir, f"export_dwpose_from_wsg_{base_name}_vis.png")
                    pred_img.save(vis_path)
                    print(f"    保存骨骼可视化: {vis_path}")
                    success_count += 1
                    
            except Exception as e:
                print(f"    骨骼可视化失败: {str(e)}")
        
        # 生成叠加可视化图像
        if 'vis_overlay' in modes and dwpose_data is not None:
            try:
                transparent_vis = draw_pose(
                    pose=dwpose_data,
                    H=height,
                    W=width,
                    include_body=True,
                    include_hand=True,
                    include_face=True,
                    transparent=True,
                    line_scale=3.0,
                    point_scale=2.0
                )
                
                if isinstance(transparent_vis, np.ndarray):
                    if transparent_vis.shape[0] == 4 and len(transparent_vis.shape) == 3:
                        transparent_vis = np.transpose(transparent_vis, (1, 2, 0))
                    transparent_img = Image.fromarray(transparent_vis, 'RGBA')
                
                original_rgba = original_pil.convert('RGBA')
                overlay_img = Image.alpha_composite(original_rgba, transparent_img)
                
                overlay_path = os.path.join(output_dir, f"export_dwpose_from_wsg_{base_name}_overlay.png")
                overlay_img.save(overlay_path)
                print(f"    保存叠加可视化: {overlay_path}")
                success_count += 1
                
            except Exception as e:
                print(f"    叠加可视化失败: {str(e)}")
        
        # 保存NPZ数据
        if 'save' in modes and dwpose_data is not None:
            try:
                npz_path = os.path.join(output_dir, f"export_dwpose_from_wsg_{base_name}.npz")
                np.savez_compressed(npz_path, **dwpose_data)
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
            output_path = os.path.join(parent_dir, f"export_dwpose_from_{folder_name}")
        
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
                
                for future in tqdm(futures, desc="提取DWPose"):
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as e:
                        print(f"任务失败: {str(e)}")
        else:
            # 单线程处理
            for task in tqdm(tasks, desc="提取DWPose"):
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
    print("WSG DWPose提取和可视化工具")
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
