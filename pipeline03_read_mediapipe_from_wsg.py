import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import cv2
import json

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

# Equivalent to the current MediaPipe `get_default_pose_landmarks_style()`
# and `drawing_utils.draw_landmarks()` defaults, expressed in OpenCV BGR.
OFFICIAL_CONNECTION_COLOR = (224, 224, 224)
OFFICIAL_LEFT_COLOR = (0, 138, 255)
OFFICIAL_RIGHT_COLOR = (231, 217, 0)
OFFICIAL_NOSE_COLOR = (224, 224, 224)
OFFICIAL_LEFT_LANDMARKS = frozenset(
    [1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
)
OFFICIAL_RIGHT_LANDMARKS = frozenset(
    [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
)

PERSON_ID_CHANNEL = 95
KEYPOINT_ID_CHANNEL = 96
SCORE_CHANNEL = 98
Z_CHANNEL = 97
STORED_VIS_CHANNELS = slice(99, 102)

# 33点的逆向映射
def get_mediapipe_keypoint_mapping():
    return {i: f"pose_{i}" for i in range(33)}

def extract_mediapipe_from_wsg(wsg_array):
    if wsg_array is None:
        return None
    height, width = wsg_array.shape[:2]
    keypoint_mapping = get_mediapipe_keypoint_mapping()
    person_channel = wsg_array[:, :, PERSON_ID_CHANNEL]
    keypoint_channel = wsg_array[:, :, KEYPOINT_ID_CHANNEL]
    score_channel = wsg_array[:, :, SCORE_CHANNEL]
    z_channel = wsg_array[:, :, Z_CHANNEL]
    valid_mask = (person_channel != -1) & (keypoint_channel != -1) & (score_channel != -1)
    valid_coords = np.where(valid_mask)
    if len(valid_coords[0]) == 0:
        return None
    persons_raw_data = {}
    for i in range(len(valid_coords[0])):
        y, x = valid_coords[0][i], valid_coords[1][i]
        person_id = int(person_channel[y, x])
        keypoint_id = int(keypoint_channel[y, x])
        score = float(score_channel[y, x])
        z = float(z_channel[y, x])
        if person_id < 0 or keypoint_id not in keypoint_mapping:
            continue
        if person_id not in persons_raw_data:
            persons_raw_data[person_id] = {}
        # 归一化坐标
        normalized_x = x / width
        normalized_y = y / height
        persons_raw_data[person_id][keypoint_id] = (normalized_x, normalized_y, z, score)
    if not persons_raw_data:
        return None
    mediapipe_data = {}
    for person_id, raw_data in persons_raw_data.items():
        keypoints = np.full((33, 4), -1.0, dtype=np.float32)
        for kp_idx, (x, y, z, score) in raw_data.items():
            if 0 <= kp_idx < 33:
                keypoints[kp_idx, 0] = x
                keypoints[kp_idx, 1] = y
                keypoints[kp_idx, 2] = z
                keypoints[kp_idx, 3] = score
        mediapipe_data[f'person_{person_id}'] = keypoints
    return mediapipe_data


def extract_stored_mediapipe_visualization(wsg_array):
    """Extract the dense black-background pose visualization from channels 99-101."""
    if wsg_array is None or wsg_array.shape[-1] < STORED_VIS_CHANNELS.stop:
        return None
    stored = np.asarray(wsg_array[:, :, STORED_VIS_CHANNELS], dtype=np.float32)
    if stored.shape[-1] != 3 or np.all(stored == -1):
        return None
    stored = np.where(stored == -1, 0.0, stored)
    finite = stored[np.isfinite(stored)]
    if finite.size == 0:
        return None
    if float(finite.max()) <= 1.0:
        stored = stored * 255.0
    return np.clip(stored, 0.0, 255.0).astype(np.uint8)

def _keypoint_to_pixel(keypoint, width, height):
    x, y = float(keypoint[0]), float(keypoint[1])
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return min(int(x * width), width - 1), min(int(y * height), height - 1)


def _draw_official_pose(image, keypoints, width, height, confidence_threshold):
    """Reproduce MediaPipe's default Pose `draw_landmarks` visual style."""
    visible = {
        index: _keypoint_to_pixel(keypoint, width, height)
        for index, keypoint in enumerate(keypoints)
        if keypoint[3] >= confidence_threshold
    }
    visible = {index: pixel for index, pixel in visible.items() if pixel is not None}

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in visible and end_idx in visible:
            cv2.line(
                image,
                visible[start_idx],
                visible[end_idx],
                OFFICIAL_CONNECTION_COLOR,
                2,
            )

    for index, pixel in visible.items():
        if index in OFFICIAL_LEFT_LANDMARKS:
            color = OFFICIAL_LEFT_COLOR
        elif index in OFFICIAL_RIGHT_LANDMARKS:
            color = OFFICIAL_RIGHT_COLOR
        else:
            color = OFFICIAL_NOSE_COLOR
        cv2.circle(image, pixel, 3, OFFICIAL_CONNECTION_COLOR, 2)
        cv2.circle(image, pixel, 2, color, 2)


def _draw_legacy_pose(image, keypoints, width, height, color, confidence_threshold):
    for start_idx, end_idx in POSE_CONNECTIONS:
        if (keypoints[start_idx, 3] > confidence_threshold and
                keypoints[end_idx, 3] > confidence_threshold):
            start_point = _keypoint_to_pixel(keypoints[start_idx], width, height)
            end_point = _keypoint_to_pixel(keypoints[end_idx], width, height)
            if start_point is not None and end_point is not None:
                cv2.line(image, start_point, end_point, color, 3)

    for keypoint in keypoints:
        if keypoint[3] > confidence_threshold:
            point = _keypoint_to_pixel(keypoint, width, height)
            if point is not None:
                cv2.circle(image, point, 6, color, -1)
                cv2.circle(image, point, 6, (255, 255, 255), 2)


def _draw_pose(image, mediapipe_data, width, height, confidence_threshold, style):
    if style not in {"official", "legacy"}:
        raise ValueError(f"Unsupported MediaPipe pose drawing style: {style}")

    if style == "official":
        for keypoints in mediapipe_data.values():
            _draw_official_pose(image, keypoints, width, height, confidence_threshold)
        return image

    # 创建黑色背景
    # 为不同的人分配不同的颜色
    colors = [
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色 (OpenCV是BGR格式)
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 品红色
        (0, 255, 255),  # 黄色
        (128, 0, 128),  # 紫色
        (255, 165, 0),  # 橙色
    ]
    
    for person_idx, (person_id, keypoints) in enumerate(mediapipe_data.items()):
        color = colors[person_idx % len(colors)]
        _draw_legacy_pose(image, keypoints, width, height, color, confidence_threshold)
    return image


def draw_pose_on_black_background(
    mediapipe_data, width, height, confidence_threshold=0.5, style="official"
):
    """Draw the pose on black, using the official MediaPipe style by default."""
    pose_image = np.zeros((height, width, 3), dtype=np.uint8)
    _draw_pose(pose_image, mediapipe_data, width, height, confidence_threshold, style)
    return pose_image


def draw_pose_overlay(
    original_rgb,
    mediapipe_data,
    width,
    height,
    confidence_threshold=0.5,
    alpha=0.7,
    style="official",
):
    """Overlay the pose, using the official MediaPipe style by default."""
    overlay = original_rgb.copy()
    _draw_pose(overlay, mediapipe_data, width, height, confidence_threshold, style)
    result = cv2.addWeighted(original_rgb, 1-alpha, overlay, alpha, 0)
    return result

def process_single_wsg_file(args):
    wsg_path, output_dir, modes = args
    try:
        base_name = os.path.splitext(os.path.basename(wsg_path))[0]
        wsg_array = np.load(wsg_path, allow_pickle=True)['wsg_data']
        mediapipe_data = extract_mediapipe_from_wsg(wsg_array)
        stored_visualization = extract_stored_mediapipe_visualization(wsg_array)

        if mediapipe_data is None and stored_visualization is None:
            print(f"  跳过 {base_name}: 未找到mediapipe数据")
            return False
        
        height, width = wsg_array.shape[:2]
        
        # 获取原始RGB图像并转换为正确格式
        wsg_rgb_channels = wsg_array[:, :, 0:3]
        if wsg_rgb_channels.max() <= 1.0:
            original_rgb = (wsg_rgb_channels * 255).astype(np.uint8)
        else:
            original_rgb = wsg_rgb_channels.astype(np.uint8)
        
        # 转换为BGR格式供OpenCV使用
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        
        # 黑色背景骨骼可视化
        if 'vis' in modes and mediapipe_data is not None:
            pose_image = draw_pose_on_black_background(mediapipe_data, width, height)
            vis_path = os.path.join(output_dir, f"export_mediapipe_from_wsg_{base_name}_vis.png")
            cv2.imwrite(vis_path, pose_image)
            print(f"    保存骨骼可视化(黑背景): {vis_path}")

        if 'vis' in modes and stored_visualization is not None:
            stored_path = os.path.join(
                output_dir,
                f"export_mediapipe_from_wsg_{base_name}_stored.png",
            )
            cv2.imwrite(stored_path, cv2.cvtColor(stored_visualization, cv2.COLOR_RGB2BGR))
            print(f"    保存通道99-101原始姿势可视化: {stored_path}")

        # 叠加可视化
        if 'vis_overlay' in modes and mediapipe_data is not None:
            overlay_image = draw_pose_overlay(original_bgr, mediapipe_data, width, height)
            overlay_path = os.path.join(output_dir, f"export_mediapipe_from_wsg_{base_name}_overlay.png")
            cv2.imwrite(overlay_path, overlay_image)
            print(f"    保存叠加可视化: {overlay_path}")
        
        # 保存原始图像（用于对比）
        if 'original' in modes:
            original_path = os.path.join(output_dir, f"export_mediapipe_from_wsg_{base_name}_original.png")
            cv2.imwrite(original_path, original_bgr)
            print(f"    保存原始图像: {original_path}")
        
        # 保存NPZ数据
        if 'save' in modes and mediapipe_data is not None:
            npz_path = os.path.join(output_dir, f"export_mediapipe_from_wsg_{base_name}.npz")
            np.savez_compressed(npz_path, **mediapipe_data)
            print(f"    保存NPZ数据: {npz_path}")
        
        # 保存JSON数据
        if 'json' in modes and mediapipe_data is not None:
            json_path = os.path.join(output_dir, f"export_mediapipe_from_wsg_{base_name}.json")
            with open(json_path, 'w') as f:
                json.dump({k: v.tolist() for k, v in mediapipe_data.items()}, f, indent=2)
            print(f"    保存JSON数据: {json_path}")
        
        return True
    except Exception as e:
        print(f"  处理文件失败 {wsg_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_wsg_input(input_path, output_path, modes, max_workers=4):
    if os.path.isfile(input_path) and input_path.endswith('.wsg'):
        if output_path is None:
            output_path = os.path.dirname(input_path)
        os.makedirs(output_path, exist_ok=True)
        print(f"处理单个WSG文件: {input_path}")
        print(f"输出目录: {output_path}")
        success = process_single_wsg_file((input_path, output_path, modes))
        print(f"处理完成! 成功: {'是' if success else '否'}")
    elif os.path.isdir(input_path):
        if output_path is None:
            parent_dir = os.path.dirname(input_path.rstrip('/'))
            folder_name = os.path.basename(input_path.rstrip('/'))
            output_path = os.path.join(parent_dir, f"export_mediapipe_from_{folder_name}")
        os.makedirs(output_path, exist_ok=True)
        wsg_files = glob.glob(os.path.join(input_path, "*.wsg"))
        wsg_files.sort()
        if not wsg_files:
            print(f"在目录 {input_path} 中没有找到WSG文件")
            return
        print(f"处理WSG文件夹: {input_path}")
        print(f"输出目录: {output_path}")
        print(f"找到 {len(wsg_files)} 个WSG文件")
        tasks = [(wsg_path, output_path, modes) for wsg_path in wsg_files]
        success_count = 0
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_single_wsg_file, task) for task in tasks]
                for future in tqdm(futures, desc="提取mediapipe"):
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as e:
                        print(f"任务失败: {str(e)}")
        else:
            for task in tqdm(tasks, desc="提取mediapipe"):
                if process_single_wsg_file(task):
                    success_count += 1
        print(f"处理完成! 成功: {success_count}/{len(wsg_files)}")
    else:
        print(f"无效的输入路径: {input_path}")

def main():
    # 自动遍历所有视频文件夹
    input_base = "../output/test_dataset_results"
    modes = ['vis', 'vis_overlay', 'original', 'save', 'json']  # 可选: 'vis', 'vis_overlay', 'original', 'save', 'json'
    max_workers = mp.cpu_count() // 2
    
    print("=" * 50)
    print("修复版本 - WSG mediapipe提取和可视化工具")
    print("=" * 50)
    print("输出模式说明:")
    print("  vis: 黑色背景骨骼图")
    print("  vis_overlay: 原图叠加骨骼")
    print("  original: 原始图像")
    print("  save: NPZ格式数据")
    print("  json: JSON格式数据")
    print("=" * 50)
    
    folders = [f for f in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, f))]
    for folder in folders:
        wsg_dir = os.path.join(input_base, folder, "wsg")
        if os.path.isdir(wsg_dir):
            print(f"\n>>> 处理: {wsg_dir}")
            process_wsg_input(wsg_dir, None, modes, max_workers)
    
    print("\n✅ 所有提取任务完成!")

if __name__ == '__main__':
    main()
