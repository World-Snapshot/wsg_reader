import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import cv2
from normalized_wsg_loader import load_preferred_wsg_data


def valid_flow_mask(flow_data):
    """Exclude explicit and border-connected min-value WSG padding."""
    finite = np.all(np.isfinite(flow_data), axis=-1)
    valid = finite & np.all(flow_data != -1.0, axis=-1)
    if not valid.any():
        return valid

    valid_vectors = flow_data[valid]
    minima = np.min(valid_vectors, axis=0)
    spans = np.ptp(valid_vectors, axis=0)
    tolerance = np.maximum(spans * 1e-5, 1e-6)
    min_pair = valid & np.all(np.abs(flow_data - minima) <= tolerance, axis=-1)
    near_min_tolerance = np.maximum(spans * 0.05, 1e-4)
    near_min_pair = valid & np.all(np.abs(flow_data - minima) <= near_min_tolerance, axis=-1)
    ratio = float(min_pair.mean())
    # VAE decoding can soften the last pixel row of an otherwise constant
    # letterbox. Detect a broad min-pair band and extend it to the image edge.
    row_hits = np.flatnonzero(near_min_pair.mean(axis=1) >= 0.20)
    col_hits = np.flatnonzero(near_min_pair.mean(axis=0) >= 0.20)
    if 4 <= row_hits.size <= int(flow_data.shape[0] * 0.60) and row_hits[-1] >= flow_data.shape[0] - 5 and row_hits.size / float(flow_data.shape[0] - row_hits[0]) >= 0.80:
        valid[row_hits[0] :, :] = False
    if 4 <= row_hits.size <= int(flow_data.shape[0] * 0.60) and row_hits[0] <= 4 and row_hits.size / float(row_hits[-1] + 1) >= 0.80:
        valid[: row_hits[-1] + 1, :] = False
    if 4 <= col_hits.size <= int(flow_data.shape[1] * 0.60) and col_hits[-1] >= flow_data.shape[1] - 5 and col_hits.size / float(flow_data.shape[1] - col_hits[0]) >= 0.80:
        valid[:, col_hits[0] :] = False
    if 4 <= col_hits.size <= int(flow_data.shape[1] * 0.60) and col_hits[0] <= 4 and col_hits.size / float(col_hits[-1] + 1) >= 0.80:
        valid[:, : col_hits[-1] + 1] = False
    if 0.01 <= ratio <= 0.60:
        count, labels = cv2.connectedComponents(min_pair.astype(np.uint8), connectivity=4)
        padding = np.zeros_like(min_pair)
        min_size = max(1, int(min_pair.size * 0.01))
        for label in range(1, count):
            component = labels == label
            if int(component.sum()) < min_size:
                continue
            if component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any():
                padding |= component
        valid &= ~padding
    return valid


def median_centered_flow(flow_data, valid_flow, radius_multiplier=1.5):
    """Return dominant-motion-compensated flow for Reader visualization."""
    if not valid_flow.any():
        return np.zeros_like(flow_data), np.zeros(2, dtype=np.float32), 1.0, 0.0
    center = np.median(flow_data[valid_flow], axis=0).astype(np.float32)
    centered = np.where(valid_flow[..., None], flow_data - center, 0.0)
    magnitude = np.linalg.norm(centered, axis=-1)
    useful = magnitude[valid_flow]
    p95 = float(np.percentile(useful, 95)) if useful.size else 0.0
    radius = max(p95, 1.0) * float(radius_multiplier)
    return centered, center, radius, p95


def process_single_wsg_file(args):
    """处理单个WSG文件"""
    wsg_path, output_dir, modes = args

    try:
        base_name = os.path.splitext(os.path.basename(wsg_path))[0]

        wsg_array, _used_normalized = load_preferred_wsg_data(
            wsg_path, [0, 1, 2, 41, 42]
        )

        if wsg_array is None:
            print(f"  跳过 {base_name}: WSG数据加载失败")
            return False

        # 提取opticalflow数据 (channels 41-42)
        flow_data = wsg_array[:, :, 41:43].astype(np.float32, copy=False)
        valid_flow = valid_flow_mask(flow_data)
        flow_for_display = np.where(valid_flow[..., None], flow_data, 0.0)
        centered_flow, flow_center, centered_radius, centered_p95 = median_centered_flow(
            flow_data, valid_flow, radius_multiplier=1.5
        )

        def display_radius():
            magnitude = np.linalg.norm(flow_for_display, axis=-1)
            useful = magnitude[valid_flow]
            p95 = float(np.percentile(useful, 95)) if useful.size else 0.0
            return max(p95, 1.0), p95

        success_count = 0

        # 生成opticalflow可视化图像
        if 'vis' in modes:
            try:
                import torch
                from ptlflow.utils import flow_utils
                radius, p95 = display_radius()
                flows = torch.from_numpy(flow_for_display).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                flows = flows.to(torch.float32).to(torch.device('cuda'))
                flow_rgb = flow_utils.flow_to_rgb(
                    flows, flow_max_radius=radius, background="dark"
                )
                flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
                flow_rgb = flow_rgb.detach().cpu().numpy()
                flow_color = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)
                flow_color = (flow_color * 255).clip(0, 255).astype(np.uint8)

                vis_path = os.path.join(output_dir, f"export_opticalflow_from_wsg_{base_name}_vis.png")
                cv2.imwrite(vis_path, flow_color)
                print(f"    保存Optical Flow可视化 (valid-P95={p95:.2f}): {vis_path}")
                success_count += 1

                # Default paper-facing Reader style: subtract the valid-flow
                # median (dominant background/camera motion), then map centered
                # P95 to two thirds of the bright PTLFlow color-wheel radius.
                # The raw WSG tensor and dark-background visualization remain
                # unchanged; only this display view is motion compensated.
                soft_flows = torch.from_numpy(centered_flow).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                soft_flows = soft_flows.to(torch.float32).to(torch.device("cuda"))
                soft_radius = centered_radius
                flow_rgb_soft = flow_utils.flow_to_rgb(
                    soft_flows, flow_max_radius=soft_radius, background="bright"
                )
                flow_rgb_soft = flow_rgb_soft[0, 0].permute(1, 2, 0)
                flow_rgb_soft = flow_rgb_soft.detach().cpu().numpy()
                flow_soft_color = cv2.cvtColor(flow_rgb_soft, cv2.COLOR_RGB2BGR)
                flow_soft_color = (flow_soft_color * 255).clip(0, 255).astype(np.uint8)
                vis_soft_path = os.path.join(
                    output_dir, f"export_opticalflow_from_wsg_{base_name}_vis_soft.png"
                )
                cv2.imwrite(vis_soft_path, flow_soft_color)
                print(
                    f"    保存Optical Flow浅色可视化 "
                    f"(PTLFlow median-centered, center={flow_center.tolist()}, "
                    f"P95={centered_p95:.2f}, radius={soft_radius:.2f}): {vis_soft_path}"
                )
                success_count += 1
            except Exception as e:
                print(f"    visualize failed: {str(e)}")
        
        # 生成叠加可视化图像
        if 'vis_overlay' in modes:
            try:
                # 获取原始RGB图像（前3个通道）
                import torch
                from ptlflow.utils import flow_utils
                original_rgb = wsg_array[:, :, :3].astype(np.uint8)
                original_pil = Image.fromarray(original_rgb)
                original_cv2 = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
                flows = torch.from_numpy(centered_flow).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                flows = flows.to(torch.float32).to(torch.device('cuda'))
                flow_rgb = flow_utils.flow_to_rgb(
                    flows, flow_max_radius=centered_radius, background="bright"
                )
                flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
                flow_rgb = flow_rgb.detach().cpu().numpy()
                flow_color = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)
                flow_color = (flow_color * 255).clip(0, 255).astype(np.uint8)
                flow_overlay = cv2.addWeighted(original_cv2, 0.5, flow_color, 0.5, 0)
                overlay_path = os.path.join(output_dir, f"export_opticalflow_from_wsg_{base_name}_overlay.png")
                cv2.imwrite(overlay_path, flow_overlay)
                print(f"    保存叠加可视化 (median-centered P95={centered_p95:.2f}, "
                      f"radius={centered_radius:.2f}): {overlay_path}")
                success_count += 1
            except Exception as e:
                print(f"    overlay visualize failed: {str(e)}")
        
        # 保存NPZ数据
        if 'save' in modes:
            try:
                npz_path = os.path.join(output_dir, f"export_opticalflow_from_wsg_{base_name}.npz")
                np.savez_compressed(npz_path, flow_data=flow_data)
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
            output_path = os.path.join(parent_dir, f"export_opticalflow_from_{folder_name}")
        
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
                
                for future in tqdm(futures, desc="提取opticalflow"):
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as e:
                        print(f"任务失败: {str(e)}")
        else:
            # 单线程处理
            for task in tqdm(tasks, desc="提取opticalflow"):
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
    #input_path = "../output/AVIS_train_results/_wco68VEGxE-00.03.30.631-00.04.36.384-seg2/wsg"

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
    print("WSG Optical Flow提取和可视化工具")
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
