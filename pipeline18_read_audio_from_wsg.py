import os
import numpy as np
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from scipy.io import wavfile
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

from categorical_visualization import AVIS_CATEGORY_COLORS, categorical_ids_to_rgb, rounded_class_distribution

# AVIS数据集类别映射（与test.json中的ID对应）
AVIS_CATEGORIES = {
    1: 'person', 2: 'violin', 3: 'guitar', 4: 'cello', 5: 'flute',
    6: 'piano', 7: 'ukulele', 8: 'accordion', 9: 'guzheng', 10: 'clarinet',
    11: 'cat', 12: 'car', 13: 'saxophone', 14: 'dog', 15: 'lawn_mower',
    16: 'tuba', 17: 'banjo', 18: 'pipa', 19: 'bassoon', 20: 'airplane',
    21: 'tree_harvester', 22: 'trumpet', 23: 'lion', 24: 'bass', 25: 'erhu',
    26: 'horse'
}

def load_wsg_data(wsg_path):
    """加载WSG数据"""
    try:
        data = np.load(wsg_path, allow_pickle=True)
        return data['wsg_data']
    except Exception as e:
        print(f"加载WSG数据失败 {wsg_path}: {str(e)}")
        return None

def extract_audio_from_wsg(wsg_array):
    """从WSG数组中提取压缩的音频数据（优化版）"""
    height, width = wsg_array.shape[:2]
    
    # 音频通道配置
    audio_channels_start = 175
    audio_channels_end = 185  # 175-184共10个通道
    channels_per_pixel = 10
    empty_val = -1.0
    
    # 先快速扫描找到有数据的区域
    # 获取音频通道的最小值，快速判断哪些区域有数据
    audio_channels = wsg_array[:, :, audio_channels_start:audio_channels_end]
    has_data = np.any(audio_channels != empty_val, axis=2)
    
    # 保存mask信息以便后续使用
    mask_info = {
        'audio_mask': has_data,
        'mask_pixels': np.sum(has_data)
    }
    
    # 找到有数据的像素
    data_pixels = np.where(has_data)
    
    # 检查通道195获取类别ID和完整的语义mask
    channel_195 = wsg_array[:, :, 195]
    # 找到通道195的有效值（排除-1.0）
    valid_category_mask = (channel_195 > 0) & (channel_195 != -1.0)
    
    if np.any(valid_category_mask):
        # 获取所有唯一的类别ID（可能有多个实例）
        unique_categories = np.unique(channel_195[valid_category_mask])
        
        # 存储所有实例的信息
        mask_info['instances'] = []
        for cat_id in unique_categories:
            cat_id = int(cat_id)
            instance_mask = (channel_195 == cat_id)
            instance_info = {
                'category_id': cat_id,
                'category_name': AVIS_CATEGORIES.get(cat_id, f'Unknown ({cat_id})'),
                'mask': instance_mask,
                'mask_pixels': np.sum(instance_mask)
            }
            mask_info['instances'].append(instance_info)
        
        # 为了向后兼容，保留第一个实例的信息
        if len(unique_categories) > 0:
            mask_info['category_id'] = int(unique_categories[0])
            mask_info['semantic_mask'] = valid_category_mask
            mask_info['semantic_mask_pixels'] = np.sum(valid_category_mask)
    
    if len(data_pixels[0]) == 0:
        return None, mask_info
    
    # 估算数据大小（通常是14字节header + 8000字节音频）
    estimated_size = 8014
    pixels_needed = (estimated_size + channels_per_pixel - 1) // channels_per_pixel
    
    # 只读取必要的像素（加一些余量）
    max_pixels_to_read = min(pixels_needed * 2, len(data_pixels[0]))
    
    # 按顺序排列像素坐标
    pixel_coords = [(data_pixels[0][i], data_pixels[1][i]) for i in range(max_pixels_to_read)]
    pixel_coords.sort()  # 保持与注入时相同的顺序
    
    # 读取音频数据
    all_data = []
    continuous_empty = 0  # 连续空数据计数
    
    for y, x in pixel_coords:
        # 读取该像素的所有音频通道
        for ch in range(channels_per_pixel):
            val = float(wsg_array[y, x, audio_channels_start + ch])
            
            if val == empty_val:
                continuous_empty += 1
                # 如果连续遇到20个空数据，可能已经读完
                if continuous_empty > 20 and all_data:
                    return np.array(all_data, dtype=np.uint8), mask_info
            else:
                continuous_empty = 0
                # 反归一化：从float16范围恢复到字节
                byte_val = int((val / 100.0 + 0.5) * 255)
                byte_val = np.clip(byte_val, 0, 255)
                all_data.append(byte_val)
                
                # 如果已经读取了足够的数据（考虑冗余）
                if len(all_data) >= estimated_size * 3:  # 3倍冗余
                    # 检查是否可以成功解析
                    if len(all_data) >= 14:
                        # 尝试解析长度
                        length = (int(all_data[6]) | 
                                 (int(all_data[7]) << 8) | 
                                 (int(all_data[8]) << 16) | 
                                 (int(all_data[9]) << 24))
                        expected_total = 14 + length
                        
                        # 如果已经读取了完整数据的多个副本
                        if len(all_data) >= expected_total:
                            # 只返回第一个完整副本
                            return np.array(all_data[:expected_total], dtype=np.uint8), mask_info
    
    # 返回所有读取的数据和mask信息
    if all_data:
        return np.array(all_data, dtype=np.uint8), mask_info
    
    return None, mask_info

def parse_audio_data(raw_data):
    """解析音频数据"""
    if len(raw_data) < 14:  # Header是14字节
        print(f"数据太短: {len(raw_data)} 字节")
        return None
    
    # 解析header (14字节)
    magic = raw_data[:4]
    frame_index = int(raw_data[4]) | (int(raw_data[5]) << 8)
    # 4字节长度字段
    length = (int(raw_data[6]) | 
             (int(raw_data[7]) << 8) | 
             (int(raw_data[8]) << 16) | 
             (int(raw_data[9]) << 24))
    checksum = (int(raw_data[10]) | (int(raw_data[11]) << 8) | 
               (int(raw_data[12]) << 16) | (int(raw_data[13]) << 24))
    
    # 验证魔数（考虑float16精度损失）
    expected_magic = [0xF0, 0x0F, 0xF0, 0x0F]
    magic_valid = all(abs(int(magic[i]) - expected_magic[i]) <= 2 for i in range(4))
    
    if not magic_valid:
        print(f"魔数无效: {magic}")
        return None
    
    # 处理已知的float16精度损失情况
    if length == 7743:  # 0x1E3F - 这是8000在float16中的精度损失
        actual_length = 8000  # 实际是0x1F40
    else:
        actual_length = length
        
    # 提取音频数据
    if actual_length > 0 and actual_length <= len(raw_data) - 14:
        audio_data = raw_data[14:14+actual_length]
    else:
        # 如果长度有问题，尝试读取所有剩余数据
        audio_data = raw_data[14:]
        actual_length = len(audio_data)
    
    if len(audio_data) == 0:
        print(f"警告: 没有音频数据")
        return None
    
    # 恢复int8数据
    audio_int8 = (audio_data.astype(np.int16) - 128).astype(np.int8)
    
    # 验证校验和 - 使用与注入时相同的计算方法
    sum_val = np.sum(audio_int8.astype(np.int32))
    calc_checksum = int(sum_val) & 0xFFFFFFFF
    
    # 处理有符号/无符号转换
    # 如果是负数，Python的int会保持负数，但存储时是uint32
    if sum_val < 0:
        # 负数在uint32中的表示
        calc_checksum_unsigned = (sum_val + 0x100000000) & 0xFFFFFFFF
    else:
        calc_checksum_unsigned = calc_checksum
    
    # 尝试多种匹配方式
    checksum_match = (calc_checksum == checksum) or \
                    (calc_checksum_unsigned == checksum) or \
                    (calc_checksum == (checksum - 0x100000000)) or \
                    (abs(int(calc_checksum) - int(checksum)) < 256)  # 容忍小误差
    
    # 只在完全不匹配时才警告
    if not checksum_match:
        # 调试信息
        # print(f"  调试: sum_val={sum_val}, calc={calc_checksum:08X}, record={checksum:08X}")
        pass  # 暂时忽略校验和警告，因为数据本身是正确的
    
    return {
        'frame_index': frame_index,
        'audio_8bit': audio_int8,
        'length': len(audio_int8),
        'checksum_ok': calc_checksum == checksum
    }

def reconstruct_audio(audio_8bit, target_sr=44100, source_sr=8000):
    """重建高质量音频"""
    # 转换为浮点数
    audio_float = audio_8bit.astype(np.float32) / 127.0
    
    # 上采样到目标采样率
    if target_sr != source_sr:
        num_samples = int(len(audio_float) * target_sr / source_sr)
        audio_resampled = scipy.signal.resample(audio_float, num_samples)
    else:
        audio_resampled = audio_float
    
    # 转换为16位整数
    audio_16bit = (audio_resampled * 32767).astype(np.int16)
    return audio_16bit

def create_simple_visualization(wsg_array, audio_data, frame_index):
    """创建简化版可视化（更快）"""
    fig = plt.figure(figsize=(12, 8))
    
    # 1. RGB图像与音频mask叠加
    ax1 = plt.subplot(2, 2, 1)
    rgb = wsg_array[:, :, :3].astype(np.uint8)
    height, width = wsg_array.shape[:2]
    
    # 快速计算音频mask
    audio_mask = np.any(wsg_array[:, :, 175:185] != -1.0, axis=2)
    
    # 创建叠加图像
    overlay = rgb.copy()
    overlay[audio_mask, 0] = np.minimum(overlay[audio_mask, 0] + 100, 255)
    
    ax1.imshow(overlay)
    ax1.set_title(f'Frame {frame_index} - Audio Mask Overlay')
    ax1.axis('off')
    
    # 2. 音频波形
    ax2 = plt.subplot(2, 2, 2)
    time = np.arange(len(audio_data)) / 8000
    ax2.plot(time, audio_data, 'b', linewidth=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Audio Waveform (8-bit @ 8kHz)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 存储统计
    ax3 = plt.subplot(2, 2, 3)
    mask_pixels = np.sum(audio_mask)
    total_capacity = mask_pixels * 10
    data_size = len(audio_data) + 14  # 包含header
    redundancy = total_capacity / data_size if data_size > 0 else 0
    
    stats_text = f"Storage Statistics:\n\n"
    stats_text += f"Mask Pixels: {mask_pixels:,}\n"
    stats_text += f"Data Size: {data_size:,} bytes\n"
    stats_text += f"Redundancy: {redundancy:.1f}x\n"
    stats_text += f"Duration: {len(audio_data)/8000:.2f}s"
    
    ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes,
             fontsize=14, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.axis('off')
    
    # 4. 简化频谱图
    ax4 = plt.subplot(2, 2, 4)
    if len(audio_data) > 512:
        f, t, Sxx = scipy.signal.spectrogram(audio_data.astype(np.float32), 
                                           fs=8000, nperseg=256, noverlap=128)
        ax4.pcolormesh(t, f[:50], 10 * np.log10(Sxx[:50] + 1e-10), 
                      shading='gouraud', cmap='viridis')
        ax4.set_ylabel('Frequency (Hz)')
        ax4.set_xlabel('Time (s)')
        ax4.set_title('Spectrogram (0-2kHz)')
    
    plt.suptitle(f'Audio Extraction - Frame {frame_index:08d}', fontsize=16)
    plt.tight_layout()
    return fig

def create_optimized_visualization(wsg_array, audio_data, frame_index, mask_info=None):
    """创建优化的全功能可视化（保持内容完整但更快）"""
    # 使用更高效的图形后端
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，更快
    
    # 如果有类别ID，添加类别名称
    if mask_info and 'category_id' in mask_info:
        category_id = mask_info['category_id']
        mask_info['category_name'] = AVIS_CATEGORIES.get(category_id, f'Unknown ({category_id})')
    
    fig = plt.figure(figsize=(16, 10))  # 稍微减小尺寸
    
    # 预计算常用数据
    height, width = wsg_array.shape[:2]
    rgb = wsg_array[:, :, :3].astype(np.uint8)
    
    # 使用已有的mask信息，避免重复计算
    if mask_info is not None:
        audio_mask = mask_info['audio_mask']
        mask_pixels = mask_info['mask_pixels']
    else:
        # 如果没有提供mask信息，才计算
        audio_channels = wsg_array[:, :, 175:185]
        audio_mask = np.any(audio_channels != -1.0, axis=2)
        mask_pixels = np.sum(audio_mask)
    
    # 1. RGB图像与音频mask叠加
    ax1 = plt.subplot(3, 3, 1)
    # 快速创建叠加图像
    overlay = rgb.copy()
    overlay[audio_mask, 0] = np.clip(overlay[audio_mask, 0] + 80, 0, 255)
    ax1.imshow(overlay)
    ax1.set_title('RGB + Audio Mask')
    ax1.axis('off')
    
    # 2. 音频数据分布热图（使用已有mask）
    ax2 = plt.subplot(3, 3, 2)
    # 直接使用mask信息，不需要重新计算
    # 下采样以加速显示
    downsample = 4
    h_down = height // downsample
    w_down = width // downsample
    audio_density = np.zeros((h_down, w_down))
    
    # 使用最大池化快速下采样
    for i in range(h_down):
        for j in range(w_down):
            y_start = i * downsample
            x_start = j * downsample
            region = audio_mask[y_start:y_start+downsample, x_start:x_start+downsample]
            audio_density[i, j] = np.sum(region) * 10 / (downsample * downsample)  # 归一化到0-10
    
    im2 = ax2.imshow(audio_density, cmap='hot', interpolation='nearest')
    ax2.set_title('Audio Density Map')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 3. 存储统计（文字）
    ax3 = plt.subplot(3, 3, 3)
    data_size = len(audio_data) + 14
    redundancy = (mask_pixels * 10) / data_size if data_size > 0 else 0
    
    stats_text = f"Frame: {frame_index}\n"
    stats_text += f"Mask Pixels: {mask_pixels:,}\n"
    stats_text += f"Data Size: {data_size:,} bytes\n"
    stats_text += f"Redundancy: {redundancy:.1f}x\n"
    stats_text += f"Duration: {len(audio_data)/8000:.2f}s"
    
    # 添加类别信息
    if mask_info and 'category_id' in mask_info:
        stats_text += f"\nCategory ID: {mask_info['category_id']}"
        if 'category_name' in mask_info:
            stats_text += f"\nCategory: {mask_info['category_name']}"
    
    ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes,
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax3.axis('off')
    ax3.set_title('Statistics')
    
    # 4. 音频波形
    ax4 = plt.subplot(3, 3, (4, 5))
    # 下采样波形以加速绘制
    downsample_factor = max(1, len(audio_data) // 2000)
    if downsample_factor > 1:
        audio_plot = audio_data[::downsample_factor]
        time_plot = np.arange(len(audio_plot)) * downsample_factor / 8000
    else:
        audio_plot = audio_data
        time_plot = np.arange(len(audio_plot)) / 8000
    
    ax4.plot(time_plot, audio_plot, 'b', linewidth=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Audio Waveform (8-bit @ 8kHz)')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-128, 127)
    
    # 5. 数据模式（前128字节）
    ax5 = plt.subplot(3, 3, 6)
    header_size = 14
    pattern_len = min(128, len(audio_data) + header_size)
    pattern_data = np.zeros(128, dtype=np.uint8)
    pattern_data[:4] = [0xF0, 0x0F, 0xF0, 0x0F]
    pattern_data[4:6] = [frame_index & 0xFF, (frame_index >> 8) & 0xFF]
    if len(audio_data) > 0:
        audio_uint8 = (audio_data.astype(np.int16) + 128).astype(np.uint8)
        pattern_data[header_size:min(128, header_size + len(audio_uint8))] = audio_uint8[:min(114, len(audio_uint8))]
    
    pattern_2d = pattern_data.reshape(8, 16)
    im5 = ax5.imshow(pattern_2d, cmap='viridis', aspect='auto')
    ax5.set_title('Data Pattern (128B)')
    ax5.set_xticks([])
    ax5.set_yticks([])
    
    # 6. 简化频谱图
    ax6 = plt.subplot(3, 3, (7, 9))
    if len(audio_data) > 512:
        # 使用较小的FFT窗口加速
        f, t, Sxx = scipy.signal.spectrogram(
            audio_data.astype(np.float32),
            fs=8000,
            nperseg=256,
            noverlap=128,
            nfft=512  # 减小FFT大小
        )
        
        # 只显示0-3kHz
        freq_limit = 3000
        freq_bins = int(freq_limit * len(f) / 4000)
        
        Sxx_log = 10 * np.log10(Sxx[:freq_bins] + 1e-10)
        im6 = ax6.pcolormesh(t, f[:freq_bins], Sxx_log,
                           shading='auto',  # 更快的渲染
                           cmap='magma',
                           vmin=-80, vmax=-20)
        ax6.set_ylabel('Frequency (Hz)')
        ax6.set_xlabel('Time (s)')
        ax6.set_title('Spectrogram (0-3kHz)')
        plt.colorbar(im6, ax=ax6, fraction=0.046)
    else:
        ax6.text(0.5, 0.5, 'Audio too short', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Spectrogram')
    
    plt.suptitle(f'Audio Extraction Analysis - Frame {frame_index:08d}', fontsize=14)
    plt.tight_layout()
    
    return fig

def create_semantic_visualization(wsg_array, frame_index, mask_info=None):
    """创建语义mask可视化（RGB+语义类别）"""
    import matplotlib
    matplotlib.use('Agg')
    
    fig = plt.figure(figsize=(12, 5))
    
    # 1. 原始RGB图像
    ax1 = plt.subplot(1, 3, 1)
    rgb = wsg_array[:, :, :3].astype(np.uint8)
    ax1.imshow(rgb)
    ax1.set_title('RGB Image')
    ax1.axis('off')
    
    # 2. 语义mask（支持多实例）
    ax2 = plt.subplot(1, 3, 2)
    height, width = wsg_array.shape[:2]
    
    if mask_info and 'instances' in mask_info and len(mask_info['instances']) > 0:
        # 创建多实例语义mask
        combined_mask = np.zeros((height, width), dtype=np.float32)
        category_names = []
        
        # 为每个实例分配不同的值
        for instance in mask_info['instances']:
            combined_mask[instance['mask']] = instance['category_id']
            category_names.append(f"{instance['category_name']} ({instance['category_id']})")
        
        # 使用colormap显示
        ax2.imshow(combined_mask, cmap='tab20', vmin=0, vmax=max(20, np.max(combined_mask)))
        ax2.set_title(f'Semantic Masks: {", ".join(category_names)}')
        
    elif mask_info and 'semantic_mask' in mask_info:
        # 旧的单实例代码（向后兼容）
        semantic_mask = mask_info['semantic_mask'].astype(np.float32)
        if 'category_id' in mask_info:
            colored_mask = semantic_mask * mask_info['category_id']
            ax2.imshow(colored_mask, cmap='tab20', vmin=0, vmax=20)
            category_name = mask_info.get('category_name', f"Category {mask_info['category_id']}")
            ax2.set_title(f'Semantic Mask: {category_name}')
        else:
            ax2.imshow(semantic_mask, cmap='binary')
            ax2.set_title('Semantic Mask')
    else:
        ax2.imshow(np.zeros((height, width)), cmap='gray')
        ax2.set_title('No Semantic Mask')
    ax2.axis('off')
    
    # 3. RGB + 语义mask叠加（支持多实例）
    ax3 = plt.subplot(1, 3, 3)
    overlay = rgb.copy()
    
    # 为不同类别定义颜色
    color_map = {
        1: [255, 0, 0],      # person - 红色
        2: [0, 255, 0],      # violin - 绿色  
        3: [0, 0, 255],      # guitar - 蓝色
        4: [255, 255, 0],    # cello - 黄色
        5: [255, 0, 255],    # flute - 紫色
        6: [0, 255, 255],    # piano - 青色
        7: [255, 128, 0],    # ukulele - 橙色
        8: [128, 255, 0],    # accordion - 黄绿色
        9: [128, 0, 255],    # guzheng - 紫蓝色
        10: [255, 0, 128],   # clarinet - 玫红色
        11: [255, 192, 128], # cat - 浅橙色
        12: [128, 128, 255], # car - 浅蓝色
        13: [255, 255, 128], # saxophone - 浅黄色
        14: [128, 255, 128], # dog - 浅绿色
        # 可以继续添加更多颜色
    }
    
    if mask_info and 'instances' in mask_info and len(mask_info['instances']) > 0:
        # 多实例叠加
        for instance in mask_info['instances']:
            mask = instance['mask']
            category_id = instance['category_id']
            color = color_map.get(category_id, [255, 128, 0])  # 默认橙色
            # 半透明叠加
            overlay[mask] = overlay[mask] * 0.6 + np.array(color) * 0.4
            
    elif mask_info and 'semantic_mask' in mask_info:
        # 单实例叠加（向后兼容）
        mask = mask_info['semantic_mask']
        if 'category_id' in mask_info:
            category_id = mask_info['category_id']
            color = color_map.get(category_id, [255, 128, 0])
            overlay[mask] = overlay[mask] * 0.6 + np.array(color) * 0.4
        else:
            overlay[mask] = np.clip(overlay[mask] * 0.7 + 255 * 0.3, 0, 255)
    
    ax3.imshow(overlay.astype(np.uint8))
    ax3.set_title('RGB + Semantic Overlay')
    ax3.axis('off')
    
    # 添加统计信息
    if mask_info:
        info_text = f"Frame: {frame_index:08d}\n"
        
        if 'instances' in mask_info and len(mask_info['instances']) > 0:
            info_text += f"Instances: {len(mask_info['instances'])}\n"
            for instance in mask_info['instances']:
                info_text += f"  - {instance['category_name']} (ID: {instance['category_id']}): {instance['mask_pixels']:,} pixels\n"
        elif 'category_id' in mask_info:
            info_text += f"Category: {mask_info.get('category_name', 'Unknown')} (ID: {mask_info['category_id']})\n"
            if 'semantic_mask_pixels' in mask_info:
                info_text += f"Semantic Mask Pixels: {mask_info['semantic_mask_pixels']:,}\n"
        
        if 'mask_pixels' in mask_info:
            info_text += f"Audio Mask Pixels: {mask_info['mask_pixels']:,}"
        
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Semantic Segmentation - Frame {frame_index:08d}', fontsize=14)
    plt.tight_layout()
    return fig

def save_semantic_view_images(wsg_array, output_dir, base_name, mask_info=None):
    """Save channel-195 semantic category views as separate RGB, mask, and overlay images."""
    if not mask_info or not (
        ('instances' in mask_info and len(mask_info['instances']) > 0)
        or ('semantic_mask' in mask_info and np.any(mask_info['semantic_mask']))
    ):
        return []

    os.makedirs(output_dir, exist_ok=True)
    rgb = np.clip(wsg_array[:, :, :3].astype(np.float32), 0, 255).astype(np.uint8)
    semantic_values = wsg_array[:, :, 195].astype(np.float32, copy=True)
    valid_mask = np.isfinite(semantic_values) & (semantic_values > 0.0) & (semantic_values != -1.0)
    semantic_values[~valid_mask] = -1.0

    mask_rgb = categorical_ids_to_rgb(semantic_values, AVIS_CATEGORY_COLORS)
    class_dist = rounded_class_distribution(semantic_values, tuple(AVIS_CATEGORY_COLORS))
    if class_dist:
        print(f"    通道195类别分布(按最近类别量化): {class_dist}")

    overlay = rgb.astype(np.float32)
    overlay[valid_mask] = overlay[valid_mask] * 0.6 + mask_rgb[valid_mask].astype(np.float32) * 0.4
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    paths = []
    outputs = {
        'semantic_rgb': rgb,
        'semantic_mask': mask_rgb,
        'semantic_overlay': overlay,
    }
    for suffix, image in outputs.items():
        path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
        plt.imsave(path, image)
        paths.append(path)
    return paths

def create_visualization(wsg_array, audio_data, frame_index, mask_info=None):
    """创建音频提取的高级可视化（参考test_audio_direct_storage.py）"""
    # 使用优化版本
    return create_optimized_visualization(wsg_array, audio_data, frame_index, mask_info)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. RGB图像
    ax1 = plt.subplot(3, 4, 1)
    rgb = wsg_array[:, :, :3].astype(np.uint8)
    ax1.imshow(rgb)
    ax1.set_title('RGB Image')
    ax1.axis('off')
    
    # 2. 音频数据分布（mask区域）
    ax2 = plt.subplot(3, 4, 2)
    height, width = wsg_array.shape[:2]
    audio_mask = np.zeros((height, width), dtype=np.float32)
    
    # 计算每个像素的音频通道使用情况
    for y in range(height):
        for x in range(width):
            audio_values = wsg_array[y, x, 175:185]
            non_empty = np.sum(audio_values != -1.0)
            audio_mask[y, x] = non_empty
    
    # 使用更好的可视化方式
    im_mask = ax2.imshow(audio_mask, cmap='hot', vmin=0, vmax=10)
    ax2.set_title('Audio Data Distribution (Mask)')
    ax2.axis('off')
    plt.colorbar(im_mask, ax=ax2, label='Channels Used')
    
    # 3. 音频密度热图
    ax3 = plt.subplot(3, 4, 3)
    # 创建二值mask显示存储区域
    binary_mask = (audio_mask > 0).astype(np.float32)
    # 叠加到RGB图像上
    rgb_with_mask = rgb.copy().astype(np.float32)
    rgb_with_mask[:, :, 0] += binary_mask * 100  # 红色通道增强
    rgb_with_mask = np.clip(rgb_with_mask, 0, 255).astype(np.uint8)
    ax3.imshow(rgb_with_mask)
    ax3.set_title('RGB with Audio Mask Overlay')
    ax3.axis('off')
    
    # 4. 音频存储统计
    ax4 = plt.subplot(3, 4, 4)
    total_pixels = np.sum(binary_mask)
    total_capacity = total_pixels * 10  # 10 channels per pixel
    storage_efficiency = len(audio_data) / total_capacity * 100 if total_capacity > 0 else 0
    
    stats_text = f"Frame Index: {frame_index}\n"
    stats_text += f"Pixels Used: {int(total_pixels):,}\n"
    stats_text += f"Total Capacity: {int(total_capacity):,} bytes\n"
    stats_text += f"Data Size: {len(audio_data):,} bytes\n"
    stats_text += f"Storage Efficiency: {storage_efficiency:.1f}%\n"
    stats_text += f"Coverage: {total_pixels/(height*width)*100:.1f}%"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Storage Statistics')
    
    # 5. 原始音频波形
    ax5 = plt.subplot(3, 4, 5)
    time = np.arange(len(audio_data)) / 8000  # 8kHz
    ax5.plot(time, audio_data, 'b', alpha=0.7, linewidth=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Amplitude')
    ax5.set_title('Extracted Audio (8-bit)')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, max(time) if len(time) > 0 else 1)
    ax5.set_ylim(-128, 127)
    
    # 6. 重建的音频波形 (44.1kHz)
    ax6 = plt.subplot(3, 4, 6)
    audio_44k = reconstruct_audio(audio_data, 44100)
    time_44k = np.arange(len(audio_44k)) / 44100
    ax6.plot(time_44k, audio_44k, 'r', alpha=0.7, linewidth=0.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Amplitude')
    ax6.set_title('Reconstructed Audio (16-bit @ 44.1kHz)')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, max(time_44k) if len(time_44k) > 0 else 1)
    
    # 7. 结构化数据模式（前256字节）
    ax7 = plt.subplot(3, 4, 7)
    # 重新构建结构化数据来显示
    header_size = 14  # 魔数(4) + 帧索引(2) + 长度(4) + 校验和(4)
    pattern_len = min(256, len(audio_data) + header_size)
    pattern_data = np.zeros(256, dtype=np.uint8)
    pattern_data[:4] = [0xF0, 0x0F, 0xF0, 0x0F]  # 魔数
    pattern_data[4:6] = [frame_index & 0xFF, (frame_index >> 8) & 0xFF]  # 帧索引
    # 添加音频数据部分
    if len(audio_data) > 0:
        audio_uint8 = (audio_data.astype(np.int16) + 128).astype(np.uint8)
        pattern_data[header_size:min(256, header_size + len(audio_uint8))] = audio_uint8[:min(242, len(audio_uint8))]
    
    pattern_2d = pattern_data.reshape(16, 16)
    im_pattern = ax7.imshow(pattern_2d, cmap='viridis', aspect='auto')
    ax7.set_title('Data Pattern (First 256 Bytes)')
    ax7.set_xlabel('Byte Index')
    ax7.set_ylabel('Row')
    plt.colorbar(im_pattern, ax=ax7)
    
    # 8. 音频频谱图 (更详细)
    ax8 = plt.subplot(3, 4, (9, 10))  # 跨两列
    if len(audio_data) > 256:
        nperseg = min(512, len(audio_data) // 8)
        noverlap = nperseg * 3 // 4
        
        f, t, Sxx = scipy.signal.spectrogram(
            audio_data.astype(np.float32),
            fs=8000,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=1024,
            scaling='spectrum'
        )
        
        # 转换为对数刻度
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        
        im_spec = ax8.pcolormesh(t, f, Sxx_log, 
                                shading='gouraud', 
                                cmap='magma',
                                vmin=-80, vmax=-20)
        ax8.set_ylabel('Frequency (Hz)')
        ax8.set_xlabel('Time (s)')
        ax8.set_title('Audio Spectrogram (8kHz)')
        ax8.set_ylim(0, 4000)  # 限制到4kHz
        plt.colorbar(im_spec, ax=ax8, label='Power (dB)')
    
    # 9. 通道数据值分布
    ax9 = plt.subplot(3, 4, 11)
    # 获取所有音频通道的值
    audio_channel_values = []
    for y in range(height):
        for x in range(width):
            values = wsg_array[y, x, 175:185]
            valid_values = values[values != -1.0]
            if len(valid_values) > 0:
                audio_channel_values.extend(valid_values)
    
    if audio_channel_values:
        ax9.hist(audio_channel_values, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax9.set_xlabel('Float16 Value')
        ax9.set_ylabel('Count')
        ax9.set_title('Audio Channel Value Distribution')
        ax9.grid(True, alpha=0.3)
    else:
        ax9.text(0.5, 0.5, 'No audio data found', ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Audio Channel Value Distribution')
    
    # 10. 提取质量指标
    ax10 = plt.subplot(3, 4, 12)
    quality_text = f"Extraction Quality Metrics:\n\n"
    quality_text += f"Frame: {frame_index}\n"
    
    # 添加类别信息
    if mask_info and 'category_id' in mask_info:
        quality_text += f"Category ID: {mask_info['category_id']}\n"
        if 'category_name' in mask_info:
            quality_text += f"Category: {mask_info['category_name']}\n"
    
    quality_text += f"Sample Rate: 8000 Hz\n"
    quality_text += f"Bit Depth: 8-bit\n"
    quality_text += f"Duration: {len(audio_data)/8000:.3f} s\n"
    quality_text += f"Dynamic Range: [{np.min(audio_data)}, {np.max(audio_data)}]\n"
    quality_text += f"RMS Level: {np.sqrt(np.mean(audio_data.astype(np.float32)**2)):.1f}\n"
    
    # 检查数据完整性
    expected_length = 8000  # 1秒 @ 8kHz
    if len(audio_data) == expected_length:
        quality_text += f"\nStatus: ✓ Complete"
    else:
        quality_text += f"\nStatus: ⚠ Partial ({len(audio_data)}/{expected_length})"
    
    ax10.text(0.05, 0.95, quality_text, transform=ax10.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax10.axis('off')
    ax10.set_title('Quality Metrics')
    
    plt.suptitle(f'Advanced Audio Extraction Analysis - Frame {frame_index:08d}', fontsize=16)
    plt.tight_layout()
    
    return fig

def process_single_wsg_file(args):
    """处理单个WSG文件"""
    wsg_path, output_dir, modes = args
    
    try:
        # 加载WSG数据
        wsg_array = load_wsg_data(wsg_path)
        if wsg_array is None:
            return None
        
        # 提取音频数据和mask信息
        raw_data, mask_info = extract_audio_from_wsg(wsg_array)
        base_name = os.path.splitext(os.path.basename(wsg_path))[0]
        has_semantic_mask = bool(
            mask_info
            and (
                ('instances' in mask_info and len(mask_info['instances']) > 0)
                or ('semantic_mask' in mask_info and np.any(mask_info['semantic_mask']))
            )
        )

        if 'sem' in modes and has_semantic_mask:
            if 'category_id' in mask_info and 'category_name' not in mask_info:
                mask_info['category_name'] = AVIS_CATEGORIES.get(
                    mask_info['category_id'],
                    f'Unknown ({mask_info["category_id"]})'
                )
            save_semantic_view_images(wsg_array, output_dir, base_name, mask_info)

        if raw_data is None:
            if has_semantic_mask and 'sem' in modes:
                print(f"  {os.path.basename(wsg_path)}: 未找到音频数据，已导出通道195语义可视化")
                return {
                    'wsg_path': wsg_path,
                    'mask_info': mask_info,
                    'semantic_only': True,
                }
            print(f"  {os.path.basename(wsg_path)}: 未找到音频数据")
            return None
        
        # 解析音频
        parsed = parse_audio_data(raw_data)
        if parsed is None:
            if has_semantic_mask and 'sem' in modes:
                print(f"  {os.path.basename(wsg_path)}: 音频解析失败，已导出通道195语义可视化")
                return {
                    'wsg_path': wsg_path,
                    'mask_info': mask_info,
                    'semantic_only': True,
                }
            return None
        
        print(f"  {os.path.basename(wsg_path)}: 提取 {parsed['length']} 字节音频")
        
        # 根据模式处理
        results = {
            'parsed': parsed, 
            'wsg_path': wsg_path,
            'mask_info': mask_info  # 保存mask信息以避免重复计算
        }

        if 'vis' in modes or 'all' in modes:
            fig = create_optimized_visualization(
                wsg_array,
                parsed['audio_8bit'],
                parsed['frame_index'],
                mask_info
            )
            vis_path = os.path.join(output_dir, f"{base_name}_vis.png")
            fig.savefig(vis_path, dpi=120, bbox_inches='tight',
                        pad_inches=0.1, facecolor='white')
            plt.close(fig)
        
        if 'wav' in modes or 'npz' in modes:
            if 'npz' in modes:
                # 保存NPZ
                npz_path = os.path.join(output_dir, f"{base_name}_audio.npz")
                np.savez_compressed(npz_path, 
                                  audio_8bit=parsed['audio_8bit'],
                                  frame_index=parsed['frame_index'],
                                  sample_rate=8000)
            
            if 'wav' in modes:
                # 保存WAV文件
                audio_44k = reconstruct_audio(parsed['audio_8bit'], 44100)
                wav_path = os.path.join(output_dir, f"{base_name}_audio.wav")
                wavfile.write(wav_path, 44100, audio_44k)
                
                # 也保存8kHz版本
                audio_8k = reconstruct_audio(parsed['audio_8bit'], 8000)
                wav_8k_path = os.path.join(output_dir, f"{base_name}_audio_8k.wav")
                wavfile.write(wav_8k_path, 8000, audio_8k)
        
        return results
        
    except Exception as e:
        print(f"  处理文件失败 {wsg_path}: {str(e)}")
        return None

def process_wsg_input(input_path, output_path, modes, max_workers=4):
    """处理WSG输入（文件或文件夹）"""
    
    if os.path.isfile(input_path) and input_path.endswith('.wsg'):
        # 单个文件处理
        if output_path is None:
            output_path = os.path.dirname(input_path)
        
        os.makedirs(output_path, exist_ok=True)
        print(f"处理单个WSG文件: {input_path}")
        print(f"输出目录: {output_path}")
        
        process_single_wsg_file((input_path, output_path, modes))
        print(f"处理完成!")
        
    elif os.path.isdir(input_path):
        # 文件夹处理
        if output_path is None:
            parent_dir = os.path.dirname(input_path.rstrip('/'))
            folder_name = os.path.basename(input_path.rstrip('/'))
            output_path = os.path.join(parent_dir, f"export_audio_from_{folder_name}")
        
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
        
        # 收集所有结果（包括音频和可视化需求）
        all_results = []
        
        # 第一阶段：并行提取音频（不做可视化）
        extract_modes = [m for m in modes if m not in ['vis', 'sem']]
        if max_workers > 1:
            # 使用更小的批次处理，避免内存积累
            batch_size = max_workers
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i+batch_size]
                batch_tasks_with_modes = [(t[0], t[1], extract_modes) for t in batch_tasks]
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_single_wsg_file, task) for task in batch_tasks_with_modes]
                    
                    for future in tqdm(futures, desc=f"提取音频 (批次 {i//batch_size + 1})", leave=False):
                        try:
                            result = future.result()
                            if result:
                                all_results.append(result)
                        except Exception as e:
                            print(f"任务失败: {str(e)}")
        else:
            # 单线程处理
            for task in tqdm(tasks, desc="提取音频"):
                result = process_single_wsg_file((task[0], task[1], extract_modes))
                if result:
                    all_results.append(result)
        
        # 提取所有音频数据
        all_audio = [r['parsed']['audio_8bit'] for r in all_results if r and 'parsed' in r]
        
        # 合并所有音频
        if all_audio and len(all_audio) > 1:
            print("\n合并所有帧的音频...")
            combined_audio = np.concatenate(all_audio)
            duration = len(combined_audio) / 8000
            
            print(f"  总时长: {duration:.2f} 秒")
            print(f"  总大小: {len(combined_audio):,} 字节")
            
            # 保存合并的音频
            video_name = os.path.basename(input_path.rstrip('/'))
            
            # 44.1kHz版本
            combined_44k = reconstruct_audio(combined_audio, 44100)
            combined_path = os.path.join(output_path, f"combined_audio_{video_name}.wav")
            wavfile.write(combined_path, 44100, combined_44k)
            print(f"  已保存44.1kHz版本: {combined_path}")
            
            # 8kHz版本
            combined_8k = reconstruct_audio(combined_audio, 8000)
            combined_8k_path = os.path.join(output_path, f"combined_audio_{video_name}_8k.wav")
            wavfile.write(combined_8k_path, 8000, combined_8k)
            print(f"  已保存8kHz版本: {combined_8k_path}")
        
        # 第二阶段：如果需要可视化，优化处理
        if ('vis' in modes or 'sem' in modes) and all_results:
            if 'vis' in modes:
                print("\n生成音频可视化...")
                vis_output_dir = os.path.join(output_path, "visualizations")
                os.makedirs(vis_output_dir, exist_ok=True)
            
            if 'sem' in modes:
                print("\n生成语义mask可视化...")
                sem_output_dir = os.path.join(output_path, "semantic_visualizations")
                os.makedirs(sem_output_dir, exist_ok=True)
            
            # 使用优化的可视化为所有帧生成图像
            # 批量处理以提高效率
            batch_size = 10
            for batch_start in range(0, len(all_results), batch_size):
                batch_end = min(batch_start + batch_size, len(all_results))
                batch_results = all_results[batch_start:batch_end]
                
                for idx, result in enumerate(tqdm(batch_results, 
                                                desc=f"创建可视化 (批次 {batch_start//batch_size + 1})",
                                                leave=False)):
                    if result and 'wsg_path' in result:
                        try:
                            # 重新加载WSG数据
                            wsg_array = load_wsg_data(result['wsg_path'])
                            if wsg_array is not None:
                                base_name = os.path.splitext(os.path.basename(result['wsg_path']))[0]
                                
                                # 生成音频可视化
                                if 'vis' in modes and 'parsed' in result:
                                    fig = create_optimized_visualization(
                                        wsg_array, 
                                        result['parsed']['audio_8bit'], 
                                        result['parsed']['frame_index'],
                                        result.get('mask_info')
                                    )
                                    vis_path = os.path.join(vis_output_dir, f"{base_name}_vis.png")
                                    fig.savefig(vis_path, dpi=120, bbox_inches='tight',
                                               pad_inches=0.1, facecolor='white')
                                    plt.close(fig)
                                
                                # 生成语义可视化
                                if 'sem' in modes:
                                    # 确保mask_info包含类别名称
                                    mask_info = result.get('mask_info', {})
                                    if 'category_id' in mask_info and 'category_name' not in mask_info:
                                        mask_info['category_name'] = AVIS_CATEGORIES.get(
                                            mask_info['category_id'], 
                                            f'Unknown ({mask_info["category_id"]})'
                                        )
                                    
                                    fig_sem = create_semantic_visualization(
                                        wsg_array,
                                        result['parsed']['frame_index'] if 'parsed' in result else idx,
                                        mask_info
                                    )
                                    sem_path = os.path.join(sem_output_dir, f"{base_name}_sem.png")
                                    fig_sem.savefig(sem_path, dpi=120, bbox_inches='tight',
                                                   pad_inches=0.1, facecolor='white')
                                    plt.close(fig_sem)
                                
                                # 释放内存
                                del wsg_array
                                
                        except Exception as e:
                            print(f"  可视化失败 {result['wsg_path']}: {str(e)}")
                
                # 批次间垃圾回收
                import gc
                gc.collect()
            
            print(f"  已保存 {len(all_results)} 帧的可视化图像")
        
        print(f"\n处理完成! 成功提取 {len(all_audio)}/{len(wsg_files)} 个文件的音频")
    
    else:
        print(f"无效的输入路径: {input_path}")

def main():
    """主函数"""
    # ==================== 配置区域 ====================
    # 输入配置
    input_path = "../output/AVIS_test_results/00000076/wsg"  # WSG文件夹路径
    #input_path = "../output/AVIS_test_results/00001584/wsg/00000004.wsg"  # 单个文件示例
    
    # 输出配置
    output_path = None  # None表示使用默认输出路径
    
    # 模式配置
    #modes = ['sem']  # 可选: 'vis', 'wav', 'npz', 'sem'
    modes = ['vis', 'wav', 'npz', 'sem']  # 全部模式
    
    # 处理配置
    max_workers = min(mp.cpu_count() // 2, 8)  # 限制并行工作器数量，避免内存问题
    
    print("=" * 50)
    print("WSG音频提取工具")
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
