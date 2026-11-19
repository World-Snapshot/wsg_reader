#!/usr/bin/env python3
"""
Pipeline 19: 字幕音频读取工具
从WSG文件的全帧区域读取字幕音频（通道185-194）
"""

import os
import numpy as np
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 尝试导入语音识别库
try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False
    print("警告: speech_recognition未安装")

# 尝试导入whisper
try:
    import whisper
    HAS_WHISPER = True
    print("检测到Whisper语音识别")
except ImportError:
    HAS_WHISPER = False
    print("警告: whisper未安装，将尝试其他方法")

# 空值定义
empty_val = -1.0

def load_wsg_data(wsg_path):
    """加载WSG数据"""
    try:
        data = np.load(wsg_path, allow_pickle=True)
        return data['wsg_data']
    except Exception as e:
        print(f"加载WSG数据失败 {wsg_path}: {str(e)}")
        return None

def extract_caption_audio_from_wsg(wsg_array):
    """从WSG数组提取字幕音频数据（全帧存储）- 优化版"""
    # 字幕音频通道配置
    caption_channels_start = 185
    caption_channels_end = 195  # 185-194共10个通道
    channels_per_pixel = 10
    
    height, width = wsg_array.shape[:2]
    total_pixels = height * width
    
    # 提取所有字幕通道数据并重塑为一维数组
    caption_data = wsg_array[:, :, caption_channels_start:caption_channels_end]
    caption_data_flat = caption_data.reshape(total_pixels * channels_per_pixel)
    
    # 找到第一个空值的位置
    empty_mask = (caption_data_flat == empty_val)
    empty_indices = np.where(empty_mask)[0]
    
    if len(empty_indices) > 0:
        # 找到第一个空值的位置
        first_empty = empty_indices[0]
        # 只处理到第一个空值之前的数据
        valid_data = caption_data_flat[:first_empty]
    else:
        # 没有空值，使用所有数据
        valid_data = caption_data_flat
    
    # 如果数据太少，返回None
    if len(valid_data) < 14:
        return None
    
    # 批量反归一化
    byte_vals = ((valid_data / 100.0 + 0.5) * 255).astype(np.int32)
    byte_vals = np.clip(byte_vals, 0, 255).astype(np.uint8)
    
    return byte_vals

def parse_caption_audio_data(raw_data):
    """解析字幕音频数据"""
    if raw_data is None or len(raw_data) < 14:
        return None
    
    # 解析header (14字节)
    magic = raw_data[:4]
    video_id = int(raw_data[4]) | (int(raw_data[5]) << 8)
    
    # 4字节长度字段
    length = (int(raw_data[6]) | 
             (int(raw_data[7]) << 8) | 
             (int(raw_data[8]) << 16) | 
             (int(raw_data[9]) << 24))
    
    checksum = (int(raw_data[10]) | 
               (int(raw_data[11]) << 8) | 
               (int(raw_data[12]) << 16) | 
               (int(raw_data[13]) << 24))
    
    # 验证魔数（考虑float16精度损失）
    expected_magic = [0xF0, 0x0F, 0xF0, 0x0F]
    magic_valid = all(abs(int(magic[i]) - expected_magic[i]) <= 2 for i in range(4))
    
    if not magic_valid:
        print(f"魔数无效: {magic}")
        return None
    
    # 提取音频数据
    # 修复：使用实际数据长度而不是header中的长度
    actual_data_length = len(raw_data) - 14
    
    if length > actual_data_length:
        print(f"警告: Header长度({length})大于实际数据({actual_data_length})")
        audio_uint8 = raw_data[14:]
    elif length < actual_data_length:
        # Header长度小于实际数据，这可能是注入时的bug
        print(f"注意: Header长度({length})小于实际数据({actual_data_length})，使用实际数据")
        audio_uint8 = raw_data[14:]  # 使用所有可用数据
    else:
        audio_uint8 = raw_data[14:14+length]
    
    # 转换为int8
    audio_8bit = audio_uint8.astype(np.int16) - 128
    audio_8bit = audio_8bit.astype(np.int8)
    
    # 验证校验和
    actual_checksum = int(np.sum(audio_8bit.astype(np.int32))) & 0xFFFFFFFF
    
    # 由于可能的数据截断，只在完整数据时验证校验和
    if len(audio_uint8) == length and actual_checksum != checksum:
        print(f"警告: 校验和不匹配 (预期: {checksum}, 实际: {actual_checksum})")
    
    return {
        'video_id': video_id,
        'audio_8bit': audio_8bit,
        'length': length,
        'checksum_valid': actual_checksum == checksum if len(audio_uint8) == length else None
    }

def create_simple_visualization(wsg_array, caption_audio_data, frame_index=0, caption_text=None):
    """创建简化的字幕音频可视化（快速版）"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. RGB图像
    ax1 = axes[0, 0]
    rgb = wsg_array[:, :, :3].astype(np.uint8)
    ax1.imshow(rgb)
    ax1.set_title('RGB Image')
    ax1.axis('off')
    
    # 2. 字幕文本显示
    ax2 = axes[0, 1]
    ax2.axis('off')
    
    if caption_text:
        # 显示识别的字幕文本
        wrapped_text = "\n".join([caption_text[i:i+40] for i in range(0, len(caption_text), 40)])
        ax2.text(0.5, 0.5, f"Caption Text:\n\n{wrapped_text}", 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, "Caption Text:\n\n[Speech recognition not available\nor failed to recognize]", 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 3. 音频波形（简化版）
    ax3 = axes[1, 0]
    if caption_audio_data and 'audio_8bit' in caption_audio_data:
        audio = caption_audio_data['audio_8bit']
        # 降采样显示
        downsample = max(1, len(audio) // 1000)
        audio_display = audio[::downsample]
        ax3.plot(audio_display, 'b-', linewidth=0.5)
        ax3.set_xlabel(f'Samples (downsampled {downsample}x)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title(f'Caption Audio ({len(audio)/8000:.2f}s)')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Audio Data', ha='center', va='center')
        ax3.set_title('Caption Audio')
    
    # 4. 信息文本
    ax4 = axes[1, 1]
    ax4.axis('off')
    if caption_audio_data:
        info_text = f"Frame: {frame_index:08d}\n"
        info_text += f"Video ID: {caption_audio_data.get('video_id', 'N/A')}\n"
        info_text += f"Audio Length: {caption_audio_data.get('length', 0)} bytes\n"
        info_text += f"Duration: {caption_audio_data.get('length', 0)/8000:.2f} seconds\n"
        info_text += f"Checksum: {'Valid' if caption_audio_data.get('checksum_valid') else 'Invalid/Unknown'}"
        
        # 计算数据存储信息
        caption_channels = wsg_array[:, :, 185:195]
        has_data = np.any(caption_channels != empty_val, axis=2)
        data_pixels = np.sum(has_data)
        total_pixels = wsg_array.shape[0] * wsg_array.shape[1]
        coverage = (data_pixels / total_pixels) * 100
        info_text += f"\n\nStorage: {data_pixels:,} pixels ({coverage:.1f}%)"
    else:
        info_text = "No Caption Audio Found"
    
    ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Caption Audio Analysis - Frame {frame_index:08d}', fontsize=14)
    plt.tight_layout()
    
    return fig

def create_visualization(wsg_array, caption_audio_data, frame_index=0):
    """创建字幕音频可视化"""
    fig = plt.figure(figsize=(14, 8))
    
    # 1. RGB图像
    ax1 = plt.subplot(2, 3, 1)
    rgb = wsg_array[:, :, :3].astype(np.uint8)
    ax1.imshow(rgb)
    ax1.set_title('RGB Image')
    ax1.axis('off')
    
    # 2. 字幕音频通道热力图
    ax2 = plt.subplot(2, 3, 2)
    caption_channels = wsg_array[:, :, 185:195]
    has_data = np.any(caption_channels != empty_val, axis=2)
    ax2.imshow(has_data.astype(float), cmap='hot', interpolation='nearest')
    ax2.set_title('Caption Audio Data Distribution')
    ax2.axis('off')
    
    # 3. 音频波形
    ax3 = plt.subplot(2, 3, 3)
    if caption_audio_data and 'audio_8bit' in caption_audio_data:
        audio = caption_audio_data['audio_8bit']
        time_ms = np.arange(len(audio)) / 8.0  # 8kHz采样率，转换为毫秒
        ax3.plot(time_ms, audio, 'b-', linewidth=0.5)
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title(f'Caption Audio Waveform ({len(audio)/8000:.2f}s)')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Audio Data', ha='center', va='center')
        ax3.set_title('Caption Audio Waveform')
    
    # 4. 数据统计
    ax4 = plt.subplot(2, 3, 4)
    if caption_audio_data:
        stats_text = f"Frame: {frame_index:08d}\n"
        stats_text += f"Video ID: {caption_audio_data.get('video_id', 'N/A')}\n"
        stats_text += f"Audio Length: {caption_audio_data.get('length', 0)} bytes\n"
        stats_text += f"Duration: {caption_audio_data.get('length', 0)/8000:.2f} seconds\n"
        stats_text += f"Checksum: {'Valid' if caption_audio_data.get('checksum_valid') else 'Invalid/Unknown'}"
        
        # 计算存储使用情况
        total_pixels = wsg_array.shape[0] * wsg_array.shape[1]
        data_pixels = np.sum(has_data)
        usage = (data_pixels / total_pixels) * 100
        stats_text += f"\n\nStorage Usage:\n"
        stats_text += f"Pixels Used: {data_pixels:,}/{total_pixels:,}\n"
        stats_text += f"Usage: {usage:.2f}%"
    else:
        stats_text = "No Caption Audio Data Found"
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    
    # 5. 音频频谱图
    ax5 = plt.subplot(2, 3, 5)
    if caption_audio_data and 'audio_8bit' in caption_audio_data:
        audio = caption_audio_data['audio_8bit']
        if len(audio) > 256:
            from scipy import signal
            f, t, Sxx = signal.spectrogram(
                audio.astype(np.float32), 
                fs=8000,
                nperseg=256,
                noverlap=128
            )
            ax5.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            ax5.set_ylabel('Frequency (Hz)')
            ax5.set_xlabel('Time (s)')
            ax5.set_title('Spectrogram')
            ax5.set_ylim(0, 4000)  # 显示到4kHz
        else:
            ax5.text(0.5, 0.5, 'Audio too short', ha='center', va='center')
            ax5.set_title('Spectrogram')
    else:
        ax5.text(0.5, 0.5, 'No Audio Data', ha='center', va='center')
        ax5.set_title('Spectrogram')
    
    # 6. 通道值分布
    ax6 = plt.subplot(2, 3, 6)
    channel_values = []
    for ch in range(10):
        ch_data = wsg_array[:, :, 185 + ch].flatten()
        valid_data = ch_data[ch_data != empty_val]
        if len(valid_data) > 0:
            channel_values.extend(valid_data)
    
    if channel_values:
        ax6.hist(channel_values, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax6.set_xlabel('Float16 Value')
        ax6.set_ylabel('Count')
        ax6.set_title('Caption Channel Value Distribution')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No Data', ha='center', va='center')
        ax6.set_title('Caption Channel Value Distribution')
    
    plt.suptitle(f'Caption Audio Analysis - Frame {frame_index:08d}', fontsize=14)
    plt.tight_layout()
    
    return fig

def audio_to_text_from_wav(wav_path, debug=True):
    """从WAV文件识别文本"""
    if not os.path.exists(wav_path):
        print(f"  ❌ WAV文件不存在: {wav_path}")
        return None
        
    print(f"  尝试语音识别: {os.path.basename(wav_path)}")
    
    # 检查音频文件信息
    try:
        file_size = os.path.getsize(wav_path)
        print(f"  WAV文件大小: {file_size} bytes")
        
        # 读取WAV文件信息
        sr_file, audio_data = wavfile.read(wav_path)
        duration = len(audio_data) / sr_file
        print(f"  音频信息: 采样率={sr_file}Hz, 长度={len(audio_data)}样本, 时长={duration:.2f}秒")
        
        # 检查音频内容
        if len(audio_data) == 0:
            print(f"  ❌ 音频为空")
            return None
        
        # 检查音频能量
        audio_energy = np.mean(np.abs(audio_data))
        max_val = np.max(np.abs(audio_data))
        print(f"  音频能量: 平均={audio_energy:.2f}, 最大值={max_val}")
        
        if max_val < 100:  # 如果音频太安静
            print(f"  ⚠️ 音频可能太安静 (最大值仅为{max_val})")
            
    except Exception as e:
        print(f"  ❌ 读取WAV文件信息失败: {e}")
    
    try:
        text = None
        
        # 优先使用本地Whisper
        if HAS_WHISPER and text is None:
            try:
                print("  使用Whisper进行本地识别...")
                # 加载模型（第一次会下载）
                if not hasattr(audio_to_text_from_wav, 'whisper_model'):
                    print("  加载Whisper模型 (第一次需要下载)...")
                    audio_to_text_from_wav.whisper_model = whisper.load_model("base")
                
                # 进行识别
                result = audio_to_text_from_wav.whisper_model.transcribe(wav_path, language='en')
                text = result["text"].strip()
                
                if text:
                    print(f"  ✓ Whisper识别成功: \"{text}\"")
                else:
                    print(f"  Whisper返回空文本")
                    
            except Exception as e:
                print(f"  Whisper识别失败: {e}")
                import traceback
                if debug:
                    traceback.print_exc()
        
        # 如果Whisper失败，尝试speech_recognition
        if HAS_SPEECH_RECOGNITION and text is None:
            try:
                print("  使用speech_recognition...")
                recognizer = sr.Recognizer()
                
                # 调整识别器参数
                recognizer.energy_threshold = 50  # 降低能量阈值
                recognizer.pause_threshold = 1.0
                
                with sr.AudioFile(wav_path) as source:
                    print("    读取音频数据...")
                    audio_data = recognizer.record(source)
                    print(f"    音频数据长度: {len(audio_data.frame_data)} bytes")
                
                # 尝试sphinx（完全离线）
                if text is None:
                    try:
                        print("    尝试Sphinx识别...")
                        text = recognizer.recognize_sphinx(audio_data)
                        if text:
                            print(f"    ✓ Sphinx识别成功: \"{text}\"")
                        else:
                            print(f"    Sphinx返回空文本")
                    except sr.UnknownValueError:
                        print(f"    Sphinx无法理解音频")
                    except Exception as e:
                        print(f"    Sphinx识别错误: {e}")
                
                # 如果离线失败，尝试在线
                if text is None:
                    try:
                        print("    尝试Google识别...")
                        text = recognizer.recognize_google(audio_data, language='en-US')
                        if text:
                            print(f"    ✓ Google识别成功: \"{text}\"")
                        else:
                            print(f"    Google返回空文本")
                    except sr.UnknownValueError:
                        print(f"    Google无法理解音频")
                    except sr.RequestError as e:
                        print(f"    Google请求错误 (需要网络): {e}")
                    except Exception as e:
                        print(f"    Google识别错误: {e}")
                        
            except Exception as e:
                print(f"  Speech recognition总体错误: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
        
        if text is None:
            print(f"  ⚠️ 所有语音识别方法失败")
            print(f"  💡 提示: 可以手动播放 {wav_path} 检查音频内容")
        
        return text
            
    except Exception as e:
        print(f"  语音识别异常: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return None

def process_single_wsg(wsg_path, output_dir, modes, need_recognition=False):
    """处理单个WSG文件"""
    try:
        # 加载WSG数据
        wsg_array = load_wsg_data(wsg_path)
        if wsg_array is None:
            return None
        
        # 提取字幕音频
        raw_data = extract_caption_audio_from_wsg(wsg_array)
        if raw_data is None:
            return {
                'wsg_path': wsg_path,
                'success': False,
                'error': '没有找到字幕音频数据'
            }
        
        # 解析数据
        parsed = parse_caption_audio_data(raw_data)
        if parsed is None:
            return {
                'wsg_path': wsg_path,
                'success': False,
                'error': '数据解析失败'
            }
        
        base_name = os.path.splitext(os.path.basename(wsg_path))[0]
        
        result = {
            'wsg_path': wsg_path,
            'success': True,
            'parsed': parsed,
            'base_name': base_name,
            'caption_text': None
        }
        
        # 保存音频文件
        wav_path = None
        if 'wav' in modes:
            wav_path = os.path.join(output_dir, f"{base_name}_caption.wav")
            # 恢复到16位用于保存
            audio_16bit = parsed['audio_8bit'].astype(np.float32) / 127.0
            wavfile.write(wav_path, 8000, (audio_16bit * 32767).astype(np.int16))
            result['wav_path'] = wav_path
        
        # 语音识别
        if need_recognition:
            print(f"  准备语音识别...")
            if wav_path and os.path.exists(wav_path):
                # 使用已保存的WAV文件
                print(f"  使用已保存的WAV文件: {wav_path}")
                caption_text = audio_to_text_from_wav(wav_path)
                result['caption_text'] = caption_text
                print(f"  识别结果: {caption_text if caption_text else '失败'}")
            else:
                # 如果没有保存WAV，创建临时WAV用于识别
                print(f"  创建临时WAV文件进行识别...")
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='_caption.wav', delete=False) as tmp_file:
                    tmp_wav_path = tmp_file.name
                audio_16bit = parsed['audio_8bit'].astype(np.float32) / 127.0
                wavfile.write(tmp_wav_path, 8000, (audio_16bit * 32767).astype(np.int16))
                caption_text = audio_to_text_from_wav(tmp_wav_path)
                result['caption_text'] = caption_text
                os.unlink(tmp_wav_path)  # 删除临时文件
            
        # 保存numpy数组
        if 'npz' in modes:
            npz_path = os.path.join(output_dir, f"{base_name}_caption.npz")
            np.savez_compressed(npz_path, 
                               audio_8bit=parsed['audio_8bit'],
                               video_id=parsed['video_id'],
                               checksum_valid=parsed['checksum_valid'])
            result['npz_path'] = npz_path
        
        return result
        
    except Exception as e:
        return {
            'wsg_path': wsg_path,
            'success': False,
            'error': str(e)
        }

def extract_caption_from_folder(wsg_folder, output_path=None, modes=['vis', 'wav'], max_files=None, max_workers=None):
    """从WSG文件夹提取所有字幕音频"""
    if not os.path.exists(wsg_folder):
        print(f"文件夹不存在: {wsg_folder}")
        return
    
    # 获取所有WSG文件
    wsg_files = glob.glob(os.path.join(wsg_folder, "*.wsg"))
    wsg_files.sort()
    
    if not wsg_files:
        print(f"没有找到WSG文件在: {wsg_folder}")
        return
    
    if max_files:
        wsg_files = wsg_files[:max_files]
    
    print(f"找到 {len(wsg_files)} 个WSG文件需要处理")
    
    # 设置输出目录
    if output_path is None:
        parent_dir = os.path.dirname(wsg_folder)
        output_path = os.path.join(parent_dir, "export_caption_audio_from_wsg")
    
    os.makedirs(output_path, exist_ok=True)
    print(f"输出目录: {output_path}")
    
    # 设置工作线程数
    if max_workers is None:
        max_workers = min(mp.cpu_count() // 4, 4)  # 降低线程数避免过载
    
    # 如果需要调试语音识别，使用单线程
    if 'vis' in modes or 'txt' in modes:
        print("  注意: 启用语音识别，使用单线程模式以便查看调试信息")
        max_workers = 1
    
    # 第一阶段：提取音频数据
    extract_modes = [m for m in modes if m != 'vis']
    all_results = []
    
    # 判断是否需要语音识别
    need_recognition = ('vis' in modes or 'txt' in modes)
    
    if extract_modes:
        print(f"\n提取模式: {', '.join(extract_modes)}")
        if need_recognition:
            print(f"  语音识别: 启用")
        
        # 处理所有文件
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for wsg_path in wsg_files:
                future = executor.submit(process_single_wsg, wsg_path, output_path, extract_modes, need_recognition)
                futures.append(future)
            
            # 收集结果
            if max_workers == 1:
                # 单线程模式，不使用tqdm以避免覆盖调试输出
                print(f"  处理 {len(futures)} 个文件...")
                for i, future in enumerate(futures):
                    print(f"\n  [{i+1}/{len(futures)}] 处理中...")
                    result = future.result()
                    if result:
                        all_results.append(result)
                        if result['success']:
                            audio_length = len(result['parsed']['audio_8bit'])
                            duration = audio_length / 8000
                            print(f"  {result['base_name']}.wsg: 提取 {audio_length} 字节音频 ({duration:.2f}秒)")
                        else:
                            print(f"  {os.path.basename(result['wsg_path'])}: {result['error']}")
            else:
                # 多线程模式使用进度条
                for future in tqdm(as_completed(futures), total=len(futures), desc="提取字幕音频"):
                    result = future.result()
                    if result:
                        all_results.append(result)
                        if result['success']:
                            audio_length = len(result['parsed']['audio_8bit'])
                            duration = audio_length / 8000
                            print(f"  {result['base_name']}.wsg: 提取 {audio_length} 字节音频 ({duration:.2f}秒)")
                        else:
                            print(f"  {os.path.basename(result['wsg_path'])}: {result['error']}")
    
    # 第二阶段：生成可视化
    if 'vis' in modes and all_results:
        print("\n生成可视化...")
        vis_output_dir = os.path.join(output_path, "visualizations")
        os.makedirs(vis_output_dir, exist_ok=True)
        
        # 只为成功的结果生成可视化
        successful_results = [r for r in all_results if r['success']]
        
        # 检查是否有识别结果
        recognized_count = sum(1 for r in successful_results if r.get('caption_text'))
        print(f"  识别成功的文件数: {recognized_count}/{len(successful_results)}")
        
        # 使用简单的进度条处理可视化（全量）
        for idx, result in enumerate(tqdm(successful_results, desc="创建可视化")):
            if result['success']:
                try:
                    # 重新加载WSG数据
                    wsg_array = load_wsg_data(result['wsg_path'])
                    if wsg_array is not None:
                        fig = create_simple_visualization(
                            wsg_array, 
                            result['parsed'],
                            idx,
                            result.get('caption_text')
                        )
                        
                        vis_path = os.path.join(vis_output_dir, f"{result['base_name']}_caption_vis.png")
                        fig.savefig(vis_path, dpi=100, bbox_inches='tight',
                                   pad_inches=0.1, facecolor='white')
                        plt.close(fig)
                        
                        del wsg_array
                        
                except Exception as e:
                    print(f"  可视化失败 {result['base_name']}: {str(e)}")
    
    # 保存识别的文本
    if all_results and any(r.get('caption_text') for r in all_results if r['success']):
        caption_texts = []
        for r in all_results:
            if r['success'] and r.get('caption_text'):
                caption_texts.append(f"{r['base_name']}: {r['caption_text']}")
        
        if caption_texts:
            txt_path = os.path.join(output_path, "recognized_captions.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("Recognized Caption Texts\n")
                f.write("=" * 50 + "\n\n")
                for text in caption_texts:
                    f.write(text + "\n")
            print(f"\n保存识别文本: {txt_path}")
    
    # 统计结果
    success_count = sum(1 for r in all_results if r['success'])
    print(f"\n处理完成! 成功: {success_count}/{len(all_results)}")

def process_video_or_file(input_path, output_path=None, modes=['vis', 'wav'], max_files=None):
    """处理视频文件夹或单个文件"""
    if os.path.isdir(input_path):
        # 处理文件夹
        if input_path.endswith('/wsg') or '/wsg' in input_path:
            # 直接是WSG文件夹
            extract_caption_from_folder(input_path, output_path, modes, max_files)
        else:
            # 可能是视频文件夹，查找wsg子目录
            wsg_dir = os.path.join(input_path, "wsg")
            if os.path.exists(wsg_dir):
                extract_caption_from_folder(wsg_dir, output_path, modes, max_files)
            else:
                print(f"没有找到wsg子目录在: {input_path}")
    
    elif input_path.endswith('.wsg'):
        # 处理单个文件
        output_dir = output_path or os.path.dirname(input_path)
        need_recognition = ('vis' in modes or 'txt' in modes)
        result = process_single_wsg(input_path, output_dir, modes, need_recognition)
        
        if result and result['success']:
            print(f"成功提取字幕音频: {input_path}")
            
            # 如果需要可视化
            if 'vis' in modes:
                wsg_array = load_wsg_data(input_path)
                if wsg_array is not None:
                    fig = create_simple_visualization(wsg_array, result['parsed'], 0, result.get('caption_text'))
                    base_name = os.path.splitext(os.path.basename(input_path))[0]
                    vis_path = os.path.join(output_dir, f"{base_name}_caption_vis.png")
                    fig.savefig(vis_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"保存可视化: {vis_path}")
        else:
            print(f"提取失败: {result.get('error', '未知错误')}")
    
    else:
        print(f"无效的输入路径: {input_path}")

def main():
    """主函数"""
    # ==================== 配置区域 ====================
    # 输入配置
    input_path = "../output/AVIS_test_results/00002559/wsg"  # WSG文件夹路径
    
    # 输出配置
    output_path = None  # None表示使用默认输出路径
    
    # 模式配置
    modes = ['vis', 'wav']  # 可选: 'vis', 'wav', 'npz'
    
    # 处理配置
    max_workers = mp.cpu_count() // 2
    max_files = None  # None表示处理所有文件
    
    print("=" * 50)
    print("WSG字幕音频提取工具 (Pipeline 19)")
    print("=" * 50)
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path or '自动设置'}")
    print(f"模式: {', '.join(modes)}")
    print(f"最大文件数: {max_files or '全部'}")
    print(f"工作线程: {max_workers}")
    print("=" * 50)
    
    try:
        process_video_or_file(input_path, output_path, modes, max_files)
        print("\n✅ 处理完成!")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()