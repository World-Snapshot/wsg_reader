
import os
import shutil
import glob
from pathlib import Path

def copy_avis_to_test_dataset():
    """
    将 input/avis 里的文件夹内容复制到 output/{train,val,test}_results
    并将所有图片放在对应的 crop_frame 文件夹里
    """
    
    # 设置路径
    input_avis_dir = "../input/test"  # 你可以改成 ../input/val 或 ../input/test
    input_base = os.path.basename(os.path.normpath(input_avis_dir))
    if input_base not in ["train", "val", "test"]:
        raise ValueError("输入目录名必须是 train、val 或 test")
    output_test_dir = f"../output/{input_base}_results"
    
    # 检查输入目录是否存在
    if not os.path.exists(input_avis_dir):
        print(f"错误: 输入目录不存在: {input_avis_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_test_dir, exist_ok=True)
    print(f"输出目录: {output_test_dir}")
    
    # 获取 avis 目录下的所有子文件夹
    avis_folders = [f for f in os.listdir(input_avis_dir) 
                    if os.path.isdir(os.path.join(input_avis_dir, f))]
    
    if not avis_folders:
        print(f"在 {input_avis_dir} 目录下没有找到任何文件夹")
        return
    
    print(f"找到 {len(avis_folders)} 个文件夹需要处理:")
    for folder in sorted(avis_folders):
        print(f"  - {folder}")
    
    # 处理每个文件夹
    for folder_name in sorted(avis_folders):
        input_folder_path = os.path.join(input_avis_dir, folder_name)
        output_folder_path = os.path.join(output_test_dir, folder_name)
        
        print(f"\n处理文件夹: {folder_name}")
        
        # 创建输出文件夹和 crop_frame 子目录
        crop_frame_dir = os.path.join(output_folder_path, "crop_frame")
        os.makedirs(crop_frame_dir, exist_ok=True)
        
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP']
        
        # 收集所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder_path, ext)))
            # 也搜索子目录中的图片
            image_files.extend(glob.glob(os.path.join(input_folder_path, "**", ext), recursive=True))
        
        # 去重并排序
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            print(f"  警告: 在 {folder_name} 中没有找到图片文件")
            continue
        
        print(f"  找到 {len(image_files)} 个图片文件")
        
        # 复制图片文件到 crop_frame 目录，按顺序重命名
        copied_count = 0
        for i, image_path in enumerate(image_files, 1):
            try:
                # 获取文件扩展名
                _, ext = os.path.splitext(image_path)
                
                # 生成新的文件名 (8位数字格式)
                new_filename = f"{i:08d}{ext.lower()}"
                output_path = os.path.join(crop_frame_dir, new_filename)
                
                # 复制文件
                shutil.copy2(image_path, output_path)
                copied_count += 1
                
                if i <= 5 or i % 100 == 0:  # 显示前5个和每100个的进度
                    print(f"    {i:3d}: {os.path.basename(image_path)} -> {new_filename}")
                    
            except Exception as e:
                print(f"  错误: 复制文件 {image_path} 时出错: {str(e)}")
                continue
        
        print(f"  成功复制 {copied_count} 个图片文件到 {crop_frame_dir}")
        
        # 复制其他非图片文件到主目录（可选）
        other_files = []
        for file_path in glob.glob(os.path.join(input_folder_path, "*")):
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file_path)
                if ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    other_files.append(file_path)
        
        if other_files:
            print(f"  同时复制 {len(other_files)} 个其他文件到主目录")
            for file_path in other_files:
                try:
                    shutil.copy2(file_path, output_folder_path)
                except Exception as e:
                    print(f"    错误: 复制文件 {file_path} 时出错: {str(e)}")
    
    print(f"\n✅ 完成! 所有文件夹已复制到 {output_test_dir}")
    print("你现在可以运行后续的 pipeline 脚本来处理这些数据")


def create_summary():
    """创建处理结果摘要"""
    # 自动根据输入目录名决定输出目录
    input_avis_dir = "../input/train"
    input_base = os.path.basename(os.path.normpath(input_avis_dir))
    output_test_dir = f"../output/{input_base}_results"
    
    if not os.path.exists(output_test_dir):
        print("输出目录不存在")
        return
    
    print(f"\n处理结果摘要:")
    print(f"输出目录: {output_test_dir}")
    
    folders = [f for f in os.listdir(output_test_dir) 
               if os.path.isdir(os.path.join(output_test_dir, f))]
    
    total_images = 0
    for folder in sorted(folders):
        crop_frame_dir = os.path.join(output_test_dir, folder, "crop_frame")
        if os.path.exists(crop_frame_dir):
            image_count = len([f for f in os.listdir(crop_frame_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            total_images += image_count
            print(f"  {folder}: {image_count} 张图片")
        else:
            print(f"  {folder}: 无 crop_frame 目录")
    
    print(f"\n总计: {len(folders)} 个视频文件夹, {total_images} 张图片")


if __name__ == "__main__":
    print("开始将 input/avis 内容复制到 output/{train,val,test}_results...")
    copy_avis_to_test_dataset()
    create_summary()