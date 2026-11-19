#!/usr/bin/env python3
"""
MatSynth数据集下载器
下载用户指定数量的parquet文件，自动合并为单个文件
"""

import os
import sys
import requests
from pathlib import Path
from typing import List, Optional
import argparse
from tqdm import tqdm


class MatSynthDownloader:
    def __init__(self, base_url: str = "https://huggingface.co/datasets/gvecchio/MatSynth"):
        self.base_url = base_url
        self.api_url = "https://huggingface.co/api/datasets/gvecchio/MatSynth"
        self.session = requests.Session()

    def get_file_list(self) -> List[dict]:
        """获取数据集中所有parquet文件的列表"""
        try:
            response = self.session.get(f"{self.api_url}/tree/main")
            response.raise_for_status()

            files_data = response.json()
            parquet_files = []

            # 递归搜索所有parquet文件
            def extract_parquet_files(items):
                for item in items:
                    if item.get('type') == 'file' and item.get('path', '').endswith('.parquet'):
                        parquet_files.append({
                            'path': item['path'],
                            'size': item.get('size', 0),
                            'url': f"https://huggingface.co/datasets/gvecchio/MatSynth/resolve/main/{item['path']}"
                        })
                    elif item.get('type') == 'directory':
                        # 如果有子目录，递归获取
                        try:
                            subdir_response = self.session.get(f"{self.api_url}/tree/main/{item['path']}")
                            if subdir_response.status_code == 200:
                                subdir_data = subdir_response.json()
                                extract_parquet_files(subdir_data)
                        except:
                            pass

            extract_parquet_files(files_data)
            return parquet_files

        except Exception as e:
            print(f"获取文件列表失败: {e}")
            return []

    def download_file(self, file_info: dict, download_dir: Path) -> bool:
        """下载单个文件"""
        # 确保下载目录存在
        download_dir.mkdir(parents=True, exist_ok=True)

        # 直接将文件放在下载目录根目录
        filename = Path(file_info['path']).name
        file_path_str = str(download_dir / filename)

        # 检查文件是否已存在
        if os.path.exists(file_path_str):
            if os.path.getsize(file_path_str) == file_info['size']:
                print(f"文件已存在，跳过: {filename}")
                return True

        try:
            print(f"开始下载: {filename}")
            response = self.session.get(file_info['url'], stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            # 先下载到临时文件，然后重命名
            temp_file = file_path_str + ".tmp"

            with open(temp_file, 'wb') as f:
                with tqdm(
                        desc=f"下载 {filename}",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # 重命名临时文件为最终文件
            os.rename(temp_file, file_path_str)
            print(f"下载完成: {filename}")
            return True

        except Exception as e:
            print(f"下载失败 {filename}: {e}")
            # 清理临时文件
            temp_file = file_path_str + ".tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

    def merge_parquet_files(self, download_dir: Path):
        """合并所有parquet文件为一个文件并删除原文件"""
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
        except ImportError:
            print("❌ 合并功能需要安装 pyarrow: pip install pyarrow")
            return

        # 查找所有parquet文件
        parquet_files = [f for f in download_dir.glob("*.parquet") if f.name != "matsynth_data.parquet"]

        if not parquet_files:
            print("未找到parquet文件")
            return

        if len(parquet_files) == 1:
            # 只有一个文件，直接重命名
            old_file = parquet_files[0]
            new_file = download_dir / "matsynth_data.parquet"
            old_file.rename(new_file)
            print(f"文件已重命名为: matsynth_data.parquet")
            return

        print(f"开始合并 {len(parquet_files)} 个parquet文件...")

        try:
            # 读取所有parquet文件
            tables = []
            for file_path in tqdm(parquet_files, desc="读取文件"):
                try:
                    table = pq.read_table(file_path)
                    tables.append(table)
                except Exception as e:
                    print(f"读取失败 {file_path.name}: {e}")

            if not tables:
                print("没有成功读取任何文件")
                return

            # 合并所有表
            merged_table = pa.concat_tables(tables)

            # 写入合并文件
            output_file = download_dir / "matsynth_data.parquet"
            pq.write_table(merged_table, output_file, compression='snappy')

            # 删除原始文件
            for file_path in parquet_files:
                file_path.unlink()
                print(f"已删除: {file_path.name}")

            file_size = output_file.stat().st_size / (1024 ** 3)
            print(f"✅ 合并完成: matsynth_data.parquet ({file_size:.2f} GB)")

        except Exception as e:
            print(f"合并失败: {e}")

    def download_files(self, num_files: Optional[int] = None, download_dir: str = "./input/MatsynthInput"):
        """下载指定数量的文件"""
        download_path = Path(download_dir).resolve()  # 使用绝对路径

        print("正在获取文件列表...")
        file_list = self.get_file_list()

        if not file_list:
            print("未找到parquet文件")
            return

        total_files = len(file_list)
        print(f"找到 {total_files} 个parquet文件")

        # 如果没有指定数量，提示用户输入
        if num_files is None:
            while True:
                try:
                    user_input = input(
                        f"\n请输入要下载的文件数量 (1-{total_files}, 输入0下载所有文件, 默认10): ").strip()
                    if user_input == "":
                        num_files = 10
                        break
                    num_files = int(user_input)
                    if num_files == 0:
                        num_files = total_files
                        break
                    elif 1 <= num_files <= total_files:
                        break
                    else:
                        print(f"❌ 请输入1到{total_files}之间的数字")
                except ValueError:
                    print("❌ 请输入有效的数字")

        # 确定要下载的文件
        files_to_download = file_list[:num_files]
        print(f"\n将下载 {len(files_to_download)} 个文件")

        total_size = sum(f['size'] for f in files_to_download)
        print(f"总大小: {total_size / (1024 ** 3):.2f} GB")
        print(f"下载目录: {download_path}")
        print()

        # 开始下载
        success_count = 0
        for i, file_info in enumerate(files_to_download, 1):
            print(f"[{i}/{len(files_to_download)}] ", end="")
            if self.download_file(file_info, download_path):
                success_count += 1

        print(f"\n下载完成! 成功: {success_count}/{len(files_to_download)}")

        # 合并文件
        if success_count > 0:
            print("\n开始处理文件...")
            self.merge_parquet_files(download_path)


def main():
    parser = argparse.ArgumentParser(description='MatSynth数据集下载器')
    parser.add_argument('-n', '--num-files', type=int,
                        help='要下载的文件数量 (如不指定则会提示输入)')
    parser.add_argument('-d', '--download-dir', type=str, default='./input/MatsynthInput',
                        help='下载目录 (默认: ./input/MatsynthInput)')
    parser.add_argument('--list-files', action='store_true',
                        help='只列出文件，不下载')

    args = parser.parse_args()
    downloader = MatSynthDownloader()

    if args.list_files:
        print("正在获取文件列表...")
        file_list = downloader.get_file_list()
        print(f"\n找到 {len(file_list)} 个parquet文件:")
        for i, file_info in enumerate(file_list, 1):
            size_mb = file_info['size'] / (1024 ** 2)
            print(f"{i:3d}. {file_info['path']} ({size_mb:.1f} MB)")
    else:
        downloader.download_files(args.num_files, args.download_dir)


if __name__ == "__main__":
    main()