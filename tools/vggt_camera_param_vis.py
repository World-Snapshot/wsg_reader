import pycolmap
import sys
import os
import numpy as np

def read_vggt_results(video_name):
    base_dir = "../output/test_dataset_results"
    sparse_dir = os.path.join(base_dir, video_name, "results_vggt_sparse")
    
    if not os.path.exists(sparse_dir):
        print(f"❌ Directory not found: {sparse_dir}")
        return
    
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read(sparse_dir)
    
    print(f"=== VGGT Results for {video_name} ===")
    print(f"Directory: {sparse_dir}")
    print(f"Cameras: {len(reconstruction.cameras)}")
    print(f"Images: {len(reconstruction.images)}")
    print(f"3D Points: {len(reconstruction.points3D)}")
    
    # 相机参数
    print(f"\n📷 Camera Parameters:")
    for camera_id, camera in reconstruction.cameras.items():
        print(f"  Camera {camera_id}: {camera.model}")
        print(f"    Resolution: {camera.width} x {camera.height}")
        print(f"    Focal/Principal: {camera.params}")
    
    # 图像信息
    print(f"\n🖼️  Images (first 3):")
    for i, (image_id, image) in enumerate(list(reconstruction.images.items())[:3]):
        print(f"  {image.name}")
        print(f"    Position: {image.cam_from_world.translation}")
    
    # 点云统计
    if len(reconstruction.points3D) > 0:
        points = np.array([pt.xyz for pt in reconstruction.points3D.values()])
        print(f"\n🌐 3D Points Statistics:")
        print(f"    Count: {len(points)}")
        print(f"    X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"    Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"    Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

def show_all_results():
    """显示所有可用的 VGGT 结果"""
    base_dir = "../output/test_dataset_results"
    
    # 获取所有有 results_vggt_sparse 的视频
    available_videos = []
    for name in os.listdir(base_dir):
        vggt_dir = os.path.join(base_dir, name, "results_vggt_sparse")
        if os.path.exists(vggt_dir) and os.path.isdir(os.path.join(base_dir, name)):
            available_videos.append(name)
    
    if not available_videos:
        print("❌ No VGGT results found")
        return
    
    print(f"📊 Found {len(available_videos)} videos with VGGT results:")
    print("=" * 60)
    
    for video_name in sorted(available_videos):
        print(f"\n📹 Video: {video_name}")
        read_vggt_results(video_name)
        print("-" * 60)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Available videos:")
        base_dir = "../output/test_dataset_results"
        available_videos = []
        for name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, name)):  # 只检查目录
                vggt_dir = os.path.join(base_dir, name, "results_vggt_sparse")
                if os.path.exists(vggt_dir):
                    print(f"  ✅ {name}")
                    available_videos.append(name)
                else:
                    print(f"  ❌ {name} (no resultsresults_vggt_sparse)")
        
        print(f"\nUsage:")
        print(f"  python {sys.argv[0]} <video_name>     # 查看单个视频")
        print(f"  python {sys.argv[0]} --all           # 查看所有视频")
        
        if available_videos:
            print(f"\nExamples:")
            for vid in available_videos[:2]:  # 显示前两个例子
                print(f"  python {sys.argv[0]} \"{vid}\"")
    
    elif len(sys.argv) == 2:
        if sys.argv[1] == "--all":
            show_all_results()
        else:
            read_vggt_results(sys.argv[1])
    else:
        print("Usage: python vggt_camera_param_vis.py [video_name|--all]")