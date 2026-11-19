#!/usr/bin/env python3

"""
优化的SMPLer-X数据处理模块
使用PyTorch CUDA（最广泛可用）和优化的NumPy算法
无需额外安装CuPy等复杂库
"""

import numpy as np
import time

# 检查GPU可用性 - 优先使用PyTorch，更广泛可用
GPU_AVAILABLE = False
GPU_BACKEND = "none"

try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_BACKEND = "pytorch"
        print(f"✅ PyTorch CUDA available - GPU acceleration enabled ({torch.cuda.get_device_name(0)})")
        print(f"   CUDA Version: {torch.version.cuda}, GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    else:
        print("❌ PyTorch available but no CUDA GPU detected")
except ImportError:
    print("❌ PyTorch not available, using optimized CPU algorithms")

# 如果没有GPU，启用优化的NumPy计算
OPTIMIZED_NUMPY = not GPU_AVAILABLE


def optimized_vertex_projection(vertices_3d, cam_param):
    """
    优化的顶点投影计算
    支持PyTorch CUDA和优化的NumPy
    """
    print(f"      [OPTIM] Starting optimized vertex projection for {len(vertices_3d)} vertices...")
    start_time = time.time()
    
    if GPU_AVAILABLE and GPU_BACKEND == "pytorch":
        return pytorch_vertex_projection(vertices_3d, cam_param)
    else:
        return optimized_cpu_vertex_projection(vertices_3d, cam_param)


def pytorch_vertex_projection(vertices_3d, cam_param):
    """
    PyTorch CUDA版本的顶点投影
    """
    device = torch.device('cuda')
    
    # 转移到GPU
    vertices_gpu = torch.from_numpy(vertices_3d).float().to(device)
    focal = torch.tensor(cam_param['focal'], device=device, dtype=torch.float32)
    princpt = torch.tensor(cam_param['princpt'], device=device, dtype=torch.float32)
    
    # 并行投影计算
    valid_mask = vertices_gpu[:, 2] > 0
    
    # 避免除零
    z_safe = torch.where(vertices_gpu[:, 2] > 0, vertices_gpu[:, 2], torch.ones_like(vertices_gpu[:, 2]))
    px_float = focal[0] * vertices_gpu[:, 0] / z_safe + princpt[0]
    py_float = focal[1] * vertices_gpu[:, 1] / z_safe + princpt[1]
    
    # 添加抖动
    torch.manual_seed(42)
    jitter = torch.rand((len(vertices_3d), 2), device=device) * 3.0 - 1.5  # [-1.5, 1.5]
    px_int = torch.round(px_float + jitter[:, 0]).int()
    py_int = torch.round(py_float + jitter[:, 1]).int()
    
    # 转回CPU
    px_int_cpu = px_int.cpu().numpy()
    py_int_cpu = py_int.cpu().numpy()
    valid_mask_cpu = valid_mask.cpu().numpy()
    
    duration = time.time() - start_time
    rate = len(vertices_3d) / duration if duration > 0 else 0
    print(f"      [PYTORCH] Projection completed: {rate:.0f} vertices/second")
    
    return px_int_cpu, py_int_cpu, valid_mask_cpu


def optimized_cpu_vertex_projection(vertices_3d, cam_param):
    """
    优化的CPU版本的顶点投影（使用向量化NumPy操作）
    """
    focal = np.array(cam_param['focal'])
    princpt = np.array(cam_param['princpt'])
    
    # 向量化计算
    valid_mask = vertices_3d[:, 2] > 0
    
    # 使用向量化操作而不是循环
    z_safe = np.where(vertices_3d[:, 2] > 0, vertices_3d[:, 2], 1.0)
    px_float = focal[0] * vertices_3d[:, 0] / z_safe + princpt[0]
    py_float = focal[1] * vertices_3d[:, 1] / z_safe + princpt[1]
    
    # 向量化抖动
    np.random.seed(42)
    jitter = np.random.uniform(-1.5, 1.5, (len(vertices_3d), 2))
    px_int = np.round(px_float + jitter[:, 0]).astype(np.int32)
    py_int = np.round(py_float + jitter[:, 1]).astype(np.int32)
    
    duration = time.time() - start_time
    rate = len(vertices_3d) / duration if duration > 0 else 0
    print(f"      [CPU-OPT] Projection completed: {rate:.0f} vertices/second")
    
    return px_int, py_int, valid_mask


def optimized_face_generation(vertices, k_neighbors=12):
    """
    优化的面生成算法
    """
    if len(vertices) < 3:
        return None
        
    print(f"      [OPTIM] Creating dense faces for {len(vertices)} vertices...")
    start_time = time.time()
    
    if GPU_AVAILABLE and GPU_BACKEND == "pytorch" and len(vertices) > 1000:
        return pytorch_face_generation(vertices, k_neighbors)
    else:
        return optimized_cpu_face_generation(vertices, k_neighbors)


def pytorch_face_generation(vertices, k_neighbors=12):
    """
    PyTorch CUDA版本的面生成
    """
    start_time = time.time()  # Fix: Add missing start_time
    device = torch.device('cuda')
    
    try:
        print(f"      [PYTORCH] Creating dense faces using GPU...")
        vertices_gpu = torch.from_numpy(vertices).float().to(device)
        
        n = len(vertices)
        faces_list = []
        
        # 分块处理以避免内存溢出
        chunk_size = min(500, n)  # 减小块大小避免GPU内存不足
        
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            chunk_vertices = vertices_gpu[i:end_i]
            
            # 计算距离矩阵（GPU并行）
            diff = chunk_vertices.unsqueeze(1) - vertices_gpu.unsqueeze(0)
            distances = torch.sum(diff ** 2, dim=2)
            
            # 找到最近邻
            for j in range(len(chunk_vertices)):
                vertex_idx = i + j
                
                # 排除自己，找到k个最近邻
                vertex_distances = distances[j].clone()
                vertex_distances[vertex_idx] = float('inf')
                
                # 获取最近邻索引
                _, nearest_indices = torch.topk(vertex_distances, k_neighbors, largest=False)
                nearest_indices = nearest_indices.cpu().numpy()
                
                # 生成三角形面
                for k in range(len(nearest_indices) - 1):
                    if nearest_indices[k] < n and nearest_indices[k+1] < n:
                        faces_list.append([vertex_idx, nearest_indices[k], nearest_indices[k+1]])
        
        faces = np.array(faces_list) if faces_list else None
        
        duration = time.time() - start_time
        if faces is not None:
            rate = len(faces) / duration if duration > 0 else 0
            print(f"      [PYTORCH] Created {len(faces)} faces in {duration:.2f}s ({rate:.0f} faces/second)")
        
        return faces
        
    except Exception as e:
        print(f"      [PYTORCH] GPU face generation failed: {str(e)}, falling back to CPU")
        return optimized_cpu_face_generation(vertices, k_neighbors)


def optimized_cpu_face_generation(vertices, k_neighbors=12):
    """
    优化的CPU版本面生成
    """
    start_time = time.time()  # Fix: Add missing start_time
    
    try:
        # 使用scipy进行快速距离计算
        from scipy.spatial.distance import cdist
        
        faces = []
        n = len(vertices)
        
        print(f"      [CPU-OPT] Using vectorized distance calculation...")
        
        # 分块处理大数据集
        if n > 3000:
            print(f"      Large vertex set ({n}), using optimized sampling")
            # 智能采样：保持重要顶点，随机采样其他
            sample_size = 3000
            indices = np.random.choice(n, sample_size, replace=False)
            sampled_vertices = vertices[indices]
            distances = cdist(sampled_vertices, sampled_vertices)
            
            for i in range(sample_size):
                # 找到k个最近邻（排除自己）
                nearest_indices = np.argpartition(distances[i], k_neighbors + 1)[1:k_neighbors + 1]
                
                # 生成三角形面
                for j in range(len(nearest_indices) - 1):
                    # 映射回原始索引
                    face = [indices[i], indices[nearest_indices[j]], indices[nearest_indices[j+1]]]
                    faces.append(face)
        else:
            # 小数据集：完整计算
            distances = cdist(vertices, vertices)
            
            for i in range(n):
                # 找到k个最近邻（排除自己）
                nearest_indices = np.argpartition(distances[i], k_neighbors + 1)[1:k_neighbors + 1]
                
                # 生成三角形面
                for j in range(len(nearest_indices) - 1):
                    faces.append([i, nearest_indices[j], nearest_indices[j+1]])
        
        faces = np.array(faces) if faces else None
        
        duration = time.time() - start_time
        if faces is not None:
            rate = len(faces) / duration if duration > 0 else 0
            print(f"      [CPU-OPT] Created {len(faces)} faces in {duration:.2f}s ({rate:.0f} faces/second)")
        
        return faces
        
    except Exception as e:
        print(f"      [CPU-OPT] Face generation failed: {str(e)}")
        return None


def optimized_storage_search(px_array, py_array, valid_mask, wsg_array, vertex_index_channel, width, height):
    """
    优化的存储位置搜索
    """
    print(f"      [OPTIM] Starting optimized storage search...")
    start_time = time.time()
    
    # 这个操作主要是内存访问，GPU优势不大，使用优化的CPU算法
    return optimized_cpu_storage_search(px_array, py_array, valid_mask, wsg_array, vertex_index_channel, width, height)


def optimized_cpu_storage_search(px_array, py_array, valid_mask, wsg_array, vertex_index_channel, width, height):
    """
    优化的CPU版本存储搜索
    """
    stored_positions = []
    search_radii = [1, 2, 3, 5]  # 优化过的搜索半径
    
    # 预计算有效顶点索引以避免重复检查
    valid_indices = np.where(valid_mask)[0]
    
    # 向量化边界检查
    px_valid = px_array[valid_indices]
    py_valid = py_array[valid_indices]
    
    in_bounds = (px_valid >= 0) & (px_valid < width) & (py_valid >= 0) & (py_valid < height)
    bounded_indices = valid_indices[in_bounds]
    bounded_px = px_valid[in_bounds]
    bounded_py = py_valid[in_bounds]
    
    # 检查中心点
    storage_channel = wsg_array[:, :, vertex_index_channel]
    
    for idx, (i, px, py) in enumerate(zip(bounded_indices, bounded_px, bounded_py)):
        stored = False
        
        # 中心点检查
        if storage_channel[py, px] == -1:
            stored_positions.append((i, py, px))
            storage_channel[py, px] = i
            stored = True
        
        # 扩展搜索
        if not stored:
            for radius in search_radii:
                if stored:
                    break
                
                # 生成搜索偏移（优化的圆形搜索模式）
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx*dx + dy*dy > radius*radius:  # 圆形搜索而非方形
                            continue
                        if abs(dx) < radius and abs(dy) < radius:
                            continue  # 跳过内部已检查的像素
                        
                        tx, ty = px + dx, py + dy
                        if 0 <= tx < width and 0 <= ty < height:
                            if storage_channel[ty, tx] == -1:
                                stored_positions.append((i, ty, tx))
                                storage_channel[ty, tx] = i
                                stored = True
                                break
                    if stored:
                        break
    
    duration = time.time() - start_time
    rate = len(stored_positions) / duration if duration > 0 else 0
    print(f"      [CPU-OPT] Storage search completed: {len(stored_positions)} positions found ({rate:.0f} pos/second)")
    
    return stored_positions


def check_gpu_availability():
    """检查GPU可用性"""
    return GPU_AVAILABLE


def get_backend_info():
    """获取当前后端信息"""
    if GPU_AVAILABLE:
        if GPU_BACKEND == "pytorch":
            return f"PyTorch CUDA on {torch.cuda.get_device_name(0)}"
        elif GPU_BACKEND == "cupy":
            return "CuPy CUDA"
    return "Optimized CPU (NumPy vectorized)"