import os
import numpy as np
import glob
import json
import cv2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

# Import optimized processing module (supports PyTorch CUDA + optimized NumPy)
try:
    from utils.optimized_processing import (
        optimized_vertex_projection,
        optimized_face_generation,
        optimized_storage_search,
        check_gpu_availability,
        get_backend_info
    )
    OPTIMIZATION_AVAILABLE = True
    print(f"✅ Optimized processing module loaded successfully")
    print(f"    Backend: {get_backend_info()}")
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("❌ Optimized processing not available, using basic CPU only")

# Optional imports for 3D mesh rendering
try:
    import os
    # Set OpenGL platform for headless rendering
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    import trimesh
    import pyrender
    RENDERING_AVAILABLE = True
except ImportError:
    RENDERING_AVAILABLE = False
    print("Warning: trimesh/pyrender not available. 3D mesh rendering will be disabled.")


# SMPLer-X joint ID reverse mapping
SMPLERX_JOINT_REVERSE_MAPPING = {
    0: 'global_orient',
    **{i: 'body_pose' for i in range(1, 22)},
    **{i: 'left_hand_pose' for i in range(22, 37)},
    **{i: 'right_hand_pose' for i in range(37, 52)},
    52: 'jaw_pose',
    53: 'leye_pose', 
    54: 'reye_pose',
    55: 'transl'
}

# WSG channel definitions
PERSON_ID_CHANNEL = 72
JOINT_ID_CHANNEL = 73
JOINT_X_CHANNEL = 74
JOINT_Y_CHANNEL = 75
JOINT_Z_CHANNEL = 76
BETAS_EXPRESSION_TYPE_CHANNEL = 77
BETAS_EXPRESSION_CHANNELS = list(range(78, 88))
BBOX_CHANNEL = 88
BBOX_MMDET_CHANNEL = 89
VERTEX_INDEX_CHANNEL = 90    # 顶点序号 (0-10474)
VERTEX_X_CHANNEL = 91        # 校准后的X坐标
VERTEX_Y_CHANNEL = 92        # 校准后的Y坐标  
VERTEX_Z_CHANNEL = 93        # 校准后的Z坐标
STORED_VISUALIZATION_CHANNELS = slice(90, 93)
SMPLX_MODEL_ROOT = os.environ.get("WSM_SMPLX_MODEL_ROOT", "")
_COMPLETE_SMPLX_MODEL = None


def create_faces_from_vertices(vertices):
    """Create face connectivity from vertices using Delaunay triangulation"""
    try:
        from scipy.spatial import Delaunay
        
        # Use only XY coordinates for 2D Delaunay triangulation
        points_2d = vertices[:, :2]
        
        # Create Delaunay triangulation
        tri = Delaunay(points_2d)
        
        # The triangulation gives us the face connectivity
        faces = tri.simplices
        
        print(f"      Created {len(faces)} faces from {len(vertices)} vertices using Delaunay triangulation")
        return faces
        
    except Exception as e:
        print(f"      Failed to create faces from vertices: {str(e)}")
        return None


def load_smplx_faces():
    """Load SMPL-X face connectivity for mesh rendering"""
    try:
        # Try to load from NPZ file first (pre-saved SMPL-X faces)
        smplx_faces_path = os.path.join(
            SMPLX_MODEL_ROOT, "smplx", "SMPLX_NEUTRAL.npz"
        )
        if SMPLX_MODEL_ROOT and os.path.exists(smplx_faces_path):
            smplx_data = np.load(smplx_faces_path)
            if 'f' in smplx_data:
                return smplx_data['f']

        # Try to import an explicitly configured SMPL-X helper.
        import sys
        helper_root = os.environ.get("WSM_HUMAN_MODELS_PYTHONPATH", "")
        if helper_root:
            sys.path.append(helper_root)
        from utils.human_models import smpl_x
        
        # Get SMPL-X instance and return faces
        if hasattr(smpl_x, 'face') and smpl_x.face is not None:
            return smpl_x.face
        else:
            print("      SMPL-X faces not found, using fallback method")
            return None
    except Exception as e:
        print(f"      Failed to load SMPL-X faces: {str(e)}")
        return None


def filter_faces_for_vertices(faces, vertex_indices, num_vertices, min_vertices_per_face=3):
    """Filter SMPL-X faces to include those with sufficient vertices present"""
    if faces is None or vertex_indices is None:
        return None
        
    # Create a mapping from original indices to new indices
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}
    vertex_set = set(vertex_indices)
    
    # Filter faces where at least min_vertices_per_face vertices are present
    filtered_faces = []
    
    for face in faces:
        present_vertices = [v for v in face if v in vertex_set]
        
        if len(present_vertices) >= min_vertices_per_face:
            # For faces with all 3 vertices, use as normal triangle
            if len(present_vertices) == 3:
                new_face = [vertex_map[v] for v in face]
                filtered_faces.append(new_face)
            
            # For faces with 2 vertices, create degenerate triangle (line)
            elif len(present_vertices) == 2 and min_vertices_per_face <= 2:
                # Duplicate the last vertex to create a degenerate triangle
                new_face = [vertex_map[present_vertices[0]], 
                           vertex_map[present_vertices[1]], 
                           vertex_map[present_vertices[1]]]
                filtered_faces.append(new_face)
    
    if len(filtered_faces) > 0:
        print(f"      Filtered {len(filtered_faces)} faces from {len(faces)} original faces (min_vertices={min_vertices_per_face})")
        return np.array(filtered_faces)
    else:
        print(f"      No valid faces found for given vertices")
        return None


def create_dense_faces_from_vertices(vertices_3d, vertex_indices):
    """Create dense face connectivity for better mesh representation"""
    if len(vertices_3d) < 3:
        return None
    
    try:
        # Method 1: Use stored 2D positions for more accurate triangulation
        print(f"      Creating dense faces for {len(vertices_3d)} vertices...")
        
        # Use optimized processing if available
        if OPTIMIZATION_AVAILABLE and len(vertices_3d) > 500:
            print(f"      Using optimized face generation")
            return optimized_face_generation(vertices_3d, k_neighbors=12)
        
        # CPU fallback
        print(f"      Using CPU for face generation")
        from scipy.spatial.distance import cdist
        
        # Calculate pairwise distances in 3D
        distances = cdist(vertices_3d, vertices_3d)
        
        # For each vertex, connect to its nearest neighbors more densely
        faces = []
        k_neighbors = min(12, len(vertices_3d) - 1)  # Connect to 12 nearest neighbors for even denser mesh
        
        for i in range(len(vertices_3d)):
            # Find k nearest neighbors (excluding self)
            neighbor_indices = np.argsort(distances[i])[1:k_neighbors+1]
            
            # Create triangles with this vertex and pairs of neighbors
            # Use a fan triangulation pattern for better coverage
            if len(neighbor_indices) >= 2:
                for j in range(len(neighbor_indices)):
                    for k in range(j+1, len(neighbor_indices)):
                        # Only add triangles with reasonable edge lengths
                        # Check if triangle is not too elongated
                        v1, v2, v3 = i, neighbor_indices[j], neighbor_indices[k]
                        
                        # Calculate edge lengths
                        edge1 = distances[v1, v2]
                        edge2 = distances[v2, v3] 
                        edge3 = distances[v1, v3]
                        
                        # Only add triangle if edges are reasonably proportioned
                        max_edge = max(edge1, edge2, edge3)
                        min_edge = min(edge1, edge2, edge3)
                        
                        # Relax aspect ratio threshold for better hole filling
                        if max_edge / (min_edge + 1e-8) < 4.0:  # More lenient aspect ratio
                            face = [v1, v2, v3]
                            faces.append(face)
        
        # Additional step: Try to fill remaining gaps with longer-range connections
        # For vertices with few connections, add extra triangles
        vertex_connection_count = [0] * len(vertices_3d)
        for face in faces:
            for v in face:
                vertex_connection_count[v] += 1
        
        # Find vertices with few connections and add more triangles
        for i in range(len(vertices_3d)):
            if vertex_connection_count[i] < 20:  # If vertex has fewer than 20 connections
                # Find additional neighbors at longer range
                long_range_neighbors = np.argsort(distances[i])[k_neighbors+1:k_neighbors+6]  # Next 5 neighbors
                for j in range(min(3, len(long_range_neighbors))):
                    for k in range(j+1, min(3, len(long_range_neighbors))):
                        if j < len(long_range_neighbors) and k < len(long_range_neighbors):
                            v1, v2, v3 = i, long_range_neighbors[j], long_range_neighbors[k]
                            
                            # More lenient check for gap-filling triangles
                            edge1 = distances[v1, v2]
                            edge2 = distances[v2, v3]
                            edge3 = distances[v1, v3]
                            max_edge = max(edge1, edge2, edge3)
                            min_edge = min(edge1, edge2, edge3)
                            
                            if max_edge / (min_edge + 1e-8) < 5.0:  # Very lenient for gap filling
                                face = [v1, v2, v3]
                                faces.append(face)
        
        if len(faces) > 0:
            print(f"      Created {len(faces)} faces using nearest neighbor connectivity")
            return np.array(faces)
        else:
            # Fallback to Delaunay triangulation on XY projection
            from scipy.spatial import Delaunay
            points_2d = vertices_3d[:, :2]
            tri = Delaunay(points_2d)
            new_faces = tri.simplices
            print(f"      Created {len(new_faces)} faces using Delaunay triangulation fallback")
            return new_faces
        
    except Exception as e:
        print(f"      Dense face creation failed: {str(e)}")
        return None


def perspective_projection(vertices, cam_param):
    """Project 3D vertices to 2D using camera parameters"""
    fx, fy = cam_param['focal']
    cx, cy = cam_param['princpt']
    
    # Create a copy to avoid modifying original vertices
    vertices_2d = vertices.copy()
    
    # Project to image plane: px = fx * x / z + cx, py = fy * y / z + cy
    vertices_2d[:, 0] = vertices_2d[:, 0] * fx / vertices_2d[:, 2] + cx
    vertices_2d[:, 1] = vertices_2d[:, 1] * fy / vertices_2d[:, 2] + cy
    
    return vertices_2d[:, :2]  # Return only X, Y coordinates


def render_3d_mesh_silhouette_with_indices(vertices_3d, vertex_indices, img_shape, cam_param):
    """Render 3D mesh silhouette with proper face filtering and pyrender"""
    height, width = img_shape[:2]
    silhouette = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(vertices_3d) < 3:
        return silhouette
    
    try:
        # First, try to use pyrender with filtered SMPL-X faces
        if RENDERING_AVAILABLE and vertex_indices is not None:
            # Load SMPL-X faces
            smplx_faces = load_smplx_faces()
            if smplx_faces is not None:
                # Filter faces for available vertices
                filtered_faces = filter_faces_for_vertices(smplx_faces, vertex_indices, len(vertices_3d))
                
                if filtered_faces is not None and len(filtered_faces) > 100:
                    try:
                        # Create trimesh with filtered faces
                        mesh = trimesh.Trimesh(vertices_3d, filtered_faces, process=False)
                        
                        # Apply rotation
                        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                        mesh.apply_transform(rot)
                        
                        # Create material
                        material = pyrender.MetallicRoughnessMaterial(
                            metallicFactor=0.0,
                            alphaMode='OPAQUE',
                            baseColorFactor=(1.0, 1.0, 1.0, 1.0)
                        )
                        
                        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material)
                        
                        # Scene setup
                        scene = pyrender.Scene(ambient_light=(1.0, 1.0, 1.0))
                        scene.add(mesh_pyrender)
                        
                        # Camera
                        focal, princpt = cam_param['focal'], cam_param['princpt']
                        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], 
                                                         cx=princpt[0], cy=princpt[1])
                        scene.add(camera)
                        
                        # Render with error handling
                        renderer = None
                        try:
                            renderer = pyrender.OffscreenRenderer(viewport_width=width, 
                                                                viewport_height=height)
                            rgb, depth = renderer.render(scene)
                            
                            # Convert to silhouette
                            mask = (depth > 0)
                            silhouette[mask] = [255, 255, 255]
                            
                            print(f"      Successfully rendered with pyrender using filtered faces")
                            return silhouette
                        finally:
                            if renderer is not None:
                                try:
                                    renderer.delete()
                                except Exception as e:
                                    # Ignore EGL cleanup errors
                                    if "eglDestroyContext" not in str(e):
                                        print(f"      Warning during renderer cleanup: {str(e)}")
                        
                    except Exception as e:
                        print(f"      Pyrender with filtered faces failed: {str(e)}")
        
        # Fallback: Project to 2D and use point cloud visualization
        print(f"      Using 2D projection fallback (pyrender not available or insufficient faces)")
        vertices_2d = perspective_projection(vertices_3d, cam_param)
        
        # Filter vertices within image bounds
        valid_mask = (vertices_2d[:, 0] >= 0) & (vertices_2d[:, 0] < width) & \
                     (vertices_2d[:, 1] >= 0) & (vertices_2d[:, 1] < height) & \
                     (vertices_3d[:, 2] > 0)  # In front of camera
        
        if np.sum(valid_mask) >= 10:
            valid_vertices_2d = vertices_2d[valid_mask]
            
            # Draw all vertices as points to show human shape
            for pt in valid_vertices_2d:
                cv2.circle(silhouette, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
        
    except Exception as e:
        print(f"      3D silhouette rendering failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return silhouette


def render_human_mesh_overlay_with_stored_positions(vertices_3d, vertices_2d, vertex_indices, background_img, alpha=0.7):
    """Render human mesh overlay using stored 2D positions"""
    overlay = background_img.copy()
    height, width = background_img.shape[:2]
    
    if len(vertices_2d) < 10:
        return overlay
    
    try:
        # Filter valid vertices within image bounds
        valid_mask = (vertices_2d[:, 0] >= 0) & (vertices_2d[:, 0] < width) & \
                     (vertices_2d[:, 1] >= 0) & (vertices_2d[:, 1] < height)
        
        valid_vertices_2d = vertices_2d[valid_mask]
        valid_vertices_3d = vertices_3d[valid_mask]
        
        if len(valid_vertices_2d) >= 10:
            # Use 3D depth information for enhanced visualization
            z_values = valid_vertices_3d[:, 2]
            if np.max(z_values) > np.min(z_values):
                z_normalized = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
            else:
                z_normalized = np.zeros(len(z_values))
            
            # Create depth-based semi-transparent overlay
            overlay_mask = np.zeros((height, width), dtype=np.float32)
            
            # Draw each vertex with depth-based influence
            for pos_2d, z_norm in zip(valid_vertices_2d, z_normalized):
                x_2d, y_2d = int(pos_2d[0]), int(pos_2d[1])
                
                # Closer points have larger influence radius
                influence_radius = max(2, int(6 * (1 - z_norm * 0.5)))  # 6 to 3
                
                # Create circular influence around each vertex
                y_start = max(0, y_2d - influence_radius)
                y_end = min(height, y_2d + influence_radius + 1)
                x_start = max(0, x_2d - influence_radius)
                x_end = min(width, x_2d + influence_radius + 1)
                
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        distance = np.sqrt((x - x_2d)**2 + (y - y_2d)**2)
                        if distance <= influence_radius:
                            # Gaussian-like falloff
                            influence = np.exp(-(distance / influence_radius)**2)
                            overlay_mask[y, x] = max(overlay_mask[y, x], influence)
            
            # Apply semi-transparent overlay based on mesh density
            overlay_mask = np.clip(overlay_mask, 0, 1)
            mask_3d = np.stack([overlay_mask] * 3, axis=2)
            
            # Semi-transparent white overlay
            overlay_color = np.full((height, width, 3), 255, dtype=np.uint8)
            
            overlay = overlay.astype(np.float32)
            overlay = overlay * (1 - mask_3d * alpha) + overlay_color * mask_3d * alpha
            overlay = overlay.astype(np.uint8)
            
            # Draw vertex points with depth-based visualization
            for pos_2d, z_norm in zip(valid_vertices_2d, z_normalized):
                x_2d, y_2d = int(pos_2d[0]), int(pos_2d[1])
                
                # Closer points are larger and more prominent
                radius = max(1, int(2 * (1 - z_norm * 0.4)))  # 2 to 1.2
                intensity = 255 - int(100 * z_norm)  # 255 to 155
                
                cv2.circle(overlay, (x_2d, y_2d), radius, 
                         (intensity, intensity, 0), -1)  # Yellow dots
                
    except Exception as e:
        print(f"      Overlay rendering with stored positions failed: {str(e)}")
    
    return overlay


def render_human_mesh_overlay(vertices_2d, vertices_3d, vertex_indices, background_img, cam_param, alpha=0.7):
    """Render human mesh overlay with proper body shape"""
    overlay = background_img.copy()
    height, width = background_img.shape[:2]
    
    if len(vertices_2d) < 10:
        return overlay
    
    try:
        # Filter valid vertices
        valid_mask = (vertices_2d[:, 0] >= 0) & (vertices_2d[:, 0] < width) & \
                     (vertices_2d[:, 1] >= 0) & (vertices_2d[:, 1] < height) & \
                     (vertices_3d[:, 2] > 0)
        
        valid_vertices = vertices_2d[valid_mask]
        
        if len(valid_vertices) >= 10:
            # Create body shape using clustering
            from sklearn.cluster import DBSCAN
            
            try:
                # Cluster vertices to identify body parts
                clustering = DBSCAN(eps=30, min_samples=5).fit(valid_vertices)
                labels = clustering.labels_
                
                # Create semi-transparent overlay for each cluster
                overlay_mask = np.zeros((height, width), dtype=np.uint8)
                
                for label in set(labels):
                    if label == -1:
                        continue
                    
                    cluster_points = valid_vertices[labels == label]
                    if len(cluster_points) >= 3:
                        try:
                            hull = ConvexHull(cluster_points)
                            hull_points = cluster_points[hull.vertices]
                            cv2.fillPoly(overlay_mask, [hull_points.astype(np.int32)], 255)
                            # Draw outline
                            cv2.polylines(overlay, [hull_points.astype(np.int32)], True, (0, 255, 0), 2)
                        except:
                            pass
                
                # Apply semi-transparent white overlay
                mask_3d = np.stack([overlay_mask] * 3, axis=2) / 255.0
                overlay_color = np.full((height, width, 3), 255, dtype=np.uint8)
                
                overlay = overlay.astype(np.float32)
                overlay = overlay * (1 - mask_3d * alpha) + overlay_color * mask_3d * alpha
                overlay = overlay.astype(np.uint8)
                
            except:
                # Fallback to simple convex hull
                if len(valid_vertices) >= 3:
                    hull = ConvexHull(valid_vertices)
                    hull_points = valid_vertices[hull.vertices]
                    
                    # Create mask
                    mesh_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(mesh_mask, [hull_points.astype(np.int32)], 255)
                    
                    # Apply overlay
                    mask_3d = np.stack([mesh_mask] * 3, axis=2) / 255.0
                    overlay_color = np.full((height, width, 3), 255, dtype=np.uint8)
                    
                    overlay = overlay.astype(np.float32)
                    overlay = overlay * (1 - mask_3d * alpha) + overlay_color * mask_3d * alpha
                    overlay = overlay.astype(np.uint8)
                    
                    # Draw outline
                    cv2.polylines(overlay, [hull_points.astype(np.int32)], True, (0, 255, 0), 2)
            
            # Draw vertex points
            for pt in valid_vertices[::10]:  # Sample every 10th point
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 1, (255, 255, 0), -1)
                
    except Exception as e:
        print(f"      Overlay rendering failed: {str(e)}")
    
    return overlay


def render_3d_mesh_silhouette_using_stored_vertices(vertices_3d, img_shape, cam_param):
    """Render 3D mesh silhouette using stored vertices without Z adjustment"""
    height, width = img_shape[:2]
    silhouette = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(vertices_3d) < 3:
        return silhouette
    
    try:
        # Use vertices as-is from SMPLer-X (they are already in camera coordinates)
        # NO Z adjustment - this was causing the problem!
        
        if RENDERING_AVAILABLE:
            # Try to load SMPL-X faces for proper mesh rendering
            faces = load_smplx_faces()
            
            # If we have partial vertices, we need to filter faces
            if faces is not None and len(vertices_3d) < 10475:
                # We have partial vertices, need to create a mapping
                print(f"      Filtering faces for {len(vertices_3d)} vertices")
                # For now, skip pyrender for partial vertices
                faces = None
            
            if faces is not None:
                try:
                    # Create trimesh object with proper faces
                    mesh = trimesh.Trimesh(vertices_3d, faces, process=False)
                    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                    mesh.apply_transform(rot)
                    
                    # Create white material for silhouette
                    material = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.0, 
                        alphaMode='OPAQUE',
                        baseColorFactor=(1.0, 1.0, 1.0, 1.0)
                    )
                    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material)
                    
                    # Scene setup
                    scene = pyrender.Scene(ambient_light=(1.0, 1.0, 1.0))
                    scene.add(mesh_pyrender)
                    
                    # Camera
                    focal, princpt = cam_param['focal'], cam_param['princpt']
                    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
                    scene.add(camera)
                    
                    # Render with error handling
                    renderer = None
                    try:
                        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
                        rgb, depth = renderer.render(scene)
                        
                        # Convert to silhouette (white where mesh exists, black elsewhere)
                        mask = (depth > 0)
                        silhouette[mask] = [255, 255, 255]
                        
                        print(f"      Successfully rendered with pyrender")
                    finally:
                        if renderer is not None:
                            try:
                                renderer.delete()
                            except Exception as e:
                                # Ignore EGL cleanup errors
                                if "eglDestroyContext" not in str(e):
                                    print(f"      Warning during renderer cleanup: {str(e)}")
                    return silhouette
                    
                except Exception as e:
                    print(f"      Pyrender failed: {str(e)}, falling back to 2D projection")
        
        # Fallback: Project to 2D and use improved shape reconstruction
        print(f"      Using 2D projection fallback method with proper human shape")
        vertices_2d = perspective_projection(vertices_3d, cam_param)
        
        # Filter vertices within image bounds
        valid_mask = (vertices_2d[:, 0] >= 0) & (vertices_2d[:, 0] < width) & \
                     (vertices_2d[:, 1] >= 0) & (vertices_2d[:, 1] < height) & \
                     (vertices_3d[:, 2] > 0)  # In front of camera
        
        if np.sum(valid_mask) >= 100:
            valid_vertices_2d = vertices_2d[valid_mask]
            
            # For human body, we should create a more human-like silhouette
            # Use alpha shape or concave hull for better human shape
            try:
                from shapely.geometry import MultiPoint
                from shapely.ops import unary_union
                
                # Create alpha shape (concave hull) for human-like shape
                points = MultiPoint(valid_vertices_2d)
                alpha_shape = points.convex_hull
                
                # Convert to polygon points
                if hasattr(alpha_shape, 'exterior'):
                    hull_points = np.array(alpha_shape.exterior.coords[:-1])
                    cv2.fillPoly(silhouette, [hull_points.astype(np.int32)], (255, 255, 255))
            except:
                # Fallback to convex hull
                hull = ConvexHull(valid_vertices_2d)
                hull_points = valid_vertices_2d[hull.vertices]
                cv2.fillPoly(silhouette, [hull_points.astype(np.int32)], (255, 255, 255))
            
            # Add all vertex points to create denser human shape
            for x, y in valid_vertices_2d:
                cv2.circle(silhouette, (int(x), int(y)), 2, (255, 255, 255), -1)
            
        else:
            print(f"      Not enough valid vertices ({np.sum(valid_mask)}) for shape reconstruction")
            
    except Exception as e:
        print(f"      3D silhouette rendering failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return silhouette


def render_3d_mesh_silhouette_from_2d(vertices_2d, img_shape):
    """Render mesh silhouette from 2D vertex locations"""
    height, width = img_shape[:2]
    silhouette = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(vertices_2d) < 3:
        return silhouette
    
    try:
        # Create convex hull from 2D points
        hull = ConvexHull(vertices_2d)
        hull_points = vertices_2d[hull.vertices]
        
        # Fill the convex hull with white
        cv2.fillPoly(silhouette, [hull_points.astype(np.int32)], (255, 255, 255))
        
        # Optional: Add vertex points for density
        for x, y in vertices_2d[::10]:  # Sample every 10th point
            cv2.circle(silhouette, (int(x), int(y)), 1, (255, 255, 255), -1)
            
    except Exception as e:
        print(f"      Convex hull failed: {str(e)}, using simple points")
        # Fallback: draw all points
        for x, y in vertices_2d:
            cv2.circle(silhouette, (int(x), int(y)), 2, (255, 255, 255), -1)
    
    return silhouette


def render_mesh_without_config(img, mesh, face, cam_param, mesh_as_vertices=False):
    """Render mesh without depending on config module"""
    
    def perspective_projection_local(vertices, cam_param):
        # vertices: [N, 3]
        # cam_param: [3]
        vertices = vertices.copy()
        fx, fy= cam_param['focal']
        cx, cy = cam_param['princpt']
        vertices[:, 0] = vertices[:, 0] * fx / vertices[:, 2] + cx
        vertices[:, 1] = vertices[:, 1] * fy / vertices[:, 2] + cy
        return vertices
    
    def vis_keypoints_local(img, kps, alpha=1, radius=3, color=None):
        """Draw keypoints on image"""
        kp_mask = np.copy(img)
        
        if color is None:
            color = (0, 0, 255)  # Red color for vertices
        
        # Draw the keypoints
        for i in range(len(kps)):
            p = int(kps[i][0]), int(kps[i][1])
            if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]:
                cv2.circle(kp_mask, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)
        
        # Blend the keypoints
        return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
    
    if mesh_as_vertices:
        # to run on cluster where headless pyrender is not supported for A100/V100
        vertices_2d = perspective_projection_local(mesh, cam_param)
        img = vis_keypoints_local(img, vertices_2d, alpha=0.8, radius=2, color=(0, 0, 255))
    else:
        # mesh
        mesh_obj = trimesh.Trimesh(mesh, face)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh_obj.apply_transform(rot)
        
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2, 
            roughnessFactor=0.8,
            alphaMode='OPAQUE', 
            baseColorFactor=(1.0, 1.0, 0.9, 1.0)
        )
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_obj, material=material, smooth=True)
        scene = pyrender.Scene(ambient_light=(0.4, 0.4, 0.4))
        scene.add(mesh_pyrender, 'mesh')

        focal, princpt = cam_param['focal'], cam_param['princpt']
        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
        scene.add(camera)

        # renderer with error handling
        renderer = None
        try:
            renderer = pyrender.OffscreenRenderer(
                viewport_width=img.shape[1], 
                viewport_height=img.shape[0], 
                point_size=1.0
            )

            # Enhanced lighting for smooth appearance
            # Main light (front)
            light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
            light_pose1 = np.eye(4)
            light_pose1[:3, 3] = np.array([0, -1, 1])
            scene.add(light1, pose=light_pose1)
            
            # Fill light (side)
            light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.6)
            light_pose2 = np.eye(4)
            light_pose2[:3, 3] = np.array([1, 0, 1])
            scene.add(light2, pose=light_pose2)
            
            # Back light (rim lighting)
            light3 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.4)
            light_pose3 = np.eye(4)
            light_pose3[:3, 3] = np.array([0, 1, -1])
            scene.add(light3, pose=light_pose3)

            # render
            print(f"      [DEBUG] Starting pyrender.render()...")
            rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            print(f"      [DEBUG] Pyrender completed, processing results...")
            rgb = rgb[:,:,:3].astype(np.float32)
            valid_mask = (depth > 0)[:,:,None]

            # save to image
            img = rgb * valid_mask + img * (1-valid_mask)
            print(f"      [DEBUG] Image blending completed")
        finally:
            # Safe cleanup with error handling
            if renderer is not None:
                try:
                    renderer.delete()
                except Exception as e:
                    # Ignore EGL cleanup errors which are not critical
                    if "eglDestroyContext" not in str(e):
                        print(f"      Warning during renderer cleanup: {str(e)}")

    return img


def apply_position_correction(standard_img, vertices_2d, background_img, cam_param):
    """Apply position correction using bounding box alignment instead of center"""
    
    height, width = background_img.shape[:2]
    result_img = background_img.copy()
    
    # Find non-black pixels in the rendered image (the mesh)
    mesh_mask = np.any(standard_img > 10, axis=2)
    
    if not np.any(mesh_mask):
        print(f"        No mesh pixels found in standard rendering")
        return result_img
    
    # Calculate bounding box of the rendered mesh
    mesh_pixels = np.where(mesh_mask)
    mesh_y_min = np.min(mesh_pixels[0])
    mesh_y_max = np.max(mesh_pixels[0])
    mesh_x_min = np.min(mesh_pixels[1])
    mesh_x_max = np.max(mesh_pixels[1])
    mesh_center_x = (mesh_x_min + mesh_x_max) / 2
    mesh_center_y = (mesh_y_min + mesh_y_max) / 2
    
    # Calculate target bounding box based on stored 2D coordinates
    if vertices_2d is not None and len(vertices_2d) > 0:
        vertices_2d_array = np.array(vertices_2d)
        target_x_min = np.min(vertices_2d_array[:, 0])
        target_x_max = np.max(vertices_2d_array[:, 0])
        target_y_min = np.min(vertices_2d_array[:, 1])
        target_y_max = np.max(vertices_2d_array[:, 1])
        target_center_x = (target_x_min + target_x_max) / 2
        target_center_y = (target_y_min + target_y_max) / 2
    else:
        # Fallback to image center if no 2D data
        target_center_x = width / 2
        target_center_y = height / 2
    
    # Calculate translation offset based on bounding box centers
    offset_x = target_center_x - mesh_center_x
    offset_y = target_center_y - mesh_center_y
    
    print(f"        Mesh bbox: X({mesh_x_min:.1f}-{mesh_x_max:.1f}), Y({mesh_y_min:.1f}-{mesh_y_max:.1f})")
    print(f"        Target bbox: X({target_x_min:.1f}-{target_x_max:.1f}), Y({target_y_min:.1f}-{target_y_max:.1f})")
    print(f"        Mesh center: ({mesh_center_x:.1f}, {mesh_center_y:.1f})")
    print(f"        Target center: ({target_center_x:.1f}, {target_center_y:.1f})")
    print(f"        Translation offset: ({offset_x:.1f}, {offset_y:.1f})")
    
    # Apply translation using cv2.warpAffine
    import cv2
    translation_matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    translated_mesh = cv2.warpAffine(standard_img, translation_matrix, (width, height), 
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # Create mask for the translated mesh
    translated_mask = np.any(translated_mesh > 10, axis=2)
    
    # Blend the translated mesh onto the background
    result_img[translated_mask] = translated_mesh[translated_mask].astype(np.uint8)
    
    mesh_pixels_after = np.sum(translated_mask)
    print(f"        Applied translation: {mesh_pixels_after} mesh pixels positioned")
    
    return result_img


def render_3d_mesh_using_original_method(vertices_3d, vertices_2d, vertex_indices, person_data, background_img):
    """Render 3D mesh using exact projection matching"""
    
    print(f"      Rendering 3D mesh with exact projection matching using {len(vertices_3d)} vertices")
    
    height, width = background_img.shape[:2]
    
    # Calculate exact camera parameters from vertex correspondences
    if vertices_2d is not None and len(vertices_2d) > 0:
        vertices_2d_array = np.array(vertices_2d)
        
        # Calculate 2D center and size
        x_center_2d = (np.min(vertices_2d_array[:, 0]) + np.max(vertices_2d_array[:, 0])) / 2
        y_center_2d = (np.min(vertices_2d_array[:, 1]) + np.max(vertices_2d_array[:, 1])) / 2
        bbox_2d_size = max(
            np.max(vertices_2d_array[:, 0]) - np.min(vertices_2d_array[:, 0]),
            np.max(vertices_2d_array[:, 1]) - np.min(vertices_2d_array[:, 1])
        )
        
        # Calculate 3D size
        bbox_3d_size = max(
            np.max(vertices_3d[:, 0]) - np.min(vertices_3d[:, 0]),
            np.max(vertices_3d[:, 1]) - np.min(vertices_3d[:, 1])
        )
        
        # Calculate optimal focal length and principal point
        # Use the 2D center as principal point for direct alignment
        standard_focal = 1500.0  # Base focal length
        princpt = [x_center_2d, y_center_2d]  # Use actual 2D center
        cam_param = {'focal': [standard_focal, standard_focal], 'princpt': princpt}
        
        print(f"      Using projection-matched camera params: focal={standard_focal}, princpt=({princpt[0]:.1f}, {princpt[1]:.1f})")
    else:
        # Fallback to standard parameters
        standard_focal = 1500.0
        standard_princpt = [width/2, height/2]
        cam_param = {'focal': [standard_focal, standard_focal], 'princpt': standard_princpt}
        print(f"      Using standard camera params: focal={standard_focal}, princpt={standard_princpt}")
    
    # Step 2: Calculate optimal Z depth to match original 2D size
    vertices_3d_adjusted = vertices_3d.copy()
    
    # Calculate current Z range
    z_min, z_max = np.min(vertices_3d[:, 2]), np.max(vertices_3d[:, 2])
    z_center = np.mean(vertices_3d[:, 2])
    
    # Calculate 3D bounding box size
    bbox_3d_width = np.max(vertices_3d[:, 0]) - np.min(vertices_3d[:, 0])
    bbox_3d_height = np.max(vertices_3d[:, 1]) - np.min(vertices_3d[:, 1])
    
    # Calculate 2D bounding box size from stored coordinates
    if vertices_2d is not None and len(vertices_2d) > 0:
        vertices_2d_array = np.array(vertices_2d)
        bbox_2d_width = np.max(vertices_2d_array[:, 0]) - np.min(vertices_2d_array[:, 0])
        bbox_2d_height = np.max(vertices_2d_array[:, 1]) - np.min(vertices_2d_array[:, 1])
        
        # Use the larger dimension for scale calculation
        bbox_2d_size = max(bbox_2d_width, bbox_2d_height)
        bbox_3d_size = max(bbox_3d_width, bbox_3d_height)
        
        # Calculate optimal Z distance: Z = focal * 3D_size / 2D_size
        # We want: 2D_size = focal * 3D_size / Z
        # So: Z = focal * 3D_size / 2D_size
        optimal_z = standard_focal * bbox_3d_size / bbox_2d_size
        
        print(f"      2D bbox size: {bbox_2d_size:.1f}px, 3D bbox size: {bbox_3d_size:.3f}m")
        print(f"      Calculated optimal Z distance: {optimal_z:.1f}m")
        
        # Apply reasonable bounds (2-20 meters)
        target_z = max(2.0, min(20.0, optimal_z))
    else:
        # Fallback: use reasonable default
        target_z = 8.0
        print(f"      No 2D data available, using default Z distance: {target_z:.1f}m")
    
    z_offset = target_z - z_center
    vertices_3d_adjusted[:, 2] += z_offset
    
    print(f"      Adjusted Z depth from {z_min:.1f}-{z_max:.1f} to {target_z:.1f} (offset: {z_offset:.1f})")
    
    try:
        # Get SMPL-X faces and filter for available vertices
        if RENDERING_AVAILABLE and vertex_indices is not None:
            smplx_faces = load_smplx_faces()
            if smplx_faces is not None:
                # Try with relaxed filtering first (allow 2-vertex faces)
                filtered_faces = filter_faces_for_vertices(smplx_faces, vertex_indices, len(vertices_3d), min_vertices_per_face=2)
                
                # Always add Delaunay triangulation for better coverage
                if True:  # Always use Delaunay for dense mesh
                    print(f"      Insufficient SMPL-X faces ({len(filtered_faces) if filtered_faces is not None else 0}), adding Delaunay triangulation")
                    delaunay_faces = create_dense_faces_from_vertices(vertices_3d, vertex_indices)
                    
                    if delaunay_faces is not None:
                        if filtered_faces is not None:
                            # Combine SMPL-X faces with Delaunay faces
                            all_faces = np.vstack([filtered_faces, delaunay_faces])
                        else:
                            all_faces = delaunay_faces
                        filtered_faces = all_faces
                        print(f"      Total faces after combining: {len(filtered_faces)}")
                
                if filtered_faces is not None and len(filtered_faces) > 50:
                    print(f"      Using {len(filtered_faces)} filtered faces for mesh rendering")
                    
                    # Render directly at correct position (no translation needed)
                    result_img = render_mesh_without_config(background_img.astype(np.float32), 
                                                           vertices_3d_adjusted, 
                                                           filtered_faces, 
                                                           cam_param, 
                                                           mesh_as_vertices=False)
                    
                    print(f"      Successfully rendered with position correction using {len(filtered_faces)} faces")
                    print(f"      [DEBUG] Converting result image and returning...")
                    
                    # Force immediate cleanup to prevent hanging
                    del filtered_faces, vertices_3d_adjusted
                    import gc
                    gc.collect()
                    
                    return result_img.astype(np.uint8)
        
        # Fallback: Use vertex-based rendering
        print(f"      Using vertex-based fallback rendering")
        
        # Render directly at correct position
        result_img = render_mesh_without_config(background_img.astype(np.float32), 
                                                vertices_3d_adjusted, 
                                                None,  # No faces for vertex rendering
                                                cam_param, 
                                                mesh_as_vertices=True)
        
        return result_img.astype(np.uint8)
        
    except Exception as e:
        print(f"      Original method rendering failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return background_img


def render_3d_mesh_using_stored_2d_positions(vertices_3d, vertices_2d, vertex_indices, img_shape):
    """Render 3D mesh at correct spatial position based on stored 2D locations"""
    height, width = img_shape[:2]
    silhouette = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(vertices_3d) != len(vertices_2d):
        print(f"      Mismatch: {len(vertices_3d)} 3D vertices vs {len(vertices_2d)} 2D positions")
        return silhouette
    
    print(f"      Rendering 3D mesh with {len(vertices_3d)} vertices at correct 2D position")
    
    try:
        # Calculate 2D center of mass to determine where the person is located
        vertices_2d_array = np.array(vertices_2d)
        center_2d = np.mean(vertices_2d_array, axis=0)
        print(f"      Person center at 2D position: ({center_2d[0]:.1f}, {center_2d[1]:.1f})")
        
        # Fallback: Enhanced point cloud using stored 2D positions
        print(f"      Using enhanced point cloud with stored 2D positions")
        
        # Use 3D depth information for enhanced visualization
        z_values = vertices_3d[:, 2]
        if len(z_values) > 0 and np.max(z_values) > np.min(z_values):
            z_normalized = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
            
            # Filter vertices within image bounds
            valid_mask = (vertices_2d[:, 0] >= 0) & (vertices_2d[:, 0] < width) & \
                        (vertices_2d[:, 1] >= 0) & (vertices_2d[:, 1] < height)
            
            if np.sum(valid_mask) >= 10:
                valid_vertices_2d = vertices_2d[valid_mask]
                valid_z_norm = z_normalized[valid_mask]
                
                # Draw vertices with depth-based visualization
                for pos_2d, z_norm in zip(valid_vertices_2d, valid_z_norm):
                    x_2d, y_2d = int(pos_2d[0]), int(pos_2d[1])
                    
                    # Closer points (smaller Z) are larger and brighter
                    intensity = int(255 * (1 - z_norm * 0.3))  # 255 to 178
                    radius = max(1, int(3 * (1 - z_norm * 0.4)))  # 3 to 1.8
                    
                    cv2.circle(silhouette, (x_2d, y_2d), radius, 
                             (intensity, intensity, intensity), -1)
        
    except Exception as e:
        print(f"      3D rendering with stored positions failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return silhouette


def render_3d_mesh_overlay_multi_person_original(smplerx_data, background_img, cam_param):
    """Render 3D mesh overlay for multiple persons using original SMPLer-X method"""
    
    result_img = background_img.copy()
    print(f"    Rendering 3D mesh overlay for {len(smplerx_data)} persons using original method...")
    
    # Colors for different persons
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for person_idx, (person_id, person_data) in enumerate(smplerx_data.items()):
        color = colors[person_idx % len(colors)]
        
        if 'vertices' not in person_data or person_data['vertices'] is None or len(person_data['vertices']) == 0:
            print(f"    {person_id}: No 3D vertices, skipping")
            continue
            
        vertices_3d = person_data['vertices']
        vertices_2d = person_data.get('vertices_2d', None)
        vertex_indices = person_data.get('vertex_indices', None)
        
        print(f"    {person_id}: Rendering {len(vertices_3d)} vertices...")
        
        # Use original method for rendering
        print(f"    {person_id}: Starting render_3d_mesh_using_original_method...")
        result_img = render_3d_mesh_using_original_method(
            vertices_3d, vertices_2d, vertex_indices, person_data, result_img
        )
        print(f"    {person_id}: Completed render_3d_mesh_using_original_method")
        
        # Add detection box visualization
        # Draw MMDet bbox (information storage area)
        if 'bbox_mmdet' in person_data:
            bbox_mm = person_data['bbox_mmdet']
            x1_mm, y1_mm, w_mm, h_mm = int(bbox_mm[0]), int(bbox_mm[1]), int(bbox_mm[2]), int(bbox_mm[3])
            
            # Draw solid rectangle for mmdet bbox (actual storage area)
            cv2.rectangle(result_img, (x1_mm, y1_mm), (x1_mm + w_mm, y1_mm + h_mm), color, 2)
            
            # Left top corner: Person ID
            cv2.putText(result_img, person_id, (x1_mm, y1_mm - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Right top corner: Information storage label
            cv2.putText(result_img, "Info Storage", (x1_mm + w_mm - 90, y1_mm - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    
    print(f"    [DEBUG] Multi-person overlay rendering completed, returning result")
    return result_img


def render_3d_mesh_silhouette_multi_person(smplerx_data, img_shape, cam_param):
    """Render 3D mesh silhouette for multiple persons using stored 2D positions"""
    height, width = img_shape[:2]
    final_silhouette = np.zeros((height, width, 3), dtype=np.uint8)
    
    print(f"    Rendering 3D mesh silhouette for {len(smplerx_data)} persons...")
    
    for person_id, person_data in smplerx_data.items():
        if 'vertices' not in person_data or person_data['vertices'] is None or len(person_data['vertices']) == 0:
            print(f"    {person_id}: No 3D vertices, skipping")
            continue
            
        vertices_3d = person_data['vertices']
        vertices_2d = person_data.get('vertices_2d', None)
        vertex_indices = person_data.get('vertex_indices', None)
        
        print(f"    {person_id}: Rendering {len(vertices_3d)} vertices...")
        
        # Use original method for better quality rendering
        background_for_silhouette = np.zeros((height, width, 3), dtype=np.uint8)
        person_result = render_3d_mesh_using_original_method(
            vertices_3d, vertices_2d, vertex_indices, person_data, background_for_silhouette
        )
        
        # Keep the original 3D rendering with lighting and shading
        # Only extract non-black pixels to preserve 3D appearance
        mask = np.any(person_result > 10, axis=2)
        
        # Debug: Show person count
        person_pixels = np.sum(mask)
        print(f"    {person_id}: Generated 3D silhouette with {person_pixels} pixels")
        
        # Combine with final silhouette, preserving the 3D shading
        # Use additive blending to ensure both people are visible
        # Only add non-black pixels to avoid overwriting
        mask = np.any(person_result > 10, axis=2)
        final_silhouette[mask] = person_result[mask]
        
        # Debug: Check if person was actually rendered
        if person_pixels > 0:
            print(f"    {person_id}: ✅ Successfully added to final silhouette")
        else:
            print(f"    {person_id}: ❌ No pixels generated for final silhouette")
    
    return final_silhouette


def render_3d_mesh_silhouette(vertices, faces, img_shape, cam_param):
    """Render 3D mesh to silhouette using pyrender"""
    # Always try pyrender first, but fallback gracefully
    print(f"      Attempting 3D mesh silhouette rendering with {len(vertices)} vertices...")
    
    try:
        height, width = img_shape[:2]
        
        # Create triangular mesh from convex hull if no faces provided
        if faces is None:
            if len(vertices) < 4:
                return np.zeros((height, width, 3), dtype=np.uint8)
            
            # Use 2D convex hull for face generation (simple triangulation)
            try:
                hull = ConvexHull(vertices[:, :2])  # Use X,Y for 2D hull
                # Simple triangulation: connect all hull points to centroid
                centroid = np.mean(vertices, axis=0)
                vertices_with_center = np.vstack([vertices, centroid])
                
                faces = []
                for i in range(len(hull.vertices)):
                    j = (i + 1) % len(hull.vertices)
                    # Triangle: hull[i] -> hull[j] -> centroid
                    faces.append([hull.vertices[i], hull.vertices[j], len(vertices)])
                
                faces = np.array(faces)
            except:
                # Fallback: create simple triangulation
                if len(vertices) >= 3:
                    faces = [[0, 1, 2]]
                    for i in range(3, len(vertices)):
                        faces.append([0, i-1, i])
                    faces = np.array(faces)
                else:
                    return np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices_with_center if 'vertices_with_center' in locals() else vertices, faces)
        
        # Apply rotation (180 degrees around X-axis)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        
        # Create material for white silhouette
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, 
            alphaMode='OPAQUE', 
            baseColorFactor=(1.0, 1.0, 1.0, 1.0)  # White color
        )
        
        # Create pyrender mesh
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        
        # Create scene
        scene = pyrender.Scene(ambient_light=(0.8, 0.8, 0.8), bg_color=(0, 0, 0))  # Black background
        scene.add(mesh_pyrender, 'mesh')
        
        # Camera setup
        focal, princpt = cam_param['focal'], cam_param['princpt']
        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
        scene.add(camera)
        
        # Lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, 0, 1])
        scene.add(light, pose=light_pose)
        
        # Render (with headless support)
        renderer = None
        try:
            try:
                renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
            except:
                # Try headless mode
                os.environ['PYOPENGL_PLATFORM'] = 'egl'
                renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
            
            rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            
            # Convert to silhouette (white on black)
            valid_mask = depth > 0
            silhouette = np.zeros((height, width, 3), dtype=np.uint8)
            silhouette[valid_mask] = [255, 255, 255]
            
            return silhouette
        finally:
            if renderer is not None:
                try:
                    renderer.delete()
                except Exception as e:
                    # Ignore EGL cleanup errors
                    if "eglDestroyContext" not in str(e):
                        print(f"      Warning during renderer cleanup: {str(e)}")
        
    except Exception as e:
        print(f"    3D mesh rendering failed: {str(e)}")
        print(f"    Falling back to enhanced convex hull silhouette...")
        return render_enhanced_mesh_silhouette(vertices, img_shape, cam_param)


def render_3d_mesh_overlay_from_2d(vertices_2d, background_img, alpha=0.7):
    """Render mesh overlay from 2D vertex locations"""
    height, width = background_img.shape[:2]
    overlay = background_img.copy()
    
    if len(vertices_2d) < 3:
        return overlay
    
    try:
        # Create convex hull from 2D points
        hull = ConvexHull(vertices_2d)
        hull_points = vertices_2d[hull.vertices]
        
        # Create a mask for the mesh area
        mesh_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mesh_mask, [hull_points.astype(np.int32)], 255)
        
        # Apply semi-transparent white overlay in mesh area
        overlay_color = np.full((height, width, 3), 255, dtype=np.uint8)
        mask_3d = np.stack([mesh_mask] * 3, axis=2) / 255.0
        
        overlay = overlay.astype(np.float32)
        overlay = overlay * (1 - mask_3d * alpha) + overlay_color * mask_3d * alpha
        overlay = overlay.astype(np.uint8)
        
        # Draw hull outline
        cv2.polylines(overlay, [hull_points.astype(np.int32)], True, (0, 255, 0), 2)
        
        # Draw vertex points (sample to avoid clutter)
        for x, y in vertices_2d[::20]:  # Sample every 20th point
            cv2.circle(overlay, (int(x), int(y)), 1, (255, 255, 0), -1)
            
    except Exception as e:
        print(f"      Overlay rendering failed: {str(e)}, using simple points")
        # Fallback: draw all points
        for x, y in vertices_2d[::10]:
            cv2.circle(overlay, (int(x), int(y)), 2, (255, 255, 255), -1)
    
    return overlay


def render_3d_mesh_overlay_multi_person(smplerx_data, background_img, cam_param, alpha=0.7):
    """Render 3D mesh overlay for multiple persons using stored 2D positions"""
    print(f"    Rendering 3D mesh overlay for {len(smplerx_data)} persons...")
    
    result_overlay = background_img.copy()
    
    for person_id, person_data in smplerx_data.items():
        if 'vertices' not in person_data or person_data['vertices'] is None or len(person_data['vertices']) == 0:
            print(f"    {person_id}: No 3D vertices, skipping")
            continue
            
        vertices_3d = person_data['vertices']
        vertices_2d = person_data.get('vertices_2d', None)
        vertex_indices = person_data.get('vertex_indices', None)
        
        print(f"    {person_id}: Rendering overlay with {len(vertices_3d)} vertices...")
        
        # Use stored 2D positions for correct spatial placement
        if vertices_2d is not None and len(vertices_2d) == len(vertices_3d):
            person_overlay = render_human_mesh_overlay_with_stored_positions(
                vertices_3d, vertices_2d, vertex_indices, result_overlay, alpha
            )
        else:
            # Fallback: project 3D vertices to 2D
            vertices_2d_proj = perspective_projection(vertices_3d, cam_param)
            person_overlay = render_human_mesh_overlay(
                vertices_2d_proj, vertices_3d, vertex_indices, result_overlay, cam_param, alpha
            )
        
        result_overlay = person_overlay
    
    return result_overlay


def render_3d_mesh_overlay(vertices, faces, background_img, cam_param, alpha=0.7):
    """Render 3D mesh overlaid on background image"""
    print(f"      Attempting 3D mesh overlay rendering with {len(vertices)} vertices...")
    
    # Always try enhanced fallback for more reliable results
    return render_enhanced_mesh_overlay(vertices, background_img, cam_param, alpha)
    
    try:
        height, width = background_img.shape[:2]
        
        # Create mesh (same as silhouette function)
        if faces is None:
            if len(vertices) < 4:
                return background_img
            
            try:
                hull = ConvexHull(vertices[:, :2])
                centroid = np.mean(vertices, axis=0)
                vertices_with_center = np.vstack([vertices, centroid])
                
                faces = []
                for i in range(len(hull.vertices)):
                    j = (i + 1) % len(hull.vertices)
                    faces.append([hull.vertices[i], hull.vertices[j], len(vertices)])
                faces = np.array(faces)
            except:
                if len(vertices) >= 3:
                    faces = [[0, 1, 2]]
                    for i in range(3, len(vertices)):
                        faces.append([0, i-1, i])
                    faces = np.array(faces)
                else:
                    return background_img
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices_with_center if 'vertices_with_center' in locals() else vertices, faces)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        
        # Semi-transparent white material
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, 
            alphaMode='BLEND', 
            baseColorFactor=(1.0, 1.0, 1.0, 0.8)  # Semi-transparent white
        )
        
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        
        # Scene setup
        scene = pyrender.Scene(ambient_light=(0.4, 0.4, 0.4))
        scene.add(mesh_pyrender, 'mesh')
        
        # Camera
        focal, princpt = cam_param['focal'], cam_param['princpt']
        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
        scene.add(camera)
        
        # Lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        
        # Render (with headless support)
        renderer = None
        try:
            try:
                renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
            except:
                # Try headless mode
                os.environ['PYOPENGL_PLATFORM'] = 'egl'
                renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
            
            rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            
            # Blend with background
            rgb = rgb[:,:,:3].astype(np.float32)
            valid_mask = (depth > 0)[:,:,None]
            
            # Convert background to BGR for consistency
            if len(background_img.shape) == 3 and background_img.shape[2] == 3:
                # Already BGR
                background_bgr = background_img.astype(np.float32)
            else:
                background_bgr = cv2.cvtColor(background_img, cv2.COLOR_RGB2BGR).astype(np.float32)
            
            # Blend
            result = rgb * valid_mask * alpha + background_bgr * (1 - valid_mask * alpha)
            
            return result.astype(np.uint8)
        finally:
            if renderer is not None:
                try:
                    renderer.delete()
                except Exception as e:
                    # Ignore EGL cleanup errors
                    if "eglDestroyContext" not in str(e):
                        print(f"      Warning during renderer cleanup: {str(e)}")
        
    except Exception as e:
        print(f"3D overlay rendering failed: {str(e)}")
        return background_img


def render_enhanced_mesh_silhouette(vertices, img_shape, cam_param):
    """Enhanced fallback mesh silhouette rendering"""
    height, width = img_shape[:2]
    silhouette = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(vertices) < 3:
        return silhouette
    
    # Project vertices to 2D with proper camera parameters
    # Use the actual focal length and principal point from SMPLer-X
    # Adjust for the large Z values in SMPLer-X (around 42-43)
    focal, princpt = cam_param['focal'], cam_param['princpt']
    
    # For SMPLer-X data, we need to handle the camera coordinate system properly
    # The vertices are in camera coordinates with large Z values
    points_2d = []
    
    for vertex in vertices:
        if vertex[2] > 0:
            # Use the actual camera projection formula
            # Note: SMPLer-X vertices are already in camera coordinates
            px = int(focal[0] * vertex[0] / vertex[2] + princpt[0])
            py = int(focal[1] * vertex[1] / vertex[2] + princpt[1])
            if 0 <= px < width and 0 <= py < height:
                points_2d.append([px, py])
    
    if len(points_2d) >= 3:
        points_2d = np.array(points_2d)
        
        try:
            # Create multiple layers of convex hulls for better mesh-like appearance
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]
            
            # Fill the main convex hull
            cv2.fillPoly(silhouette, [hull_points.astype(np.int32)], (255, 255, 255))
            
            # Add density visualization - draw more points in dense areas
            # Create a density map
            for px, py in points_2d:
                # Draw small circles for vertex density
                cv2.circle(silhouette, (px, py), 1, (255, 255, 255), -1)
            
            # Skip the expensive mesh line drawing for performance
            # The filled convex hull and vertex points are sufficient for silhouette
                            
        except Exception as e:
            print(f"    Enhanced rendering failed: {str(e)}, using simple points")
            # Simple fallback: draw all points
            for px, py in points_2d:
                cv2.circle(silhouette, (px, py), 2, (255, 255, 255), -1)
    
    return silhouette


def render_convex_hull_silhouette(vertices, img_shape, cam_param):
    """Simple fallback silhouette rendering using convex hull"""
    height, width = img_shape[:2]
    silhouette = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(vertices) < 3:
        return silhouette
    
    # Project vertices to 2D
    focal, princpt = cam_param['focal'], cam_param['princpt']
    points_2d = []
    
    for vertex in vertices:
        if vertex[2] > 0:
            px = int(focal[0] * vertex[0] / vertex[2] + princpt[0])
            py = int(focal[1] * vertex[1] / vertex[2] + princpt[1])
            if 0 <= px < width and 0 <= py < height:
                points_2d.append([px, py])
    
    if len(points_2d) >= 3:
        points_2d = np.array(points_2d)
        try:
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]
            cv2.fillPoly(silhouette, [hull_points.astype(np.int32)], (255, 255, 255))
        except:
            # Fallback: draw all points
            for px, py in points_2d:
                cv2.circle(silhouette, (px, py), 2, (255, 255, 255), -1)
    
    return silhouette


def render_enhanced_mesh_overlay(vertices, background_img, cam_param, alpha=0.7):
    """Enhanced mesh overlay rendering"""
    overlay = background_img.copy()
    height, width = background_img.shape[:2]
    
    if len(vertices) < 3:
        return overlay
    
    # Project vertices to 2D
    focal, princpt = cam_param['focal'], cam_param['princpt']
    points_2d = []
    
    for vertex in vertices:
        if vertex[2] > 0:
            px = int(focal[0] * vertex[0] / vertex[2] + princpt[0])
            py = int(focal[1] * vertex[1] / vertex[2] + princpt[1])
            if 0 <= px < width and 0 <= py < height:
                points_2d.append([px, py])
    
    if len(points_2d) >= 3:
        points_2d = np.array(points_2d)
        
        try:
            # Create mesh overlay
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]
            
            # Create a mask for the mesh area
            mesh_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mesh_mask, [hull_points.astype(np.int32)], 255)
            
            # Apply semi-transparent white overlay in mesh area
            overlay_color = np.full((height, width, 3), 255, dtype=np.uint8)
            mask_3d = np.stack([mesh_mask] * 3, axis=2) / 255.0
            
            overlay = overlay.astype(np.float32)
            overlay = overlay * (1 - mask_3d * alpha) + overlay_color * mask_3d * alpha
            overlay = overlay.astype(np.uint8)
            
            # Skip the expensive mesh line drawing for performance
            # The convex hull and vertex points are sufficient
            
            # Draw hull outline
            cv2.polylines(overlay, [hull_points.astype(np.int32)], True, (0, 255, 0), 2)
            
            # Draw vertex points
            for px, py in points_2d[::10]:  # Sample every 10th point to avoid clutter
                cv2.circle(overlay, (px, py), 1, (255, 255, 0), -1)
                
        except Exception as e:
            print(f"    Enhanced overlay failed: {str(e)}, using simple points")
            # Simple fallback: draw all points
            for px, py in points_2d:
                cv2.circle(overlay, (px, py), 2, (255, 255, 255), -1)
    
    return overlay


def extract_smplerx_from_wsg(wsg_array):
    """
    Extract SMPLer-X data from WSG array
    Returns: dictionary with person data
    """
    if wsg_array is None:
        return None
    
    height, width = wsg_array.shape[:2]
    
    # Extract person regions and joint data
    person_channel = wsg_array[:, :, PERSON_ID_CHANNEL]
    joint_channel = wsg_array[:, :, JOINT_ID_CHANNEL]
    
    # Find all valid person IDs
    valid_mask = person_channel != -1
    if not np.any(valid_mask):
        return None
    
    unique_persons = np.unique(person_channel[valid_mask].astype(int))
    
    results = {}
    
    for person_id in unique_persons:
        print(f"  Extracting data for person_{person_id}")
        
        # Find all pixels for this person
        person_mask = person_channel == person_id
        person_pixels = np.where(person_mask)
        
        if len(person_pixels[0]) == 0:
            continue
        
        # Initialize parameter dictionary
        person_data = {
            'global_orient': np.zeros((1, 3)),
            'body_pose': np.zeros((21, 3)),
            'left_hand_pose': np.zeros((15, 3)),
            'right_hand_pose': np.zeros((15, 3)),
            'jaw_pose': np.zeros((1, 3)),
            'leye_pose': np.zeros((1, 3)),
            'reye_pose': np.zeros((1, 3)),
            'betas': np.zeros((1, 10)),
            'expression': np.zeros((1, 10)),
            'transl': np.zeros((1, 3))
        }
        
        # Extract joint parameters
        joint_data = {}
        for i in range(len(person_pixels[0])):
            y, x = person_pixels[0][i], person_pixels[1][i]
            joint_id = int(joint_channel[y, x])
            
            if joint_id >= 0 and joint_id in SMPLERX_JOINT_REVERSE_MAPPING:
                joint_x = wsg_array[y, x, JOINT_X_CHANNEL]
                joint_y = wsg_array[y, x, JOINT_Y_CHANNEL]
                joint_z = wsg_array[y, x, JOINT_Z_CHANNEL]
                
                if joint_id not in joint_data:
                    joint_data[joint_id] = (joint_x, joint_y, joint_z)
        
        # Fill joint parameters
        for joint_id, (x, y, z) in joint_data.items():
            joint_name = SMPLERX_JOINT_REVERSE_MAPPING[joint_id]
            
            if joint_name == 'global_orient':
                person_data['global_orient'][0] = [x, y, z]
            elif joint_name == 'body_pose':
                idx = joint_id - 1
                if 0 <= idx < 21:
                    person_data['body_pose'][idx] = [x, y, z]
            elif joint_name == 'left_hand_pose':
                idx = joint_id - 22
                if 0 <= idx < 15:
                    person_data['left_hand_pose'][idx] = [x, y, z]
            elif joint_name == 'right_hand_pose':
                idx = joint_id - 37
                if 0 <= idx < 15:
                    person_data['right_hand_pose'][idx] = [x, y, z]
            elif joint_name in ['jaw_pose', 'leye_pose', 'reye_pose', 'transl']:
                person_data[joint_name][0] = [x, y, z]
        
        # Extract betas and expression
        y_min, y_max = person_pixels[0].min(), person_pixels[0].max()
        x_min, x_max = person_pixels[1].min(), person_pixels[1].max()
        
        # Check first row for betas
        if y_min < height:
            for i in range(10):
                if x_min + i < width:
                    type_val = wsg_array[y_min, x_min + i, BETAS_EXPRESSION_TYPE_CHANNEL]
                    if type_val == 0:  # betas
                        person_data['betas'][0, i] = wsg_array[y_min, x_min + i, BETAS_EXPRESSION_CHANNELS[i]]
        
        # Check second row for expression
        if y_min + 1 < height:
            for i in range(10):
                if x_min + i < width:
                    type_val = wsg_array[y_min + 1, x_min + i, BETAS_EXPRESSION_TYPE_CHANNEL]
                    if type_val == 1:  # expression
                        person_data['expression'][0, i] = wsg_array[y_min + 1, x_min + i, BETAS_EXPRESSION_CHANNELS[i]]

        # Historical injections cleared person IDs on beta/expression pixels.
        # A single-person frame can recover these rows unambiguously.
        if len(unique_persons) == 1:
            for parameter_type, parameter_name in ((0, 'betas'), (1, 'expression')):
                parameter_mask = wsg_array[:, :, BETAS_EXPRESSION_TYPE_CHANNEL] == parameter_type
                if np.any(parameter_mask):
                    for i, channel in enumerate(BETAS_EXPRESSION_CHANNELS):
                        values = wsg_array[:, :, channel][parameter_mask]
                        values = values[values != -1]
                        if values.size:
                            person_data[parameter_name][0, i] = values[0]
        
        # Extract bbox information using person region extent
        # Since bbox corners are stored globally, we'll use the person's vertex region as bbox
        person_y_coords = person_pixels[0]
        person_x_coords = person_pixels[1]
        if len(person_y_coords) > 0:
            # Use person region extent as bounding box
            person_y_min, person_y_max = person_y_coords.min(), person_y_coords.max()
            person_x_min, person_x_max = person_x_coords.min(), person_x_coords.max()
            
            # Expand slightly to create proper bbox
            margin = 20
            bbox_x_min = max(0, person_x_min - margin)
            bbox_y_min = max(0, person_y_min - margin)  
            bbox_x_max = min(width - 1, person_x_max + margin)
            bbox_y_max = min(height - 1, person_y_max + margin)
            
            person_data['bbox'] = [bbox_x_min, bbox_y_min, 
                                  bbox_x_max - bbox_x_min + 1, 
                                  bbox_y_max - bbox_y_min + 1]
            print(f"    [DEBUG] Generated bbox for {person_id}: {person_data['bbox']}")
        
        # Extract bbox_mmdet information using tighter person region
        if len(person_y_coords) > 0:
            # Use person region extent as mmdet bbox (tighter than main bbox)
            margin_mm = 5  # Smaller margin for mmdet bbox
            bbox_mm_x_min = max(0, person_x_min - margin_mm)
            bbox_mm_y_min = max(0, person_y_min - margin_mm)
            bbox_mm_x_max = min(width - 1, person_x_max + margin_mm)
            bbox_mm_y_max = min(height - 1, person_y_max + margin_mm)
            
            person_data['bbox_mmdet'] = [bbox_mm_x_min, bbox_mm_y_min,
                                        bbox_mm_x_max - bbox_mm_x_min + 1,
                                        bbox_mm_y_max - bbox_mm_y_min + 1]
            print(f"    [DEBUG] Generated bbox_mmdet for {person_id}: {person_data['bbox_mmdet']}")
        
        # Extract vertex data with indices (filter by person_id)
        vertex_mask = (wsg_array[:, :, VERTEX_INDEX_CHANNEL] != -1) & \
                     (wsg_array[:, :, VERTEX_X_CHANNEL] != -1) & \
                     (wsg_array[:, :, VERTEX_Y_CHANNEL] != -1) & \
                     (wsg_array[:, :, VERTEX_Z_CHANNEL] != -1) & \
                     person_mask
        
        vertex_pixels = np.where(vertex_mask)
        print(f"    [DEBUG] Found {len(vertex_pixels[0])} pixels with vertex data for {person_id}")
        
        if len(vertex_pixels[0]) > 0:
            # 使用字典存储有序顶点，以便重建完整的顶点数组
            vertex_dict = {}
            vertex_2d_dict = {}  # Store 2D positions by vertex index
            
            # Track vertex indices to check for duplicates
            vertex_indices_found = []
            
            for i in range(len(vertex_pixels[0])):
                y, x = vertex_pixels[0][i], vertex_pixels[1][i]
                vertex_idx = int(wsg_array[y, x, VERTEX_INDEX_CHANNEL])
                vx = wsg_array[y, x, VERTEX_X_CHANNEL]
                vy = wsg_array[y, x, VERTEX_Y_CHANNEL]
                vz = wsg_array[y, x, VERTEX_Z_CHANNEL]
                
                vertex_indices_found.append(vertex_idx)
                vertex_dict[vertex_idx] = [vx, vy, vz]
                vertex_2d_dict[vertex_idx] = [x, y]  # Store 2D position by vertex index
            
            # Debug output to understand the difference
            unique_indices = len(set(vertex_indices_found))
            print(f"    [DEBUG] Pixels: {len(vertex_pixels[0])}, Unique vertices: {unique_indices}")
            if len(vertex_indices_found) != unique_indices:
                print(f"    [DEBUG] Some vertex indices appear multiple times in pixels!")
            
            # 重建有序的顶点数组，确保3D和2D顶点对应
            max_idx = max(vertex_dict.keys()) if vertex_dict else 0
            ordered_vertices = []
            ordered_vertices_2d = []
            valid_indices = []
            
            for idx in range(max_idx + 1):
                if idx in vertex_dict:
                    ordered_vertices.append(vertex_dict[idx])
                    ordered_vertices_2d.append(vertex_2d_dict[idx])
                    valid_indices.append(idx)
                # 注意：我们不为缺失的顶点添加None，而是只保留有效的顶点
            
            person_data['vertices'] = np.array(ordered_vertices)
            person_data['vertices_2d'] = np.array(ordered_vertices_2d)
            person_data['vertex_indices'] = np.array(valid_indices)  # 保存有效的顶点索引
            
            print(f"    Extracted {len(ordered_vertices)} vertices for {person_id}")
            print(f"    Vertex index range: {min(valid_indices)} - {max(valid_indices)}")
        else:
            print(f"    No vertices found for {person_id}")
        
        results[f'person_{person_id}'] = person_data
    
    return results


def _get_complete_smplx_model():
    """Create the official SMPL-X neutral body model once per process."""
    global _COMPLETE_SMPLX_MODEL
    if _COMPLETE_SMPLX_MODEL is None:
        if not SMPLX_MODEL_ROOT:
            raise FileNotFoundError(
                "Set WSM_SMPLX_MODEL_ROOT to the directory containing SMPL-X assets"
            )
        import smplx

        _COMPLETE_SMPLX_MODEL = smplx.create(
            SMPLX_MODEL_ROOT,
            model_type='smplx',
            gender='neutral',
            ext='npz',
            num_betas=10,
            num_expression_coeffs=10,
            use_pca=False,
            use_face_contour=True,
            create_global_orient=False,
            create_body_pose=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            create_betas=False,
            create_expression=False,
            create_transl=False,
        ).cpu().eval()
    return _COMPLETE_SMPLX_MODEL


def _create_complete_smplx_mesh(person_data):
    """Regenerate all SMPL-X vertices/faces from the stored pose parameters."""
    import torch

    model = _get_complete_smplx_model()
    with torch.no_grad():
        output = model(
            betas=torch.from_numpy(np.asarray(person_data['betas'], dtype=np.float32)),
            body_pose=torch.from_numpy(np.asarray(person_data['body_pose'], dtype=np.float32).reshape(1, -1)),
            global_orient=torch.from_numpy(np.asarray(person_data['global_orient'], dtype=np.float32)),
            right_hand_pose=torch.from_numpy(np.asarray(person_data['right_hand_pose'], dtype=np.float32).reshape(1, -1)),
            left_hand_pose=torch.from_numpy(np.asarray(person_data['left_hand_pose'], dtype=np.float32).reshape(1, -1)),
            jaw_pose=torch.from_numpy(np.asarray(person_data['jaw_pose'], dtype=np.float32)),
            leye_pose=torch.from_numpy(np.asarray(person_data['leye_pose'], dtype=np.float32)),
            reye_pose=torch.from_numpy(np.asarray(person_data['reye_pose'], dtype=np.float32)),
            expression=torch.from_numpy(np.asarray(person_data['expression'], dtype=np.float32)),
            transl=torch.from_numpy(np.asarray(person_data['transl'], dtype=np.float32)),
            return_verts=True,
        )
    return output.vertices[0].cpu().numpy(), np.asarray(model.faces)


def _estimate_complete_mesh_camera(person_data, width, height):
    """Fit intrinsics from the WSG-stored mesh projections."""
    vertices = np.asarray(person_data.get('vertices', []), dtype=np.float64)
    pixels = np.asarray(person_data.get('vertices_2d', []), dtype=np.float64)
    if len(vertices) < 10 or len(pixels) != len(vertices):
        return 1500.0, 1500.0, width / 2.0, height / 2.0
    valid = np.abs(vertices[:, 2]) > 1e-5
    vertices = vertices[valid]
    pixels = pixels[valid]
    design_x = np.column_stack((vertices[:, 0] / vertices[:, 2], np.ones(len(vertices))))
    design_y = np.column_stack((vertices[:, 1] / vertices[:, 2], np.ones(len(vertices))))
    fx, cx = np.linalg.lstsq(design_x, pixels[:, 0], rcond=None)[0]
    fy, cy = np.linalg.lstsq(design_y, pixels[:, 1], rcond=None)[0]
    if fx <= 0 or fy <= 0 or not np.all(np.isfinite([fx, fy, cx, cy])):
        return 1500.0, 1500.0, width / 2.0, height / 2.0
    return float(fx), float(fy), float(cx), float(cy)


def render_complete_smplx_black_background(wsg_array, smplerx_data):
    """Render parameter-recovered complete SMPL-X bodies on a black background."""
    if not RENDERING_AVAILABLE:
        return None
    height, width = wsg_array.shape[:2]
    result = np.zeros((height, width, 3), dtype=np.uint8)
    try:
        for person_id, person_data in smplerx_data.items():
            vertices, faces = _create_complete_smplx_mesh(person_data)
            fx, fy, cx, cy = _estimate_complete_mesh_camera(person_data, width, height)
            mesh = trimesh.Trimesh(vertices, faces, process=False)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.1,
                roughnessFactor=0.4,
                alphaMode='OPAQUE',
                emissiveFactor=(0.2, 0.2, 0.2),
                baseColorFactor=(0.7, 0.7, 0.7, 1.0),
            )
            scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(0.3, 0.3, 0.3))
            scene.add(pyrender.Mesh.from_trimesh(mesh, material=material), 'mesh')
            camera_pose = np.eye(4)
            scene.add(pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e12), pose=camera_pose)
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=camera_pose)
            renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
            try:
                rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            finally:
                renderer.delete()
            rgb = rgba[:, :, :3]
            mask = np.any(rgb != 0, axis=2)
            result[mask] = rgb[mask]
            print(
                f"    {person_id}: rendered complete SMPL-X mesh "
                f"({len(vertices)} vertices, {len(faces)} faces)"
            )
        return result
    except Exception as e:
        print(f"    Complete SMPL-X rendering failed, falling back to stored vertices: {str(e)}")
        return None


def render_complete_smplx_overlay(wsg_array, smplerx_data, alpha=0.7):
    """Overlay the complete SMPL-X render on the original RGB frame."""
    wsg_rgb = wsg_array[:, :, 0:3]
    if wsg_rgb.max() <= 1.0:
        original_rgb = (wsg_rgb * 255).astype(np.uint8)
    else:
        original_rgb = wsg_rgb.astype(np.uint8)
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    complete_rgb = render_complete_smplx_black_background(wsg_array, smplerx_data)
    if complete_rgb is None or not np.any(complete_rgb):
        return original_bgr
    complete_bgr = cv2.cvtColor(complete_rgb, cv2.COLOR_RGB2BGR)
    mask = np.any(complete_rgb != 0, axis=2)
    overlay = original_bgr.copy()
    overlay[mask] = cv2.addWeighted(original_bgr[mask], 1.0 - alpha, complete_bgr[mask], alpha, 0)
    return overlay


def render_human_silhouette(wsg_array, smplerx_data):
    """Render only vertices explicitly stored at image pixels in the WSG."""
    height, width = wsg_array.shape[:2]
    
    # Create black background
    silhouette_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # For each person, render their vertices
    for person_id, person_data in smplerx_data.items():
        if 'vertices_2d' in person_data and len(person_data['vertices_2d']) > 0:
            # Use the stored 2D pixel locations directly
            vertices_2d = person_data['vertices_2d']
            
            # Draw all vertex points to form human shape
            for px, py in vertices_2d:
                # Draw slightly larger circles to make the human shape more visible
                cv2.circle(silhouette_img, (int(px), int(py)), 2, (255, 255, 255), -1)
    
    return silhouette_img


def read_stored_pose_visualization(wsg_array):
    """Read the pre-rendered black-background SMPLer-X visualization from channels 90-92."""
    stored = np.asarray(wsg_array[:, :, STORED_VISUALIZATION_CHANNELS])
    if stored.shape[2] != 3 or np.all(stored == -1):
        return None
    valid = np.all(stored != -1, axis=2)
    image = np.zeros(stored.shape, dtype=np.float32)
    image[valid] = stored[valid]
    if image.size and np.nanmax(image) <= 1.0:
        image *= 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def render_overlay_visualization(wsg_array, smplerx_data, alpha=0.5):
    """Render SMPLer-X data overlaid on original image"""
    height, width = wsg_array.shape[:2]
    
    # Get original RGB image
    wsg_rgb_channels = wsg_array[:, :, 0:3]
    if wsg_rgb_channels.max() <= 1.0:
        original_img = (wsg_rgb_channels * 255).astype(np.uint8)
    else:
        original_img = wsg_rgb_channels.astype(np.uint8)
    
    # Convert to BGR for OpenCV
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    
    # Create overlay
    overlay = original_bgr.copy()
    
    # Colors for different persons
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for person_idx, (person_id, person_data) in enumerate(smplerx_data.items()):
        color = colors[person_idx % len(colors)]
        
        # Skip drawing convex hull - we only want to show the vertices and bbox
        
        # Draw MMDet bbox (information storage area)
        if 'bbox_mmdet' in person_data:
            bbox_mm = person_data['bbox_mmdet']
            x1_mm, y1_mm, w_mm, h_mm = int(bbox_mm[0]), int(bbox_mm[1]), int(bbox_mm[2]), int(bbox_mm[3])
            
            # Draw solid rectangle for mmdet bbox (actual storage area)
            cv2.rectangle(overlay, (x1_mm, y1_mm), (x1_mm + w_mm, y1_mm + h_mm), color, 2)
            
            # Left top corner: Person ID
            cv2.putText(overlay, person_id, (x1_mm, y1_mm - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Right top corner: Information storage label
            cv2.putText(overlay, "Info Storage", (x1_mm + w_mm - 90, y1_mm - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    
    # Draw vertices as small circles at the actual WSG storage pixels (filtered by person_id)
    person_channel = wsg_array[:, :, PERSON_ID_CHANNEL]
    vertex_mask = (wsg_array[:, :, VERTEX_INDEX_CHANNEL] != -1) & \
                 (wsg_array[:, :, VERTEX_X_CHANNEL] != -1) & \
                 (wsg_array[:, :, VERTEX_Y_CHANNEL] != -1) & \
                 (wsg_array[:, :, VERTEX_Z_CHANNEL] != -1)
    
    vertex_pixels = np.where(vertex_mask)
    for i in range(len(vertex_pixels[0])):
        y, x = vertex_pixels[0][i], vertex_pixels[1][i]
        pixel_person_id = person_channel[y, x]
        
        # Only draw vertices that belong to extracted persons
        if pixel_person_id != -1 and f'person_{int(pixel_person_id)}' in smplerx_data:
            cv2.circle(overlay, (x, y), 1, (0, 255, 255), -1)  # Yellow dots for vertices
    
    # Blend with original image
    result = cv2.addWeighted(original_bgr, 1-alpha, overlay, alpha, 0)
    return result


def process_single_wsg_file(args):
    """Process single WSG file"""
    wsg_path, output_dir, modes = args
    extremely_tolerant = 'extremely_tolerant' in modes
    visualization_mode = 'extremely_tolerant' if extremely_tolerant else 'trusted'
    
    try:
        base_name = os.path.splitext(os.path.basename(wsg_path))[0]
        
        # Load WSG data
        wsg_array = np.load(wsg_path, allow_pickle=True)['wsg_data']
        stored_img = (
            read_stored_pose_visualization(wsg_array)
            if 'vis_stored' in modes
            else None
        )
        
        # Extract SMPLer-X data
        smplerx_data = extract_smplerx_from_wsg(wsg_array)
        
        if smplerx_data is None or len(smplerx_data) == 0:
            if stored_img is not None:
                stored_path = os.path.join(
                    output_dir, f"export_smplerx_from_wsg_{base_name}_stored.png"
                )
                cv2.imwrite(stored_path, cv2.cvtColor(stored_img, cv2.COLOR_RGB2BGR))
                print(f"  {base_name}: Saved stored pose visualization without structured SMPLer-X data: {stored_path}")
                return True
            print(f"  Skip {base_name}: No SMPLer-X data found")
            return False
        
        print(f"  {base_name}: Extracted SMPLer-X data for {len(smplerx_data)} persons")
        print(f"    SMPLer-X visualization mode: {visualization_mode}")
        
        success_count = 0
        
        # Save NPZ data
        if 'save' in modes:
            npz_path = os.path.join(output_dir, f"export_smplerx_from_wsg_{base_name}.npz")
            np.savez_compressed(npz_path, **smplerx_data)
            print(f"    Saved NPZ data: {npz_path}")
            success_count += 1
        
        # Save NPZ data (excluding vertices to keep file size manageable)
        if 'save_light' in modes:
            light_data = {}
            # Define vertex-related keys to exclude
            vertex_keys = {'vertices', 'vertices_2d', 'vertex_indices'}
            
            for person_id, person_data in smplerx_data.items():
                light_person = {}
                for key, value in person_data.items():
                    if key not in vertex_keys:  # Skip all vertex-related data
                        light_person[key] = value
                light_data[person_id] = light_person
            
            light_npz_path = os.path.join(output_dir, f"export_smplerx_from_wsg_{base_name}_light.npz")
            np.savez_compressed(light_npz_path, **light_data)
            print(f"    Saved light NPZ data: {light_npz_path}")
            success_count += 1
        
        # Render pure white silhouette on black background
        if 'vis_silhouette' in modes:
            silhouette_img = (
                render_complete_smplx_black_background(wsg_array, smplerx_data)
                if extremely_tolerant
                else render_human_silhouette(wsg_array, smplerx_data)
            )
            if silhouette_img is None:
                silhouette_img = render_human_silhouette(wsg_array, smplerx_data)
            silhouette_path = os.path.join(output_dir, f"export_smplerx_from_wsg_{base_name}_silhouette.png")
            cv2.imwrite(silhouette_path, silhouette_img)
            print(f"    Saved silhouette visualization: {silhouette_path}")
            success_count += 1

        if 'vis_stored' in modes:
            if stored_img is not None:
                stored_path = os.path.join(
                    output_dir, f"export_smplerx_from_wsg_{base_name}_stored.png"
                )
                cv2.imwrite(stored_path, cv2.cvtColor(stored_img, cv2.COLOR_RGB2BGR))
                print(f"    Saved stored pose visualization: {stored_path}")
                success_count += 1
            else:
                print("    No stored SMPLer-X visualization found in channels 90-92")
        
        # Render overlay visualization
        if 'vis_overlay' in modes:
            overlay_img = render_overlay_visualization(wsg_array, smplerx_data)
            overlay_path = os.path.join(output_dir, f"export_smplerx_from_wsg_{base_name}_overlay.png")
            cv2.imwrite(overlay_path, overlay_img)
            print(f"    Saved overlay visualization: {overlay_path}")
            success_count += 1
        
        # Render 3D mesh silhouette (pyrender)
        if 'vis_3d_silhouette' in modes:
            try:
                # Get camera parameters from metadata (should match what was used during SMPLer-X processing)
                camera_params = None
                # Use the actual focal and princpt values from SMPLer-X metadata
                # These were stored during injection but we need to retrieve them
                # For now, use the typical SMPLer-X values
                img_height, img_width = wsg_array.shape[:2]
                # From the metadata we saw: focal is around 12185
                focal = [12185.44, 12185.44]  # Use actual SMPLer-X focal length
                princpt = [img_width/2, img_height/2]  # Approximate center
                camera_params = {'focal': focal, 'princpt': princpt}
                
                if camera_params:
                    if extremely_tolerant:
                        mesh_silhouette = render_complete_smplx_black_background(wsg_array, smplerx_data)
                    else:
                        mesh_silhouette = render_3d_mesh_silhouette_multi_person(
                            smplerx_data, wsg_array.shape, camera_params
                        )
                    if mesh_silhouette is None:
                        mesh_silhouette = render_human_silhouette(wsg_array, smplerx_data)
                    silhouette_3d_path = os.path.join(output_dir, f"export_smplerx_from_wsg_{base_name}_3d_silhouette.png")
                    cv2.imwrite(silhouette_3d_path, mesh_silhouette)
                    print(f"    Saved 3D mesh silhouette: {silhouette_3d_path}")
                    success_count += 1
                else:
                    print(f"    No camera parameters found for 3D silhouette rendering")
            except Exception as e:
                print(f"    3D silhouette rendering failed: {str(e)}")
        
        # Render 3D mesh overlay (pyrender)
        if 'vis_3d_overlay' in modes:
            try:
                # Get original image
                wsg_rgb = wsg_array[:, :, 0:3]
                if wsg_rgb.max() <= 1.0:
                    original_img = (wsg_rgb * 255).astype(np.uint8)
                else:
                    original_img = wsg_rgb.astype(np.uint8)
                original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
                
                # Use optimized camera parameters (same as 3D silhouette)
                # This ensures consistent rendering between silhouette and overlay
                img_height, img_width = wsg_array.shape[:2]
                # Pass dummy params - actual params will be calculated per person
                camera_params = {'focal': [1500, 1500], 'princpt': [img_width/2, img_height/2]}
                
                # Always render (params will be optimized per person)
                if True:
                    print(f"    [DEBUG] Starting 3D overlay rendering...")
                    if extremely_tolerant:
                        mesh_overlay = render_complete_smplx_overlay(wsg_array, smplerx_data)
                    else:
                        mesh_overlay = render_3d_mesh_overlay_multi_person_original(
                            smplerx_data, original_bgr, camera_params
                        )
                    print(f"    [DEBUG] 3D overlay rendering completed, saving to file...")
                    overlay_3d_path = os.path.join(output_dir, f"export_smplerx_from_wsg_{base_name}_3d_overlay.png")
                    cv2.imwrite(overlay_3d_path, mesh_overlay)
                    print(f"    Saved 3D mesh overlay: {overlay_3d_path}")
                    
                    # Force cleanup to prevent hanging
                    del mesh_overlay, original_bgr
                    import gc
                    gc.collect()
                    
                    success_count += 1
                else:
                    print(f"    No camera parameters found for 3D overlay rendering")
            except Exception as e:
                print(f"    3D overlay rendering failed: {str(e)}")
        
        # Save original image
        if 'original' in modes:
            wsg_rgb = wsg_array[:, :, 0:3]
            if wsg_rgb.max() <= 1.0:
                original_img = (wsg_rgb * 255).astype(np.uint8)
            else:
                original_img = wsg_rgb.astype(np.uint8)
            original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
            original_path = os.path.join(output_dir, f"export_smplerx_from_wsg_{base_name}_original.png")
            cv2.imwrite(original_path, original_bgr)
            print(f"    Saved original image: {original_path}")
            success_count += 1
        
        return success_count > 0
        
    except Exception as e:
        print(f"  Failed to process file {wsg_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def process_wsg_input(input_path, output_path, modes, max_workers=4):
    """Process WSG input (file or folder)"""
    
    if os.path.isfile(input_path) and input_path.endswith('.wsg'):
        # Single file processing
        if output_path is None:
            output_path = os.path.dirname(input_path)
        
        os.makedirs(output_path, exist_ok=True)
        print(f"Processing single WSG file: {input_path}")
        print(f"Output directory: {output_path}")
        
        success = process_single_wsg_file((input_path, output_path, modes))
        print(f"Processing complete! Success: {'Yes' if success else 'No'}")
        
    elif os.path.isdir(input_path):
        # Folder processing
        if output_path is None:
            parent_dir = os.path.dirname(input_path.rstrip('/'))
            folder_name = os.path.basename(input_path.rstrip('/'))
            output_path = os.path.join(parent_dir, f"export_smplerx_from_{folder_name}")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Get all WSG files
        wsg_files = glob.glob(os.path.join(input_path, "*.wsg"))
        wsg_files.sort()
        
        if not wsg_files:
            print(f"No WSG files found in directory {input_path}")
            return
        
        print(f"Processing WSG folder: {input_path}")
        print(f"Output directory: {output_path}")
        print(f"Found {len(wsg_files)} WSG files")
        
        # Prepare tasks
        tasks = [(wsg_path, output_path, modes) for wsg_path in wsg_files]
        
        # Parallel processing with reduced workers to avoid hanging
        success_count = 0
        # Reduce max workers to prevent resource conflicts and hanging
        actual_max_workers = min(max_workers, 2)  # Limit to 2 workers max
        
        if actual_max_workers > 1:
            print(f"    Using {actual_max_workers} parallel workers (reduced from {max_workers} to prevent hanging)")
            with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
                futures = [executor.submit(process_single_wsg_file, task) for task in tasks]
                
                for future in tqdm(futures, desc="Extracting SMPLer-X"):
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as e:
                        print(f"Task failed: {str(e)}")
                        import traceback
                        traceback.print_exc()
        else:
            # Single-threaded processing
            print(f"    Using single-threaded processing")
            for task in tqdm(tasks, desc="Extracting SMPLer-X"):
                try:
                    if process_single_wsg_file(task):
                        success_count += 1
                except Exception as e:
                    print(f"Task failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        print(f"Processing complete! Success: {success_count}/{len(wsg_files)}")
    
    else:
        print(f"Invalid input path: {input_path}")


def main():
    """Main function"""
    import sys
    
    # Check if command line arguments provided
    if len(sys.argv) > 1:
        # Command line mode
        if len(sys.argv) < 2:
            print("Usage: python pipeline02_read_smplerx_from_wsg.py <input_path> [modes] [output_path]")
            print("Modes (comma-separated): trusted,extremely_tolerant,save,save_light,vis_silhouette,vis_stored,vis_overlay,vis_3d_silhouette,vis_3d_overlay,original")
            return
        
        input_path = sys.argv[1]
        modes = sys.argv[2].split(',') if len(sys.argv) > 2 else ['save']
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        max_workers = 4
        
        process_wsg_input(input_path, output_path, modes, max_workers)
    
    else:
        # Default batch mode
        input_base = "../output/test_dataset_results"
        #input_base = "../output/quicktest"
        modes = ['trusted', 'save', 'save_light', 'vis_silhouette', 'vis_overlay', 'vis_3d_silhouette', 'vis_3d_overlay', 'original']
        # Reduce workers to avoid OpenGL context conflicts
        max_workers = min(4, mp.cpu_count() // 4)
        
        print("=" * 50)
        print("WSG SMPLer-X Extraction and Visualization Tool")
        print("=" * 50)
        print("Output modes:")
        print("  trusted: Default; visualize only explicitly stored WSG vertices using the historical reader")
        print("  extremely_tolerant: Reconstruct and show every detected person as a complete SMPL-X body")
        print("  save: Complete NPZ format data (includes vertices)")
        print("  save_light: Light NPZ format data (no vertices, smaller file size)")
        print("  vis_silhouette: Pure white human silhouette on black background") 
        print("  vis_stored: Directly read the pre-rendered visualization from channels 90-92")
        print("  vis_overlay: Overlay visualization on original image")
        print("  vis_3d_silhouette: 3D mesh rendered silhouette (pyrender)")
        print("  vis_3d_overlay: 3D mesh overlaid on original image (pyrender)")
        print("  original: Original image")
        print("=" * 50)
        
        folders = [f for f in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, f))]
        for folder in folders:
            wsg_dir = os.path.join(input_base, folder, "wsg")
            if os.path.isdir(wsg_dir):
                print(f"\n>>> Processing: {wsg_dir}")
                process_wsg_input(wsg_dir, None, modes, max_workers)
        
        print("\nAll extraction tasks completed!")


if __name__ == '__main__':
    main()
