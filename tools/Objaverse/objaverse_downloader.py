#!/usr/bin/env python3
"""
直接下载Objaverse模型并转换为NPZ - 不保存中间文件
"""

import os
import numpy as np


def download_and_convert_direct(num_models=100):
    """直接下载并转换为NPZ，不保存中间文件"""

    print(f"🎯 直接下载并转换 {num_models} 个Objaverse模型为NPZ")
    print("=" * 60)

    try:
        import objaverse
        import random
        import trimesh

        print("📋 获取Objaverse模型列表...")
        uids = objaverse.load_uids()
        print(f"✅ 数据库中有 {len(uids)} 个模型")

        # 选择模型
        random.seed(42)
        if len(uids) < num_models:
            sample_uids = uids
        else:
            sample_uids = random.sample(uids, num_models)

        print(f"🎯 选择 {len(sample_uids)} 个模型进行处理")

        # 下载模型
        print("📥 下载模型...")
        objects = objaverse.load_objects(
            uids=sample_uids,
            download_processes=4
        )

        print(f"✅ 下载完成，共 {len(objects)} 个文件")

        # 直接处理为模型字典
        print("\n🔄 直接转换为模型数据...")
        models = []

        for i, (uid, glb_path) in enumerate(objects.items(), 1):
            print(f"   [{i}/{len(objects)}] 处理: {uid}")

            if not os.path.exists(glb_path):
                print(f"      ❌ 文件不存在")
                continue

            try:
                # 加载GLB文件
                scene_or_mesh = trimesh.load(str(glb_path))

                # 获取mesh
                if hasattr(scene_or_mesh, 'geometry'):
                    geometries = list(scene_or_mesh.geometry.values())
                    if geometries:
                        mesh = geometries[0]
                    else:
                        print(f"      ⚠️ 场景为空")
                        continue
                else:
                    mesh = scene_or_mesh

                if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
                    print(f"      ⚠️ 无效mesh")
                    continue

                vertices = np.array(mesh.vertices, dtype=np.float32)
                faces = np.array(mesh.faces, dtype=np.int32)

                if len(vertices) == 0 or len(faces) == 0:
                    print(f"      ⚠️ 空几何体")
                    continue

                # 法向量
                try:
                    if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) == len(vertices):
                        normals = np.array(mesh.vertex_normals, dtype=np.float32)
                    else:
                        mesh.fix_normals()
                        normals = np.array(mesh.vertex_normals, dtype=np.float32)
                except:
                    normals = np.tile([0, 0, 1], (len(vertices), 1)).astype(np.float32)

                # 简单颜色处理
                try:
                    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                        colors = np.array(mesh.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
                    else:
                        # 基于位置生成颜色
                        norm_verts = vertices - vertices.min(axis=0)
                        vert_range = vertices.max(axis=0) - vertices.min(axis=0)
                        vert_range[vert_range == 0] = 1
                        colors = norm_verts / vert_range
                except:
                    colors = np.random.rand(len(vertices), 3).astype(np.float32)

                colors = np.clip(colors, 0.0, 1.0)

                # 创建模型字典
                model_dict = {
                    'vertices': vertices,
                    'faces': faces,
                    'normals': normals,
                    'colors': colors,
                    'vertex_colors': colors,
                    'uuid': uid,
                    'joints': np.array([]),
                    'bones': np.array([]),
                    'joint_names': [],
                    'root_index': 0,
                    'skinning_weights_value': np.ones(len(vertices)),
                    'skinning_weights_row': np.arange(len(vertices)),
                    'skinning_weights_col': np.zeros(len(vertices)),
                    'skinning_weights_shape': [len(vertices), 1]
                }

                models.append(model_dict)
                print(f"      ✅ 成功: {len(vertices)}顶点, {len(faces)}面")

            except Exception as e:
                print(f"      ❌ 处理失败: {e}")
                continue

        if models:
            # 创建输出目录
            output_dir = "input/ObjaverseInput"
            os.makedirs(output_dir, exist_ok=True)

            # 保存NPZ文件
            output_path = os.path.join(output_dir, "objaverse_models.npz")
            np.savez_compressed(output_path, arr_0=np.array(models, dtype=object))

            print(f"\n💾 成功保存到: {output_path}")
            print(f"📊 转换统计:")
            print(f"   下载模型: {len(objects)}")
            print(f"   成功转换: {len(models)}")
            print(f"   成功率: {len(models) / len(objects) * 100:.1f}%")

            # 显示模型信息
            print(f"\n🎯 模型详情:")
            total_vertices = 0
            total_faces = 0
            for i, model in enumerate(models[:5]):  # 显示前5个
                vertices_count = len(model['vertices'])
                faces_count = len(model['faces'])
                total_vertices += vertices_count
                total_faces += faces_count
                print(f"   {i + 1}. {model['uuid']}: {vertices_count}顶点, {faces_count}面")

            if len(models) > 5:
                print(f"   ... 还有 {len(models) - 5} 个模型")

            print(f"\n📈 总计: {total_vertices}顶点, {total_faces}面")
            print(f"🔍 验证文件: python test.py ObjaverseInput/objaverse_models.npz")
            print(f"🚀 运行pipeline: python main/pipeline15_inject_object_into_wsg.py")

            return True
        else:
            print("❌ 没有成功转换任何模型")
            return False

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("💡 请安装: pip install objaverse trimesh[easy]")
        return False
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return False


def main():
    """主函数"""

    print("🎯 Objaverse直接转换器")
    print("=" * 50)

    # 获取用户输入
    try:
        num_models = input("请输入要下载的模型数量 (默认100): ").strip()
        if not num_models:
            num_models = 100
        else:
            num_models = int(num_models)

    except ValueError:
        print("❌ 输入无效，使用默认值100")
        num_models = 100

    # 验证参数
    if num_models <= 0:
        num_models = 100
    if num_models > 500:
        confirm = input(f"⚠️ 要下载 {num_models} 个模型？这可能需要很长时间。继续吗？(y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ 已取消")
            return

    print(f"\n🎯 开始处理 {num_models} 个模型")

    if download_and_convert_direct(num_models):
        print("\n🎉 转换完成！")
        print("📄 NPZ文件: input/ObjaverseInput/objaverse_models.npz")
    else:
        print("\n❌ 转换失败")


if __name__ == "__main__":
    main()