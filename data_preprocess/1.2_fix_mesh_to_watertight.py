import pymeshfix
import os
import tqdm
import open3d as o3d
import numpy as np
import argparse
import multiprocessing
import time
from __PATHS__ import *
from common_utils import *

import time

def clean_mesh_with_timeout(vertices, faces, timeout=30):
    def _run_clean(q, vertices, faces, remove_small):
        q.put(pymeshfix.clean_from_arrays(
            vertices, faces,
            joincomp=True,
            remove_smallest_components=remove_small
        ))

    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_clean, args=(q, vertices, faces, False)
    )
    p.start()

    try:
        result = q.get(timeout=timeout) 
        p.join()
        return result
    except multiprocessing.queues.Empty:
        p.terminate()
        p.join()

    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_clean, args=(q, vertices, faces, True)
    )
    p.start()
    return q.get()



def run_one_scene(dataset, split, scene_token, out_dir, timeout=None):
    mesh_dir = os.path.join(out_dir, 'meshes', scene_token)
    mesh_files = os.listdir(mesh_dir)
    mesh_files = [item for item in mesh_files if ((scene_token not in item) and ('fixed' not in item))]

    pb = tqdm.tqdm(total=len(mesh_files), leave=True, desc=f'fixing mesh to watertight {dataset} => {split} => {scene_token}...')
    for mesh_file in mesh_files:
        mesh_file = os.path.join(mesh_dir, mesh_file)
        repaired_file = mesh_file.replace('.ply', '_fixed.ply')
        # pymeshfix.clean_from_file(mesh_file, repaired_file)

        # 1. open3d 读取原始网格
        mesh = o3d.io.read_triangle_mesh(mesh_file)

        # 2. 提取顶点和面片，强制转换为 float32（PyMeshFix 要求）
        vertices = np.asarray(mesh.vertices).astype(np.float32)  # 必须 float32！
        faces = np.asarray(mesh.triangles).astype(np.int32)      # 必须 int32！
        
        # 3. 修复网格
        if timeout is None:
            v_clean, f_clean = pymeshfix.clean_from_arrays(
                vertices, 
                faces,
                joincomp=True,
                remove_smallest_components=False
                )
        else:
            v_clean, f_clean = clean_mesh_with_timeout(vertices, faces, timeout)

        # 4. 转回 Open3D 翻转物体面片法线 并保存
        mesh_repaired = o3d.geometry.TriangleMesh()
        mesh_repaired.vertices = o3d.utility.Vector3dVector(v_clean)
        # mesh_repaired.triangles = o3d.utility.Vector3iVector(f_clean)
        mesh_repaired.triangles = o3d.utility.Vector3iVector(f_clean[:, ::-1])  # 翻转面片顺序
        mesh_repaired.compute_vertex_normals()  # 重新计算法线
        
        # 5. 保存修复后的网格
        o3d.io.write_triangle_mesh(repaired_file, mesh_repaired)
        
        pb.update()
    pb.close()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta Data Construction Pipeline.")
    parser.add_argument('--dataset', choices=['nuscenes', 'lyft', 'waymo'], default='lyft', help="Dataset to construct.")
    parser.add_argument('--split', choices=['train', 'val'], default='val', help="Dataset to construct.")
    parser.add_argument('--timeout', default=None, help='handle pymeshfix bug...')
    args = parser.parse_args()
    
    dataset = args.dataset
    split = args.split
    timeout = float(args.timeout)
    
    devkit, mminfo, out_dir = load_dataset_devkit(dataset, split)
    _metadata = mminfo['metadata']
    mminfo = mminfo['infos']
    
    all_scene_tokens = get_scenes_from_mminfo(mminfo)
    n_scenes = len(all_scene_tokens)
    
    pb = tqdm.tqdm(total=len(all_scene_tokens), leave=True, desc=f'fixing mesh to watertight : {dataset} => {split}')
    for i, scene_token in enumerate(all_scene_tokens):
        run_one_scene(dataset, split, scene_token, out_dir, timeout)
        pb.update()
    pb.close()
