import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import tqdm
import pickle
import torch
import argparse

from multiprocessing import Pool

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from common_utils import *

from __PATHS__ import *

from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
)
from pytorch3d.renderer.mesh.shader import ShaderBase

import open3d as o3d

import time

def load_ply_as_mesh(ply_file, device):
    mesh_o3d = o3d.io.read_triangle_mesh(ply_file)
    
    # o3d to pytorch3d
    verts = torch.tensor(mesh_o3d.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh_o3d.triangles, dtype=torch.int64, device=device)
    
    return Meshes(verts=[verts], faces=[faces])
    

def merge_meshes(mesh_list):
    if not mesh_list:
        raise ValueError("mesh_list不能为空")
    
    device = mesh_list[0].verts_list()[0].device
    
    all_verts = []
    all_faces = []
    face_offset = 0
    
    for mesh in mesh_list:
        # 获取当前mesh的顶点和面
        verts = mesh.verts_list()[0]  # [V, 3]
        faces = mesh.faces_list()[0]  # [F, 3]
        
        all_verts.append(verts)
        all_faces.append(faces + face_offset)
        face_offset += verts.shape[0]  # 更新面索引偏移量
    
    # 创建合并后的mesh
    return Meshes(
        verts=[torch.cat(all_verts, dim=0)],
        faces=[torch.cat(all_faces, dim=0)],
        textures=None  # 注意: 这里忽略了纹理合并
    ).to(device)
    
    
def transform_mesh(mesh, transform):
    if transform.shape != (4, 4):
        raise ValueError("transform必须是4x4矩阵")
    
    device = mesh.verts_list()[0].device
    transform = torch.tensor(transform).to(device).float()
    
    # 获取顶点并转换为齐次坐标 [V, 3] -> [V, 4]
    verts = mesh.verts_list()[0]
    verts_homo = torch.cat([
        verts,
        torch.ones(verts.shape[0], 1, device=device)
    ], dim=1)
    
    # 应用变换 [V,4] @ [4,4].T -> [V,4]
    transformed_verts = (verts_homo @ transform.T)[:, :3]
    
    # 创建新mesh (保留原始面和纹理)
    return Meshes(
        verts=[transformed_verts],
        faces=mesh.faces_list(),
        textures=mesh.textures
    ).to(device)
    

def filter_mesh_before_cam(mesh, depth_range=[1, 100]):
    verts = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    device = verts.device
    
    verts_mask = (verts[:, 2] > depth_range[0]) & (verts[:, 2] < depth_range[1])
    
    # 筛选面片：仅保留所有顶点都在verts_mask中的三角形
    faces_mask = torch.all(verts_mask[faces], dim=1)
    valid_faces = faces[faces_mask]
    
    # 筛选被用到的顶点（避免保留无用顶点）
    used_verts_mask = torch.zeros_like(verts_mask)
    used_verts_mask[valid_faces.unique()] = True  # 仅保留被面片引用的顶点
    new_verts = verts[used_verts_mask]
    
    # 重新映射面片顶点索引
    vert_id_map = torch.cumsum(used_verts_mask.int(), dim=0) - 1  # 旧索引→新索引
    remapped_faces = vert_id_map[valid_faces]  # 更新面片中的索引

    # 构建新mesh 
    return Meshes(
        verts=[new_verts], 
        faces=[remapped_faces.long()],  # 确保为int64类型
        textures=mesh.textures
    )


class DepthShader(ShaderBase):
    def forward(self, fragments, meshes_world, **kwargs):
        return fragments.zbuf  # [N, H, W, 1]


def create_mesh_renderer(batch_K, img_w, img_h, device):
    #=== intrinsics
    focal_length = torch.tensor([[_K[0, 0], _K[1, 1]] for _K in batch_K], dtype=torch.float32)
    principal_point = torch.tensor([[_K[0, 2], _K[1, 2]] for _K in batch_K], dtype=torch.float32)
    image_size = torch.tensor([[img_h, img_w]] * len(batch_K), dtype=torch.float32)

    #=== extrinsics(trans to pytorch3d cam coord system)
    R = torch.tensor([[-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 1]]).unsqueeze(0).float()
    t = torch.tensor([0, 0, 0]).unsqueeze(0).float()
    
    #=== init camera and raster
    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=image_size,
        R=R,  
        T=t,  
        device=device,
        in_ndc=False
    )

    raster_settings = RasterizationSettings(
        image_size=(img_h, img_w), 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        cull_backfaces=True
    )
    
    #=== init renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings),
        shader=DepthShader(device=device)  
    )

    return renderer



def run_one_scene(dataset, split, scene_token, devkit, mminfo, out_dir, device, downsample=1):
    mminfo_token2idx = {_info['token']: idx for idx, _info in enumerate(mminfo)}
    sample_tokens = get_sample_tokens_from_scene(devkit, scene_token)
    # ! check if this scene is runned
    ret_flag = [0, 0]
    for sample_token in sample_tokens:
        _info = mminfo[mminfo_token2idx[sample_token]]
        for sensor in _info['cams'].keys():
            result_file = os.path.join(out_dir, 'depths', 'mesh_depth', sensor, f'{sample_token}.png')
            if os.path.exists(result_file): ret_flag[0] += 1
            ret_flag[1] += 1
    if ret_flag[0] / ret_flag[1] == 1: return
    
    mm_scene_objs = MMSceneObjects(devkit, mminfo, scene_token, dataset, use_sweeps=False)
    sample_tokens = mm_scene_objs.sample_tokens

    # ! begin to render mesh depth
    # load static scene mesh (in global coord)
    background_scene_mesh_file = os.path.join(out_dir, 'meshes', scene_token, f'{scene_token}.ply')
    background_scene_mesh = load_ply_as_mesh(background_scene_mesh_file, device=device)

    all_objs_token = [k for k, v in mm_scene_objs.instance_static.items()]
    # load dynamic obj mesh (in local coord)
    all_objs_mesh = {}
    for _token in all_objs_token:
        _mesh_file = os.path.join(out_dir, 'meshes', scene_token, f'{_token}_fixed.ply')
        if not os.path.exists(_mesh_file):
            all_objs_mesh[_token] = None
            continue
        _obj_mesh = load_ply_as_mesh(_mesh_file, device=device)
        all_objs_mesh[_token] = _obj_mesh
    
    
    pb = tqdm.tqdm(total=len(sample_tokens), leave=True, desc=f'generating mesh depths {dataset} => {split} => {scene_token}...')
    for sample_token, sample_cam_objs in zip(sample_tokens, mm_scene_objs.cam_objs):
        # create mesh in this frame (trans dynamic obj mesh to this frame coord)
        mesh_list = [background_scene_mesh]
        for i in range(len(sample_cam_objs.lidar_boxes)):
            _token = sample_cam_objs.lidar_instance_tokens[i]
            if _token in all_objs_token:
                obj_mesh = all_objs_mesh[_token]
                if obj_mesh is None: continue
                T_local2global = sample_cam_objs.T_local2globals[i]
                obj_mesh = transform_mesh(obj_mesh, T_local2global)
                mesh_list.append(obj_mesh)
        
        sample_mesh = merge_meshes(mesh_list)   # @ global coord system
            
        # render per camera mesh depth
        for sensor in sample_cam_objs.cam_objs.keys():
            renderer = create_mesh_renderer(
                batch_K=sample_cam_objs.cam_objs[sensor].K[None, ...] // downsample,
                img_w=int(sample_cam_objs.cam_objs[sensor].cam_w // downsample),
                img_h=int(sample_cam_objs.cam_objs[sensor].cam_h // downsample),
                device=device
            )
            
            # NOTE: we use mesh before camera
            mesh2render = transform_mesh(sample_mesh, sample_cam_objs.cam_objs[sensor].T_global2cam)
            mesh2render = filter_mesh_before_cam(mesh2render, depth_range=[1, 100])
            
            mesh_depth = renderer(mesh2render)  # (1, h, w, 1)
            mesh_depth = mesh_depth[0, ..., 0].cpu().numpy()  # (h, w)
            mesh_depth[mesh_depth < 0] = 0
            mesh_depth[mesh_depth > 655.35] = 655
            
            if downsample != 1:
                mesh_depth = cv2.resize(mesh_depth, (sample_cam_objs.cam_objs[sensor].cam_w, sample_cam_objs.cam_objs[sensor].cam_h), interpolation=cv2.INTER_NEAREST)
            
            # save
            result_file = os.path.join(out_dir, 'depths', 'mesh_depth', sensor, f'{sample_token}.png')
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            save_depth_map(result_file, mesh_depth) 
                
        pb.update()
    pb.close()    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta Data Construction Pipeline.")
    parser.add_argument('--dataset', choices=['nuscenes', 'lyft', 'waymo'], default='nuscenes', help="Dataset to construct.")
    parser.add_argument('--split', choices=['train', 'val'], default='train', help="Dataset to construct.")
    parser.add_argument('--downsample', type=float, default=1, help="downsample factor")
    args = parser.parse_args()
    
    dataset = args.dataset
    split = args.split
    downsample = float(args.downsample)
    device = 'cuda'
    
    devkit, mminfo, out_dir = load_dataset_devkit(dataset, split)
    _metadata = mminfo['metadata']
    mminfo = mminfo['infos']
    
    all_scene_tokens = get_scenes_from_mminfo(mminfo)
    n_scenes = len(all_scene_tokens)
    
    pb = tqdm.tqdm(total=len(all_scene_tokens), leave=True, desc=f'generating mesh depths : {dataset} => {split}')
    for i, scene_token in enumerate(all_scene_tokens):
        run_one_scene(dataset, split, scene_token, devkit, mminfo, out_dir, device, downsample)
        torch.cuda.empty_cache()
        pb.update()
    pb.close()
