import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import tqdm
import pickle
import torch
import argparse

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
from easydict import EasyDict
from scipy.spatial import KDTree

MIN_PTS_PER_INSTANCE = 10
MAX_PTS_PER_INSTANCE = 1000000

def load_mesh_as_pointcloud(mesh_file, sample_density=0.01):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    if not mesh.has_triangles():
        raise ValueError("Mesh必须包含三角形面片")
    
    # 计算表面积和所需点数（每 density^2 面积一个点）
    area = mesh.get_surface_area()
    num_points = int(area / (sample_density ** 2))
    # num_points = max(num_points, 10)
    num_points = np.clip(num_points, MIN_PTS_PER_INSTANCE, MAX_PTS_PER_INSTANCE)

    # 均匀采样点云
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    return np.asarray(pcd.points)


def run_one_scene(dataset, split, scene_token, devkit, mminfo, out_dir, device, MAX_VIEW_PER_CAM=999999, MAX_VIEW_PER_POINT=3):
    # if os.path.exists(os.path.join(out_dir, 'objs_model', scene_token)):
    #     return
    
    print(f'running on {dataset} => {split} => {scene_token} ...')
    
    mminfo_token2idx = {_info['token']: idx for idx, _info in enumerate(mminfo)}
    mm_scene_objs = MMSceneObjects(devkit, mminfo, scene_token, dataset, use_sweeps=False)
    sample_tokens = mm_scene_objs.sample_tokens
    
    # ! Step 1 : load all available obj mesh and trans to pointclouds model
    print('loading mesh...')
    mesh_dir = os.path.join(out_dir, 'meshes', scene_token)
    mesh_files = os.listdir(mesh_dir)
    mesh_files = [item for item in mesh_files if 'fixed' in item]
    
    instance_local_pts = {}
    for mesh_file in mesh_files:
        instance_token = mesh_file.split('.')[0].replace('_fixed', '')
        mesh_file = os.path.join(mesh_dir, mesh_file)
        instance_local_pts[instance_token] = torch.from_numpy(load_mesh_as_pointcloud(mesh_file, sample_density=0.01)).float().to(device)
    
    # ! Step 2 : collect all camera's rgb and depth
    all_imgs = []
    all_depths = []
    all_imgs_idx = []
    cam_names = None
    pb = tqdm.tqdm(total=len(sample_tokens), leave=True, desc=f'collecting rgb and depth ...')
    for i, sample_token in enumerate(sample_tokens):
        _info = mminfo[mminfo_token2idx[sample_token]]
        if cam_names is None: cam_names = list(_info['cams'].keys())
        for j, sensor in enumerate(_info['cams'].keys()):
            img_file = _info['cams'][sensor]['data_path']
            depth_file = os.path.join(out_dir, 'depths', 'dense_depth_SPNorm', sensor, f'{sample_token}.png')
            
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            depth = read_depth_map(depth_file)
            
            all_imgs.append(torch.from_numpy(img).to(device))
            all_depths.append(torch.from_numpy(depth).float().to(device))
            all_imgs_idx.append(j + i*len(cam_names))
            
        pb.update()
    pb.close()
    # all_imgs = torch.stack(all_imgs)
    # all_depths = torch.stack(all_depths)
    all_imgs_idx = torch.tensor(all_imgs_idx).to(device).long()
            
            
    # ! Step 3 : allocate color of each pts from near to far respective to the camera
    pb = tqdm.tqdm(total=len(instance_local_pts), leave=True, desc=f'allocating color...')
    for instance_token, local_pts in instance_local_pts.items():
        # pts_idx = torch.arange(len(local_pts)).to(device)
        # allo_mask = torch.zeros((len(local_pts)), dtype=bool).to(device)
        
        allo_tex = torch.zeros((len(local_pts), 3), dtype=torch.int16).to(device)
        allo_nviews = torch.zeros((len(local_pts)), dtype=torch.int16).to(device)
        
        # Step 3.1 : find out which camera the obj has occurs
        all_views = []
        for i, sample_token in enumerate(sample_tokens):
            cam_objs = mm_scene_objs.cam_objs[i].cam_objs   # objs in all camera of one sample
            for sensor, cam_obj in cam_objs.items():
                if instance_token not in cam_obj.instance_tokens: continue
                _view = recursive_defaultdict()
                _idx = np.where(np.array(cam_obj.instance_tokens) == instance_token)[0][0]
                
                _view['sample_token'] = sample_token
                _view['sensor'] = sensor
                _view['T_cam2img'] = torch.from_numpy(cam_obj.T_cam2img).to(device)
                _view['cam_h'] = cam_obj.cam_h
                _view['cam_w'] = cam_obj.cam_w
                _view['T_local2cam'] = torch.from_numpy(cam_obj.T_local2cams[_idx]).to(device)
                _view['depth'] = cam_obj.cam_corners[_idx][:, 2].mean()
                _view['box2d_size'] = get_box_size(cam_obj.proj_box2d[_idx])
                
                _view = EasyDict(_view)
                all_views.append(_view)
                
        if len(all_views) == 0: 
            pb.update()
            continue
        
        # Step 3.2 : select views: for each camera, we select views which obj-to-cam-depth between different views are larger then 1m
        # and we collect all this views and search texture in these imgs, the final texture will take the mean of them
        used_views = []
        for cam_name in cam_names:
            cam_views = [item for item in all_views if item.sensor==cam_name]
            cam_views = sorted(cam_views, key=lambda x: x.depth)
            for i in range(len(cam_views)):
                if i == 0: 
                    used_views.append(cam_views[i])
                    tmp_depth = cam_views[i].depth
                    n_view = 1
                else:
                    if n_view > MAX_VIEW_PER_CAM: break
                    if cam_views[i].depth - tmp_depth < 1: continue
                    used_views.append(cam_views[i])
                    tmp_depth = cam_views[i].depth
                    n_view += 1
                    
        # Step 3.3 : collect all views' T_local2img and info
        T_local2img_used = []
        cam_h_used = []
        cam_w_used = []
        imgs_idx_used = []
        for _view in used_views:
            T_local2img_used.append(_view.T_cam2img @ _view.T_local2cam)
            cam_h_used.append(_view.cam_h)
            cam_w_used.append(_view.cam_w)
            imgs_idx_used.append(int(cam_names.index(_view.sensor) + sample_tokens.index(_view.sample_token)*len(cam_names)))
        T_local2img_used = torch.stack(T_local2img_used).float().to(device)   # (n_view, 3, 4)
        cam_h_used = torch.tensor(cam_h_used).to(device)        # (n_view)
        cam_w_used = torch.tensor(cam_w_used).to(device)        # (n_view)
        # imgs_idx_used = torch.tensor(imgs_idx_used).long().to(device)
        
        # img_used = all_imgs[imgs_idx_used]          # (n_view, h, w, 3)
        # depth_used = all_depths[imgs_idx_used]      # (n_view, h, w)
        img_used = [all_imgs[i] for i in imgs_idx_used]
        depth_used = [all_depths[i] for i in imgs_idx_used]
        
        # Step 3.4 : start to search texture
        img_pts = torch.matmul(cart2homo_cuda(local_pts), (T_local2img_used).permute(0, 2, 1))     # (n_pts, 4) @ (n_view, 4, 3) -> (n_view, n_pts, 3)
        pts_depth = img_pts[..., 2]                         # (n_view, n_pts)
        pts_uv = img_pts[..., 0:2] / img_pts[..., 2:3]      # (n_view, n_pts, 2)
        pts_uv = pts_uv.long()
        mask = (pts_depth > 0) & \
               (pts_uv[..., 0] >= 0) & (pts_uv[..., 0] < cam_w_used[:, None]) & \
               (pts_uv[..., 1] >= 0) & (pts_uv[..., 1] < cam_h_used[:, None])    # (n_view, n_pts)
                   
        for i in range(len(pts_depth)):            
            # check occlusion
            _pts_depth = pts_depth[i]
            _pts_uv = pts_uv[i]
            _m = mask[i] & (allo_nviews <= MAX_VIEW_PER_POINT)
            
            pts_idx = torch.arange(len(_pts_depth)).to(device)
        
            _pts_depth = _pts_depth[_m]
            _pts_uv = _pts_uv[_m]
            pts_idx = pts_idx[_m]
            
            _m = torch.abs(
                depth_used[i][_pts_uv[:, 1], _pts_uv[:, 0]] - _pts_depth
            ) < 0.1
            _pts_depth = _pts_depth[_m]
            _pts_uv = _pts_uv[_m]
            pts_idx = pts_idx[_m]
            
            # get color
            pts_color = img_used[i][_pts_uv[:, 1], _pts_uv[:, 0]]
            allo_tex[pts_idx] += pts_color
            allo_nviews[pts_idx] += 1

        
        # Step 3.5 : get mean texture
        allo_mask = (allo_nviews!=0)
        allo_tex = allo_tex.float()
        allo_tex[allo_mask] /= allo_nviews[allo_mask, None]
        allo_tex = torch.clamp(allo_tex, 0, 255).to(torch.uint8)
        
        # ! key value to numpy
        local_pts = local_pts.cpu().numpy()
        allo_mask = allo_mask.cpu().numpy()
        allo_tex = allo_tex.cpu().numpy()
            
        # Step 3.6 : use mirror prior to help fill the texture of another unseen side
        sym_pts = local_pts.copy()
        sym_pts[:, 1] = -sym_pts[:, 1]  # y → -y
        tree = KDTree(sym_pts)
        _, sym_idx = tree.query(local_pts, k=1)
        sym_tex = allo_tex[sym_idx]     # color of the mirror point for each point
        
        mask = ~allo_mask
        allo_tex[mask] = sym_tex[mask] 
        allo_mask = (allo_mask | allo_mask[sym_idx])
        
        # Step 3.7 save obj texture pointcloud model
        result_file = os.path.join(out_dir, 'objs_model', scene_token, f'{instance_token}_full.ply')
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        save_texture_point_cloud_to_ply(
            local_pts,
            allo_tex,
            result_file
        )
        
        color_mask = ~np.all(allo_tex==0, axis=1)
        result_file = os.path.join(out_dir, 'objs_model', scene_token, f'{instance_token}.ply')
        save_texture_point_cloud_to_ply(
            local_pts[color_mask],
            allo_tex[color_mask],
            result_file
        )
        
        pb.update()
    pb.close()
      
      

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta Data Construction Pipeline.")
    parser.add_argument('--dataset', choices=['nuscenes', 'lyft', 'waymo'], default='nuscenes', help="Dataset to construct.")
    parser.add_argument('--split', choices=['train', 'val'], default='train', help="Dataset to construct.")
    args = parser.parse_args()
    
    dataset = args.dataset
    split = args.split
    device = 'cuda'
    
    devkit, mminfo, out_dir = load_dataset_devkit(dataset, split)
    _metadata = mminfo['metadata']
    mminfo = mminfo['infos']
    
    all_scene_tokens = get_scenes_from_mminfo(mminfo)
    
    n_scenes = len(all_scene_tokens)
    
    pb = tqdm.tqdm(total=len(all_scene_tokens), leave=True, desc=f'constructing instance model : {dataset} => {split}')
    for i, scene_token in enumerate(all_scene_tokens):
        
        run_one_scene(dataset, split, scene_token, devkit, mminfo, out_dir, device)
        
        torch.cuda.empty_cache()
        pb.update()
    pb.close()
    
    