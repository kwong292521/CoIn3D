set_cpu = True
import psutil
import os
pid = os.getpid()
if set_cpu:
    cpu2use = 8
    cpu_scan_time = 10
    cpu_list = [
        
        ]
    if len(cpu_list) == 0:
        cpu_usage = psutil.cpu_percent(interval=cpu_scan_time, percpu=True)
        sorted_usage = sorted(enumerate(cpu_usage), key=lambda x: x[1])
        cpu_list = [index for index, value in sorted_usage[:cpu2use]]
    os.sched_setaffinity(pid, cpu_list)
affinity = psutil.Process(pid).cpu_affinity()
cpu_info = f'Process {pid} is running on CPUs: {affinity}'
print(cpu_info)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

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

from easydict import EasyDict


# DOWNSAMPLE_RATIO = 2
DOWNSAMPLE_RATIO = 1


def fit_plane_cuda(points):
    """
    fit ax + by + cz + d = 0
    set z = ax + by + d (c=-1)
    """
    A = torch.column_stack((points[:, :2], torch.ones(len(points), dtype=points.dtype, device=points.device)))  # (x,y,1)
    b = points[:, 2:3]      # z
    x = torch.linalg.inv(A.T @ A) @ A.T @ b
    x = x.squeeze()
    return [x[0], x[1], -1, x[2]] 


def compute_plane_depths_cuda(pts_cam, K, pts_uv, device, depth_threshold=1000.0):
    """
    使用 GPU 计算地面深度
    
    输入:
        pts_cam: (4, 3) 的地面点 3D 坐标 (cam 坐标系下)
        K: (3, 3) 的相机内参
        pts_uv: (N, 2) 的像素坐标 (u, v)
        device: GPU 设备 id (例如 0，1 或 'cuda:0')
        depth_threshold: 判定深度是否合理的最大值（米）
        
    输出:
        depths: (N,) 的数组，表示每个像素点对应的地面深度（无效返回 -1）
    """
    # Step 1: numpy -> tensor
    pts_cam_tensor = torch.tensor(pts_cam.copy(), dtype=torch.float32, device=device)
    K_tensor = torch.tensor(K, dtype=torch.float32, device=device)
    pts_uv_tensor = torch.tensor(pts_uv.copy(), dtype=torch.float32, device=device)
    
    # Step 2: fit plane ax + by + cz + d = 0 (we define cam plane as y=ax+cz+b)
    a, c, b, d = fit_plane_cuda(pts_cam_tensor[:, [0, 2, 1]])

    # Step 3: fit the plane depth for each pixels
    # (1) u = fx * X / Z + cx -> X = (u - cx) / fx * Z  -> X = u_coef * Z
    # (2) v = fy * Y / Z + cy -> Y = (v - cy) / fy * Z  -> Y = v_coef * Z
    # (3) aX + bY + cZ + d = 0 -> a * u_coef * Z + b * v_coef * Z + c * Z + d = 0 -> (a*u_coef + b*v_coef + c) * Z + d = 0
    # (1)(2)(3) -> Z = - d / (a*u_coef + b*v_coef + c)
    fx, fy = K_tensor[0, 0], K_tensor[1, 1]
    cx, cy = K_tensor[0, 2], K_tensor[1, 2]
    u = pts_uv_tensor[:, 0]
    v = pts_uv_tensor[:, 1]
    
    u_coef = (u - cx) / fx
    v_coef = (v - cy) / fy
    
    plane_z = - d / (a * u_coef + b * v_coef + c)
    
    # Step 6: 合理性判断
    invalid_mask = (plane_z <= 0) | (plane_z > depth_threshold) | torch.isnan(plane_z)
    plane_z[invalid_mask] = -1
    
    return plane_z.cpu().numpy() 



def run_one_scene(dataset, split, scene_token, out_dir, device):
    mm_scene_objs_file = os.path.join('./devkits', 'mm_scene_objs', dataset, split, f'{scene_token}.pkl')
    assert os.path.exists(mm_scene_objs_file)
    with open(mm_scene_objs_file, 'rb') as f:
        mm_scene_objs = pickle.load(f) 
    assert mm_scene_objs.scene_token == scene_token
    sample_tokens = mm_scene_objs.sample_tokens
    
    # ! Step 2 : collect all camera's rgb and depth
    pb = tqdm.tqdm(total=len(sample_tokens), leave=True, desc=f'inpainting foreground depth {dataset} => {split} => {scene_token} ... ')
    for frame_idx, sample_token in enumerate(sample_tokens):
        sample_mminfo_file = os.path.join('./devkits', 'sample_mminfo', dataset, split, f'{sample_token}.pkl')
        assert os.path.exists(sample_mminfo_file)
        with open(sample_mminfo_file, 'rb') as f:
            _info = pickle.load(f)
        
        for sensor in _info['cams'].keys():   
            cam_objs = mm_scene_objs.cam_objs[frame_idx].cam_objs
            
            dense_depth_file = os.path.join(out_dir, 'depths', 'dense_depth_SPNorm', sensor, f'{sample_token}.png')
            dense_depth = read_depth_map(dense_depth_file)
            
            key_mask_file = os.path.join(out_dir, 'masks', 'key_mask2d', sensor, f'{sample_token}.png')
            key_mask = cv2.imread(key_mask_file)
            
            # h, w = cam_objs[sensor]['cam_h'], cam_objs[sensor]['cam_w']
            raw_h, raw_w = cam_objs[sensor]['cam_h'], cam_objs[sensor]['cam_w']
            h, w = raw_h // DOWNSAMPLE_RATIO, raw_w // DOWNSAMPLE_RATIO
            
            dense_depth = cv2.resize(dense_depth, (w, h), interpolation=cv2.INTER_NEAREST)
            key_mask = cv2.resize(key_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            result_file = os.path.join(out_dir, 'inpainted', 'depth', sensor, f'{sample_token}.png')
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            
            cam_corners_bottom = cam_objs[sensor]['cam_corners'][:, 4:]     # (n_objs, 4)
            # proj_box2d = cam_objs[sensor]['proj_box2d']
            proj_box2d = cam_objs[sensor]['proj_box2d'] // DOWNSAMPLE_RATIO
            # K = cam_objs[sensor]['K']
            K = cam_objs[sensor]['K'] // DOWNSAMPLE_RATIO
            
            background_depth = np.zeros((h, w))
            ground_depth = np.ones((h, w)) * 999999
            
            for _corners, _box in zip(cam_corners_bottom, proj_box2d):
                mask = np.zeros((h, w), dtype=np.bool_)
                
                mask[_box[1]:_box[3], _box[0]:_box[2]] = True
                pts_uv = np.argwhere(mask)[:, ::-1]
                
                # pts_depth = compute_plane_depths(
                pts_depth = compute_plane_depths_cuda(
                    pts_cam=_corners,
                    K=K,
                    pts_uv=pts_uv,
                    device=device
                )
            
                _m = (pts_depth > 0)
                pts_uv = pts_uv[_m]
                pts_depth = pts_depth[_m]
                
                # ground depth
                ground_depth[pts_uv[:, 1], pts_uv[:, 0]] = np.minimum(
                    ground_depth[pts_uv[:, 1], pts_uv[:, 0]],
                    pts_depth
                )
                
                # background depth
                _background_depth = np.zeros((h, w))
                _upper_mean = np.mean(dense_depth[max(0, _box[1]-10):_box[1]], axis=0, keepdims=True)
                _background_depth = _background_depth + _upper_mean
                background_depth[mask] = np.maximum(
                    background_depth[mask],
                    _background_depth[mask]
                )
                
            background_depth[background_depth==0] = 999999
            
            inpainted_depth = np.minimum(background_depth, ground_depth)
            ep_m = key_mask[..., 2] == 250
            inpainted_depth[ep_m] = np.minimum(inpainted_depth[ep_m], dense_depth[ep_m])
            inpainted_depth[inpainted_depth==999999] = 0
            
            inpainted_depth = cv2.resize(inpainted_depth, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
            
            save_depth_map(result_file, inpainted_depth)

        pb.update()
    pb.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta Data Construction Pipeline.")
    parser.add_argument('--dataset', choices=['nuscenes', 'lyft', 'waymo'], default='lyft', help="Dataset to construct.")
    parser.add_argument('--split', choices=['train', 'val'], default='train', help="Dataset to construct.")
    args = parser.parse_args()
    
    dataset = args.dataset
    split = args.split
    device = 'cuda'
    
    out_dir_mapping = {
        'nuscenes': NUSC_OUT_DIR,
        'lyft': LYFT_OUT_DIR,
        'waymo': WAYMO_OUT_DIR
    }
    out_dir = out_dir_mapping[dataset]
    
    all_scene_tokens = os.listdir(os.path.join('./devkits', 'mm_scene_objs', dataset, split))
    all_scene_tokens = [item.split('.')[0] for item in all_scene_tokens]
    
    pb = tqdm.tqdm(total=len(all_scene_tokens), leave=True, desc=f'inpainting foreground depth : {dataset} => {split}')
    for i, scene_token in enumerate(all_scene_tokens):
    
        run_one_scene(dataset, split, scene_token, out_dir, device)
            
        torch.cuda.empty_cache()
        pb.update()
    pb.close()
    