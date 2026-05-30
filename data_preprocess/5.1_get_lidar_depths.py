set_cpu = True
import psutil
import os
pid = os.getpid()
if set_cpu:
    cpu2use = 24
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

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import tqdm
import pickle
import argparse
from multiprocessing import Pool
from scipy.sparse import load_npz

import torch
from pyquaternion import Quaternion

from __PATHS__ import *
from common_utils import *

MAX_DEPTH = 200.

def run_one_sample(args):
    sample_token, mminfo_file, lidar_depth_root, dataset = args

    with open(mminfo_file, 'rb') as f:
        mminfo = pickle.load(f)
    cam_names = list(mminfo['cams'].keys())
    
    lidar_path = mminfo['lidar_path']
    if dataset in ['nuscenes', 'lyft']:
        lidar_pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    elif dataset in ['waymo']:
        lidar_pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 6)[:, :3]
        
    T_lidar2imgs = []
    for cam in cam_names:
        T_cam2lidar = np.eye(4)
        T_cam2lidar[:3, :3] = np.array(mminfo['cams'][cam]['sensor2lidar_rotation'])
        T_cam2lidar[:3, 3] = np.array(mminfo['cams'][cam]['sensor2lidar_translation'])
        T_lidar2cam = np.linalg.inv(T_cam2lidar)
        
        T_cam2img = np.eye(4)
        T_cam2img[:3, :3] = np.array(mminfo['cams'][cam]['cam_intrinsic'])
        
        T_lidar2img = T_cam2img @ T_lidar2cam
        T_lidar2imgs.append(T_lidar2img)
    T_lidar2imgs = np.stack(T_lidar2imgs)   # (n_cam, 4, 4)
    
    # batch transform
    img_pts = (cart2homo(lidar_pts) @ T_lidar2imgs.transpose(0, 2, 1))[:, :, :3]     # (n_cam, n_pts, 3)
       
    for i, cam in enumerate(cam_names):
        img_h, img_w = mminfo['cams'][cam]['img_h'], mminfo['cams'][cam]['img_w']
        
        pts = img_pts[i]    # (n_pts, 3)
        pts_depth = pts[:, 2]
        pts_uv = pts[:, :2] / pts_depth[:, None]
        pts_uv = np.int32(np.round(pts_uv))
        
        # Remove points outside image
        inside_mask = (pts_uv[:, 0] >= 0) & (pts_uv[:, 0] < img_w) \
                    & (pts_uv[:, 1] >= 0) & (pts_uv[:, 1] < img_h) \
                    & (pts_depth < MAX_DEPTH) & (pts_depth > 0)
        pts_depth = pts_depth[inside_mask]
        pts_uv = pts_uv[inside_mask]
    
        # project by z-buffer
        ranks = pts_uv[:, 1] * img_w + pts_uv[:, 0]
        sort = np.argsort(ranks + pts_depth/MAX_DEPTH)
    
        pts_depth = pts_depth[sort]
        pts_uv = pts_uv[sort]
        ranks = ranks[sort]
        
        saved_mask = np.ones(len(pts_depth), dtype=bool)
        # example: sort: [0.2, 1.3, 2.3, 3.2, 4.1, 4.2, 4.3, 5.3, 6.1, 6.2, 6.3]
        # init saved_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # ranks[1:] :     [1, 2, 3, 4, 4, 4, 5, 6, 6, 6]
        # ranks[:-1]:     [0, 1, 2, 3, 4, 4, 4, 5, 6, 6]
        # save_mask[1:]:  [1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
        # save_mask:      [1,  1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
        saved_mask[1:] = (ranks[1:] != ranks[:-1])
        pts_depth = pts_depth[saved_mask]
        pts_uv = pts_uv[saved_mask]
        
        
        # create depth and save
        depth = np.zeros((img_h, img_w))
        depth[pts_uv[:, 1], pts_uv[:, 0]] = pts_depth
        
        depth_file = os.path.join(lidar_depth_root, cam, f'{sample_token}.png')
        os.makedirs(os.path.dirname(depth_file), exist_ok=True)

        save_depth_map(depth_file, depth)
    

if __name__ == '__main__':
    for dataset in [
        'nuscenes',
        'lyft',
        'waymo'
    ]:
        mminfo_root = f''
        lidar_depth_root = f'/meta_data/depths/lidar_depths'
        all_mminfo_files = os.listdir(mminfo_root)
        all_sample_tokens = [item.split('.')[0] for item in all_mminfo_files]
        all_mminfo_files = [os.path.join(mminfo_root, item) for item in all_mminfo_files]    
        
        pb = tqdm.tqdm(total=len(all_sample_tokens), leave=True, desc=f'Generating Lidar Depths For {dataset}...')
        for i in range(len(all_sample_tokens)):
            run_one_sample((all_sample_tokens[i], all_mminfo_files[i], lidar_depth_root, dataset))
            pb.update()
        pb.close()
    

