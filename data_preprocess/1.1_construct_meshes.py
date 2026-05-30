import ipdb
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import tqdm
import pickle
import torch
import argparse

import time

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from common_utils import *
from vdb_fusion_utils import *

from __PATHS__ import *



def run_one_scene(dataset, split, scene_token, devkit, mminfo, out_dir):
    if os.path.exists(os.path.join(out_dir, 'meshes', scene_token)):
        return
    
    # ipdb.set_trace()
    mm_scene_objs = MMSceneObjects(devkit, mminfo, scene_token, dataset)
    instance_static = mm_scene_objs.instance_static
    
    # collect all lidar pointclouds
    print('collecting all lidar pointclouds ...')
    lidar_pts = []
    lidar_objs = []
    for i, lidar_data_token in enumerate(mm_scene_objs.all_lidar_data_tokens):
        lidar_file = os.path.join(
            devkit.dataroot,
            devkit.get('sample_data', lidar_data_token)['filename'])
        
        pts_lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, LIDAR_FEAT[dataset])[:, :3]
        
        # range filter
        pts_lidar = pts_lidar[np.linalg.norm(pts_lidar, axis=1) <= 100.]
        pts_lidar = pts_lidar[np.linalg.norm(pts_lidar, axis=1) >= 3.]
        
        pts_lidar = pts_lidar.astype(np.float64)
        
        lidar_pts.append(pts_lidar)
        mmobjs = mm_scene_objs.lidar_objs[i]
        lidar_objs.append(mmobjs)
        
    
    # create scene meshes(exclude dynamic objects) & collect objects point cloud
    instance_local_pts = {}
    for instance_token in instance_static.keys(): 
        instance_local_pts[instance_token] = []
    
    pts4mesh = []
    poses4mesh = []
    
    for pts_lidar, mmobjs in zip(lidar_pts, lidar_objs):
        # Step 1: 收集静态场景点云 这时框往外扩降噪 
        pts_obj_idxs, _ = mmobjs.get_lidar_pts_in_boxes(pts_lidar, _res=-0.1, _res_inbottom_only=False)
        # Step 2: 收集物体点云 这是框的底部往内缩 把地面滤除 
        _, obj_local_pts = mmobjs.get_lidar_pts_in_boxes(pts_lidar, _res=0.05, _res_inbottom_only=True)
        
        _dynamic_idx = []
        for i, instance_token in enumerate(mmobjs.lidar_instance_tokens):
            if not instance_static[instance_token]: 
                _dynamic_idx.append(i)
            if obj_local_pts[i] is not None:
                instance_local_pts[instance_token].append(obj_local_pts[i])
    
        # _static_mask = ~np.isin(pts_obj_idxs, _dynamic_idx)
        _bg_mask = (pts_obj_idxs==-1)
        
        # trans to global coord
        T_lidar2global = mmobjs.T_lidar2global
        pts_lidar = np.concatenate([pts_lidar, np.ones((pts_lidar.shape[0], 1))], axis=1)
        pts_lidar = (pts_lidar @ T_lidar2global.T)[:, :3]
        
        pts4mesh.append(pts_lidar[_bg_mask])
        poses4mesh.append(mmobjs.T_lidar2global)
        
    
    # ! begin to create scene mesh & obj mesh model
    print('begin to create mesh...')
    vdbfusion_pipeline = VDBFusionPipeline(
        config=make_vdbfusion_config(
            out_dir=os.path.join(out_dir, 'meshes', scene_token)),
        result_name=scene_token
    )
    vdbfusion_pipeline.run(
        scans=pts4mesh, 
        poses=poses4mesh)
    
    
    for instance_token in instance_local_pts.keys():
        if len(instance_local_pts[instance_token]) == 0: continue
        
        vdbfusion_pipeline = VDBFusionPipeline(
            config=make_vdbfusion_config(
                out_dir=os.path.join(out_dir, 'meshes', scene_token)),
            result_name=instance_token
        )
        
        # _scans = [item for item in instance_local_pts[instance_token] if len(item) != 0]   
        # NOTE: 利用物体的对称
        _scans = []
        for item in instance_local_pts[instance_token]:
            if len(item) == 0: continue
            item_mirror = item.copy()
            item_mirror[:, 1] *= -1
            item = np.concatenate([item, item_mirror], axis=0)
            _scans.append(item)
        
        _poses = [np.eye(4) for _ in range(len(_scans))]

        vdbfusion_pipeline.run(
            scans=_scans,
            poses=_poses)
    
    print('Done ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta Data Construction Pipeline.")
    parser.add_argument('--dataset', choices=['nuscenes', 'lyft', 'waymo'], default='nuscenes', help="Dataset to construct.")
    parser.add_argument('--split', choices=['train', 'val'], default='train', help="Dataset to construct.")
    args = parser.parse_args()
    
    dataset = args.dataset
    split = args.split
    
    devkit, mminfo, out_dir = load_dataset_devkit(dataset, split)
    _metadata = mminfo['metadata']
    mminfo = mminfo['infos']
    
    all_scene_tokens = get_scenes_from_mminfo(mminfo)
    n_scenes = len(all_scene_tokens)
    
    pb = tqdm.tqdm(total=len(all_scene_tokens), leave=True, desc=f'running scene & obj mesh construction : {dataset} => {split}')
    for i, scene_token in enumerate(all_scene_tokens):
        print(f'[INFO] [{i} / {n_scenes}] : running scene & obj mesh construction : {dataset} => {split} => {scene_token}')
        run_one_scene(dataset, split, scene_token, devkit, mminfo, out_dir)
        
        pb.update()
    pb.close()
    