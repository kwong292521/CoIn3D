set_cpu = True
import psutil
import os
pid = os.getpid()
if set_cpu:
    cpu2use = 16
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


def load_texture_point_cloud_from_ply(filename):
    with open(filename, 'rb') as f:
        while True:
            line = f.readline().decode('utf-8').strip()
            if line == "end_header":
                break
        
        dtype = np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
        ])
        
        data = np.fromfile(f, dtype=dtype)
        geo = np.column_stack((data['x'], data['y'], data['z']))
        tex = np.column_stack((data['r'], data['g'], data['b'])) 
        
    return geo, tex

def read_depth_map(depth_map_path):
    depth_image = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
    depth_map = depth_image / 256.0

    # Discard depths less than 10cm from the camera
    depth_map[depth_map < 0.1] = 0.0

    return depth_map.astype(np.float32)

# def cart2homo(pts):
#     assert pts.shape[-1] == 3
#     return torch.cat([pts, torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)], axis=1)

def cart2homo(pts):
    assert pts.shape[-1] == 3
    return np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)

def map_z2scale(z_val, z_1, z_2, s_1, s_2):
    z_clamped = torch.clamp(z_val.clone(), float(min(z_1, z_2)), float(max(z_1, z_2)))
    z_scale = (s_2 - s_1)/(z_2 - z_1) * (z_clamped - z_1) + s_1
    return z_scale        
        

def construct_T_from_vector(translation_vector, rotation_vector):
    assert len(translation_vector) == 3
    assert len(rotation_vector) == 4
    
    T = np.eye(4)
    R = Quaternion(*rotation_vector).rotation_matrix
    T[:3, :3] = R
    T[:3, 3] = translation_vector
    return T


def depth_to_ego_pts_mask(depth, K, T_cam2ego, mask):
    v_fg, u_fg = torch.where(mask)
    depth_fg = depth[v_fg, u_fg]

    homo_K = torch.eye(4).to(K)
    homo_K[:3, :3] = K
    T_img2cam = torch.inverse(homo_K)
    T_img2ego = T_cam2ego @ T_img2cam
    
    pts_img_homo = torch.stack([
        u_fg * depth_fg,
        v_fg * depth_fg,
        depth_fg,
        torch.ones_like(depth_fg)
    ], dim=-1)
    
    pts_ego = (pts_img_homo @ T_img2ego.T)[..., :3]
    pts_2d = torch.stack([u_fg, v_fg], dim=-1)
    
    return pts_ego, pts_2d



def get_render_mask(key_mask):
    h, w, _ = key_mask.shape

    overlap_mask = (key_mask[..., 1] == 255) 
    dilated_edge_mask = (key_mask[..., 0] == 125)
    eroded_edge_mask = (key_mask[..., 0] == 250)
    egopart_mask = (key_mask[..., 2] == 250)
    fg_inpainted_mask = (key_mask[..., 2] == 125)
    
    # normal_render_mask = np.ones((h, w), dtype=np.bool_)
    # normal_render_mask[overlap_mask] = False
    # # normal_render_mask[dilated_edge_mask] = False
    # normal_render_mask[egopart_mask] = False
    
    # inpainted_render_mask = (fg_inpainted_mask & eroded_edge_mask) | egopart_mask
    inpainted_render_mask = ((key_mask[..., 0] != 0) & fg_inpainted_mask) | egopart_mask
    inpainted_render_mask[overlap_mask] = False
    
    return overlap_mask, egopart_mask, inpainted_render_mask


def run_one_sample(args):
    dataset, scene_token, sample_tokens, device = args
        
    # Step 1 : init
    log_dir = os.path.join('./logs/pregen_sample_gaussians', dataset)
    
    all_mminfo = {}
    all_instances_tokens = []
    for sample_token in sample_tokens:
        mminfo_file = os.path.join('./devkits/sample_mminfo_ssd', dataset, f'{sample_token}.pkl')
        with open(mminfo_file, 'rb') as f:
            mminfo = pickle.load(f)
        all_mminfo[sample_token] = mminfo
        all_instances_tokens.extend(mminfo['instance_tokens'])
    all_instances_tokens = list(set(all_instances_tokens))

    if dataset == 'nuscenes':
        meta_data_root = NUSC_OUT_DIR
    elif dataset == 'lyft':
        meta_data_root = LYFT_OUT_DIR
    elif dataset == 'waymo':
        meta_data_root = WAYMO_OUT_DIR
        
    n_cams = len(mminfo['cams'])
    
    # hard-encoded gaussian scales
    gaussian_scale_ada_fg = 0.0025
    gaussian_scale_ada_obj = 0.005
    gaussian_scale_ada_inpaint = 0.001
    gaussian_scale_ada_bg=[0.02, 0.001, 5] # 0m->10m, 0.25->0.001 (linearly decrease to fill the road)
    obj_downsample_ratio = 2
        
    # Step 2 : load all obj models
    n_frames = len(sample_tokens)
    all_objs_model = {}
    for _token in all_instances_tokens:
        obj_model_file = os.path.join(meta_data_root, 'objs_model', scene_token, f'{_token}.ply')
        obj_model_mask_file = os.path.join(meta_data_root, 'masks', 'obj_render_mask', scene_token, f'{_token}.npz')
        if (not os.path.exists(obj_model_file)) or (not os.path.exists(obj_model_mask_file)):
            all_objs_model[_token] = None
            continue
        pts_3d, pts_tex = load_texture_point_cloud_from_ply(obj_model_file)
        obj_mask = load_npz(obj_model_mask_file).toarray().reshape(n_frames, -1)
        # downsample NOTE: because obj model are construct from the raw size image
        obj_mask[:, ::obj_downsample_ratio] = False
        all_objs_model[_token] = [torch.tensor(pts_3d).to(device).float(), torch.tensor(pts_tex).to(device).float(), torch.tensor(obj_mask).to(device)]
        

    # Step 3 : collect gaussians
    for sample_token in sample_tokens:
        gs_file = os.path.join(meta_data_root, 'gaussians', f'{sample_token}.npz')
        if os.path.exists(gs_file):
            continue
        os.makedirs(os.path.dirname(gs_file), exist_ok=True)
        
        mminfo = all_mminfo[sample_token]
        frame_idx = mminfo['frame_idx']
    
        means3D = []        # save by float16
        rgbs = []           # save by uint8
        scales = []         # save by float16
        masks = []          # save by bool
    
        # Step 3.1 collect gaussians from image
        for sensor in mminfo['cams'].keys():
            img_file = mminfo['cams'][sensor]['data_path']
            dense_depth_file = os.path.join(meta_data_root, 'depths', 'dense_depth_SPNorm', sensor, f'{sample_token}.png')
            inpainted_img_file = os.path.join(meta_data_root, 'inpainted', 'img', sensor, f'{sample_token}.png')
            inpainted_depth_file = os.path.join(meta_data_root, 'inpainted', 'depth', sensor, f'{sample_token}.png')
            key_mask_file = os.path.join(meta_data_root, 'masks', 'key_mask2d', sensor, f'{sample_token}.png')
            
            key_mask = cv2.imread(key_mask_file)
            overlap_mask, egopart_mask, inpainted_render_mask = get_render_mask(key_mask)
            bg_mask = torch.tensor(key_mask[..., 0] == 0).to(device)
            
            img = cv2.imread(img_file)[:, :, ::-1]      # BGR to RGB
            inpainted_img = cv2.imread(inpainted_img_file)[:, :, ::-1]      
            h, w, _ = img.shape
            
            dense_depth = read_depth_map(dense_depth_file)
            dense_depth = torch.tensor(dense_depth).to(device)
            inpainted_depth = read_depth_map(inpainted_depth_file)
            inpainted_depth = torch.tensor(inpainted_depth).to(device)
            
            K = torch.tensor(mminfo['cams'][sensor]['cam_intrinsic']).float().to(device)
            T_cam2ego = torch.tensor(construct_T_from_vector(
                mminfo['cams'][sensor]['sensor2ego_translation'], 
                mminfo['cams'][sensor]['sensor2ego_rotation'])).float().to(device)
                
            # collect raw gaussians
            pts_3d, pts_2d = depth_to_ego_pts_mask(dense_depth, K, T_cam2ego, torch.tensor(~egopart_mask).to(device))
            _scales = torch.ones((len(pts_2d), )).to(K) * gaussian_scale_ada_fg
            _bg_m = bg_mask[pts_2d[:, 1], pts_2d[:, 0]]
            _bg_z = pts_3d[_bg_m, 2]
            _bg_z_scale = map_z2scale(_bg_z, 0, gaussian_scale_ada_bg[2], 
                                                gaussian_scale_ada_bg[0], 
                                                gaussian_scale_ada_bg[1])
            _scales[_bg_m] = _bg_z_scale
            
            # collect inpainted gaussians
            pts_3d = pts_3d.cpu().numpy()
            pts_2d = pts_2d.cpu().numpy()
            _scales = _scales.cpu().numpy()
            
            means3D.append(pts_3d.astype(np.float16))
            rgbs.append(img[pts_2d[:, 1], pts_2d[:, 0]])
            scales.append(_scales.astype(np.float16))
            masks.append(~overlap_mask[pts_2d[:, 1], pts_2d[:, 0]])
            
            # collect inpainted gaussians
            pts_3d, pts_2d = depth_to_ego_pts_mask(inpainted_depth, K, T_cam2ego, torch.tensor(inpainted_render_mask).to(device))
            _scales = torch.ones((len(pts_2d), )).to(K) * gaussian_scale_ada_inpaint
            
            pts_3d = pts_3d.cpu().numpy()
            pts_2d = pts_2d.cpu().numpy()
            _scales = _scales.cpu().numpy()
            
            means3D.append(pts_3d.astype(np.float16))
            rgbs.append(inpainted_img[pts_2d[:, 1], pts_2d[:, 0]])
            scales.append(_scales.astype(np.float16))
            masks.append(np.ones(len(pts_2d), dtype=bool))

        # Step 3.2 collect objs gaussians    
        T_local2egos = torch.tensor(mminfo['T_local2egos']).to(device).float()       # (N, 4, 4)
        for i, _token in enumerate(mminfo['instance_tokens']):
            if all_objs_model[_token] is None: continue
            
            pts_3d, pts_tex, obj_mask = all_objs_model[_token]
            pts_3d = pts_3d[obj_mask[frame_idx]]
            pts_tex = pts_tex[obj_mask[frame_idx]]
            T_local2ego = T_local2egos[i]
            
            pts_3d = (cart2homo(pts_3d) @ T_local2ego.T)[:, :3]
            
            means3D.append(pts_3d.cpu().numpy().astype(np.float16))
            rgbs.append(pts_tex.cpu().numpy().astype(np.uint8))
            scales.append(np.ones(len(pts_3d), dtype=np.float16) * gaussian_scale_ada_obj)
            masks.append(np.ones(len(pts_3d), dtype=bool))

        # Step 3.3 collect blind areas
        blind_area_file = os.path.join(meta_data_root, 'inpainted', 'blind_area_pts', f'{sample_token}.ply')
        hole_pts_3d, hole_pts_tex = load_texture_point_cloud_from_ply(blind_area_file)
    
        means3D.append(hole_pts_3d.astype(np.float16))
        rgbs.append(hole_pts_tex.astype(np.uint8))
        scales.append((np.ones(len(hole_pts_3d)) * gaussian_scale_ada_bg[0]).astype(np.float16))
        masks.append(np.ones(len(hole_pts_3d), dtype=bool))

        # Step 3.4 collect lidar
        lidar_path = mminfo['lidar_path']
        if dataset in ['nuscene', 'lyft']:
            lidar_pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        elif dataset in ['waymo']:
            lidar_pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 6)[:, :3]
        lidar_pts = lidar_pts[:, :3].astype(np.float16)

        # Step 3.5 save
        means3D = np.concatenate(means3D, axis=0)
        rgbs = np.concatenate(rgbs, axis=0)
        scales = np.concatenate(scales, axis=0)
        masks = np.concatenate(masks, axis=0)
        
        np.savez_compressed(gs_file, means3D=means3D, rgbs=rgbs, scales=scales, masks=masks, lidar_pts=lidar_pts)
        
    # Done and log
    open(os.path.join(log_dir, scene_token), 'w').close()
    torch.cuda.empty_cache()
    

if __name__ == '__main__':
    for dataset in [
        'nuscenes', 
        'lyft', 
        'waymo'
        ]:
        scene_sample_mapping = pickle.load(open(os.path.join('./devkits', f'{dataset}_scene_samples.pkl'), 'rb'))
        
        undo_scene_tokens = list(scene_sample_mapping.keys())
        
        pb = tqdm.tqdm(total=len(undo_scene_tokens), leave=True, desc=f'Pregenerating Gaussian Map for {dataset}...')
        for i in range(len(undo_scene_tokens)):
            run_one_sample((dataset, undo_scene_tokens[i], scene_sample_mapping[undo_scene_tokens[i]], torch.device(f'cuda:0')))
            pb.update()
        pb.close()
                        
               