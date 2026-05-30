import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import tqdm
import pickle
import argparse

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from common_utils import *

from scipy.spatial import ConvexHull

from __PATHS__ import *

DEPTH_MATCH_ERROR = 0.1
DIST2SEARCH_THRES = 0.25    # NOTE: 两帧之间位移超过这个值才去寻找inpaint的像素

def sample_points_in_convex_hull(pts_hole_xy, grid_spacing=0.1):
    """
    找到凸包并在其内部以规则间隔采样点。
    
    参数:
        pts_hole_xy: (N, 2) 的输入点
        grid_spacing: 网格间隔（越小点越密）

    返回:
        sampled_points: (M, 2) array，落在凸包内的采样点坐标
        hull_points: (K, 2) array，凸包的顶点坐标（顺时针）
    """
    # 1. 计算凸包
    hull = ConvexHull(pts_hole_xy)
    hull_vertices = pts_hole_xy[hull.vertices]

    # 2. 创建包含凸包的meshgrid
    min_xy = hull_vertices.min(axis=0)
    max_xy = hull_vertices.max(axis=0)

    x_vals = np.arange(min_xy[0], max_xy[0], grid_spacing)
    y_vals = np.arange(min_xy[1], max_xy[1], grid_spacing)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

    # 3. 用 OpenCV 判断哪些点在凸包内部
    # OpenCV 要求 int32 点（注意坐标顺序要是整数！但这里为了保精度用 float32）
    contour = hull_vertices.astype(np.float32).reshape((-1, 1, 2))

    inside_mask = np.array([
        cv2.pointPolygonTest(contour, (float(x), float(y)), measureDist=False) >= 0
        for x, y in grid_points
    ])
    sampled_points = grid_points[inside_mask]

    return sampled_points, hull_vertices




def run_one_scene(dataset, split, scene_token, devkit, mminfo, out_dir, device):
    print(f'running on {dataset} => {split} => {scene_token} ...')
    
    mminfo_token2idx = {_info['token']: idx for idx, _info in enumerate(mminfo)}
    sample_tokens = get_sample_tokens_from_scene(devkit, scene_token)
    
    # ! Step 1 : collect all info
    sequence_info = recursive_defaultdict()
    cam_names = None
    
    pb = tqdm.tqdm(total=len(sample_tokens), leave=True, desc=f'collect all info ...')
    for sample_idx, sample_token in enumerate(sample_tokens):
        _info = mminfo[mminfo_token2idx[sample_token]]
        if cam_names is None: cam_names = list(_info['cams'].keys())
        
        pts_inpainted = []      # NOTE: including pts_ego_part and pts_blind_area
        pts_ep_reverse_idx = []    # NOTE: indicated pts_ego_part's raw index in raw image
        pts_hole = []           # NOTE: blind-area corner pts
        for cam_idx, sensor in enumerate(_info['cams'].keys()): 
            key_mask_file = os.path.join(out_dir, 'masks', 'key_mask2d', sensor, f'{sample_token}.png')
            key_mask = cv2.imread(key_mask_file)
            ego_part_mask = (key_mask[..., 2] == 250)

            sequence_info[sample_token][sensor]['key_mask'] = torch.from_numpy(key_mask[..., 0]).to(device)
            sequence_info[sample_token][sensor]['ego_part_mask'] = torch.from_numpy(ego_part_mask).to(device)

            img_file = _info['cams'][sensor]['data_path']
            dense_depth_file = os.path.join(out_dir, 'depths', 'dense_depth_SPNorm', sensor, f'{sample_token}.png')
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            depth = read_depth_map(dense_depth_file)
            
            img = torch.from_numpy(img).to(device)
            depth = torch.from_numpy(depth).to(device)
            
            sequence_info[sample_token][sensor]['img'] = img
            sequence_info[sample_token][sensor]['depth'] = depth
            
            # calibrations
            K = torch.tensor(_info['cams'][sensor]['cam_intrinsic']).float().to(device)
            T_cam2img = torch.zeros((3, 4)).float().to(device)
            T_cam2img[:3, :3] = K
            sequence_info[sample_token][sensor]['K'] = K
            sequence_info[sample_token][sensor]['T_cam2img'] = T_cam2img
            T_cam2ego = construct_T_from_vector(_info['cams'][sensor]['sensor2ego_translation'], _info['cams'][sensor]['sensor2ego_rotation'])
            T_ego2global = construct_T_from_vector(_info['cams'][sensor]['ego2global_translation'], _info['cams'][sensor]['ego2global_rotation'])
            T_cam2ego, T_ego2global = torch.from_numpy(T_cam2ego).float().to(device), torch.from_numpy(T_ego2global).float().to(device)
            T_cam2global = T_ego2global @ T_cam2ego
            T_global2cam = torch.linalg.inv(T_cam2global)
            sequence_info[sample_token][sensor]['T_cam2global'] = T_cam2global
            sequence_info[sample_token][sensor]['T_global2cam'] = T_global2cam
            sequence_info[sample_token][sensor]['T_ego2global'] = T_ego2global
            sequence_info[sample_token][sensor]['T_global2ego'] = torch.linalg.inv(T_ego2global)
              
            # ego-part pts3d
            ep_mask = sequence_info[sample_token][sensor]['ego_part_mask']
            # if ep_mask is not None: 
            if torch.any(ep_mask): 
                pts_2d_ep, pts_3d_ep = get_pts2d3d_from_dense_depth_cuda(depth, K, ep_mask)
                pts_3d_ep = (cart2homo_cuda(pts_3d_ep) @ T_cam2ego.T)[:, :3]
                pts_inpainted.append(pts_3d_ep)
                
                pts_ep_reverse_idx.append(
                    torch.cat([torch.ones((len(pts_2d_ep), 1), dtype=torch.int64).to(device) * sample_idx,
                               torch.ones((len(pts_2d_ep), 1), dtype=torch.int64).to(device) * cam_idx,
                               pts_2d_ep.long()], dim=1))    # (n_pts, 4)
                
                
            # generate bottom hole corners
            T_ego2cam = torch.linalg.inv(T_cam2ego)
            cam_H = T_ego2cam[1, 3].cpu().numpy()
            f = K[0, 0].cpu().numpy()
            cv = K[1, 2].cpu().numpy()
            _D = f * cam_H / (h - cv)
            
            pts_hole_cam = get_pts_2d_to_3d(
                pts_2d=np.array([[0, h-1],
                                [w-1, h-1]]),
                pts_depth=np.array([_D, _D]),
                K=K.cpu().numpy()
            )
            pts_hole_ego = (cart2homo(pts_hole_cam) @ T_cam2ego.cpu().numpy().T)[:, :3]
            pts_hole.append(pts_hole_ego)
                        
        pts_hole = np.concatenate(pts_hole, axis=0)     # (n_cam*2, 3) @ ego coord

        # generate anchor pts in hole
        pts_blind_area, _ = sample_points_in_convex_hull(pts_hole[:, :-1], grid_spacing=0.02)
        pts_blind_area = np.concatenate([pts_blind_area, np.zeros((len(pts_blind_area), 1))], axis=1)   # NOTE: these pts are on the ground
        pts_inpainted.append(torch.from_numpy(pts_blind_area).float().to(device))
        n_pts_blind_area = len(pts_blind_area)

        # concatenate pts2inpainted
        pts_inpainted = torch.cat(pts_inpainted, dim=0)
        n_pts_all = len(pts_inpainted)
        n_pts_ep = n_pts_all - n_pts_blind_area
        
        if len(pts_ep_reverse_idx) != 0: 
            pts_ep_reverse_idx = torch.cat(pts_ep_reverse_idx, dim=0) 
        
        sequence_info[sample_token]['pts_inpainted'] = pts_inpainted
        sequence_info[sample_token]['pts_inpainted_allo_mask'] = torch.zeros((len(pts_inpainted)), dtype=bool, device=device)
        sequence_info[sample_token]['pts_inpainted_allo_tex'] = torch.zeros((len(pts_inpainted), 3), dtype=torch.uint8, device=device)
        
        sequence_info[sample_token]['pts_ep_reverse_idx'] = pts_ep_reverse_idx
        sequence_info[sample_token]['n_pts_ep'] = n_pts_ep

        pb.update()
    pb.close()    
    

    # ! Step 2 : we first need to inpaint ego part pts
    len_seq = len(sample_tokens)
    pb = tqdm.tqdm(total=len_seq, leave=True, desc=f'allocate color for each pts_inpainted(EGO-PART) in each frame...')
    for i, src_sample in enumerate(sample_tokens):
        if dataset == 'waymo': break
        
        pts_ep_reverse_idx = sequence_info[src_sample]['pts_ep_reverse_idx']
        n_pts_ep = sequence_info[src_sample]['n_pts_ep']
        pts_inpainted = sequence_info[src_sample]['pts_inpainted'][:n_pts_ep]
        pts_inpainted_allo_mask = sequence_info[src_sample]['pts_inpainted_allo_mask'][:n_pts_ep]
        pts_inpainted_allo_tex = sequence_info[src_sample]['pts_inpainted_allo_tex'][:n_pts_ep]
        
        src_ego2global_translation = sequence_info[src_sample][cam_names[0]]['T_ego2global'][:3, 3]
        
        # Step 2.1 : begin to inpainting
        for j in range(len_seq//2):
            # forward check and backward check  (including cur frame, 因为凸包内的点可能包含一些不是盲区的)
            for _sign in [1, -1]:
                tgt_idx = i + j*_sign
                if tgt_idx not in list(range(len_seq)): continue
                
                tgt_sample = sample_tokens[tgt_idx]
                tgt_ego2global_translation = sequence_info[tgt_sample][cam_names[0]]['T_ego2global'][:3, 3]
                src2tgt_dist = torch.linalg.norm(src_ego2global_translation - tgt_ego2global_translation)
                if src2tgt_dist < DIST2SEARCH_THRES: continue
                
                for tgt_sensor in cam_names:
                    _check_idx = torch.where(~pts_inpainted_allo_mask)[0]
                    if len(_check_idx) == 0: break
                    
                    tgt_img = sequence_info[tgt_sample][tgt_sensor]['img']
                    tgt_depth = sequence_info[tgt_sample][tgt_sensor]['depth']
                    tgt_mask = sequence_info[tgt_sample][tgt_sensor]['ego_part_mask']
                
                    _pts = pts_inpainted[_check_idx].clone()
                    T_srcego2tgtimg = sequence_info[tgt_sample][tgt_sensor]['T_cam2img'] @ sequence_info[tgt_sample][tgt_sensor]['T_global2cam'] @ sequence_info[src_sample][cam_names[0]]['T_ego2global']
                        
                    _pts = cart2homo_cuda(_pts) @ T_srcego2tgtimg.T
                    _pts_z = _pts[:, 2]

                    _z_m = _pts_z > 0
                    if torch.all(~_z_m): continue
                    _pts = _pts[_z_m]
                    
                    _pts_z = _pts[:, 2]
                    _pts = (_pts[:, :2] / _pts[:, 2:3]).long()
                    
                    _inside_m = (_pts[:, 0] >= 0) & (_pts[:, 0] < tgt_img.shape[1]) & \
                                (_pts[:, 1] >= 0) & (_pts[:, 1] < tgt_img.shape[0])
                    if torch.all(~_inside_m): continue
                    _pts = _pts[_inside_m]
                    _pts_z = _pts_z[_inside_m]
                    
                    # _inbg_m = ~tgt_mask[_pts[:, 1], _pts[:, 0]] if tgt_mask is not None else torch.ones_like(_pts_z, dtype=bool)
                    _inbg_m = ~tgt_mask[_pts[:, 1], _pts[:, 0]] if torch.any(tgt_mask) else torch.ones_like(_pts_z, dtype=bool)
                    _match_m = torch.abs(_pts_z - tgt_depth[_pts[:, 1], _pts[:, 0]]) < DEPTH_MATCH_ERROR
                    _match_m = _match_m & _inbg_m
                    _match_m = _inbg_m
                    if torch.all(~_match_m): continue
                    
                    _pts = _pts[_match_m]
                    # _pts_z = _pts_z[_match_m]
                    
                    # allocate color
                    _alloc_idx = _check_idx[_z_m][_inside_m][_match_m]
                    pts_inpainted_allo_tex[_alloc_idx] = tgt_img[_pts[:, 1], _pts[:, 0]]
                    pts_inpainted_allo_mask[_alloc_idx] = True

        # Step 2.2 : refresh image and ego-part mask of each frame
        for sample_idx in torch.unique(pts_ep_reverse_idx[:, 0]):
            for cam_idx in torch.unique(pts_ep_reverse_idx[:, 1]):
                _m = (pts_ep_reverse_idx[:, 0] == sample_idx) & (pts_ep_reverse_idx[:, 1] == cam_idx)
                _pts_2d_ep = pts_ep_reverse_idx[_m, 2:]
                _allo_mask = pts_inpainted_allo_mask[_m]
                _allo_tex = pts_inpainted_allo_tex[_m]

                sequence_info[sample_tokens[sample_idx]][cam_names[cam_idx]]['img'][_pts_2d_ep[:, 1], _pts_2d_ep[:, 0]] = _allo_tex
                sequence_info[sample_tokens[sample_idx]][cam_names[cam_idx]]['ego_part_mask'][_pts_2d_ep[:, 1], _pts_2d_ep[:, 0]][_allo_mask] = False
        
        # Step 2.3 : refresh pts_inpainted
        sequence_info[src_sample]['pts_inpainted'][:n_pts_ep] = pts_inpainted
        sequence_info[src_sample]['pts_inpainted_allo_mask'][:n_pts_ep] = pts_inpainted_allo_mask
        sequence_info[src_sample]['pts_inpainted_allo_tex'][:n_pts_ep] = pts_inpainted_allo_tex        
                
        pb.update()
    pb.close()
    
    # Step 2.4* : visual inpainted results (we just need to save the inpainted part!!!)
    if dataset != 'waymo':
        for sample in sample_tokens:
            for sensor in cam_names:
                # if sequence_info[sample][sensor]['ego_part_mask'] is None: continue
                if ~torch.any(sequence_info[sample][sensor]['ego_part_mask']) : continue
                
                # _m = cv2.imread(os.path.join(out_dir, 'masks', 'ego_area_mask2d', sensor, f'{scene_token}.png'), cv2.IMREAD_GRAYSCALE) != 0
                _m = cv2.imread(os.path.join(out_dir, 'masks', 'key_mask2d', sensor, f'{sample_token}.png'))[..., 2] == 250
                inpainted_img = cv2.cvtColor(sequence_info[sample][sensor]['img'].cpu().numpy(), cv2.COLOR_RGB2BGR)
                inpainted_img[~_m] = 0
                
                inpainted_result_file = os.path.join(out_dir, 'inpainted', 'img', sensor, f'{sample}.png')
                os.makedirs(os.path.dirname(inpainted_result_file), exist_ok=True)
                cv2.imwrite(inpainted_result_file, inpainted_img)
    
    
    # ! Step 3 : we then inpaint the blind area pts and save all
    len_seq = len(sample_tokens)
    pb = tqdm.tqdm(total=len_seq, leave=True, desc=f'allocate color for each pts_inpainted(BLIND-AREA) in each frame...')
    for i, src_sample in enumerate(sample_tokens):
        n_pts_ep = sequence_info[src_sample]['n_pts_ep']
        pts_inpainted = sequence_info[src_sample]['pts_inpainted'][n_pts_ep:]
        pts_inpainted_allo_mask = sequence_info[src_sample]['pts_inpainted_allo_mask'][n_pts_ep:]
        pts_inpainted_allo_tex = sequence_info[src_sample]['pts_inpainted_allo_tex'][n_pts_ep:]
        
        src_ego2global_translation = sequence_info[src_sample][cam_names[0]]['T_ego2global'][:3, 3]
        
        # Step 3.1 : begin to inpainting
        for j in range(len_seq//2):
            # forward check and backward check  (including cur frame, 因为凸包内的点可能包含一些不是盲区的)
            for _sign in [1, -1]:
                tgt_idx = i + j*_sign
                if tgt_idx not in list(range(len_seq)): continue
                
                tgt_sample = sample_tokens[tgt_idx]
                tgt_ego2global_translation = sequence_info[tgt_sample][cam_names[0]]['T_ego2global'][:3, 3]
                src2tgt_dist = torch.linalg.norm(src_ego2global_translation - tgt_ego2global_translation)
                if src2tgt_dist < DIST2SEARCH_THRES: continue
                
                for tgt_sensor in cam_names:
                    _check_idx = torch.where(~pts_inpainted_allo_mask)[0]
                    if len(_check_idx) == 0: break
                    
                    tgt_img = sequence_info[tgt_sample][tgt_sensor]['img']
                    tgt_depth = sequence_info[tgt_sample][tgt_sensor]['depth']
                    # tgt_mask = sequence_info[tgt_sample][tgt_sensor]['ego_part_mask']
                    tgt_mask = sequence_info[tgt_sample][tgt_sensor]['key_mask']
                    tgt_mask = (tgt_mask >= 1) & (tgt_mask <= 125)
                    # if sequence_info[tgt_sample][tgt_sensor]['ego_part_mask'] is not None:
                    if torch.any(sequence_info[tgt_sample][tgt_sensor]['ego_part_mask']):
                        tgt_mask = tgt_mask | sequence_info[tgt_sample][tgt_sensor]['ego_part_mask']
                    
                
                    _pts = pts_inpainted[_check_idx].clone()
                    T_srcego2tgtimg = sequence_info[tgt_sample][tgt_sensor]['T_cam2img'] @ sequence_info[tgt_sample][tgt_sensor]['T_global2cam'] @ sequence_info[src_sample][cam_names[0]]['T_ego2global']
                        
                    _pts = cart2homo_cuda(_pts) @ T_srcego2tgtimg.T
                    _pts_z = _pts[:, 2]

                    _z_m = _pts_z > 0
                    if torch.all(~_z_m): continue
                    _pts = _pts[_z_m]
                    
                    _pts_z = _pts[:, 2]
                    _pts = (_pts[:, :2] / _pts[:, 2:3]).long()
                    
                    _inside_m = (_pts[:, 0] >= 0) & (_pts[:, 0] < tgt_img.shape[1]) & \
                                (_pts[:, 1] >= 0) & (_pts[:, 1] < tgt_img.shape[0])
                    if torch.all(~_inside_m): continue
                    _pts = _pts[_inside_m]
                    _pts_z = _pts_z[_inside_m]
                    
                    _inbg_m = ~tgt_mask[_pts[:, 1], _pts[:, 0]] if tgt_mask is not None else torch.ones_like(_pts_z, dtype=bool)
                    _match_m = torch.abs(_pts_z - tgt_depth[_pts[:, 1], _pts[:, 0]]) < DEPTH_MATCH_ERROR
                    _match_m = _match_m & _inbg_m
                    _match_m = _inbg_m
                    if torch.all(~_match_m): continue
                    
                    _pts = _pts[_match_m]
                    # _pts_z = _pts_z[_match_m]
                    
                    # allocate color
                    _alloc_idx = _check_idx[_z_m][_inside_m][_match_m]
                    pts_inpainted_allo_tex[_alloc_idx] = tgt_img[_pts[:, 1], _pts[:, 0]]
                    pts_inpainted_allo_mask[_alloc_idx] = True

        # # Step 3.2 : refresh pts_inpainted
        # sequence_info[src_sample]['pts_inpainted'][n_pts_ep:] = pts_inpainted
        # sequence_info[src_sample]['pts_inpainted_allo_mask'][n_pts_ep:] = pts_inpainted_allo_mask
        # sequence_info[src_sample]['pts_inpainted_allo_tex'][n_pts_ep:] = pts_inpainted_allo_tex

        # # Step 3.3 : extract full pts_inpainted
        # pts_inpainted = sequence_info[src_sample]['pts_inpainted']
        # pts_inpainted_allo_tex = sequence_info[src_sample]['pts_inpainted_allo_tex']
        
        # Step 3.4 : save result
        inpainted_pts_file = os.path.join(out_dir, 'inpainted', 'blind_area_pts', f'{src_sample}.ply')
        os.makedirs(os.path.dirname(inpainted_pts_file), exist_ok=True)
        save_texture_point_cloud_to_ply(pts_inpainted.cpu().numpy(), 
                                        pts_inpainted_allo_tex.cpu().numpy(), 
                                        inpainted_pts_file)
        
        pb.update()
    pb.close()
    
                        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta Data Construction Pipeline.")
    parser.add_argument('--dataset', choices=['nuscenes', 'lyft', 'waymo'], default='waymo', help="Dataset to construct.")
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
    
    pb = tqdm.tqdm(total=len(all_scene_tokens), leave=True, desc=f'inpainting ego and blind area : {dataset} => {split}')
    for i, scene_token in enumerate(all_scene_tokens):
  
        run_one_scene(dataset, split, scene_token, devkit, mminfo, out_dir, device)
        
        torch.cuda.empty_cache()
        pb.update()
    pb.close()
    