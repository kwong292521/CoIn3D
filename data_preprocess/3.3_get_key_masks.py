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
import torch.nn.functional as F

import open3d as o3d
from scipy.sparse import csr_matrix, coo_matrix, save_npz, load_npz

from scipy.ndimage import label as label_connected_components

# ! parameters
# OVERLAP_REPROJ_DEPTH_THRES = 0.1
OVERLAP_REPROJ_DEPTH_THRES = 999999
# ratio = kernel_size / sqrt(fg_area) 
EDGE_ERODE_KERNEL_SIZE_RATIO = 0.3
EDGE_DILATE_KERNEL_SIZE_RATIO = 0.1

# MESH_DEPTH_RENDER_DOWNSAMPLE = 2
MESH_DEPTH_RENDER_DOWNSAMPLE = 1

# ! utils
def merge_meshes_for_batch_rendering(mesh_list):
    verts = [m.verts_list()[0] for m in mesh_list]
    faces = [m.faces_list()[0] for m in mesh_list]
    return Meshes(verts=verts, faces=faces)
    

def extract_edges_from_mask(mask, erode_kernel_size=5, dilate_kernel_size=5, erode_keep_upside_only=True):
    mask = mask.astype(np.uint8) * 255
    
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, erode_kernel, iterations=1)
    
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    
    # edges = dilated_mask - eroded_mask
    eroded_edges = mask - eroded_mask
    dilated_edges = dilated_mask - mask
    
    # edges = edges!=0
    eroded_edges = eroded_edges!=0
    dilated_edges = dilated_edges!=0
    
    if erode_keep_upside_only:
        uv = np.argwhere(eroded_edges)[:, ::-1]
        if len(uv) != 0:
            max_u_idx, min_u_idx = np.argmax(uv[:, 0]), np.argmin(uv[:, 0])
            used_v = int((uv[max_u_idx, 1] + uv[min_u_idx, 1]) / 2)
            eroded_edges[used_v:] = False
    
    edges = eroded_edges | dilated_edges
    
    return edges, eroded_edges, dilated_edges


def load_ply_as_mesh(ply_file, device):
    mesh_o3d = o3d.io.read_triangle_mesh(ply_file)
    
    # o3d to pytorch3d
    verts = torch.tensor(mesh_o3d.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh_o3d.triangles, dtype=torch.int64, device=device)
    
    return Meshes(verts=[verts], faces=[faces])
    
        
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
    
    # 应用变换 [V,4] @ [4,4].T -> [V,3]
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
        cull_backfaces=True,
        # bin_size=256
        # max_faces_per_bin=100000,
    )
    
    #=== init renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings),
        shader=DepthShader(device=device)  
    )

    return renderer


def _render_mask(mask, img, color, alpha=0.5):
    _m = mask.copy()
    mask = mask.astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask *= color.astype(np.uint8)
    img[_m] = img[_m] * (1 - alpha) + mask[_m] * alpha
    return img


def vis_masks(masks, img):
    img_vis = img.copy()
    for i, mask in enumerate(masks):
        # color = (np.random.rand(3) * 255).astype(np.uint8)
        color = np.array([0, 255, 0]).astype(np.uint8)
        img_vis = _render_mask(mask, img_vis, color)
    return img_vis


def postprocess_mask(mask, dilate_iter=3, kernel_size=5):
    """
    对二值 mask 进行最大连通域提取 + 膨胀处理。

    参数:
        mask: (H, W) 布尔值或 0/1 的 mask
        dilate_iter: 膨胀迭代次数
        kernel_size: 膨胀核大小

    返回:
        processed_mask: (H, W) bool mask
    """
    # 1. 提取最大连通域
    labeled, num_features = label_connected_components(mask)
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)

    # 统计每个区域的面积（跳过 label=0 的背景）
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # 忽略背景
    max_label = np.argmax(counts)

    largest_component = (labeled == max_label)

    # 2. 膨胀处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    largest_component_uint8 = largest_component.astype(np.uint8)
    dilated = cv2.dilate(largest_component_uint8, kernel, iterations=dilate_iter)

    return dilated.astype(bool)


# ! main functions
def get_obj_and_overlap_mask(dataset, split, scene_token, devkit, mminfo, out_dir, device):
    mm_scene_objs = MMSceneObjects(devkit, mminfo, scene_token, dataset, use_sweeps=False)
    sample_tokens = mm_scene_objs.sample_tokens
    
    pb = tqdm.tqdm(total=len(sample_tokens), leave=True, desc=f'getting obj and overlap mask {dataset} => {split} => {scene_token}...')
    
    overlap_check_map = {
        'waymo': {
            'CAM_FRONT': None,
            'CAM_FRONT_LEFT': ['CAM_FRONT@l'],      # NOTE: this denote use left-side overlap pixel of cam_front to replace corresponding pixels in front_left
            'CAM_FRONT_RIGHT': ['CAM_FRONT@r'],
            'CAM_SIDE_LEFT': ['CAM_FRONT_LEFT@l'],
            'CAM_SIDE_RIGHT': ['CAM_FRONT_RIGHT@r']
        },
        'nuscenes': {
            'CAM_FRONT': None,
            'CAM_FRONT_LEFT': ['CAM_FRONT@l'],
            'CAM_FRONT_RIGHT': ['CAM_FRONT@r'],
            'CAM_BACK': None,
            'CAM_BACK_LEFT': ['CAM_FRONT_LEFT@l', 'CAM_BACK@r'],
            'CAM_BACK_RIGHT': ['CAM_FRONT_RIGHT@r', 'CAM_BACK@l']
        },
        'lyft': {
            'CAM_FRONT': None,
            'CAM_FRONT_LEFT': ['CAM_FRONT@l'],
            'CAM_FRONT_RIGHT': ['CAM_FRONT@r'],
            'CAM_BACK': None,
            'CAM_BACK_LEFT': ['CAM_FRONT_LEFT@l', 'CAM_BACK@r'],
            'CAM_BACK_RIGHT': ['CAM_FRONT_RIGHT@r', 'CAM_BACK@l']
        }
    }
    
    
    # ! Step 1 : load all mesh and pointclouds model
    all_objs_token = [k for k, v in mm_scene_objs.instance_static.items()]
    # load dynamic obj mesh (in local coord)
    all_objs_mesh = {}
    all_objs_local_pts = {}
    all_objs_render_mask = {}
    for _token in all_objs_token:
        _mesh_file = os.path.join(out_dir, 'meshes', scene_token, f'{_token}_fixed.ply')
        _obj_file = os.path.join(out_dir, 'objs_model', scene_token, f'{_token}.ply')
        
        if (not os.path.exists(_mesh_file)) or (not os.path.exists(_obj_file)):
            all_objs_mesh[_token] = None
            all_objs_local_pts[_token] = None
            all_objs_render_mask[_token] = None
            continue
        
        _obj_mesh = load_ply_as_mesh(_mesh_file, device=device)
        all_objs_mesh[_token] = _obj_mesh
    
        _pts, _tex = load_texture_point_cloud_from_ply(_obj_file)
        if len(_pts) == 0:
            all_objs_mesh[_token] = None
            all_objs_local_pts[_token] = None
            all_objs_render_mask[_token] = None
            continue
            
        all_objs_local_pts[_token] = torch.from_numpy(_pts).to(device)
        all_objs_render_mask[_token] = torch.zeros((len(sample_tokens), len(_pts)), dtype=bool).to(device)
    
    # ! Step 2 :  render obj-wise mesh depth and get render mask for each frame
    for sample_token, sample_cam_objs in zip(sample_tokens, mm_scene_objs.cam_objs):
        # ! Step 2.1: collect all obj's mesh/pts in this frame @ global coord
        mesh_sample = []
        pts_sample = []
        token_sample = []
        for i in range(len(sample_cam_objs.lidar_boxes)):
            _token = sample_cam_objs.lidar_instance_tokens[i]
            if _token in all_objs_token:
                obj_mesh = all_objs_mesh[_token]
                obj_pts = all_objs_local_pts[_token]
                if obj_mesh is None: continue
                T_local2global = torch.from_numpy(sample_cam_objs.T_local2globals[i]).float().to(device)
                obj_mesh = transform_mesh(obj_mesh, T_local2global)
                obj_pts = (cart2homo_cuda(obj_pts) @ T_local2global.T)[:, :3]
                
                mesh_sample.append(obj_mesh)
                pts_sample.append(obj_pts)
                token_sample.append(_token)
                    
        # ! Step 2.2: collect all frames' dense depth from G2 (overlap area we only use one part to save memory)
        dense_depths = {}
        T_cams = {}
        for sensor in sample_cam_objs.cam_objs.keys():
            dense_depth_file = os.path.join(out_dir, 'depths', 'dense_depth_SPNorm', sensor, f'{sample_token}.png')
            dense_depth = read_depth_map(dense_depth_file)
            dense_depths[sensor] = torch.from_numpy(dense_depth).float().to(device)

            T_cams[sensor] = {
                'K': torch.from_numpy(sample_cam_objs.cam_objs[sensor].K).float().to(device),
                'T_cam2img': torch.from_numpy(sample_cam_objs.cam_objs[sensor].T_cam2img).float().to(device),
                'T_cam2ego': torch.from_numpy(sample_cam_objs.cam_objs[sensor].T_cam2ego).float().to(device),
                'T_ego2cam': torch.from_numpy(sample_cam_objs.cam_objs[sensor].T_ego2cam).float().to(device)
            }
                    
        # ! Step 2.3: get edge mask and other, render per camera obj mesh depth and create key mask
        for sensor in sample_cam_objs.cam_objs.keys():
            K = torch.from_numpy(sample_cam_objs.cam_objs[sensor].K).float().to(device)
            cam_w = sample_cam_objs.cam_objs[sensor].cam_w
            cam_h = sample_cam_objs.cam_objs[sensor].cam_h
            
            mask = torch.zeros((3, cam_h, cam_w), dtype=torch.uint8).to(device)
            
            # ! Step 2.3.1 get overlap render mask 
            cur_dense_depth = dense_depths[sensor]
            if overlap_check_map[dataset][sensor] is not None:
                for flag in overlap_check_map[dataset][sensor]:
                    tgt_sensor, tgt_side = flag.split('@')
                    tgt_depth = dense_depths[sensor]
                    _m = torch.zeros_like(tgt_depth).to(bool)
                    if tgt_side == 'l':
                        _m[:, :cam_w//2] = True
                    elif tgt_side == 'r':
                        _m[:, cam_w//2:] = True
                    _, pts_3d = get_pts2d3d_from_dense_depth_cuda(
                        tgt_depth,
                        T_cams[tgt_sensor]['K'],
                        _m)
                    
                    T_tgtcam2curimg = T_cams[sensor]['T_cam2img'] @ T_cams[sensor]['T_ego2cam'] @ T_cams[tgt_sensor]['T_cam2ego']
                    pts_img = cart2homo_cuda(pts_3d) @ T_tgtcam2curimg.T
                    pts_depth = pts_img[:, 2]
                    pts_uv = pts_img[:, 0:2] / pts_depth[:, None]
                    pts_uv = pts_uv.long()
                    _m = (pts_depth > 0) & \
                         (pts_uv[:, 0] >= 0) & (pts_uv[:, 0] < cam_w) & \
                         (pts_uv[:, 1] >= 0) & (pts_uv[:, 1] < cam_h)
                    pts_depth = pts_depth[_m]
                    pts_uv = pts_uv[_m]
                    
                    _m = torch.abs(cur_dense_depth[pts_uv[:,1], pts_uv[:,0]] - pts_depth) < OVERLAP_REPROJ_DEPTH_THRES
                    pts_uv = pts_uv[_m]
                    
                    mask[1, pts_uv[:, 1], pts_uv[:, 0]] = 255
            
            
            renderer = create_mesh_renderer(
                batch_K=K[None, ...] / MESH_DEPTH_RENDER_DOWNSAMPLE,
                img_w=int(cam_w / MESH_DEPTH_RENDER_DOWNSAMPLE),
                img_h=int(cam_h / MESH_DEPTH_RENDER_DOWNSAMPLE),
                device=device
            )
            
            mesh_cam = []
            pts_cam = []
            mesh_z_cam = []
            token_cam = []
            pts_idx = []
            
            # ! Step 2.3.2 collect objs mesh and pts 
            for mesh, pts, token in zip(mesh_sample, pts_sample, token_sample):
                # NOTE: we use mesh/pts before camera
                mesh2render = transform_mesh(mesh, sample_cam_objs.cam_objs[sensor].T_global2cam)
                mesh2render = filter_mesh_before_cam(mesh2render, depth_range=[1, 100])
                
                pts2render = (cart2homo_cuda(pts) @ torch.from_numpy(sample_cam_objs.cam_objs[sensor].T_global2cam).float().to(device).T)[:, :3]
                pts2render = pts2render[pts2render[:, 2] > 0]
                
                if len(mesh2render.verts_list()[0]) < 10: continue
                mesh_z = mesh2render.verts_list()[0][:, 2].mean().cpu().numpy()
                
                mesh_cam.append(mesh2render)
                mesh_z_cam.append(mesh_z)
                pts_cam.append(pts2render)
                token_cam.append(token)
                pts_idx.append(torch.arange(len(pts2render), dtype=torch.int64).to(device))
                
            # z_sort = np.argsort(mesh_z_cam)[::-1]
            z_sort = np.argsort(mesh_z_cam)
            mesh_cam = [mesh_cam[i] for i in z_sort]
            pts_cam = [pts_cam[i] for i in z_sort]
            mesh_z_cam = [mesh_z_cam[i] for i in z_sort]
            token_cam = [token_cam[i] for i in z_sort]
            pts_idx = [pts_idx[i] for i in z_sort]
            
            # ! Step 2.3.3 get key mask and obj render mask NOTE: from near to far in this version
            for i, mesh2render in enumerate(mesh_cam):
                mesh_depth = renderer(mesh2render)  # (1, h, w, 1)
                mesh_depth = mesh_depth[0, ..., 0]#.cpu().numpy()  # (h, w)
                mesh_depth[mesh_depth < 0] = 0
                mesh_depth[mesh_depth > 655.35] = 655
                mesh_depth = F.interpolate(mesh_depth[None, None, ...], size=(cam_h, cam_w), mode='nearest')[0, 0]
            
                _fg = (mesh_depth!=0)
                _fg_size = int(_fg.sum().sqrt())
                if _fg_size == 0: continue
                erode_kernel_size = int(max((_fg_size * EDGE_ERODE_KERNEL_SIZE_RATIO)//2*2+1, 3))
                dilate_kernel_size = int(max((_fg_size * EDGE_DILATE_KERNEL_SIZE_RATIO)//2*2+1, 3))
                
                _, _erode_edge, _dilate_edge = extract_edges_from_mask(_fg.cpu().numpy(), 
                                                                    erode_kernel_size=erode_kernel_size,
                                                                    dilate_kernel_size=dilate_kernel_size,
                                                                    erode_keep_upside_only=True)
                
                _erode_edge = torch.from_numpy(_erode_edge).to(device)
                _dilate_edge = torch.from_numpy(_dilate_edge).to(device)
                
                _existed_fg = (mask[0] != 0)
                _intersection = _existed_fg & _fg
                _existed_erode = (mask[0] == 250)
                _existed_erode_in_itsec = _existed_erode & _intersection               
                
                _fg[_existed_fg] = False   # foreground mask need occlude by near one
                _dilate_edge[_existed_fg] = False # dilate edge(handle depth distortion) need occlude by near one
                mask[0, _fg] = i+1
                mask[0, _dilate_edge] = 125
                
                # for the erode edge(used to define inpainted area to construct gaussian 3d), edge locate in the intersection between current foreground and existed foreground should be delete
                _erode_edge[_intersection] = False                
                mask[0, _erode_edge] = 250
                mask[0, _existed_erode_in_itsec] = 1
                
                # ! get obj render mask
                pts2render = pts_cam[i]
                _idx = pts_idx[i]
                
                img_pts = pts2render @ torch.from_numpy(sample_cam_objs.cam_objs[sensor].K).float().to(device).T
                pts_depth = img_pts[:, 2]
                pts_uv = img_pts[:, 0:2] / pts_depth[:, None]
                pts_uv = pts_uv.long()
                
                _m = (pts_depth > 0) & \
                     (pts_uv[:, 0] >= 0) & (pts_uv[:, 0] < cam_w) & \
                     (pts_uv[:, 1] >= 0) & (pts_uv[:, 1] < cam_h)

                _blind_area_m = (pts_depth > 0) & (pts_uv[:, 1] > cam_h)
                _blind_area_idx = _idx[_blind_area_m]
                all_objs_render_mask[token_cam[i]][sample_tokens.index(sample_token), _blind_area_idx] = True

                if ~torch.any(_m): continue
                pts_depth = pts_depth[_m]
                pts_uv = pts_uv[_m]
                _idx = _idx[_m]
                
                # pts faces to camera
                _m = torch.abs(mesh_depth[pts_uv[:, 1], pts_uv[:, 0]] - pts_depth) < 0.1
                if ~torch.any(_m): continue
                pts_depth = pts_depth[_m]
                pts_uv = pts_uv[_m]
                _idx = _idx[_m]
                
                # pts occlude in raw scene & pts on the dilated edge
                _m = ((cur_dense_depth[pts_uv[:, 1], pts_uv[:, 0]] - pts_depth) < 0) | \
                     (mask[0, pts_uv[:, 1], pts_uv[:, 0]] == 125)
                if ~torch.any(_m): continue
                pts_depth = pts_depth[_m]
                pts_uv = pts_uv[_m]
                _idx = _idx[_m]
                
                # downsample pts according to the unique uv
                tmp = torch.zeros((cam_h, cam_w), dtype=torch.int64, device=device) - 1
                tmp[pts_uv[:, 1], pts_uv[:, 0]] = _idx
                _idx = torch.unique(tmp)[1:]
                
                # refresh render mask in this frame(sample)   P.S. render_mask shape(n_sample, n_pts)
                all_objs_render_mask[token_cam[i]][sample_tokens.index(sample_token), _idx] = True
                

            # save key mask and render mask results
            mask_file = os.path.join(out_dir, 'masks', 'key_mask2d', sensor, f'{sample_token}.png')
            os.makedirs(os.path.dirname(mask_file), exist_ok=True)
            cv2.imwrite(mask_file, mask.cpu().numpy().transpose(1, 2, 0))
            
        pb.update()
    pb.close()
    
    # save obj render mask  NOTE: use sparse matrix to save (too many pts will raise libpng warning: Image width exceeds user limit in IHDR)
    for _token, mask in all_objs_render_mask.items():
        if mask is None: continue        
        mask_file = os.path.join(out_dir, 'masks', 'obj_render_mask', scene_token, f'{_token}.npz')
        os.makedirs(os.path.dirname(mask_file), exist_ok=True)
        mask = mask.cpu().numpy()
        mask = csr_matrix(mask.reshape(-1))
        save_npz(mask_file, mask)
        

def get_zits_inpaint_mask(dataset, split, scene_token, devkit, mminfo, out_dir):
    mm_scene_objs = MMSceneObjects(devkit, mminfo, scene_token, dataset, use_sweeps=False)
    mminfo_token2idx = {_info['token']: idx for idx, _info in enumerate(mminfo)}
    sample_tokens = get_sample_tokens_from_scene(devkit, scene_token)

    pb = tqdm.tqdm(total=len(sample_tokens), leave=True, desc=f'getting zits inpaint mask {dataset} => {split} => {scene_token}...')
    for frame_idx, sample_token in enumerate(sample_tokens):
        _info = mminfo[mminfo_token2idx[sample_token]]
        for sensor in _info['cams'].keys():    
            key_mask_file = os.path.join(out_dir, 'masks', 'key_mask2d', sensor, f'{sample_token}.png')
            assert os.path.exists(key_mask_file)
            key_mask = cv2.imread(key_mask_file)
            key_mask_size = key_mask.shape[0] * key_mask.shape[1]
            
            # NOTE: refresh the 125 for debug
            _m = key_mask[..., 2] == 125
            key_mask[_m, 2] = 0
            
            cam_objs = mm_scene_objs.cam_objs[frame_idx].cam_objs
            proj_box2d = cam_objs[sensor]['proj_box2d']
            
            for _box in proj_box2d:
                _box_size = (_box[3] - _box[1]) * (_box[2] - _box[0])
                if _box_size / key_mask_size < 0.5:
                    key_mask[_box[1]:_box[3], _box[0]:_box[2], 2] = 125
            
            cv2.imwrite(key_mask_file, key_mask)
        
        pb.update()
    pb.close()
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta Data Construction Pipeline.")
    parser.add_argument('--dataset', choices=['nuscenes', 'lyft', 'waymo'], default='lyft', help="Dataset to construct.")
    parser.add_argument('--split', choices=['train', 'val'], default='val', help="Dataset to construct.")
    args = parser.parse_args()
    
    dataset = args.dataset
    split = args.split
    device = 'cuda'
    
    devkit, mminfo, out_dir = load_dataset_devkit(dataset, split)
    _metadata = mminfo['metadata']
    mminfo = mminfo['infos']
    
    all_scene_tokens = get_scenes_from_mminfo(mminfo)
    n_scenes = len(all_scene_tokens)
    
    pb = tqdm.tqdm(total=len(all_scene_tokens), leave=True, desc=f'generating key masks : {dataset} => {split}')
    for i, scene_token in enumerate(all_scene_tokens):
        
        get_obj_and_overlap_mask(dataset, split, scene_token, devkit, mminfo, out_dir, device)
        get_zits_inpaint_mask(dataset, split, scene_token, devkit, mminfo, out_dir)
        
        torch.cuda.empty_cache()
        pb.update()
    pb.close()
    