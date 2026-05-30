import os
import numpy as np
import cv2
import tqdm
import time
from collections import defaultdict
from scipy.spatial.transform import Rotation
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from functools import partial

from pyquaternion import Quaternion
import torch

from gaussian3d_kernel.gaussian import GaussianRenderer

from typing import List


# ! =========== for Gaussian Rendering =============
def convert_to_gaussian_tensor(geo, tex, opacity_val=1.0, scale_val=0.01):
    """
    geo: (N, 3) 点云坐标
    tex: (N, 3) RGB颜色，范围可以是[0,255]或[0,1]
    opacity_val: 所有点默认不透明度值
    scale_val: 每个高斯球体的尺度（等比放缩）
    """
    N = geo.shape[0]
    assert geo.shape == tex.shape == (N, 3)

    # Normalize color if needed
    if tex.max() > 1.0:
        tex = tex / 255.0  # 把[0,255]变为[0,1]

    # Mean
    means3D = geo.float()  # (N, 3)

    # RGB
    rgbs = tex.float().clamp(0, 1)  # (N, 3)

    # Opacity: (N, 1)
    opacity = torch.full((N, 1), fill_value=opacity_val, dtype=torch.float32, device=geo.device)

    # Rotation: 单位四元数 (N, 4)
    rotations = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=geo.device).repeat(N, 1)

    # Scales: 各向同性球 (N, 3)
    scales = torch.full((N, 3), fill_value=scale_val, dtype=torch.float32, device=geo.device)

    # 拼接为 (N, 14)
    gaussians = torch.cat([means3D, rgbs, opacity, rotations, scales], dim=1)
    return gaussians  # shape: (N, 14)



class CameraOps:
    """
    Single Camera Operations
    In nuscenes:
        ego coord: x y z -> forward, left, up
        camera coord: x y z -> right, down, forward
        
        ego(camera): x(z) y(-x) z(-y)
        camera(ego): x(-y) y(-z) z(x)
    """
    def __init__(self, dataset):
        # 坐标轴的原点以及各个轴上的单位点 用于定义相机的位置和方向
        self.base_pts = np.array([[0, 0, 0],
                                  [1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
        
        self.base_pts_homo = np.concatenate([self.base_pts, np.ones((4, 1))], axis=1)
        
        self.dataset = dataset
        assert dataset in ['nuscenes'], f'dataset {dataset} not supported'
        
        self.set_world_cam_axis_alignment()    
        
            
    def set_world_cam_axis_alignment(self):
        """
        将轴向由world转换到cam的对齐旋转变换
        """
        if self.dataset == 'nuscenes':
            self.R_world2cam_axis_homo = np.array([[0, -1, 0, 0],
                                                   [0, 0, -1, 0],
                                                   [1, 0, 0, 0],
                                                   [0, 0, 0, 1]])
            self.R_world2cam_axis = self.R_world2cam_axis_homo[:3, :3]
            self.R_cam2world_axis_homo = self.R_world2cam_axis_homo.T
            self.R_cam2world_axis = self.R_world2cam_axis.T
            
    
    def estimate_rigid_transform(self, pts_A, pts_B):
        assert pts_A.shape == pts_B.shape
        N = pts_A.shape[0]

        # 计算中心
        centroid_A = np.mean(pts_A, axis=0)
        centroid_B = np.mean(pts_B, axis=0)

        # 去中心化
        AA = pts_A - centroid_A
        BB = pts_B - centroid_B

        # 计算协方差矩阵
        H = AA.T @ BB

        # 奇异值分解
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 修正反射（如果 det(R) = -1）
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        # 平移
        t = centroid_B - R @ centroid_A

        # 构造 4x4 变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T 
    
    
    def transmtx_to_worldpts(self, T_cam2world):
        return (self.base_pts_homo @ T_cam2world.T)[:, :3]
    
    def basepts_to_transmtx(self, pts_world):
        T_cam2world = self.estimate_rigid_transform(self.base_pts, pts_world)
        T_wolrd2cam = np.linalg.inv(T_cam2world)
        
        tmp = (self.base_pts_homo @ T_cam2world.T)[:, :3]
        assert np.allclose(tmp, pts_world), 'pts_world is not correctly setted'
        
        return T_cam2world, T_wolrd2cam
        
    def construct_camera(self, angles_deg, trans, euler_order='XYZ'):
        """
        params:
            angles_deg: (3,) 相机的旋转角度 分别是绕x, y, z轴的旋转角度
            trans: (3,) 相机的平移向量
            euler_order: 相机的旋转顺序 默认为XYZ 
        return:
            T_world2cam: (4, 4) 世界坐标系到相机坐标系的变换矩阵
        """
        assert len(angles_deg) == 3
        assert len(trans) == 3
        assert euler_order in ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']
        
        T_world2aligned = self.R_world2cam_axis_homo
        
        # 构造旋转矩阵（extrinsic）
        rot = Rotation.from_euler(euler_order, angles_deg, degrees=True)
        R_mat = rot.as_matrix()
        
        T_aligned2cam = np.eye(4)
        T_aligned2cam[:3, :3] = R_mat
        T_aligned2cam[:3, 3] = trans
        
        T_world2cam = T_aligned2cam @ T_world2aligned
        
        return T_world2cam
    
    
    def decompose_camera_extrinsic(self, T_world2cam, euler_order='XYZ'):
        """
        给定world2cam的变换 解构出camera的构造过程(绕轴的旋转和平移) camera的构造过程是在轴对齐后的世界坐标系下进行的
        params:
            T_world2cam: (4, 4) 世界坐标系到相机坐标系的变换矩阵
            euler_order: 相机的旋转顺序 默认为XYZ 
        return:
            angles_deg: (3,) 相机的旋转角度 分别是绕x, y, z轴的旋转角度
            trans: (3,) 相机的平移向量
        """
        assert T_world2cam.shape == (4, 4)
        
        # Step 1: 反求出 aligned2cam NOTE: 两个方式是等价的 第二个方式直观点
        # T_aligned2cam = T_world2cam @ self.R_cam2world_axis_homo
        T_aligned2cam = T_world2cam @ np.linalg.inv(self.R_world2cam_axis_homo)

        # Step 2: 分离旋转和平移
        R_mat = T_aligned2cam[:3, :3]
        trans = T_aligned2cam[:3, 3]
        
        # Step 3: 提取欧拉角（extrinsic，即绕世界坐标轴旋转）
        rot = Rotation.from_matrix(R_mat)
        angles_deg = rot.as_euler(euler_order, degrees=True)
        
        return angles_deg, trans


class NuscenesCameraGroups:
    """
    Nuscenes Camera Groups Setting of One Frame
    """
    def __init__(self, raw_cams):
        self.raw_cams = raw_cams
        self._cams_check_and_update()
        
        self.cam_ops = CameraOps('nuscenes')

    def _cams_check_and_update(self):
        for sensor in self.raw_cams.keys():
            assert self.raw_cams[sensor]['K'].shape == (3, 3)
            assert self.raw_cams[sensor]['T_ego2cam'].shape == (4, 4)
            assert isinstance(self.raw_cams[sensor]['img_h'], int)
            assert isinstance(self.raw_cams[sensor]['img_w'], int)
            
            # update more information
            self.raw_cams[sensor]['fu'] = self.raw_cams[sensor]['K'][0, 0]
            self.raw_cams[sensor]['fv'] = self.raw_cams[sensor]['K'][1, 1]
            self.raw_cams[sensor]['cu'] = self.raw_cams[sensor]['K'][0, 2]
            self.raw_cams[sensor]['cv'] = self.raw_cams[sensor]['K'][1, 2]
            self.raw_cams[sensor]['T_cam2ego'] = np.linalg.inv(self.raw_cams[sensor]['T_ego2cam'])
            self.raw_cams[sensor]['fovx_degree'] = 2 * np.arctan2(self.raw_cams[sensor]['img_w']/2, self.raw_cams[sensor]['fu']) / np.pi * 180
            self.raw_cams[sensor]['fovy_degree'] = 2 * np.arctan2(self.raw_cams[sensor]['img_h']/2, self.raw_cams[sensor]['fv']) / np.pi * 180
            
    @staticmethod
    def vis_surround_view(scene_imgs: List[np.ndarray], cam_names):
        
        
        # NOTE: nuscenes have the same image size
        h, w = scene_imgs[0].shape[0], scene_imgs[0].shape[1]
        
        # 创建空白大画布 (高度: 2*h + 3*padding, 宽度: 2*w + 3*padding)
        padding = 10  # 图像间间距
        canvas = np.zeros((4*h + 5*padding, 2*w + 3*padding, 3), dtype=np.uint8)
        
        # 提取各个视角图像 (按输入顺序)
        front = scene_imgs[cam_names.index('CAM_FRONT')]          # CAM_FRONT
        front_left = scene_imgs[cam_names.index('CAM_FRONT_LEFT')]     # CAM_FRONT_LEFT
        front_right = scene_imgs[cam_names.index('CAM_FRONT_RIGHT')]    # CAM_FRONT_RIGHT
        back = scene_imgs[cam_names.index('CAM_BACK')]           # CAM_BACK
        back_left = scene_imgs[cam_names.index('CAM_BACK_LEFT')]      # CAM_BACK_LEFT
        back_right = scene_imgs[cam_names.index('CAM_BACK_RIGHT')]     # CAM_BACK_RIGHT
        
        # 放置图像到对应位置 (按照你指定的布局)
        # 第一行 (FRONT)
        canvas[padding:h+padding, 
            padding+w//2:padding+w//2+w] = front
        
        # 第二行 (FRONT_LEFT 和 FRONT_RIGHT)
        canvas[h+2*padding:2*h+2*padding, 
            padding:padding+w] = front_left
        canvas[h+2*padding:2*h+2*padding, 
            padding+w+padding:padding+2*w+padding] = front_right
        
        # 第三行 (BACK_LEFT 和 BACK_RIGHT)
        canvas[2*h+3*padding:3*h+3*padding, 
            padding:padding+w] = back_left
        canvas[2*h+3*padding:3*h+3*padding, 
            padding+w+padding:padding+2*w+padding] = back_right
        
        # 第四行 (BACK)
        canvas[3*h+4*padding:4*h+4*padding, 
            padding+w//2:padding+w//2+w] = back
        
        return canvas


    def get_fov_and_cropping(self, cus, cvs, fus, fvs, imghs, imgws):
        # compute std fovs
        _L = cus
        _T = cvs
        _R = imgws - _L
        _B = imghs - _T
        
        resolution = [
            int(max(_T.max(), _B.max()) * 2),
            int(max(_L.max(), _R.max()) * 2)
        ]
        
        fovxs = 2 * torch.arctan(resolution[1] / 2 / fus)
        fovys = 2 * torch.arctan(resolution[0] / 2 / fvs)
    
        # compute crop range for each camera
        nframe = len(fus)
        crop_start = torch.zeros(nframe, 2, dtype=torch.int32)
        crop_end = torch.zeros(nframe, 2, dtype=torch.int32)
        _um = _L < _R
        _vm = _T < _B
        _um_idx = torch.where(_um)[0]
        _vm_idx = torch.where(_vm)[0]

        crop_start[_um_idx, 0] = resolution[1] - imgws[_um_idx]
        crop_start[_vm_idx, 1] = resolution[0] - imghs[_vm_idx]

        crop_end[..., 0] = crop_start[..., 0] + imgws
        crop_end[..., 1] = crop_start[..., 1] + imghs

        return fovxs, fovys, resolution, crop_start, crop_end
    
    
    def gen_cam_for_gaussian(self, trans_xyz=None, rot_xyz_deg=None, focals=None, img_sizes=None, fovs=None, ext_mode='O', int_mode='A'):
        n_cam = len(self.raw_cams)
        
        c2ws = torch.zeros((n_cam, 4, 4), dtype=torch.float32)
        cus = torch.zeros((n_cam,), dtype=torch.float32)
        cvs = torch.zeros((n_cam,), dtype=torch.float32)
        fus = torch.zeros((n_cam,), dtype=torch.float32)
        fvs = torch.zeros((n_cam,), dtype=torch.float32)
        imghs = torch.zeros((n_cam,), dtype=torch.int32)
        imgws = torch.zeros((n_cam,), dtype=torch.int32)
        
        for i, cam_name in enumerate(self.raw_cams.keys()):
            # extrinsic
            c2ws[i] = torch.from_numpy(self._cfg_func_extrinsic(cam_name, trans_xyz, rot_xyz_deg, mode=ext_mode)).float()
            
            # intrinsic
            fu, fv, cu, cv, img_w, img_h = self._cfg_func_intrinsic_fixed_size(cam_name, focals, fovs, mode=int_mode)
            fus[i] = fu
            fvs[i] = fv
            cus[i] = cu
            cvs[i] = cv
            imghs[i] = img_h
            imgws[i] = img_w
            
        # cropping to adapt omni-scene gaussian render kernel
        fovxs, fovys, resolution, crop_start, crop_end = self.get_fov_and_cropping(cus, cvs, fus, fvs, imghs, imgws)
        camera_args = {
            'resolution': resolution,
            'znear': 0.1,
            'zfar': 1000.0,
        }
        
        return c2ws, fovxs, fovys, camera_args, crop_start, crop_end
            
    
    
    def _cfg_func_extrinsic(self, cam_name, trans_xyz=None, rot_xyz_deg=None, mode='O'):
        assert mode in ['A', 'O', 'R']   # NOTE: 'A(absolue)' mean construct camera from input ; 'O(offset)' mean construct camera from (raw_cam+input) ; 'R' mean raw camera
    
        if mode == 'A':
            T_ego2cam = self.cam_ops.construct_camera(rot_xyz_deg[cam_name], trans_xyz[cam_name])
            T_cam2ego = np.linalg.inv(T_ego2cam)
        elif mode == 'O':
            rot_raw_deg, trans_raw = self.cam_ops.decompose_camera_extrinsic(self.raw_cams[cam_name]['T_ego2cam'])
            T_ego2cam = self.cam_ops.construct_camera(rot_raw_deg + np.array(rot_xyz_deg[cam_name]), 
                                                      trans_raw + np.array(trans_xyz[cam_name]))
            T_cam2ego = np.linalg.inv(T_ego2cam)
        elif mode == 'R':
            T_cam2ego = self.raw_cams[cam_name]['T_cam2ego']
        
        return T_cam2ego  
      
      
        
    def _cfg_func_intrinsic_fixed_size(self, cam_name, focals=None, fovxs=None, mode='A'):
        """
        We fix the imgsize all the time for simplicity
        params:
            focals / img_sizes / fovxs : dict ['cam_name'] = [_ux, _vy]
            for intrinsic, there are three variable fov / c / f: 
                1. fov=2*arctan(c/f) where c=size/2
                2. if two variables are given, the other will be fixed, and only two variables can be given
        rets:
            fu, fv, cu, cv, img_w, img_h
        """ 
        assert mode in ['A', 'O', 'R']   # NOTE: 'A(absolue)' mean construct camera from input ; 'O(offset)' mean construct camera from (raw_cam+input)
        
        # load raw cam intrinsic
        raw_fu = self.raw_cams[cam_name]['fu']
        raw_fv = self.raw_cams[cam_name]['fv']
        raw_cu = self.raw_cams[cam_name]['cu']
        raw_cv = self.raw_cams[cam_name]['cv']
        raw_img_w = self.raw_cams[cam_name]['img_w']
        raw_img_h = self.raw_cams[cam_name]['img_h']
        raw_fovx = self.raw_cams[cam_name]['fovx_degree']
        raw_fovy = self.raw_cams[cam_name]['fovy_degree']
        
        if mode == 'R':
            return raw_fu, raw_fv, raw_cu, raw_cv, raw_img_w, raw_img_h
        
        # ! only two variable can be given
        assert sum(x is not None for x in (focals, fovxs)) == 1, "We need and only need one variable to get the intrinsic under fixed img size"
        
        if focals is None:
            new_img_w = raw_img_w
            new_img_h = raw_img_h
            new_cu = new_img_w / 2
            new_cv = new_img_h / 2
            
            new_fovx = fovxs[cam_name][0] if mode == 'A' else raw_fovx + fovxs[cam_name][0]
            new_fu = new_cu / np.tan(new_fovx / 2 * np.pi / 180)
            new_fv = new_fu
            new_fovy = 2 * np.arctan(new_img_h / 2 / new_fv) / np.pi * 180
        elif fovxs is None:
            new_fu = focals[cam_name][0] if mode == 'A' else raw_fu + focals[cam_name][0]
            new_fv = focals[cam_name][1] if mode == 'A' else raw_fv + focals[cam_name][1]
            new_img_w = raw_img_w
            new_img_h = raw_img_h
            new_cu = new_img_w / 2
            new_cv = new_img_h / 2
            
            new_fovx = 2 * np.arctan(new_img_w / 2 / new_fu) / np.pi * 180
            new_fovy = 2 * np.arctan(new_img_h / 2 / new_fv) / np.pi * 180
            
        return new_fu, new_fv, new_cu, new_cv, new_img_w, new_img_h
        
        
class WaymoCameraGroups(NuscenesCameraGroups):
    def __init__(self, raw_cams):
        super().__init__(raw_cams)
        
    @staticmethod
    def vis_surround_view(scene_imgs: List[np.ndarray], cam_names):
        # NOTE: waymo have different img_h
        w = scene_imgs[0].shape[1]
        max_h = scene_imgs[0].shape[0]  # default is the cam_front
        
        canvas = np.zeros((max_h, 5*w, 3), dtype=np.uint8)
        
        front = scene_imgs[cam_names.index('CAM_FRONT')]          # CAM_FRONT
        front_left = scene_imgs[cam_names.index('CAM_FRONT_LEFT')]     # CAM_FRONT_LEFT
        front_right = scene_imgs[cam_names.index('CAM_FRONT_RIGHT')]    # CAM_FRONT_RIGHT
        side_left = scene_imgs[cam_names.index('CAM_SIDE_LEFT')]      # CAM_SIDE_LEFT
        side_right = scene_imgs[cam_names.index('CAM_SIDE_RIGHT')]     # CAM_SIDE_RIGHT
        
        side_h = side_left.shape[0]
        
        canvas[max_h-side_h:, :w] = side_left
        canvas[:, w:2*w] = front_left
        canvas[:, 2*w:3*w] = front
        canvas[:, 3*w:4*w] = front_right
        canvas[max_h-side_h:, 4*w:] = side_right
        
        return canvas
    
    
