import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import torch.distributed as dist
import numpy as np
import copy

#! base
class BasePriorModulatedLayer(nn.Module):
    def __init__(self, c_feat, prior_cfg, where):
        super(BasePriorModulatedLayer, self).__init__()
        assert where in ['after_neck', 'after_backbone']
        
        self.c_feat = c_feat 
        self.prior_cfg = copy.deepcopy(prior_cfg)
        self.pm_cfg = self.prior_cfg['feat_modulate']

        self.modulate_flag = self.prior_cfg['feat2modulate'][where]
        self.cat_prior_flag = self.prior_cfg['feat2cat'][where]

    def forward(self, feat, prior):
        """
        feat: [B, c_feat, H, W]
        prior: [B, c_prior, H, W]
        """
        if self.modulate_flag:
            feat = self.feat_modulate(feat, prior)

        if self.cat_prior_flag:
            feat = torch.cat([feat, prior], dim=1)
            
        return feat
    
    
    def feat_modulate(self, feat, prior):
        return feat


#! Scale+Perspective Modulated Layers
class PM_InvFMul_GDGGPRAdd(BasePriorModulatedLayer):
    def __init__(self, c_feat, prior_cfg, where):
        super(PM_InvFMul_GDGGPRAdd, self).__init__(c_feat, prior_cfg, where)
        
        if self.modulate_flag:
            self.perspective_modulate_conv = nn.Sequential(
                nn.Conv2d(2+6, c_feat, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )

    def feat_modulate(self, feat, prior):
        # Step 1: scale modulate
        focal_idx, focal_c = self.prior_cfg['map_info']['inv_focal']
        inv_focal_map = prior[:, focal_idx:focal_idx+focal_c, :, :]
        feat = feat * inv_focal_map
        
        # Step 2: perspective modulate
        idxs = []
        for item in ['ground_depth', 'gd_grad', 'plucker_ray', 'plucker_moment']:
            idx, c = self.prior_cfg['map_info'][item]
            idxs += list(range(idx, idx+c))
        
        perspective_prior = prior[:, idxs, :, :]
        perspective_bias = self.perspective_modulate_conv(perspective_prior)
        feat = feat + perspective_bias
        
        return feat


    
#! Prior Generator
class PriorMapGenerator():
    @staticmethod
    def cart2homo(pts):
        return torch.cat([pts, torch.ones((pts.shape[0], 1)).to(pts)], axis=1)  
    
    @staticmethod
    def _decompose_extrinsic(T):
        R = T[:3, :3]
        t = T[:3, 3]
        angles_deg = Rotation.from_matrix(R).as_euler('XYZ', degrees=True)
        return angles_deg, t
    
    @staticmethod
    def _recompose_extrinsic(angles_deg, t):
        R = Rotation.from_euler('XYZ', angles_deg, degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
           
    def __init__(self, cfg, num_frame):
        self.cfg = cfg

        if dist.is_initialized():
            self.device = torch.device(f'cuda:{dist.get_rank()}')
        else:
            self.device = torch.device('cuda')
        
        self.num_frame = num_frame
        
        self.T_aligned = torch.tensor([[0, 0, 1, 0],
                              [-1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, 0, 1]], dtype=torch.float32, device=self.device)
        self.T_aligned_inv = torch.linalg.inv(self.T_aligned)
        

    def get_rz(self, T_cam2ego):
        T_cam2ego_alinged = T_cam2ego @ self.T_aligned_inv
        R = T_cam2ego_alinged[:3, :3].cpu().numpy()
        angles = Rotation.from_matrix(R).as_euler('XYZ', degrees=False)
        return angles[2]
    
    def extract_scale_from_aug(self, T_img_augs):
        Ms = T_img_augs[:, :, :2, :2]
        U, S, V = torch.linalg.svd(Ms)
        # R = U @ V.transpose(-1, -2)
        S = torch.diag_embed(S)
        return S
    
    def fit_plane(self, points):
        """
        fit ax + by + cz + d = 0
        set z = ax + by + d (c=-1)
        params:
            points: (B, 4, 3)
        """
        A = torch.column_stack((points[:, :2], torch.ones(len(points), dtype=points.dtype, device=points.device)))  # (x,y,1)
        b = points[:, 2:3]      # z
        x = torch.linalg.inv(A.T @ A) @ A.T @ b
        x = x.squeeze()
        return [x[0], x[1], -1, x[2]] 
    
                
    def prepare_ground_depths(self, uv1, T_img2cams, T_cam2unpertegos, depth_thres=60.):
        bs, N, h, w, _ = uv1.shape
        
        T_unpertego2cams = torch.linalg.inv(T_cam2unpertegos)
        ego_corners = torch.tensor([[1, 1, 0],
                                    [-1, -1, 0],
                                    [1, -1, 0],
                                    [-1, 1, 0]]).to(uv1)
        cam_corners = (self.cart2homo(ego_corners) @ T_unpertego2cams.transpose(2, 3))[..., :3]  # (n_pt, 4) @ (bs, N, 4, 4) -> (bs, N, n_pt, 3)
        
        #! ground depths
        # fit plane ax + by + cz + d = 0 (we define cam plane as y=ax+cz+b)
        A, B, C, D = [torch.zeros(bs, N, 1, 1).to(uv1) for _ in range(4)]
        for i in range(bs):
            for j in range(N):
                a, c, b, d = self.fit_plane(cam_corners[i, j, :, [0, 2, 1]])    # fit ax+cz+by+d=0, b=-1
                A[i, j], B[i, j], C[i, j], D[i, j] = a, b, c, d
    
        ray = torch.einsum('bnhwc,bncd->bnhwd', uv1, T_img2cams.transpose(2, 3))   # (bs, N, h, w, 3)
        ray_x, ray_y = ray[..., 0], ray[..., 1]     # (bs, N, h, w)
        ground_depths = - D  / (A * ray_x + B * ray_y + C)
        
        invalid_mask = (ground_depths <= 0) | (ground_depths > depth_thres) | torch.isnan(ground_depths)
        ground_depths[invalid_mask] = 0
        
        #! ground pointmap
        T_img2cams_homo = torch.eye(4)[None, None, :, :].repeat(bs, N, 1, 1).to(T_img2cams)
        T_img2cams_homo[:, :, :3, :3] = T_img2cams
        T_img2unpertegos = T_cam2unpertegos @ T_img2cams_homo
        uvd1 = torch.cat([uv1, torch.ones((bs, N, h, w, 1)).to(uv1)], dim=-1)
        uvd1[..., :-1] *= ground_depths[..., None]
        gd_pointmaps = torch.einsum('bnhwc,bncd->bnhwd', uvd1, T_img2unpertegos.transpose(2, 3))
        gd_pointmaps = gd_pointmaps[..., :2]      # we only need xy coord (z is 0 for flatten assumtion)
        gd_pointmaps[invalid_mask] = 0
        
        #! ground depth grad
        gd_grads = torch.zeros_like(ground_depths)
        gd_grads[:, :, 1:, :] = ground_depths[:, :, :-1, :] - ground_depths[:, :, 1:, :]
        _m = gd_grads > 0
        gd_grads[_m] = torch.log(1/gd_grads[_m] + 1) / 2
        gd_grads[~_m] = 0.
        
        _factor = self.cfg.get('depth_norm_factor', None)
        if _factor:
            ground_depths /= _factor
            gd_pointmaps /= _factor
        
        # (bs, N, h, w)
        return ground_depths, gd_pointmaps.permute(0, 1, 4, 2, 3), gd_grads


    def prepare_plucker_ray_maps(self, uv1, T_img2cams, T_cam2egos, test_mode):
        b, N, h, w, _ = uv1.shape

        if self.cfg.get('plucker_fb_flip', False) and (np.random.rand() < 0.5) and (not test_mode):
            T_cam2egos_aligned = T_cam2egos @ self.T_aligned_inv
            T_flipedego2ego = torch.tensor(self._recompose_extrinsic([0, 0, 180], [1.6, 0, 0])).to(T_cam2egos)
            T_ego2flipedego = torch.inverse(T_flipedego2ego)
            T_cam2egos_aligned = T_ego2flipedego @ T_cam2egos_aligned
            T_cam2egos = T_cam2egos_aligned @ self.T_aligned
            
        T_ego2cams = torch.linalg.inv(T_cam2egos)
        R_cam2ego = T_cam2egos[:, :, :3, :3]
        T_img2egoR = R_cam2ego @ T_img2cams
        # ego_origin_in_cam = T_ego2cams[:, :, :3, 3]     # (b, N, 3)
        
        rays = torch.einsum('bnhwc,bncd->bnhwd', uv1, T_img2egoR.transpose(2, 3))   # (b, N, h, w, 3)
        if self.cfg['plucker_normed']:
            rays /= torch.linalg.norm(rays, axis=-1)[..., None]
        
        if self.cfg.get('plucker_moment_origin', 'ego_in_cam') == 'ego_in_cam':
            ego_origin_in_cam = T_ego2cams[:, :, :3, 3]     # (b, N, 3)
            moments = torch.cross(ego_origin_in_cam[:, :, None, None, :], rays)
        elif self.cfg.get('plucker_moment_origin', 'ego_in_cam') == 'cam_in_ego':
            cam_origin_in_ego = T_cam2egos[:, :, :3, 3]     # (b, N, 3)
            moments = torch.cross(cam_origin_in_ego[:, :, None, None, :], rays)
        
        # plucker_ray_maps = torch.cat([moments, rays], axis=-1)  # (b, N, h, w, 6)
        # plucker_ray_maps = plucker_ray_maps.permute(0, 1, 4, 2, 3)
        # return plucker_ray_maps
        
        rays = rays.permute(0, 1, 4, 2, 3)
        moments = moments.permute(0, 1, 4, 2, 3)
        return rays, moments
        


    def prepare_fov_maps(self, uv1, Ks, T_img_augs, T_cam2egos):
        b, N, h, w, _ = uv1.shape
        aug_scales = self.extract_scale_from_aug(T_img_augs)
        
        Ks_aug = Ks.clone()
        Ks_aug[:, :, :2, :2] *= aug_scales
        
        Ks_inv = torch.linalg.inv(Ks_aug)
        
        rays = torch.einsum('bnhwc,bncd->bnhwd', uv1, Ks_inv.transpose(2, 3))
        x_fov_maps = torch.arctan2(rays[..., 0], torch.tensor(1).to(rays)) + torch.pi
        y_fov_maps = torch.arctan2(rays[..., 1], torch.tensor(1).to(rays)) + torch.pi
        
        if self.cfg.get('fov_map_with_rot', False):
            for i in range(b):
                for j in range(N):
                    rz = self.get_rz(T_cam2egos[i, j])  # rad
                    if abs(rz) > np.pi/2:
                        rz = rz - np.sign(rz) * np.pi
                    x_fov_maps[i, j] -= rz
        
        # x_fov_maps[x_fov_maps < 0] = 2*torch.pi + x_fov_maps[x_fov_maps < 0]
        # x_fov_maps[x_fov_maps > 2*torch.pi] = 2*torch.pi - x_fov_maps[x_fov_maps > 2*torch.pi]
        # y_fov_maps[y_fov_maps < 0] = 2*torch.pi + y_fov_maps[y_fov_maps < 0]
        # y_fov_maps[y_fov_maps > 2*torch.pi] = 2*torch.pi - y_fov_maps[y_fov_maps > 2*torch.pi]
        
        return x_fov_maps, y_fov_maps
        
    
    def prepare_focal_maps(self, uv1, Ks, T_img_augs):
        b, N, h, w, _ = uv1.shape
        aug_scales = self.extract_scale_from_aug(T_img_augs)
        
        Ks_aug = Ks.clone()
        Ks_aug[:, :, :2, :2] *= aug_scales
        
        _focal = Ks_aug[:, :, 0:1, 0:1]
        _focal /= self.cfg['focal_norm_factor']     
        
        if self.cfg.get('square_focal', False):
            _focal = _focal**2
        
        focal_map = torch.ones((b, N, h, w)).to(uv1)
        
        inv_focal_map = focal_map * (1/_focal)
        rescale_focal_map = focal_map * (_focal)
        
        return rescale_focal_map, inv_focal_map
                    
    
    def downsample_maps(self, prior_maps, ds, mode='mean'):
        """
        (b, n_cam, c_prior, h, w)
        """
        assert isinstance(ds, int)
        assert mode in ['min', 'max', 'mean']
        
        downsample_maps = []
        # downsample_grid_maps = []
        for prior_map in prior_maps:
            B, N, C, H, W = prior_map.shape
            prior_map = prior_map.view(B, N, C, H//ds, ds, W//ds, ds)
            prior_map = prior_map.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(B, N, C, H//ds, W//ds, -1)
            # downsample_grid_maps.append(prior_map)
            if mode == 'mean':
                ds_prior_map = getattr(torch, mode)(prior_map, dim=-1)
            else:
                ds_prior_map = getattr(torch, mode)(prior_map, dim=-1).values
                
            # process depth TODO: process all ground-related prior maps
            idxs = []
            if 'ground_depth' in self.cfg['map2use']:
                idxs.append(self.cfg['map_info']['ground_depth'][0])
            if 'gd_pointmap' in self.cfg['map2use']:
                idxs.append(self.cfg['map_info']['gd_pointmap'][0])
                idxs.append(self.cfg['map_info']['gd_pointmap'][0]+1)
            if 'gd_grad' in self.cfg['map2use']:
                idxs.append(self.cfg['map_info']['gd_grad'][0])
            if len(idxs) > 0:
                _m = prior_map[:, :, idxs, ...] != 0
                gd_prior = prior_map[:, :, idxs, ...]
                if mode == 'min':
                    tmp = torch.where(~_m, 999999*torch.ones_like(gd_prior), gd_prior)
                    ds_prior_map[:, :, idxs, ...] = torch.min(tmp, dim=-1).values
                elif mode == 'max':
                    ds_prior_map[:, :, idxs, ...] = torch.max(gd_prior, dim=-1).values
                elif mode == 'mean':
                    gd_prior = gd_prior.sum(dim=-1) / _m.sum(dim=-1)
                    gd_prior[torch.isnan(gd_prior)] = 0
                    ds_prior_map[:, :, idxs, ...] = gd_prior
            
            downsample_maps.append(ds_prior_map)
        
        return downsample_maps#, downsample_grid_maps
        
        
    def __call__(self, img_inputs, test_mode):
        """
        perception field encoding + ground depth encoding
        params:
            img_inputs: tuple, each is (b, n_cam*n_adj, ...) N = n_cam*n_adj,
                0: bNchw imgs ; 1: bN44 T_sensor2egos; 2: bN44 T_ego2globals; 3: bN33 Ks; 4: bN33 img_rot_aug; 5: bN3 img_trans_aug; 6: b44 BEV_Rt_aug
        rets:
            prior_maps:
                list, each for different timestamp  [cur_frame, [adj_frames, ...]
                each frame is (RAW_SCALE, 16x_SCALE, 32x_SCALE) 
                each scale is (b, n_cam, c_prior, h, w)
        """
        imgs, T_sensor2egos, T_ego2globals, Ks, img_rot_aug, img_trans_aug, bda = img_inputs
        
        b, N, c, h, w = imgs.shape
        n_cam = N // self.num_frame
                
        vv, uu = torch.meshgrid(
            torch.arange(h).to(imgs), torch.arange(w).to(imgs))
        uu = uu[None, None, ...].expand(b, N, -1, -1)       # (b, N, h, w)
        vv = vv[None, None, ...].expand(b, N, -1, -1)
    
        uv1 = torch.stack([uu, vv, torch.ones_like(uu)], axis=-1)   # (b, N, h, w, 3)
        
        T_img_augs = img_rot_aug.clone()
        T_img_augs[:, :, :2, 2] = img_trans_aug[:, :, :2].clone()
        
        T_cam2imgs = T_img_augs @ Ks      # (b, N, 3, 3)
        
        T_img2cams = torch.linalg.inv(T_cam2imgs)
        if self.cfg.get('use_unpert_ego', True):
            raise NotImplementedError   # for debug
        else:
            T_cam2unpertegos = T_sensor2egos
        
        maps_dict = {}
        if 'ground_depth' in self.cfg['map2use']:
            ground_depths, gd_pointmaps, gd_grads = self.prepare_ground_depths(uv1, T_img2cams, T_cam2unpertegos)
            maps_dict['ground_depth'] = ground_depths
            maps_dict['gd_pointmap'] = gd_pointmaps
            maps_dict['gd_grad'] = gd_grads
        if ('hori_fov' in self.cfg['map2use']) or ('vert_fov' in self.cfg['map2use']):
            x_fov_maps, y_fov_maps = self.prepare_fov_maps(uv1, Ks, T_img_augs, T_cam2unpertegos)
            maps_dict['hori_fov'] = x_fov_maps
            maps_dict['vert_fov'] = y_fov_maps
        if ('plucker_ray' in self.cfg['map2use']) or ('plucker_moment' in self.cfg['map2use']):
            plucker_rays, plucker_moments = self.prepare_plucker_ray_maps(uv1, T_img2cams, T_cam2unpertegos.clone(), test_mode)
            maps_dict['plucker_ray'] = plucker_rays
            maps_dict['plucker_moment'] = plucker_moments
        if ('focal' in self.cfg['map2use']) or ('inv_focal' in self.cfg['map2use']):
            focal_maps, inv_focal_maps = self.prepare_focal_maps(uv1, Ks, T_img_augs)
            maps_dict['focal'] = focal_maps
            maps_dict['inv_focal'] = inv_focal_maps
        
        prior_maps = []     # len=n_frame, each is (b, n_cam, c, h, w)
        for i in range(self.num_frame):
            prior_map = []
            for _name in self.cfg['map2use']:
                if self.cfg['map_info'][_name][1] == 1:
                    prior_map.append(maps_dict[_name][:, n_cam*i:n_cam*(i+1), :, :][:, :, None, :, :])
                else:
                    prior_map.append(maps_dict[_name][:, n_cam*i:n_cam*(i+1), :, :])
            prior_map = torch.cat(prior_map, dim=2)
            prior_maps.append(prior_map)
            
        
        return prior_maps
        

#! aux
def get_prior_modulated_module(pm_type):
    return globals()[pm_type]
    