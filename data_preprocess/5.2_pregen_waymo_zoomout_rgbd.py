set_cpu = True
import psutil
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
pid = os.getpid()
if set_cpu:
    cpu2use = 8
    cpu_scan_time = 5
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

import numpy as np
import cv2
import torch
import pickle
import tqdm
import copy
import time
from pyquaternion import Quaternion
from mmdet3d.gaussian3d_kernel.gaussian import GaussianRenderer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NuscenesCameraGroups:
    """
    Nuscenes Camera Groups Setting of One Frame
    """
    # T_cam2ego = T_cam2ego_aligned @ T_aligned : 分解成两次变换 第一次是对齐cam和ego的xyz轴 然后再平移旋转
    T_aligned = np.array([[0, 0, 1, 0],
                          [-1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, 0, 1]])
    T_aligned_inv = np.linalg.inv(T_aligned)
    
    def __init__(self, raw_cams, aug_ego2global, aug_cam2ego, aug_K, integrate_extrinsic_aug, custom_aug=None):
        self.raw_cams = raw_cams
        self._cams_check_and_update()
        self.new_cams = copy.deepcopy(self.raw_cams)
        
        self.aug_ego2global = aug_ego2global
        self.aug_cam2ego = aug_cam2ego
        self.aug_K = aug_K
        self.integrate_extrinsic_aug = integrate_extrinsic_aug
        
        self.custom_aug = custom_aug
        # if custom_aug is not None:
        #     assert custom_aug in ['lyft', 'waymo']

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
        
        fovxs = 2 * np.arctan(resolution[1] / 2 / fus)
        fovys = 2 * np.arctan(resolution[0] / 2 / fvs)
    
        # compute crop range for each camera
        nframe = len(fus)
        crop_start = np.zeros((nframe, 2), dtype=np.int32)
        crop_end = np.zeros((nframe, 2), dtype=np.int32)
        _um = _L < _R
        _vm = _T < _B
        _um_idx = np.where(_um)[0]
        _vm_idx = np.where(_vm)[0]

        crop_start[_um_idx, 0] = resolution[1] - imgws[_um_idx]
        crop_start[_vm_idx, 1] = resolution[0] - imghs[_vm_idx]

        crop_end[..., 0] = crop_start[..., 0] + imgws
        crop_end[..., 1] = crop_start[..., 1] + imghs

        return fovxs, fovys, resolution, crop_start, crop_end

    def gen_raw_cam_for_gaussian_with_focalaug(self, aug_cfg):
        n_cam = len(self.raw_cams)
        
        c2ws = np.zeros((n_cam, 4, 4), dtype=np.float32)
        cus = np.zeros((n_cam,), dtype=np.float32)
        cvs = np.zeros((n_cam,), dtype=np.float32)
        fus = np.zeros((n_cam,), dtype=np.float32)
        fvs = np.zeros((n_cam,), dtype=np.float32)
        imghs = np.zeros((n_cam,), dtype=np.int32)
        imgws = np.zeros((n_cam,), dtype=np.int32)
        
        for i, cam_name in enumerate(self.raw_cams.keys()):
            if aug_cfg[0] == 'ratio':
                ratio = np.random.uniform(aug_cfg[1], aug_cfg[2])
            elif aug_cfg == 'range':
                ratio = np.random.uniform(aug_cfg[1], aug_cfg[2]) / fus[i]
            self.new_cams[cam_name]['K'][:2, :2] *= ratio
            
            c2ws[i] = np.linalg.inv(self.raw_cams[cam_name]['T_ego2cam']).astype(np.float32)
            fus[i] = self.raw_cams[cam_name]['fu'] * ratio
            fvs[i] = self.raw_cams[cam_name]['fv'] * ratio
            cus[i] = self.raw_cams[cam_name]['cu']
            cvs[i] = self.raw_cams[cam_name]['cv']
            imghs[i] = self.raw_cams[cam_name]['img_h']
            imgws[i] = self.raw_cams[cam_name]['img_w']
            
        fovxs, fovys, resolution, crop_start, crop_end = self.get_fov_and_cropping(cus, cvs, fus, fvs, imghs, imgws)
        camera_args = {
            'resolution': resolution,
            'znear': 0.1,
            'zfar': 1000.0,
        }
        
        return c2ws, fovxs, fovys, camera_args, crop_start, crop_end


def save_depth_map(save_path, depth_map,
                   version='cv2', png_compression=3):
    # Convert depth map to a uint16 png
    depth_image = (depth_map * 256.0).astype(np.uint16)

    if version == 'cv2':
        ret = cv2.imwrite(save_path, depth_image, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])

        if not ret:
            raise RuntimeError('Could not save depth map')
    else:
        raise ValueError('Invalid version', version)


class WaymoDataset(Dataset):
    def __init__(self, scale=0.7):
        self.train_file = 'waymo_infos_train_bevdet_format_general_3cls_ds@4.pkl'
        self.meta_data_root = 'waymo/meta_data'
        with open(self.train_file, 'rb') as f:
            self.train_infos = pickle.load(f)['infos']
        
        # NOTE: filter out done samples
        tmp = []
        for _info in self.train_infos:
            sample_token = _info['token']
            check_file = os.path.join(self.meta_data_root, f'zoomout_rgbd_{scale}', 'img', 'CAM_FRONT', f'{sample_token}.jpg')
            if not os.path.exists(check_file):
                tmp.append(_info)
        print(f'all sample tokens: {len(self.train_infos)}')
        print(f'todo sample tokens: {len(tmp)}')
        self.train_infos = tmp
            
        self.cam_names = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_SIDE_LEFT', 'CAM_SIDE_RIGHT'
            ]
        self.scale_factor = scale
        self.max_pts = 4000000
        # self.input_size = (533, 800)        # raw size without sky crop
        # self.side_input_size = (369, 800)   # side camera size
        self.input_size = {}
        for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']:
            self.input_size[cam] = (533, 800)
        for cam in ['CAM_SIDE_LEFT', 'CAM_SIDE_RIGHT']:
            self.input_size[cam] = (369, 800)
    
    
    def __len__(self):
        return len(self.train_infos)
    
    def cart2homo(self, pts):
        assert pts.shape[-1] == 3
        return np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    
    def construct_T_from_vector(self, translation_vector, rotation_vector):
        assert len(translation_vector) == 3
        assert len(rotation_vector) == 4
        
        T = np.eye(4)
        R = Quaternion(*rotation_vector).rotation_matrix
        T[:3, :3] = R
        T[:3, 3] = translation_vector
        return T
    
    def collect_lidar_depths(self, lidar_pts, mminfo, nvs_cams):
        MAX_DEPTH = 200
        
        T_lidar2ego = self.construct_T_from_vector(
            mminfo['lidar2ego_translation'],
            mminfo['lidar2ego_rotation'])
                    
        T_lidar2imgs = []
        for cam in self.cam_names:
            T_ego2cam = nvs_cams[cam]['T_ego2cam']
            K = nvs_cams[cam]['K'].copy()
            T_cam2img = np.eye(4)
            T_cam2img[:3, :3] = K
            
            T_lidar2img = T_cam2img @ T_ego2cam @ T_lidar2ego
            T_lidar2imgs.append(T_lidar2img)
        T_lidar2imgs = np.stack(T_lidar2imgs)   # (n_cam, 4, 4)
        
        # batch transform
        img_pts = (self.cart2homo(lidar_pts) @ T_lidar2imgs.transpose(0, 2, 1))[:, :, :3]
    
        lidar_depths = []
        for i, cam in enumerate(self.cam_names):
            img_h, img_w = self.input_size[cam]
            
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
            saved_mask[1:] = (ranks[1:] != ranks[:-1])
            pts_depth = pts_depth[saved_mask]
            pts_uv = pts_uv[saved_mask]
            
            # create depth and save
            depth = np.zeros((img_h, img_w))
            depth[pts_uv[:, 1], pts_uv[:, 0]] = pts_depth
            
            lidar_depths.append(depth)
        # lidar_depths = np.stack(lidar_depths)
        
        return lidar_depths
    
    def __getitem__(self, idx):
        mminfo = self.train_infos[idx]
        
        # Step 1: collect cams
        nvs_cams = {}
        for sensor in self.cam_names:
            nvs_cams[sensor] = {}
            
            h, w = mminfo['cams'][sensor]['img_h'], mminfo['cams'][sensor]['img_w']
            
            K = np.array(mminfo['cams'][sensor]['cam_intrinsic']).astype(np.float32)
            T_cam2ego = self.construct_T_from_vector(
                mminfo['cams'][sensor]['sensor2ego_translation'], 
                mminfo['cams'][sensor]['sensor2ego_rotation']).astype(np.float32)
            T_ego2cam = np.linalg.inv(T_cam2ego)
        
            nvs_cams[sensor]['K'] = K
            nvs_cams[sensor]['T_ego2cam'] = T_ego2cam
            nvs_cams[sensor]['img_w'] = w
            nvs_cams[sensor]['img_h'] = h
        
        # Step 2 : apply focal rescale 
        cam_groups = NuscenesCameraGroups(nvs_cams, 0, 0, 0, 0)
        c2ws, fovxs, fovys, camera_args, crop_start, crop_end = \
            cam_groups.gen_raw_cam_for_gaussian_with_focalaug(('ratio', self.scale_factor, self.scale_factor))
        nvs_cams = cam_groups.new_cams
        
        
        # Step 3 : collect gaussians
        sample_token = mminfo['token']
        gs_file = os.path.join(self.meta_data_root, 'gaussians', f'{sample_token}.npz')
        gs_data = np.load(gs_file)
        
        means3D = (gs_data['means3D']).astype(np.float32)
        rgbs = (gs_data['rgbs'] / 255.).astype(np.float32)
        scales = (gs_data['scales']).astype(np.float32)[:, None].repeat(3, axis=1)
        lidar_pts = gs_data['lidar_pts'].astype(np.float32)
        
        raw_pts = len(means3D)
        n_pad = self.max_pts - raw_pts
        means3D = np.pad(means3D, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)
        rgbs = np.pad(rgbs, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)
        scales = np.pad(scales, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)

        # Step 4 : get lidar depths
        lidar_depths = self.collect_lidar_depths(lidar_pts, mminfo, nvs_cams)
        
        data_blob = {
            'sample_token': sample_token,
            'means3D': means3D,
            'rgbs': rgbs,
            'scales': scales,
            'n_raw_pts': raw_pts,
            'lidar_depths': lidar_depths,
            
            'c2ws': c2ws,
            'fovxs': fovxs,
            'fovys': fovys,
            'camera_args': camera_args,
            'crop_start': crop_start,
            'crop_end': crop_end,
            
            'cam_names': self.cam_names,
            'scale_factor': self.scale_factor,
            'input_size': self.input_size
        }
        
        return data_blob
        
        
    
def main():
    #=== predefine settings
    device = 'cuda'
    max_pts = 4000000
    opacity_placeholder = torch.full((max_pts, 1), fill_value=1.0, dtype=torch.float32, device=device)[None, ...]
    rotations_placeholder = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device).repeat(max_pts, 1)[None, ...]
    means2D_placeholder = torch.zeros((max_pts, 3), dtype=torch.float32, device=device)[None, ...]
    #===
    
    render_times = []
    
    dataset = WaymoDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    
    scale_factor = dataset.scale_factor

    pb = tqdm.tqdm(total=len(dataloader), desc='Rendering Waymo Data...', leave=True)
    for data in dataloader:
        for i in range(len(data['sample_token'])):
            sample_token = data['sample_token'][i]
            cam_names = dataset.cam_names
            
            renderer = GaussianRenderer(
                device=device,
                resolution=[data['camera_args']['resolution'][0][i], data['camera_args']['resolution'][1][i]],
                znear=data['camera_args']['znear'][i],
                zfar=data['camera_args']['zfar'][i],
                # renderer_type='fast_gauss',
            )
            
            N = data['n_raw_pts'][i]
            # render_results = renderer.render_v2(
            #     batch_means3D=data['means3D'][i:i+1, :N, :].to(device),
            #     batch_rgbs=data['rgbs'][i:i+1, :N, :].to(device),
            #     batch_opacity=opacity_placeholder[:, :N, :],
            #     batch_rotations=rotations_placeholder[:, :N, :],
            #     batch_scales=data['scales'][i:i+1, :N, :].to(device),
            #     batch_means2D=means2D_placeholder[:, :N, :],
            #     c2w=data['c2ws'][i:i+1].to(device),
            #     fovx=data['fovxs'][i:i+1].to(device),
            #     fovy=data['fovys'][i:i+1].to(device),
            #     rays_o=None,
            #     rays_d=None
            # )
            
            batch_means3D=data['means3D'][i:i+1, :N, :].to(device)
            batch_rgbs=data['rgbs'][i:i+1, :N, :].to(device)
            batch_scales=data['scales'][i:i+1, :N, :].to(device)
            c2w=data['c2ws'][i:i+1].to(device)
            fovx=data['fovxs'][i:i+1].to(device)
            fovy=data['fovys'][i:i+1].to(device)
            t1 = time.time()
            render_results = renderer.render_v2(
                batch_means3D=batch_means3D,
                batch_rgbs=batch_rgbs,
                batch_opacity=opacity_placeholder[:, :N, :],
                batch_rotations=rotations_placeholder[:, :N, :],
                batch_scales=batch_scales,
                batch_means2D=means2D_placeholder[:, :N, :],
                c2w=c2w,
                fovx=fovx,
                fovy=fovy,
                rays_o=None,
                rays_d=None
            )
            render_times.append(time.time() - t1)
            
            _nvs_imgs = render_results['image']     # b v 3 h w
            _nvs_depths = render_results['depth'].squeeze(2)    # b v h w
            
            cropped_imgs = []
            cropped_depths = []
            
            # train_h = data['input_size'][0][i]
            for j in range(_nvs_imgs.shape[1]):
                train_h = data['input_size'][cam_names[j]][0][i]
                crop_start = data['crop_start'][i][j]
                crop_end = data['crop_end'][i][j]
                
                _img = _nvs_imgs[:, j, :, crop_start[1]:crop_end[1], crop_start[0]:crop_end[0]]
                _depth = _nvs_depths[:, j, crop_start[1]:crop_end[1], crop_start[0]:crop_end[0]]
                render_h = _img.shape[2]
                h_crop_start = render_h - train_h
                _img = _img[:, :, h_crop_start:, :]
                _depth = _depth[:, h_crop_start:, :]
                
                cropped_imgs.append(_img)
                cropped_depths.append(_depth)
            
            # save results
            # for img, depth, cam in zip(cropped_imgs, data['lidar_depths'][i], cam_names):
            for cam_idx, (img, cam) in enumerate(zip(cropped_imgs, cam_names)):
                img = img[0].cpu().numpy()
                img = (img * 255).astype(np.uint8)
                img = img.transpose(1, 2, 0)[..., ::-1]
                
                # depth = depth.cpu().numpy()
                depth = data['lidar_depths'][cam_idx][i].cpu().numpy()
                
                saved_img_file = os.path.join(dataset.meta_data_root, f'zoomout_rgbd_{scale_factor}', 'img', cam, f'{sample_token}.jpg')
                saved_depth_file = os.path.join(dataset.meta_data_root, f'zoomout_rgbd_{scale_factor}', 'depth', cam, f'{sample_token}.png')
                os.makedirs(os.path.dirname(saved_img_file), exist_ok=True)
                os.makedirs(os.path.dirname(saved_depth_file), exist_ok=True)

                
                cv2.imwrite(saved_img_file, img)
                save_depth_map(saved_depth_file, depth)
        
        pb.update()
    pb.close()

    all_render_times = np.sum(render_times)
    n_all_samples = len(dataloader) * len(dataset.cam_names)
    print(f'all_render_times:{all_render_times}  n_all_samples:{n_all_samples}')
    print(f'DONE!!! rendering speed per image: {all_render_times / n_all_samples:.4f} sec/image  |  FPS: {n_all_samples / all_render_times:.4f} fps')



if __name__ == '__main__':
    main()