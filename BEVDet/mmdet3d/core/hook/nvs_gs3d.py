from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.gaussian3d_kernel.gaussian import GaussianRenderer
import torch
import torch.distributed as dist
import numpy as np
import cv2
import os

__all__  = ['NovelViewSynthesisHook']


@HOOKS.register_module()
class NovelViewSynthesisHook(Hook):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.nvs_runner = NovelViewSynthesisRunner(**kwargs)
        
        
    def before_train_iter(self, runner):
        data_batch = runner.data_batch
        processed_batch = self.nvs_runner.data_post_process(data_batch)
        runner.data_batch = processed_batch
        
    def before_val_iter(self, runner):
        data_batch = runner.data_batch
        processed_batch = self.nvs_runner.data_post_process(data_batch)
        runner.data_batch = processed_batch
        
    
class NovelViewSynthesisRunner():
    def __init__(self, max_pts=3000000, use_placeholder=True, use_pmd=False, p_pmd=0.5, device_id=0):
        if dist.is_initialized():
            self.local_rank = dist.get_rank()
            self.device = torch.device(f'cuda:{self.local_rank}')
            # print(f'DDP Mode NovelViewSynthesisRunner Created : DEVICE on {self.device}')
        else:
            # device = torch.device('cuda')
            self.device = torch.device(f'cuda:{device_id}')
            # print(f'Single GPU Mode NovelViewSynthesisRunner Created : DEVICE on {self.device}')
            
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)#, device=self.device)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)#, device=self.device)
        
        self.use_placeholder = use_placeholder
        if self.use_placeholder:
            self.max_pts = max_pts
            self.opacity_placeholder = torch.full((self.max_pts, 1), fill_value=1.0, dtype=torch.float32, device=self.device)[None, ...]
            self.rotations_placeholder = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=self.device).repeat(self.max_pts, 1)[None, ...]
            self.means2D_placeholder = torch.zeros((self.max_pts, 3), dtype=torch.float32, device=self.device)[None, ...]

        self.use_pmd = use_pmd
        if use_pmd:
            self.pmd = GpuPhotoMetricDistortion()
            self.p_pmd = p_pmd


        # self.to_device_keys = [
        #     'means3D', 'rgbs', 'scales', 'c2ws', 'fovxs', 'fovys', 'crop_start', 'crop_end'
        # ]
        
    def data_post_process(self, data_batch):
        # data_batch = to_device_recursive(data_batch, self.device)        
        # keys_to_device(data_batch, self.device, self.to_device_keys)
            
        nvs_meta = data_batch['nvs_meta']
        plain_flip_mask = data_batch['img_inputs'][4][:, :, 0, 0] < 0
        
        nvs_imgs = []
        nvs_depths = []
        
        # Step 1 : render current
        _imgs, _depths = self.render_one_frame(nvs_meta['curr'], self.device)
        nvs_imgs.append(_imgs)
        nvs_depths.append(_depths)
        
        # ! DEBUG
        # print(f'local_rank:{self.local_rank} imgs.device:{_imgs.device} imgs.mean:{_imgs.mean():.2f} depth.device:{_depths.device} depth.mean:{_depths.mean():.2f}')
        # _vis_nvs(_imgs, nvs_meta['curr']['debug_imgs']) ; _vis_nvs(_imgs, _meta['debug_imgs'])
        # _vis_nvs_batch(_imgs, nvs_meta['curr']['debug_imgs'], data_batch['img_metas'].data[0])
        # _vis_nvs_batch(_depths, nvs_meta['curr']['debug_imgs'], data_batch['img_metas'].data[0], is_depth=True)
        # _vis_nvs_batch(_imgs, _meta['debug_imgs'], data_batch['img_metas'].data[0])
        # _vis_nvs_batch(_depths, _meta['debug_imgs'], data_batch['img_metas'].data[0], is_depth=True)
        # _vis_nvs_batch(nvs_imgs[:, :6, ...], nvs_meta['curr']['debug_imgs'], data_batch['img_metas'].data[0])
        # _vis_nvs_batch(nvs_depths[:, :6, ...], nvs_meta['curr']['debug_imgs'], data_batch['img_metas'].data[0], is_depth=True)
        
        # render adjacent
        for _meta in nvs_meta['adjacent']:  
            _imgs, _depths = self.render_one_frame(_meta, self.device)
            nvs_imgs.append(_imgs)
            nvs_depths.append(_depths)
            
        nvs_imgs = torch.cat(nvs_imgs, dim=1)
        nvs_depths = torch.cat(nvs_depths, dim=1)
        
        nvs_imgs[plain_flip_mask] = nvs_imgs[plain_flip_mask].flip(-1)
        nvs_depths[plain_flip_mask] = nvs_depths[plain_flip_mask].flip(-1)
        
        # Step 2 : photometric distortion on gpu (from SparseBEV)
        if self.use_pmd:
            b, v, c, h, w = nvs_imgs.shape
            pmd_flags = torch.rand(b*v, device=self.device) < self.p_pmd
            nvs_imgs = self.pmd(nvs_imgs.view(b*v, c, h, w), pmd_flags).view(b, v, c, h, w)
        
        # Step 3 : normalization (NOTE: bug in raw bevdet rgb->bgr->norm_by_rgb_stat, but just use in test debug)
        # nvs_imgs = (nvs_imgs.flip(2) - self.rgb_mean.view(1, 1, 3, 1, 1).to(self.device)) / self.rgb_std.view(1, 1, 3, 1, 1).to(self.device)
        nvs_imgs = (nvs_imgs - self.rgb_mean.view(1, 1, 3, 1, 1).to(self.device)) / self.rgb_std.view(1, 1, 3, 1, 1).to(self.device)
        
        # Step 4 : refresh img/depth placeholder in data_batch
        data_batch['img_inputs'][0] = nvs_imgs#.unsqueeze(0)
        data_batch['gt_depth'] = nvs_depths#.unsqueeze(0)
        
        # ! adjust the order : WE ONLY NEED to adjust img_inputs[0] i.e. the image 
        n_adj = len(data_batch['nvs_meta']['adjacent'])
        n_cam = nvs_imgs.shape[1] // (n_adj + 1)
        order = [int(i+j*n_cam) for i in range(n_cam) for j in range(n_adj+1)]
        data_batch['img_inputs'][0] = data_batch['img_inputs'][0][:, order, ...].contiguous()
        data_batch['gt_depth'] = data_batch['gt_depth'][:, :n_cam].contiguous()
        
        del data_batch['nvs_meta']
        
        return data_batch 
    
    
    def render_one_frame(self, nvs_meta, device):
        nvs_imgs = []
        nvs_depths = []
        
        bs = nvs_meta['means3D'].shape[0]
        for i in range(bs):
            if not nvs_meta['use_nvs_flag'][i]:
                nvs_imgs.append(nvs_meta['raw_imgs'][i].to(device))
                nvs_depths.append(nvs_meta['raw_depths'][i].to(device))     # lidar_depth/dense_depth have been decided in dataloading
                continue
            
            renderer = GaussianRenderer(
                device=device,
                resolution=[nvs_meta['camera_args']['resolution'][0][i], nvs_meta['camera_args']['resolution'][1][i]],
                znear=nvs_meta['camera_args']['znear'][i],
                zfar=nvs_meta['camera_args']['zfar'][i],
                # renderer_type='fast_gauss',
            )
            
            N = nvs_meta['n_raw_pts'][i]
            render_results = renderer.render_v2(
                batch_means3D=nvs_meta['means3D'][i:i+1, :N, :].to(device),
                batch_rgbs=nvs_meta['rgbs'][i:i+1, :N, :].to(device),
                batch_opacity=self.opacity_placeholder[:, :N, :],
                batch_rotations=self.rotations_placeholder[:, :N, :],
                batch_scales=nvs_meta['scales'][i:i+1, :N, :].to(device),
                batch_means2D=self.means2D_placeholder[:, :N, :],
                c2w=nvs_meta['c2ws'][i:i+1].to(device),
                fovx=nvs_meta['fovxs'][i:i+1].to(device),
                fovy=nvs_meta['fovys'][i:i+1].to(device),
                rays_o=None,
                rays_d=None
            )
            
            _nvs_imgs = render_results['image']     # b v 3 h w
            _nvs_depths = render_results['depth'].squeeze(2)    # b v h w
            
            cropped_imgs = []
            cropped_depths = []
            
            train_h = nvs_meta['input_size'][0][i]
            for j in range(_nvs_imgs.shape[1]):
                crop_start = nvs_meta['crop_start'][i][j]
                crop_end = nvs_meta['crop_end'][i][j]
                
                _img = _nvs_imgs[:, j, :, crop_start[1]:crop_end[1], crop_start[0]:crop_end[0]]
                _depth = _nvs_depths[:, j, crop_start[1]:crop_end[1], crop_start[0]:crop_end[0]]
                render_h = _img.shape[2]
                h_crop_start = render_h - train_h
                _img = _img[:, :, h_crop_start:, :]
                _depth = _depth[:, h_crop_start:, :]
                
                cropped_imgs.append(_img)
                cropped_depths.append(_depth)
            
            # if nvs_meta['use_outpainting_flag'][i]:
            #     _background = torch.cat(cropped_imgs, dim=0)
            #     _foreground = nvs_meta['raw_imgs'][i].to(device)
            #     _fg_mask = _foreground != 0
            #     _background[_fg_mask] = _foreground[_fg_mask]
            #     nvs_imgs.append(_background)
            #     # nvs_imgs.append(torch.cat(cropped_imgs, dim=0))
            # else:
            nvs_imgs.append(torch.cat(cropped_imgs, dim=0))
            
            if nvs_meta['use_lidar_depth'][i]:
                nvs_depths.append(nvs_meta['raw_depths'][i].to(device))
            else:
                nvs_depths.append(torch.cat(cropped_depths, dim=0))
            
            # torch.cuda.empty_cache()
                    
        return torch.stack(nvs_imgs), torch.stack(nvs_depths)
        
    
    
class GpuPhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta / 255.
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta


    def __call__(self, imgs, pmd_flags):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        # imgs = imgs[:, [2, 1, 0], :, :]  # BGR to RGB
        # assert len(imgs) == len(pmd_flags)

        contrast_modes = []
        for _ in range(imgs.shape[0]):
            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            contrast_modes.append(np.random.randint(2))

        for idx in range(imgs.shape[0]):
            if not pmd_flags[idx]: continue
            
            # random brightness
            if np.random.randint(2):
                delta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
                imgs[idx] += delta

            if contrast_modes[idx] == 0:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                    imgs[idx] *= alpha

        # convert color from BGR to HSV
        imgs[pmd_flags] = self.rgb_to_hsv(imgs[pmd_flags])

        for idx in range(imgs.shape[0]):
            if not pmd_flags[idx]: continue
            
            # random saturation
            if np.random.randint(2):
                imgs[idx, 1] *= np.random.uniform(self.saturation_lower, self.saturation_upper)

            # random hue
            if np.random.randint(2):
                imgs[idx, 0] += np.random.uniform(-self.hue_delta, self.hue_delta)

        # imgs[:, 0][imgs[:, 0] > 360] -= 360
        # imgs[:, 0][imgs[:, 0] < 0] += 360
        imgs[pmd_flags, 0][imgs[pmd_flags, 0] > 360] -= 360
        imgs[pmd_flags, 0][imgs[pmd_flags, 0] < 0] += 360

        # convert color from HSV to BGR
        imgs[pmd_flags] = self.hsv_to_rgb(imgs[pmd_flags])

        for idx in range(imgs.shape[0]):
            if not pmd_flags[idx]: continue
            
            # random contrast
            if contrast_modes[idx] == 1:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                    imgs[idx] *= alpha

            # randomly swap channels
            if np.random.randint(2):
                imgs[idx] = imgs[idx, np.random.permutation(3)]

        # imgs = imgs[:, [2, 1, 0], :, :]  # RGB to BGR

        return imgs
    
    
    def rgb_to_hsv(self, image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

        # image = image / 255.0

        max_rgb, argmax_rgb = image.max(-3)
        min_rgb, argmin_rgb = image.min(-3)
        deltac = max_rgb - min_rgb

        v = max_rgb
        s = deltac / (max_rgb + eps)

        deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
        rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

        h1 = bc - gc
        h2 = (rc - bc) + 2.0 * deltac
        h3 = (gc - rc) + 4.0 * deltac

        h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
        h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
        h = (h / 6.0) % 1.0

        h = h * 360.0
        v = v * 255.0

        return torch.stack((h, s, v), dim=-3)


    def hsv_to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

        h: torch.Tensor = image[..., 0, :, :] / 360.0
        s: torch.Tensor = image[..., 1, :, :]
        v: torch.Tensor = image[..., 2, :, :] / 255.0

        hi: torch.Tensor = torch.floor(h * 6) % 6
        f: torch.Tensor = ((h * 6) % 6) - hi
        one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
        p: torch.Tensor = v * (one - s)
        q: torch.Tensor = v * (one - f * s)
        t: torch.Tensor = v * (one - (one - f) * s)

        hi = hi.long()
        indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
        out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
        out = torch.gather(out, -3, indices)
        # out = out * 255.0

        return out

    
        
        
        

    
# ! debug utils    
def keys_to_device(data, device, keys):
    for k, v in data['nvs_meta']['curr'].items():
        if k in keys:
            data['nvs_meta']['curr'][k] = v.to(device)
    for _data in data['nvs_meta']['adjacent']:
        for k, v in _data.items():
            if k in keys:
                _data[k] = v.to(device)
 
   
def _vis_nvs_batch(nvs_imgs, raw_imgs, img_metas, is_depth=False):
    """
    nvs_imgs: (b, v, c, h, w) / (b, v, h, w)
    raw_imgs: cam_names -> (b, c, h, w)
    img_metas: list(len=b) -> have sample_idx
    """
    bs = nvs_imgs.shape[0]
    cam_names = list(raw_imgs.keys())
    for i in range(bs):
        sample_token = img_metas[i]['sample_idx']
        
        nvs_canvas = []
        for img in nvs_imgs[i]:
            img = img.cpu().numpy()
            if is_depth:
                mask = (img==0)
                img = np.uint8(img / img.max() * 255)
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                img[mask] = 0
            else:
                img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)[..., ::-1]
            nvs_canvas.append(img)
        
        gt_canvas = []
        for cam in cam_names:
            img = raw_imgs[cam][i].cpu().numpy()
            h, w, _ = img.shape
            new_h = int(704/w*h)
            img = cv2.resize(img, (704, new_h))
            # img = img[new_h-256:, ...]
            # img = img[new_h-384:, ...]
            img = img[new_h-320:, ...]
            gt_canvas.append(img)
            
        nvs_canvas = _create_cavans(nvs_canvas, cam_names)
        gt_canvas = _create_cavans(gt_canvas, cam_names)
        
        if len(cam_names) in [3, 5]:
            canvas = np.concatenate([nvs_canvas, gt_canvas], axis=0)
        elif len(cam_names) == 6:
            canvas = np.concatenate([nvs_canvas, gt_canvas], axis=1)
            
        if is_depth:
            saved_file = os.path.join('./tmp/canvas_debug', f'{i}_{sample_token}_depth.jpg')
        else:
            saved_file = os.path.join('./tmp/canvas_debug', f'{i}_{sample_token}.jpg')
        cv2.imwrite(saved_file, canvas)
        

   
def _vis_nvs(nvs_imgs, raw_imgs):
    cam_names = list(raw_imgs.keys())
    assert len(cam_names) == len(nvs_imgs)
    
    vis_imgs = []
    for img in nvs_imgs:
        img = img.cpu().numpy()
        img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)[..., ::-1]
        vis_imgs.append(img)
    
    
    gt_imgs = []
    for i, cam_name in enumerate(cam_names):
        img = raw_imgs[cam_name][0].cpu().numpy()

        h, w, _ = vis_imgs[i].shape
        img = cv2.resize(img, (w, h))
        gt_imgs.append(img)
        
    nvs_canvas = _create_cavans(vis_imgs, cam_names)
    gt_canvas = _create_cavans(gt_imgs, cam_names)
    
    cv2.imwrite('nvs_canvas.png', nvs_canvas)
    cv2.imwrite('gt_canvas.png', gt_canvas)
        
def _create_cavans(scene_imgs, cam_names):
    if len(cam_names) == 5:
        # waymo
        w = scene_imgs[0].shape[1]
        max_h = scene_imgs[0].shape[0]
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
    
    elif len(cam_names) == 3:
        # waymo fronts
        w = scene_imgs[0].shape[1]
        max_h = scene_imgs[0].shape[0]
        canvas = np.zeros((max_h, 3*w, 3), dtype=np.uint8)
        front = scene_imgs[cam_names.index('CAM_FRONT')]          # CAM_FRONT
        front_left = scene_imgs[cam_names.index('CAM_FRONT_LEFT')]     # CAM_FRONT_LEFT
        front_right = scene_imgs[cam_names.index('CAM_FRONT_RIGHT')]    # CAM_FRONT_RIGHT
        
        canvas[:, :w] = front_left
        canvas[:, w:2*w] = front
        canvas[:, 2*w:3*w] = front_right
        
        return canvas
    
    elif len(cam_names) == 6:
        # nuscenes and lyft(6cam)
        h, w = scene_imgs[0].shape[0], scene_imgs[0].shape[1]
        padding = 10  # 图像间间距
        canvas = np.zeros((4*h + 5*padding, 2*w + 3*padding, 3), dtype=np.uint8)
        
        front = scene_imgs[cam_names.index('CAM_FRONT')]          # CAM_FRONT
        front_left = scene_imgs[cam_names.index('CAM_FRONT_LEFT')]     # CAM_FRONT_LEFT
        front_right = scene_imgs[cam_names.index('CAM_FRONT_RIGHT')]    # CAM_FRONT_RIGHT
        back = scene_imgs[cam_names.index('CAM_BACK')]           # CAM_BACK
        back_left = scene_imgs[cam_names.index('CAM_BACK_LEFT')]      # CAM_BACK_LEFT
        back_right = scene_imgs[cam_names.index('CAM_BACK_RIGHT')]     # CAM_BACK_RIGHT
        
        # 第一行 (FRONT)
        canvas[padding:h+padding, 
            padding+w//2:padding+w//2+w] = front
        
        # 第二行 (FRONT_LEFT 和 FRONT_RIGHT)
        canvas[h+2*padding:2*h+2*padding, 
            padding:padding+w] = front_left
        canvas[h+2*padding:2*h+2*padding, 
            padding+w+padding:padding+2*w+padding] = front_right
        
        # 第三行 (BACK_LEFT 和 BACK_RIGHT)
        try:
            canvas[2*h+3*padding:3*h+3*padding, 
                padding:padding+w] = back_left
            canvas[2*h+3*padding:3*h+3*padding, 
                padding+w+padding:padding+2*w+padding] = back_right
        except:
            # for nuscenes' nv at waymo
            _offset = h - back_left.shape[0]
            canvas[2*h+3*padding+_offset:3*h+3*padding, 
                padding:padding+w] = back_left
            canvas[2*h+3*padding+_offset:3*h+3*padding, 
                padding+w+padding:padding+2*w+padding] = back_right
        
        # 第四行 (BACK)
        canvas[3*h+4*padding:4*h+4*padding, 
            padding+w//2:padding+w//2+w] = back
        
        return canvas
        


    