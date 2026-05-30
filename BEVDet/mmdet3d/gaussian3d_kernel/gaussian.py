import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, 
    GaussianRasterizer
)

from .utils.ops import get_cam_info_gaussian
from .utils.typing import *


C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def strip_lowerdiag(L, device):
    # uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym, device):
    return strip_lowerdiag(sym, device)


def build_rotation(r, device):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    # R = torch.zeros((q.size(0), 3, 3), device="cuda")
    R = torch.zeros((q.size(0), 3, 3), device=device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r, device):
    # L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=device)
    R = build_rotation(r, device)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L

class Depth2Normal(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delzdelxkernel = torch.tensor(
            [
                [0.00000, 0.00000, 0.00000],
                [-1.00000, 0.00000, 1.00000],
                [0.00000, 0.00000, 0.00000],
            ]
        )
        self.delzdelykernel = torch.tensor(
            [
                [0.00000, -1.00000, 0.00000],
                [0.00000, 0.00000, 0.00000],
                [0.0000, 1.00000, 0.00000],
            ]
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        B, C, H, W = x.shape
        delzdelxkernel = self.delzdelxkernel.view(1, 1, 3, 3).to(x.device)
        delzdelx = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelxkernel, padding=1
        ).reshape(B, C, H, W)
        delzdelykernel = self.delzdelykernel.view(1, 1, 3, 3).to(x.device)
        delzdely = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelykernel, padding=1
        ).reshape(B, C, H, W)
        normal = -torch.cross(delzdelx, delzdely, dim=1)
        return normal


class GaussianRenderer:
    def __init__(
        self, 
        device,
        resolution: list = [512, 512],
        znear: float = 0.1,
        zfar: float = 100.0, 
        renderer_type: str = "vanilla", # only support "vanilla"
        **kwargs,
    ):  
        self.device = device
        assert renderer_type in ['vanilla', 'fast_gauss']
        self.renderer_type = renderer_type

        self.resolution = resolution
        self.znear = znear
        self.zfar = zfar
        # self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

        # self.normal_module = Depth2Normal().to(device)

        self.setup_functions()


    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation, self.device)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance, self.device)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def render(
        self, 
        gaussians: Float[Tensor, "B N F"], 
        c2w: Float[Tensor, "B V 4 4"],
        fovx: Float[Tensor, "B V"] = None,
        fovy: Float[Tensor, "B V"] = None,
        rays_o: Float[Tensor, "B V H W 3"] = None,
        rays_d: Float[Tensor, "B V H W 3"] = None,
        bg_color: Float[Tensor, "... 3"] = None, 
        scale_modifier: float = 1.,
    ):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        # at least one of fovx and fovy is not none
        assert fovx is not None or fovy is not None
        if fovx is None:
            fovx = fovy
        if fovy is None:
            fovy = fovx

        device = gaussians.device
        B, V = c2w.shape[:2]

        # loop of loop...
        images = []
        alphas = []
        depths = []
        for b in range(B):

            means3D = gaussians[b, :, 0:3].contiguous().float()
            rgbs = gaussians[b, :, 3:6].contiguous().float() # [N, 3]
            opacity = gaussians[b, :, 6:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            scales = gaussians[b, :, 11:].contiguous().float()
            means2D = torch.zeros_like(means3D, dtype=means3D.dtype, device=device)

            for v in range(V):
                fovx_ = fovx[b, v].clone()
                fovy_ = fovy[b, v].clone()
                c2w_ = c2w[b, v].clone()
                w2c, proj, cam_p = get_cam_info_gaussian(
                    c2w=c2w_, fovx=fovx_, fovy=fovy_, znear=self.znear, zfar=self.zfar, device=device
                )
                # render novel views
                tan_half_fovx = torch.tan(fovx_ * 0.5)
                tan_half_fovy = torch.tan(fovy_ * 0.5)

                if self.renderer_type == "vanilla":
                    raster_settings = GaussianRasterizationSettings(
                        image_height=self.resolution[0],
                        image_width=self.resolution[1],
                        tanfovx=tan_half_fovx,
                        tanfovy=tan_half_fovy,
                        bg=self.bg_color if bg_color is None else bg_color,
                        scale_modifier=scale_modifier,
                        viewmatrix=w2c,
                        projmatrix=proj,
                        sh_degree=0,
                        campos=cam_p,
                        prefiltered=False,
                        debug=False,
                    )
                    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                else:
                    raise NotImplementedError

                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                if self.renderer_type == "vanilla":
                    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                        means3D=means3D,
                        means2D=means2D,
                        shs=None,
                        colors_precomp=rgbs,
                        opacities=opacity,
                        scales=scales,
                        rotations=rotations,
                        cov3D_precomp=None,
                    )
                    rendered_normal = None
                else:
                    raise NotImplementedError

                rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)
                images.append(rendered_image)
                alphas.append(rendered_alpha)
                depths.append(rendered_depth)

        images = torch.stack(images, dim=0).view(B, V, 3, self.resolution[0], self.resolution[1])
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.resolution[0], self.resolution[1])
        depths = torch.stack(depths, dim=0).view(B, V, 1, self.resolution[0], self.resolution[1])

        return {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
            "depth": depths
        }


    def render_v2(
        self, 
        # gaussians: Float[Tensor, "B N F"], 
        batch_means3D: Float[Tensor, "B N 3"],
        batch_rgbs: Float[Tensor, "B N 3"],
        batch_opacity: Float[Tensor, "B N 1"],
        batch_rotations: Float[Tensor, "B N 4"],
        batch_scales: Float[Tensor, "B N 3"],
        batch_means2D: Float[Tensor, "B N 2"],
        c2w: Float[Tensor, "B V 4 4"],
        fovx: Float[Tensor, "B V"] = None,
        fovy: Float[Tensor, "B V"] = None,
        rays_o: Float[Tensor, "B V H W 3"] = None,
        rays_d: Float[Tensor, "B V H W 3"] = None,
        bg_color: Float[Tensor, "... 3"] = None, 
        scale_modifier: float = 1.,
    ):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        # at least one of fovx and fovy is not none
        assert fovx is not None or fovy is not None
        if fovx is None:
            fovx = fovy
        if fovy is None:
            fovy = fovx

        device = batch_means3D.device
        B, V = c2w.shape[:2]

        # loop of loop...
        images = []
        alphas = []
        depths = []
        for b in range(B):

            means3D = batch_means3D[b]
            rgbs = batch_rgbs[b]
            opacity = batch_opacity[b]
            rotations = batch_rotations[b]
            scales = batch_scales[b]
            means2D = batch_means2D[b]
                
            for v in range(V):
                fovx_ = fovx[b, v].clone()
                fovy_ = fovy[b, v].clone()
                c2w_ = c2w[b, v].clone()
                w2c, proj, cam_p = get_cam_info_gaussian(
                    c2w=c2w_, fovx=fovx_, fovy=fovy_, znear=self.znear, zfar=self.zfar, device=device
                )
                # render novel views
                tan_half_fovx = torch.tan(fovx_ * 0.5)
                tan_half_fovy = torch.tan(fovy_ * 0.5)

                if self.renderer_type == "vanilla":
                    raster_settings = GaussianRasterizationSettings(
                        image_height=self.resolution[0],
                        image_width=self.resolution[1],
                        tanfovx=tan_half_fovx,
                        tanfovy=tan_half_fovy,
                        bg=self.bg_color if bg_color is None else bg_color,
                        scale_modifier=scale_modifier,
                        viewmatrix=w2c,
                        projmatrix=proj,
                        sh_degree=0,
                        campos=cam_p,
                        prefiltered=False,
                        debug=False,
                    )
                    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                # elif self.renderer_type == 'fast_gauss':
                #     raster_settings = GaussianRasterizationSettingsFast(
                #         image_height=self.resolution[0],
                #         image_width=self.resolution[1],
                #         tanfovx=tan_half_fovx,
                #         tanfovy=tan_half_fovy,
                #         bg=self.bg_color if bg_color is None else bg_color,
                #         scale_modifier=scale_modifier,
                #         viewmatrix=w2c,
                #         projmatrix=proj,
                #         sh_degree=0,
                #         campos=cam_p,
                #         prefiltered=False,
                #         debug=False,
                #     )
                #     rasterizer = GaussianRasterizerFast(raster_settings=raster_settings)
                else:
                    raise NotImplementedError

                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                # if self.renderer_type == "vanilla":
                if self.renderer_type in ["vanilla", "fast_gauss"]:
                    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                        means3D=means3D,
                        means2D=means2D,
                        shs=None,
                        colors_precomp=rgbs,
                        opacities=opacity,
                        scales=scales,
                        rotations=rotations,
                        cov3D_precomp=None,
                        device=device
                    )
                    rendered_normal = None
                else:
                    raise NotImplementedError

                rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)
                images.append(rendered_image)
                alphas.append(rendered_alpha)
                depths.append(rendered_depth)

            # torch.cuda.empty_cache()

        images = torch.stack(images, dim=0).view(B, V, 3, self.resolution[0], self.resolution[1])
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.resolution[0], self.resolution[1])
        depths = torch.stack(depths, dim=0).view(B, V, 1, self.resolution[0], self.resolution[1])

        return {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
            "depth": depths
        }


    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    
    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians
    