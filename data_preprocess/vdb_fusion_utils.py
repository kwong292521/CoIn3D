import os
import numpy as np
import cv2
import tqdm
import pickle
import torch
import ipdb
import time

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes, create_splits_logs
from common_utils import *
from __PATHS__ import *

# vdbfusion imports
import glob
import importlib
import functools as ft
import yaml
import open3d as o3d

from typing import List
from trimesh import transform_points
from diskcache import Cache
from diskcache.core import ENOVAL, args_to_key, full_name
from easydict import EasyDict
from functools import reduce
from tqdm import trange
from vdbfusion import VDBVolume


# ! ========= vdbfusion utils ============= #
def get_cache(directory):
    return Cache(directory, timeout=1, size_limit=3e11)


def memoize(name=None, typed=False, expire=None, tag=None):
    """Same as DiskCache.memoize but ignoring the first argument(self) for the keys."""
    # Caution: Nearly identical code exists in DjangoCache.memoize
    if callable(name):
        raise TypeError(f"name {name} cannot be callable")

    def decorator(func):
        """Decorator created by memoize() for callable `func`."""
        base = (full_name(func),) if name is None else (name,)

        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            cls = args[0]
            if not cls.use_cache:
                return func(*args, **kwargs)
            key = wrapper.__cache_key__(*args, **kwargs)
            result = cls.cache.get(key, default=ENOVAL, retry=True)

            if result is ENOVAL:
                result = func(*args, **kwargs)
                if expire is None or expire > 0:
                    cls.cache.set(key, result, expire, tag=tag, retry=True)

            return result

        def __cache_key__(*args, **kwargs):
            """Make key for cache given function arguments."""
            return args_to_key(base, args, kwargs, typed, ignore={0, "self"})

        wrapper.__cache_key__ = __cache_key__
        return wrapper

    return decorator


# def load_config(config_file: str):
#     return EasyDict(yaml.safe_load(open(config_file)))
def make_vdbfusion_config(
    # VDBFUSION
    voxel_size=0.1, sdf_trunc=0.3, space_carving=False, out_dir=None,
    # Reconstruction
    fill_holes=True, min_weight=5.0,
    # Data
    apply_pose=True, min_range=2.0, max_range=100.0,
    ):
    _dict =  {
        'voxel_size': voxel_size,
        'sdf_trunc': sdf_trunc,
        'space_carving': space_carving,
        'out_dir': out_dir,
        'fill_holes': fill_holes,
        'min_weight': min_weight,
        'apply_pose': apply_pose,
        'min_range': min_range,
        'max_range': max_range,
    }
    config = EasyDict(_dict)
    return config
    
    
def write_vdbfusion_config(config: EasyDict, filename: str):
    with open(filename, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


class VDBFusionPipeline:
    """Abstract class that defines a Pipeline, derived classes must implement the dataset and config
    properties."""

    def __init__(self, config, result_name):
        self._config = config
        self.result_name = result_name
        self._tsdf_volume = VDBVolume(
            self._config.voxel_size,
            self._config.sdf_trunc,
            self._config.space_carving,
        )
        self._res = {}

    def run(self, scans, poses):
        self._run_tsdf_pipeline(scans, poses)
        self._write_ply()
        # self._write_cfg()
        # self._write_vdb()
        # self._print_tim()


    def _run_tsdf_pipeline(self, scans, poses):
        times = []
        for idx in trange(len(scans), unit=" frames"):
            scan, pose = scans[idx], poses[idx]
            tic = time.perf_counter_ns()
            self._tsdf_volume.integrate(scan, pose)
            toc = time.perf_counter_ns()
            times.append(toc - tic)
        self._res = {"mesh": self._get_o3d_mesh(self._tsdf_volume, self._config), "times": times}

    def _write_vdb(self):
        os.makedirs(self._config.out_dir, exist_ok=True)
        filename = os.path.join(self._config.out_dir, self.result_name) + ".vdb"
        self._tsdf_volume.extract_vdb_grids(filename)

    def _write_ply(self):
        os.makedirs(self._config.out_dir, exist_ok=True)
        filename = os.path.join(self._config.out_dir, self.result_name) + ".ply"
        o3d.io.write_triangle_mesh(filename, self._res["mesh"])
        
    # def _write_ply_float32(self):
    #     os.makedirs(self._config.out_dir, exist_ok=True)
    #     filename = os.path.join(self._config.out_dir, self.result_name) + ".ply"
    #     # 获取网格并强制转换顶点为 float32
    #     mesh = self._res["mesh"]
    #     mesh.vertices = o3d.utility.Vector3dVector(
    #         np.asarray(mesh.vertices).astype(np.float32)  # 关键步骤
    #     )

    #     o3d.io.write_triangle_mesh(filename, mesh)


    def _write_cfg(self):
        os.makedirs(self._config.out_dir, exist_ok=True)
        filename = os.path.join(self._config.out_dir, self.result_name) + ".yml"
        write_vdbfusion_config(dict(self._config), filename)

    def _print_tim(self):
        total_time_ns = reduce(lambda a, b: a + b, self._res["times"])
        total_time = total_time_ns * 1e-9
        total_scans = self._n_scans - self._jump
        self.fps = float(total_scans / total_time)

    @staticmethod
    def _get_o3d_mesh(tsdf_volume, cfg):
        vertices, triangles = tsdf_volume.extract_triangle_mesh(cfg.fill_holes, cfg.min_weight)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(triangles),
        )
        mesh.compute_vertex_normals()
        return mesh
