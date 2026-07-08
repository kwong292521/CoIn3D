# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
import copy
import pickle
from scipy.sparse import load_npz
from scipy.spatial.transform import Rotation
from collections import defaultdict

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera
        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        time_dim (int, optional): Which dimension to represent the timestamps
            of each points. Defaults to 4.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.time_dim = time_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, self.time_dim] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, self.time_dim] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def __call__(self, results):
        assert 'points' in results
        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype=np.int64,
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str


@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]  
        post_rots, post_trans, bda = results['img_inputs'][4:]  
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results


@PIPELINES.register_module()
class PointToMultiViewDepthFusion(PointToMultiViewDepth):
    def __call__(self, results):
        points_camego_aug = results['points'].tensor[:, :3]
        # print(points_lidar.shape)
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        points_camego = points_camego_aug - bda[:3, 3].view(1,3)
        points_camego = points_camego.matmul(torch.inverse(bda[:3,:3]).T)

        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            camego2img = cam2img.matmul(torch.inverse(cam2camego))

            points_img = points_camego.matmul(
                camego2img[:3, :3].T) + camego2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    # to_rgb = True
    # ! NOTE: Image.open is rgb so we dont need to_rgb
    to_rgb = False
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        opencv_pp=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.opencv_pp = opencv_pp

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        if not self.opencv_pp:
            img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        if self.opencv_pp:
            img = self.img_transform_core_opencv(img, post_rot, post_tran, crop)
        return img, post_rot, post_tran

    def img_transform_core_opencv(self, img, post_rot, post_tran,
                                  crop):
        img = np.array(img).astype(np.float32)
        img = cv2.warpAffine(img,
                             np.concatenate([post_rot,
                                            post_tran.reshape(2,1)],
                                            axis=1),
                             (crop[2]-crop[0], crop[3]-crop[1]),
                             flags=cv2.INTER_LINEAR)
        return img

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            random_crop_height = \
                self.data_config.get('random_crop_height', False)
            if random_crop_height:
                crop_h = int(np.random.uniform(max(0.3*newH, newH-fH),
                                               newH-fH))
            else:
                crop_h = \
                    int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
            if self.data_config.get('vflip', False) and np.random.choice([0, 1]):
                rotate += 180
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor_transforms(self, cam_info, cam_name):
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def photo_metric_distortion(self, img, pmd):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if np.random.rand()>pmd.get('rate', 1.0):
            return img

        img = np.array(img).astype(np.float32)
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,' \
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(-pmd['brightness_delta'],
                                   pmd['brightness_delta'])
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                       pmd['contrast_upper'])
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(pmd['saturation_lower'],
                                          pmd['saturation_upper'])

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-pmd['hue_delta'], pmd['hue_delta'])
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                       pmd['contrast_upper'])
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]
        return Image.fromarray(img.astype(np.uint8))

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            if self.is_train and self.data_config.get('pmd', None) is not None:
                img = self.photo_metric_distortion(img, self.data_config['pmd'])

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    if self.opencv_pp:
                        img_adjacent = \
                            self.img_transform_core_opencv(
                                img_adjacent,
                                post_rot[:2, :2],
                                post_tran[:2],
                                crop)
                    else:
                        img_adjacent = self.img_transform_core(
                            img_adjacent,
                            resize_dims=resize_dims,
                            crop=crop,
                            flip=flip,
                            rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        imgs = torch.stack(imgs)

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas'] = canvas
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotations(object):

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        return results


@PIPELINES.register_module()
class BEVAug(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
            translation_std = self.bda_aug_conf.get('tran_lim', [0.0, 0.0, 0.0])
            tran_bda = np.random.normal(scale=translation_std, size=3).T
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            tran_bda = np.zeros((1, 3), dtype=np.float32)
        return rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy, tran_bda):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, :3] = gt_boxes[:, :3] + tran_bda
        return gt_boxes, rot_mat

    def __call__(self, results):
        gt_boxes = results['gt_bboxes_3d'].tensor
        gt_boxes[:,2] = gt_boxes[:,2] + 0.5*gt_boxes[:,5]
        rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = \
            self.sample_bda_augmentation()
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy, tran_bda)
        if 'points' in results:
            points = results['points'].tensor
            points_aug = (bda_rot @ points[:, :3].unsqueeze(-1)).squeeze(-1)
            points[:,:3] = points_aug + tran_bda
            points = results['points'].new_point(points)
            results['points'] = points
        bda_mat[:3, :3] = bda_rot
        bda_mat[:3, 3] = torch.from_numpy(tran_bda)
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        if 'img_inputs' in results:
            imgs, rots, trans, intrins = results['img_inputs'][:4]
            post_rots, post_trans = results['img_inputs'][4:]
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                     post_trans, bda_mat)
        if 'voxel_semantics' in results:
            if flip_dx:
                results['voxel_semantics'] = results['voxel_semantics'][::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][::-1,...].copy()
            if flip_dy:
                results['voxel_semantics'] = results['voxel_semantics'][:,::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][:,::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][:,::-1,...].copy()
        return results


# ! BELOW IS OUR NVS PIPELINE FUNCTION
@PIPELINES.register_module()
class PrepareNVSMetaData(object):
    def __init__(
        self,
        meta_data_root,
        sequential=False,
        # gaussian create settings
        use_overlap_mask=True,
        gaussian_scale_ada_fg=0.0025,
        gaussian_scale_ada_obj=0.005,
        gaussian_scale_ada_inpaint=0.001,
        gaussian_scale_ada_bg=[0.02, 0.001, 5],  # 0m->10m, 0.25->0.001 (linearly decrease to fill the road)
        use_fg_inpaint=True,
        add_blind_area=True,
        obj_downsample_ratio=2,
        # camera configuration settings
        tgt_cam_cfg_file=None,
        # tgt_cam_idx=0,        
        align_tgt_intrinsic=True,   
        # align_tgt_extrinsic=False,
        # final size for training
        input_size=(256, 704),
        cam_names=None,
        # max pts to pad
        max_pts=3000000,
        # global flip augmentation
        global_flip_aug=True,
        plain_flip_aug=False,
        fb_flip_aug=False,
        # whether use raw augmentation policy (scale/rotation/flip) when use raw input
        use_raw_aug=False,
        aug_K_on_raw=None,    # [type, r1, r2], type=['range', 'ratio'], aug range: [f*r1, f*r2] focal augmentation cfg
        # whether to use raw images as input
        # p_curr_as_raw=0.,
        # p_adj_as_raw=0.,
        p_raw=0.,
        inpaint_ego_occ_train=False,
        # novel camera generation policy
        aug_ego2global=None,    # list, ego pose augmentation [rx, ry, rz] degrees, rx/y/z -> roll/pitch/yaw
        aug_cam2ego=None,       # list, mounting pose augmentation(by offset) [tx, ty, tz, rx, ry, rz] tz is a tuple (type, a, b)!
        aug_K=None,       # [type, r1, r2], type=['range', 'ratio'], aug range: [f*r1, f*r2]
        integrate_extrinsic_aug=True,
        # use lidar depth to supervise
        use_lidar_depth=False,
        fg_depth_only=False,
        
        # sync params
        nvs_wo_cam_sync=False,
        sync_adj_aug=False,
        
        # test mode
        test_mode=False,
        inpaint_ego_occ_test=False,
        crop_ego_occ_test=False,
        
        # customize augmentation
        custom_aug=None,
        
        # use nvs outpainting for raw resize-to-small image
        outpainting_for_raw=False,
    ):
        self.meta_data_root = meta_data_root
        self.sequential = sequential
        self.input_size = input_size
        self.cam_names = cam_names
        
        self.test_mode = test_mode
        if test_mode:
            assert not(inpaint_ego_occ_test and crop_ego_occ_test)
            self.inpaint_ego_occ_test = inpaint_ego_occ_test
            self.crop_ego_occ_test = crop_ego_occ_test
            self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            return

        self.use_overlap_mask = use_overlap_mask
        self.gaussian_scale_ada_fg = gaussian_scale_ada_fg
        self.gaussian_scale_ada_obj = gaussian_scale_ada_obj
        self.gaussian_scale_ada_inpaint = gaussian_scale_ada_inpaint
        self.gaussian_scale_ada_bg = gaussian_scale_ada_bg
        self.use_fg_inpaint = use_fg_inpaint
        self.add_blind_area = add_blind_area
        self.obj_downsample_ratio = obj_downsample_ratio
        
        self.gs_cfg = dict(
            use_overlap_mask=use_overlap_mask,
            gaussian_scale_ada_fg=gaussian_scale_ada_fg,
            gaussian_scale_ada_obj=gaussian_scale_ada_obj,
            gaussian_scale_ada_inpaint=gaussian_scale_ada_inpaint,
            gaussian_scale_ada_bg=gaussian_scale_ada_bg,
            use_fg_inpaint=use_fg_inpaint,
            add_blind_area=add_blind_area,
        )
        
        if tgt_cam_cfg_file is not None:
            with open(tgt_cam_cfg_file, 'rb') as f:
                self.tgt_cam_cfg = pickle.load(f)
            # self.tgt_cam_idx = tgt_cam_idx
            self.align_tgt_intrinsic = align_tgt_intrinsic
            # self.align_tgt_extrinsic = align_tgt_extrinsic
        else:
            self.tgt_cam_cfg = None    
            
        self.max_pts = max_pts
        
        # augmentations NOTE: we seperately define aug_ego2global and aug_cam2ego, 
        self.global_flip_aug = global_flip_aug  # change results['ann_infos'], results['curr']['ann_infos'], ignore adj(we dont need)
        self.plain_flip_aug = plain_flip_aug
        self.fb_flip_aug = fb_flip_aug
        assert not (global_flip_aug and plain_flip_aug)
        
        self.use_raw_aug = use_raw_aug
        self.aug_ego2global = aug_ego2global
        self.aug_cam2ego = aug_cam2ego
        self.aug_K = aug_K
        self.aug_K_on_raw = aug_K_on_raw
        self.integrate_extrinsic_aug = integrate_extrinsic_aug
        if aug_ego2global is not None:
            assert len(aug_ego2global) == 3
        if aug_cam2ego is not None:
            assert len(aug_cam2ego) == 6
        if aug_K is not None:
            assert len(aug_K) == 3
            assert aug_K[0] in ['range', 'ratio']
        if aug_K_on_raw is not None:
            assert len(aug_K_on_raw) == 3
            assert aug_K_on_raw[0] in ['range', 'ratio']
        
        # self.p_curr_as_raw = p_curr_as_raw
        # self.p_adj_as_raw = p_adj_as_raw
        self.p_raw = p_raw
        self.inpaint_ego_occ_train = inpaint_ego_occ_train
        
        self.use_lidar_depth = use_lidar_depth
        self.fg_depth_only = fg_depth_only
        
        # placeholders
        self.means3D_placeholder = np.zeros((self.max_pts, 3), dtype=np.float32)
        self.rgbs_placeholder = np.zeros((self.max_pts, 3), dtype=np.float32)
        self.scales_placeholder = np.zeros((self.max_pts, 3), dtype=np.float32)
        self.raw_imgs_placeholder = np.zeros((len(self.cam_names), 3, self.input_size[0], self.input_size[1]), dtype=np.float32)
        self.raw_depths_placeholder = np.zeros((len(self.cam_names), self.input_size[0], self.input_size[1]), dtype=np.float32)
    
        # sync parameters
        self.nvs_wo_cam_sync = nvs_wo_cam_sync
        self.sync_adj_aug = sync_adj_aug
    
        # customize augmentaion
        self.custom_aug = custom_aug
        if custom_aug is not None:
            assert custom_aug in ['lyft', 'waymo', 'lyft2nuscenes']
            
        # use nvs outpainting for raw resize-to-small image
        self.outpainting_for_raw = outpainting_for_raw
        if outpainting_for_raw:
            assert isinstance(outpainting_for_raw, float)       # zoomout's scale factor
            
    
    def construct_T_from_vector(self, translation_vector, rotation_vector):
        assert len(translation_vector) == 3
        assert len(rotation_vector) == 4
        
        T = np.eye(4)
        R = Quaternion(*rotation_vector).rotation_matrix
        T[:3, :3] = R
        T[:3, 3] = translation_vector
        return T
        
    
    def cart2homo(self, pts):
        assert pts.shape[-1] == 3
        return np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        
    
    def read_depth_map(self, depth_map_path):
        depth_image = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
        depth_map = depth_image / 256.0

        # Discard depths less than 10cm from the camera
        depth_map[depth_map < 0.1] = 0.0

        return depth_map.astype(np.float32)
    
    
    def collect_raw_cams(self, mminfo, data_dict):
        # collect cameras
        raw_cams = {}
        for sensor in self.cam_names:
            raw_cams[sensor] = {}
            
            h, w = mminfo['cams'][sensor]['img_h'], mminfo['cams'][sensor]['img_w']
            
            K = np.array(mminfo['cams'][sensor]['cam_intrinsic']).astype(np.float32)
            T_cam2ego = self.construct_T_from_vector(
                mminfo['cams'][sensor]['sensor2ego_translation'], 
                mminfo['cams'][sensor]['sensor2ego_rotation']).astype(np.float32)
            T_ego2cam = np.linalg.inv(T_cam2ego)
        
            raw_cams[sensor]['K'] = K
            raw_cams[sensor]['T_ego2cam'] = T_ego2cam
            raw_cams[sensor]['img_w'] = w
            raw_cams[sensor]['img_h'] = h
            
        data_dict['raw_cams'] = raw_cams
    
    
    def collect_one_frame_gaussians_gsfile(self, mminfo, data_dict, use_raw=False, use_global_flip=False, need_depth=False):
        sample_token = mminfo['token']
        
        # collect gaussians/imgs
        if not use_raw:
            gs_file = os.path.join(self.meta_data_root, 'gaussians', f'{sample_token}.npz')
            gs_data = np.load(gs_file)
            
            means3D = (gs_data['means3D']).astype(np.float32)
            rgbs = (gs_data['rgbs'] / 255.).astype(np.float32)
            scales = (gs_data['scales']).astype(np.float32)[:, None].repeat(3, axis=1)
            overlap_mask = gs_data['masks']
            lidar_pts = gs_data['lidar_pts'].astype(np.float32)
            
            if self.use_overlap_mask:
                means3D = means3D[overlap_mask]
                rgbs = rgbs[overlap_mask]
                scales = scales[overlap_mask]
            
            if use_global_flip:
                means3D[:, 1] *= -1
                
            # pad
            raw_pts = len(means3D)
            n_pad = self.max_pts - raw_pts
            means3D = np.pad(means3D, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)
            rgbs = np.pad(rgbs, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)
            scales = np.pad(scales, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)
            
            # get lidar depths
            if need_depth:
                raw_depths = self.collect_lidar_depths(lidar_pts, mminfo, data_dict, use_global_flip)
            else:
                raw_depths = self.raw_depths_placeholder
            
            
            data_dict.update({
                'means3D': means3D, 'rgbs': rgbs, 'scales': scales, 'n_raw_pts': raw_pts,\
                # placeholder
                'raw_imgs': self.raw_imgs_placeholder, 'raw_depths': raw_depths,
                'use_nvs_flag': True,
            })
    
        else:
            raw_imgs = [None for _ in self.cam_names]
            raw_depths = [None for _ in self.cam_names]
            
            for i, sensor in enumerate(self.cam_names):
                img_file = mminfo['cams'][sensor]['data_path']
                if self.use_lidar_depth:
                    if self.outpainting_for_raw:
                        depth_file = os.path.join(self.meta_data_root, f'zoomout_rgbd_{self.outpainting_for_raw}', 'depth', sensor, f'{sample_token}.png')
                    else:
                        depth_file = os.path.join(self.meta_data_root, 'depths', 'lidar_depths', sensor, f'{sample_token}.png')
                else:
                    depth_file = os.path.join(self.meta_data_root, 'depths', 'dense_depth_SPNorm', sensor, f'{sample_token}.png')
                
                img = cv2.imread(img_file)[..., ::-1]
                if need_depth: depth = self.read_depth_map(depth_file)
                
                if self.inpaint_ego_occ_train:
                    inpainted_img_file = os.path.join(self.meta_data_root, 'inpainted', 'img', sensor, f'{sample_token}.png')
                    key_mask_file = os.path.join(self.meta_data_root, 'masks', 'key_mask2d', sensor, f'{sample_token}.png')
                    inpainted_img = cv2.imread(inpainted_img_file)[..., ::-1]
                    key_mask = cv2.imread(key_mask_file)
                    
                    inpainted_mask = (key_mask[..., 2] == 250)
                    img[inpainted_mask] = inpainted_img[inpainted_mask]
                    
                h, w, _ = img.shape
                nh, nw = int(h*self.input_size[1]/w), self.input_size[1]
                img = cv2.resize(img, (nw, nh))
                if need_depth: depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
                
                img = img[nh-self.input_size[0]:, ...]
                if need_depth: depth = depth[nh-self.input_size[0]:, ...]
                
                if use_global_flip:
                    img = img[:, ::-1, :]
                    if need_depth: depth = depth[:, ::-1]
                    if 'LEFT' in sensor:
                        j = self.cam_names.index(sensor.replace('LEFT', 'RIGHT'))
                    elif 'RIGHT' in sensor:
                        j = self.cam_names.index(sensor.replace('RIGHT', 'LEFT'))
                    else:
                        # front / back
                        j = i
                    raw_imgs[j] = img
                    if need_depth: raw_depths[j] = depth
                else:
                    raw_imgs[i] = img
                    if need_depth: raw_depths[i] = depth
                
            raw_imgs = np.stack(raw_imgs).transpose(0, 3, 1, 2)
            if need_depth: 
                raw_depths = np.stack(raw_depths)
            else:
                raw_depths = self.raw_depths_placeholder
                    
            data_dict.update({
                'raw_imgs': (raw_imgs / 255.).astype(np.float32), 'raw_depths': raw_depths.astype(np.float32),
                # placeholder
                'means3D': self.means3D_placeholder, 'rgbs': self.rgbs_placeholder, 'scales': self.scales_placeholder, 'n_raw_pts': -1,
                'use_nvs_flag': False,
            })
            
            if self.outpainting_for_raw:
                # for waymo
                zoomout_imgs = [None for _ in self.cam_names]
                for i, sensor in enumerate(self.cam_names):
                    img_file = os.path.join(self.meta_data_root, f'zoomout_rgbd_{self.outpainting_for_raw}', 'img', sensor, f'{sample_token}.jpg')
                    img = cv2.imread(img_file)[..., ::-1]
                    h, w, _ = img.shape
                    nh, nw = int(h*self.input_size[1]/w), self.input_size[1]
                    img = cv2.resize(img, (nw, nh))
                    img = img[nh-self.input_size[0]:, ...]
                    zoomout_imgs[i] = img
                zoomout_imgs = np.stack(zoomout_imgs).transpose(0, 3, 1, 2)
                data_dict.update({'zoomout_imgs': (zoomout_imgs / 255.).astype(np.float32)})
            
            # ! DEBUG
            # print(f'sample_token:{sample_token} raw_imgs.shape:{raw_imgs.shape} raw_depths.shape:{raw_depths.shape}')
        
        
        # NOTE: DEBUG
        # debug_imgs = {}
        # for sensor in self.cam_names:
        #     _dbimg = cv2.imread(mminfo['cams'][sensor]['data_path'])
        #     _dbimg = _dbimg[-450:, :, :]
        #     debug_imgs[sensor] = _dbimg
        #     # debug_imgs[sensor] = cv2.imread(mminfo['cams'][sensor]['data_path'])
        # data_dict['debug_imgs'] = debug_imgs
    
    
    def collect_nvs_cameras(self, data_dict, T_ego_perturbed, raw_flag):
        def _adjust_to_train_size(nvs_cams):
            for cam in nvs_cams.keys():
                # adjust to training size
                train_w = self.input_size[1]
                train_rs = train_w / nvs_cams[cam]['img_w']
                train_h = int(nvs_cams[cam]['img_h'] * train_rs)
                train_K = nvs_cams[cam]['K']
                train_K[:2, :] *= train_rs
        
                nvs_cams[cam]['K'] = train_K
                nvs_cams[cam]['img_w'] = train_w
                nvs_cams[cam]['img_h'] = train_h
            return nvs_cams
        
        if (self.tgt_cam_cfg is None) or raw_flag:
            nvs_cams = copy.deepcopy(data_dict['raw_cams'])
        else:
            # idx = np.random.choice([-1] + list(range(len(self.tgt_cam_cfg))))
            # if idx == -1:
            #     nvs_cams = data_dict['raw_cams']
            # else:
            idx = np.random.choice(list(range(len(self.tgt_cam_cfg))))
            tgt_cams = self.tgt_cam_cfg[idx]
            nvs_cams = copy.deepcopy(data_dict['raw_cams'])
            
            # for cam in nvs_cams.keys():
            for cam in self.cam_names:
                if cam in tgt_cams.keys():
                    _cam = cam
                else:
                    if ('BACK_LEFT' in cam) or ('BACK_RIGHT' in cam):
                        # raw=['nuscenes', 'lyft'] tgt=['waymo']
                        _cam = cam.replace('BACK', 'SIDE')
                    elif 'SIDE' in cam:
                        # raw=['waymo'] tgt=['nuscenes', 'lyft']
                        _cam = cam.replace('SIDE', 'BACK')
                    elif cam == 'CAM_BACK':
                        # raw=['nuscenes', 'lyft'] tgt=['waymo']
                        _cam = 'CAM_FRONT'
                    else:
                        raise ValueError(f'Dontnot support {cam}!')
            
                if self.align_tgt_intrinsic:
                    nvs_cams[cam]['K'] = tgt_cams[_cam]['K'].astype(np.float32)
                    nvs_cams[cam]['img_w'] = tgt_cams[_cam]['img_w']
                    nvs_cams[cam]['img_h'] = tgt_cams[_cam]['img_h']
            
                # if self.align_tgt_extrinsic:
                if (cam == 'CAM_BACK') and (_cam == 'CAM_FRONT'):
                    # only adjust camera height
                    nvs_cams[cam]['T_ego2cam'][1, 3] = tgt_cams[_cam]['T_ego2cam'][1, 3].astype(np.float32)
                else:
                    nvs_cams[cam]['T_ego2cam'] = tgt_cams[_cam]['T_ego2cam'].astype(np.float32)
        
        # adjust to training size
        nvs_cams = _adjust_to_train_size(nvs_cams)
        
        cam_groups = NuscenesCameraGroups(nvs_cams, self.aug_ego2global, 
                                          self.aug_cam2ego, self.aug_K, 
                                          self.integrate_extrinsic_aug,
                                          custom_aug=self.custom_aug
                                          )
        if ((not self.aug_ego2global) and (not self.aug_cam2ego) and (not self.aug_K)) or raw_flag:
            c2ws, fovxs, fovys, camera_args, crop_start, crop_end = cam_groups.gen_raw_cam_for_gaussian()
        else:
            c2ws, fovxs, fovys, camera_args, crop_start, crop_end = cam_groups.gen_new_cam_for_gaussian(T_ego_perturbed)
            
        data_dict.update({
            # 'nvs_cams': nvs_cams,
            'nvs_cams': cam_groups.new_cams,
            'c2ws': c2ws, 'fovxs': fovxs, 'fovys': fovys, 'camera_args': camera_args, 'crop_start': crop_start, 'crop_end': crop_end,
        })
        
    
    def collect_img_inputs(self, nvs_meta, curr_mminfo, adj_mminfos, raw_flag):
        n_frame = len(adj_mminfos) + 1
        b = len(self.cam_names) * n_frame
        
        # placeholders
        imgs = np.zeros((b, 3, 10, 10), dtype=np.float32)
        gt_depth = np.zeros((len(self.cam_names), 10, 10), dtype=np.float32)
        img_rot_aug = np.eye(3, dtype=np.float32)[None, ...].repeat(b, axis=0)
        img_trans_aug = np.zeros((b, 3), dtype=np.float32)
        
        # others
        T_sensor2egos = []
        T_ego2globals = []
        Ks = []
        
        for cam in self.cam_names:
            T_sensor2egos.append(np.linalg.inv(nvs_meta['curr']['nvs_cams'][cam]['T_ego2cam']))
            if raw_flag:
                T_ego2globals.append(self.construct_T_from_vector(curr_mminfo['cams'][cam]['ego2global_translation'],
                                                                curr_mminfo['cams'][cam]['ego2global_rotation']))
            else:
                T_ego2globals.append(self.construct_T_from_vector(curr_mminfo['ego2global_translation'],
                                                                curr_mminfo['ego2global_rotation']))
            # crop cv
            K = nvs_meta['curr']['nvs_cams'][cam]['K'].copy()
            K[1, 2] -= (nvs_meta['curr']['nvs_cams'][cam]['img_h'] - self.input_size[0])
            Ks.append(K)
            
        for i in range(len(adj_mminfos)):
            for cam in self.cam_names:
                T_sensor2egos.append(np.linalg.inv(nvs_meta['adjacent'][i]['nvs_cams'][cam]['T_ego2cam']))
                if raw_flag:
                    T_ego2globals.append(self.construct_T_from_vector(adj_mminfos[i]['cams'][cam]['ego2global_translation'],
                                                                    adj_mminfos[i]['cams'][cam]['ego2global_rotation']))
                else:
                    T_ego2globals.append(self.construct_T_from_vector(adj_mminfos[i]['ego2global_translation'],
                                                                    adj_mminfos[i]['ego2global_rotation']))
                K = nvs_meta['adjacent'][i]['nvs_cams'][cam]['K'].copy()
                K[1, 2] -= (nvs_meta['adjacent'][i]['nvs_cams'][cam]['img_h'] - self.input_size[0])
                Ks.append(K)
        
        T_sensor2egos = np.stack(T_sensor2egos, axis=0)
        T_ego2globals = np.stack(T_ego2globals, axis=0)
        Ks = np.stack(Ks, axis=0)
        
        img_inputs = (imgs, T_sensor2egos, T_ego2globals, Ks, img_rot_aug, img_trans_aug)
    
        return img_inputs, gt_depth
      
    def flip_annotations(self, results, flip_lidar_boxes=False):
        """
        only flip annotations in curr frame
        """
        mminfo = results['curr']
        ego_boxes = np.array(mminfo['ann_infos'][0])
        # [x, y, z, dx(l), dy(w), dz(h), yaw, vx, vy], 1,6,8
        ego_boxes[:, [1,6,8]] *= -1
        results['ann_infos'] = list(results['ann_infos'])
        results['curr']['ann_infos'] = list(results['curr']['ann_infos'])
        
        results['ann_infos'][0] = list(ego_boxes)
        results['curr']['ann_infos'][0] = list(ego_boxes)
        
        results['ann_infos'] = tuple(results['ann_infos'])
        results['curr']['ann_infos'] = tuple(results['curr']['ann_infos'])
        
        if flip_lidar_boxes:
            lidar_boxes = mminfo['gt_boxes']
            lidar_velo = mminfo['gt_velocity']
            T_lidar2ego = construct_T_from_vector(
                mminfo['lidar2ego_translation'],
                mminfo['lidar2ego_rotation'])
            T_ego2lidar = np.linalg.inv(T_lidar2ego)
            
            xyz_lidar = lidar_boxes[:, :3]
            yaw_lidar = lidar_boxes[:, 6]
            
            # flip trans
            T_flip_ego = np.eye(4)
            T_flip_ego[1, 1] = -1
            T_flip_lidar = T_ego2lidar @ T_flip_ego @ T_lidar2ego
            R_flip_lidar = T_flip_lidar[:3, :3]
        
            # flip
            xyz_lidar = (self.cart2homo(xyz_lidar) @ T_flip_lidar.T)[:, :3]
            yaw_lidar = vector_to_yaw(yaw_to_unit_vector(yaw_lidar) @ R_flip_lidar.T)
            lidar_velo = np.concatenate([lidar_velo, np.zeros_like(lidar_velo[:, 0:1])], axis=-1)
            lidar_velo = (lidar_velo @ R_flip_lidar.T)[:, :2]

            # refresh
            results['curr']['gt_boxes'][:, :3] = xyz_lidar
            results['curr']['gt_boxes'][:, 6] = yaw_lidar
            results['curr']['gt_velocity'][:, 6] = lidar_velo
        

    def collect_lidar_depths(self, lidar_pts, mminfo, nvs_meta, use_global_flip):
        MAX_DEPTH = 200
        
        T_lidar2ego = self.construct_T_from_vector(
            mminfo['lidar2ego_translation'],
            mminfo['lidar2ego_rotation'])
        
        T_flip_ego = np.eye(4)
        if use_global_flip:
            T_flip_ego[1, 1] = -1
            
        T_lidar2imgs = []
        for cam in self.cam_names:
            T_ego2cam = nvs_meta['nvs_cams'][cam]['T_ego2cam']
            K = nvs_meta['nvs_cams'][cam]['K'].copy()
            K[1, 2] -= (nvs_meta['nvs_cams'][cam]['img_h'] - self.input_size[0])
            T_cam2img = np.eye(4)
            T_cam2img[:3, :3] = K
            
            T_lidar2img = T_cam2img @ T_ego2cam @ T_flip_ego @ T_lidar2ego
            T_lidar2imgs.append(T_lidar2img)
        T_lidar2imgs = np.stack(T_lidar2imgs)   # (n_cam, 4, 4)
        
        # batch transform
        img_pts = (self.cart2homo(lidar_pts) @ T_lidar2imgs.transpose(0, 2, 1))[:, :, :3]
    
        lidar_depths = []
        for i, cam in enumerate(self.cam_names):
            img_h, img_w = self.input_size
            
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
        lidar_depths = np.stack(lidar_depths)
        
        return lidar_depths
    
        
    def raw_style_augmentation(self, img_inputs, nvs_meta):
        def _get_rot(h):
            return np.array([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])
            
        def _img_transform_core_opencv(img, post_rot, post_tran, crop, is_depth=False):
            if is_depth:
                _flag = cv2.INTER_NEAREST
            else:
                _flag = cv2.INTER_LINEAR
                img = img.transpose(1, 2, 0)
                
                
            img = cv2.warpAffine(img.astype(np.float32),
                                 np.concatenate([post_rot,
                                                 post_tran.reshape(2,1)],
                                                 axis=1),
                                 (crop[2]-crop[0], crop[3]-crop[1]),
                                 flags=_flag)
            
            if not is_depth:
                img = img.transpose(2, 0, 1)
            
            return img
            
        def _aug_K(img, K, ratio):
            # assert ratio > 1    # we only apply zoom-in aug in raw, for zoom-out, there will occur black edge
            
            fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            if len(img.shape) == 3:
                img = img.transpose(1, 2, 0)
                h, w, _ = img.shape
                _flag = cv2.INTER_LINEAR
            else:
                h, w = img.shape
                _flag = cv2.INTER_NEAREST
            
            # dist from edge to priciple point
            l, r, t, d = cu, w - cu, cv, h - cv
            if ratio > 1:
                crop = (int(cu - l/ratio), int(cv - t/ratio), int(cu + r/ratio), int(cv + d/ratio))
                img = img[crop[1]:crop[3], crop[0]:crop[2]]
                img = cv2.resize(img, (w, h), interpolation=_flag)
            elif ratio < 1:
                new_w, new_h = int(w * ratio), int(h * ratio)
                _u, _v = int(cu - l*ratio), int(cv - t*ratio)
                tmp = np.zeros_like(img)
                img = cv2.resize(img, (new_w, new_h), interpolation=_flag)      
                tmp[_v:_v+new_h, _u:_u+new_w] = img
                img = tmp
            
            if len(img.shape) == 3:
                img = img.transpose(2, 0, 1)
                
            return img
            
            
        n_cam = len(self.cam_names)
        n_adj = len(nvs_meta['adjacent'])        
        imgs, T_sensor2egos, T_ego2globals, Ks, img_rot_aug, img_trans_aug = img_inputs
        
        if self.tgt_cam_cfg:
            idx = np.random.choice([-1] + list(range(len(self.tgt_cam_cfg))))
            
        for i in range(n_cam):
            ratio = 1.
            if self.tgt_cam_cfg:
                if idx != -1:
                    # adjust cam names
                    _src_cam = self.cam_names[i]
                    if _src_cam in self.tgt_cam_cfg[idx].keys():
                        _dst_cam = _src_cam
                    else:
                        if ('BACK_LEFT' in _src_cam) or ('BACK_RIGHT' in _src_cam):
                            # raw=['nuscenes', 'lyft'] tgt=['waymo']
                            _dst_cam = _src_cam.replace('BACK', 'SIDE')
                        elif 'SIDE' in _src_cam:
                            # raw=['waymo'] tgt=['nuscenes', 'lyft']
                            _dst_cam = _src_cam.replace('SIDE', 'BACK')
                        elif _src_cam == 'CAM_BACK':
                            # raw=['nuscenes', 'lyft'] tgt=['waymo']
                            _dst_cam = 'CAM_FRONT'
                        else:
                            raise ValueError(f'Dontnot support {cam}!')
                        
                    # _tgt_f = self.input_size[1] / self.tgt_cam_cfg[idx][self.cam_names[i]]['img_w'] * self.tgt_cam_cfg[idx][self.cam_names[i]]['K'][0, 0]
                    _tgt_f = self.input_size[1] / self.tgt_cam_cfg[idx][_dst_cam]['img_w'] * self.tgt_cam_cfg[idx][_dst_cam]['K'][0, 0]
                    _src_f = Ks[i][0, 0]
                    ratio = _tgt_f / _src_f
                    
            # NOTE:focal augmentation on raw image
            # ! 如果进行了内参增强 那么原始增强就不进行resize增强了
            if self.aug_K_on_raw:
                if self.aug_K_on_raw[0] == 'ratio':
                    ratio = np.random.uniform(ratio*self.aug_K_on_raw[1], ratio*self.aug_K_on_raw[2])
                elif self.aug_K_on_raw[0] == 'range':
                    ratio = np.random.uniform(self.aug_K_on_raw[1], self.aug_K_on_raw[2]) / (Ks[i][0, 0] * ratio)
                    
            # NOTE:ratio<1时图像要缩小我们不进行这操作 ratio=1时没变化也不进行这操作
            # if ratio > 1:
            if ratio != 1:
                if self.outpainting_for_raw:
                    zoomout_ratio = self.outpainting_for_raw
                    zoomout_K = Ks[i].copy()
                    zoomout_K[:2, :2] *= zoomout_ratio
                    nvs_meta['curr']['raw_imgs'][i] = _aug_K(nvs_meta['curr']['raw_imgs'][i], Ks[i], ratio)
                    nvs_meta['curr']['zoomout_imgs'][i] = _aug_K(nvs_meta['curr']['zoomout_imgs'][i], zoomout_K, ratio/zoomout_ratio)
                    nvs_meta['curr']['raw_depths'][i] = _aug_K(nvs_meta['curr']['raw_depths'][i], zoomout_K, ratio/zoomout_ratio)
                    Ks[i][:2, :2] *= ratio
                    zoomout_mask = (nvs_meta['curr']['raw_imgs'][i]==0)
                    nvs_meta['curr']['raw_imgs'][i][zoomout_mask] = nvs_meta['curr']['zoomout_imgs'][i][zoomout_mask]
                    # del nvs_meta['curr']['zoomout_imgs'][i]
                    for j, _meta in enumerate(nvs_meta['adjacent']):
                        zoomout_K = Ks[n_cam*(j+1)+i].copy()
                        zoomout_K[:2, :2] *= zoomout_ratio
                        nvs_meta['adjacent'][j]['raw_imgs'][i] = _aug_K(nvs_meta['adjacent'][j]['raw_imgs'][i], Ks[n_cam*(j+1)+i], ratio)
                        nvs_meta['adjacent'][j]['zoomout_imgs'][i] = _aug_K(nvs_meta['adjacent'][j]['zoomout_imgs'][i], zoomout_K, ratio/zoomout_ratio)
                        nvs_meta['adjacent'][j]['raw_depths'][i] = _aug_K(nvs_meta['adjacent'][j]['raw_depths'][i], zoomout_K, ratio/zoomout_ratio)
                        Ks[n_cam*(j+1)+i][:2, :2] *= ratio
                        zoomout_mask = (nvs_meta['adjacent'][j]['raw_imgs'][i]==0)
                        nvs_meta['adjacent'][j]['raw_imgs'][i][zoomout_mask] = nvs_meta['adjacent'][j]['zoomout_imgs'][i][zoomout_mask]
                        # del nvs_meta['adjacent'][j]['zoomout_imgs'][i]
                    
                else:
                    nvs_meta['curr']['raw_imgs'][i] = _aug_K(nvs_meta['curr']['raw_imgs'][i], Ks[i], ratio)
                    nvs_meta['curr']['raw_depths'][i] = _aug_K(nvs_meta['curr']['raw_depths'][i], Ks[i], ratio)
                    Ks[i][:2, :2] *= ratio      
                    for j, _meta in enumerate(nvs_meta['adjacent']):
                        nvs_meta['adjacent'][j]['raw_imgs'][i] = _aug_K(nvs_meta['adjacent'][j]['raw_imgs'][i], Ks[n_cam*(j+1)+i], ratio)
                        nvs_meta['adjacent'][j]['raw_depths'][i] = _aug_K(nvs_meta['adjacent'][j]['raw_depths'][i], Ks[n_cam*(j+1)+i], ratio)
                        Ks[n_cam*(j+1)+i][:2, :2] *= ratio  
                
            # sample augmentation
            h, w = self.input_size
            if self.aug_K_on_raw:
                resize = 1.
            else:
                resize = 1 + np.random.uniform(-0.06, 0.11) / w * 1600
            new_h, new_w = int(h*resize), int(w*resize)
            resize_dims = (new_w, new_h)
            crop_h = new_h - h
            crop_w = int(np.random.uniform(0, max(0, new_w - w)))
            crop = (crop_w, crop_h, crop_w + w, crop_h + h)
            rotate = np.random.uniform(-5.4, 5.4)

            # post-homography transformation
            post_rot = np.eye(2)
            post_tran = np.zeros(2)
            post_rot *= resize
            post_tran -= np.array(crop[:2])

            A = _get_rot(rotate / 180 * np.pi)
            b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A @ (-b) + b
            post_rot = A @ (post_rot)
            post_tran = A @ (post_tran) + b
            
            # apply transformation
            nvs_meta['curr']['raw_imgs'][i] = _img_transform_core_opencv(
                nvs_meta['curr']['raw_imgs'][i],
                post_rot, post_tran, crop, is_depth=False)
            nvs_meta['curr']['raw_depths'][i] = _img_transform_core_opencv(
                nvs_meta['curr']['raw_depths'][i],
                post_rot, post_tran, crop, is_depth=True)
            img_rot_aug[i, :2, :2] = post_rot
            img_trans_aug[i, :2] = post_tran
            
            for j, _meta in enumerate(nvs_meta['adjacent']):
                nvs_meta['adjacent'][j]['raw_imgs'][i] = _img_transform_core_opencv(
                    nvs_meta['adjacent'][j]['raw_imgs'][i],
                    post_rot, post_tran, crop, is_depth=False)
                nvs_meta['adjacent'][j]['raw_depths'][i] = _img_transform_core_opencv(
                    nvs_meta['adjacent'][j]['raw_depths'][i],
                    post_rot, post_tran, crop, is_depth=True)
                img_rot_aug[n_cam*(j+1)+i, :2, :2] = post_rot
                img_trans_aug[n_cam*(j+1)+i, :2] = post_tran
            
        img_inputs = (imgs, T_sensor2egos, T_ego2globals, Ks, img_rot_aug, img_trans_aug)
        
        #NOTE: delete zoomout imgs
        if self.outpainting_for_raw:
            del nvs_meta['curr']['zoomout_imgs']
            for _meta in nvs_meta['adjacent']:
                del _meta['zoomout_imgs']
        
        return img_inputs, nvs_meta
               
    
    def plain_flip_augmentation(self, img_inputs):
        T_flip = np.eye(3, dtype=np.float32)
        T_flip[0, 0] = -1
        T_flip[0, 2] = self.input_size[1]
        
        imgs, T_sensor2egos, T_ego2globals, Ks, img_rot_aug, img_trans_aug = img_inputs
        T_img_aug = img_rot_aug.copy()
        T_img_aug[:, :2, 2] = img_trans_aug[:, :2].copy()
        
        flip_mask = np.random.rand(len(img_rot_aug),) < 0.5
        T_img_aug[flip_mask] = T_flip[None, ...] @ T_img_aug[flip_mask]
        
        img_rot_aug[:, :2, :2] = T_img_aug[:, :2, :2]
        img_trans_aug[:, :2] = T_img_aug[:, :2, 2]
        
        img_inputs = (imgs, T_sensor2egos, T_ego2globals, Ks, img_rot_aug, img_trans_aug)
        return img_inputs
            
    def nvs_wo_cam_sync_adapt(self, curr_mminfo, adj_mminfos):
        update_keys = ['sensor2ego_translation', 'sensor2ego_rotation', 
                       'ego2global_translation', 'ego2global_rotation', 
                       'sensor2lidar_translation', 'sensor2lidar_rotation']
        
        for cam in self.cam_names:
            for key in update_keys:
                curr_mminfo['cams'][cam][key] = curr_mminfo['cams'][cam][f'{key}_unsync']
        for adj_mminfo in adj_mminfos:
            for cam in self.cam_names:
                for key in update_keys:
                    adj_mminfo['cams'][cam][key] = adj_mminfo['cams'][cam][f'{key}_unsync']
                
             
    def mask_fg_depth(self, depths, mminfo, img_inputs):
        ego_boxes = np.array(mminfo['ann_infos'][0])
        cls_mask = np.array(mminfo['ann_infos'][1]) == 0  # car only for experiments
        if not (self.fg_depth_only == 'car'):
            raise NotImplementedError   # for debug
            cls_mask = np.ones_like(cls_mask)   
        ego_boxes = ego_boxes[cls_mask]
        
        n_obj = len(ego_boxes)
        if n_obj == 0:
            return np.zeros_like(depths)
        
        # Step 1 : get object corners in local space
        objs_dims = ego_boxes[:, 3:6]   # dx, dy, dz
        
        local_corners = np.array([
            [ 0.5,  0.5,  0.5],
            [ 0.5, -0.5,  0.5],
            [-0.5, -0.5,  0.5],
            [-0.5,  0.5,  0.5],
            [ 0.5,  0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5,  0.5, -0.5],
        ])  # (8, 3)
        
        local_corners = local_corners[None, :, :] * objs_dims[:, None, :]    # (n_obj, 8, 3)
        
        # Step 2 : init transformation from local to cam, cam to img
        if 'nuscenes' in self.meta_data_root:
            _centers = ego_boxes[:, :3]
            _yaws = ego_boxes[:, 6]
            cos_yaw = np.cos(_yaws)
            sin_yaw = np.sin(_yaws)
            zeros = np.zeros_like(_yaws)
            ones = np.ones_like(_yaws)
            _R = np.stack([
                np.stack([cos_yaw, -sin_yaw, zeros], axis=1),
                np.stack([sin_yaw,  cos_yaw, zeros], axis=1),
                np.stack([zeros,    zeros,   ones],  axis=1),
            ], axis=1)  # (N, 3, 3)
            T_local2egos = np.eye(4)[None, :, :].repeat(n_obj, axis=0)
            T_local2egos[:, :3, :3] = _R
            T_local2egos[:, :3, 3] = _centers
        else:
            T_local2egos = mminfo['T_local2egos'][cls_mask]       # (n_obj, 4, 4)
        
        
        T_cam2egos = img_inputs[1][:len(self.cam_names)]
        T_ego2cams = np.linalg.inv(T_cam2egos)
        Ks = img_inputs[3][:len(self.cam_names)]
        T_img_aug = img_inputs[4][:len(self.cam_names)].copy()
        T_img_aug[:, :2, 2] = img_inputs[5][:len(self.cam_names)][:, :2].copy()
        T_cam2imgs = T_img_aug @ Ks
        T_cam2imgs_homo = np.eye(4)[None, ...].repeat(len(self.cam_names), axis=0)
        T_cam2imgs_homo[:, :3, :3] = T_cam2imgs
        
        # Step 3 : project each object corners to each image space and get fg mask
        depth_mask = []
        for i in range(len(self.cam_names)):
            _mask = np.zeros((self.input_size[0], self.input_size[1]), dtype=bool)
            T_local2imgs = T_cam2imgs_homo[i] @ T_ego2cams[i] @ T_local2egos
            for _corner, T_local2img in zip(local_corners, T_local2imgs):
                # get img corner
                _corner = (self.cart2homo(_corner) @ T_local2img.T)[:, :3]
                
                # scan each object
                _corner_mask = _corner[:, -1] > 0
                if np.all(~_corner_mask): continue
                
                _corner = _corner[_corner_mask]
                _corner = _corner[:, :2] / _corner[:, 2:3]  # corner uv
                
                _corner[:, 0] = np.clip(_corner[:, 0], 0, self.input_size[1])
                _corner[:, 1] = np.clip(_corner[:, 1], 0, self.input_size[0])
                _u1, _u2 = int(_corner[:, 0].min()), int(_corner[:, 0].max())
                _v1, _v2 = int(_corner[:, 1].min()), int(_corner[:, 1].max())
                
                _mask[_v1:_v2, _u1:_u2] = True
            depth_mask.append(_mask)
        depth_mask = np.stack(depth_mask, axis=0)
        
        depths[~depth_mask] = 0.
        
        return depths
                

    def get_inputs_test(self, results):
        def _collect_sample_imgs(mminfo):
            sample_token = mminfo['token']
            imgs = []
            for i, sensor in enumerate(self.cam_names):
                img_file = mminfo['cams'][sensor]['data_path']
                img = cv2.imread(img_file)[..., ::-1]

                h, w, _ = img.shape
                nh, nw = int(h*self.input_size[1]/w), self.input_size[1]
                img = cv2.resize(img, (nw, nh))
                img = img[nh-self.input_size[0]:, ...]
                
                # img = ((img[..., ::-1]/255. - self.rgb_mean) / self.rgb_std).astype(np.float32)
                img = ((img/255. - self.rgb_mean) / self.rgb_std).astype(np.float32)
                
                imgs.append(img.transpose(2, 0, 1))
            
            return np.stack(imgs)   # (v, c, h, w)
                
        def _collect_sample_Ks(mminfo):
            Ks = []
            for sensor in self.cam_names:
                h, w = mminfo['cams'][sensor]['img_h'], mminfo['cams'][sensor]['img_w']
                K = np.array(mminfo['cams'][sensor]['cam_intrinsic']).astype(np.float32)
            
                # adjust to train size
                train_w = self.input_size[1]
                train_rs = train_w / w
                train_h = int(h * train_rs)
                K[:2, :] *= train_rs
                
                # crop cv
                K[1, 2] -= (train_h - self.input_size[0])
                Ks.append(K)
            return np.stack(Ks)
            

        curr_mminfo = results['curr']
        adj_mminfos = results['adjacent']
        
        # 0. imgs (bchw, b=n_cam*n_adj)
        imgs = []
        imgs.append(_collect_sample_imgs(curr_mminfo))
        for adj_mminfo in adj_mminfos:
            imgs.append(_collect_sample_imgs(adj_mminfo))
        imgs = np.concatenate(imgs, axis=0)   # (b, c, h, w)
        # adjust the order to adapt bevdet
        n_adj = len(adj_mminfos)
        n_cam = len(imgs) // (n_adj + 1)
        order = [int(i+j*n_cam) for i in range(n_cam) for j in range(n_adj+1)]
        imgs = imgs[order]
        
        # 1. T_sensor2ego (b44) & 2. T_ego2global (b44)
        T_sensor2egos = []
        T_ego2globals = []
        for sensor in self.cam_names:
            T_sensor2egos.append(self.construct_T_from_vector(
                curr_mminfo['cams'][sensor]['sensor2ego_translation'], 
                curr_mminfo['cams'][sensor]['sensor2ego_rotation']).astype(np.float32))
            T_ego2globals.append(self.construct_T_from_vector(
                curr_mminfo['cams'][sensor]['ego2global_translation'],
                curr_mminfo['cams'][sensor]['ego2global_rotation']).astype(np.float32))
        for adj_mminfo in adj_mminfos:
            for sensor in self.cam_names:
                T_sensor2egos.append(self.construct_T_from_vector(
                    adj_mminfo['cams'][sensor]['sensor2ego_translation'], 
                    adj_mminfo['cams'][sensor]['sensor2ego_rotation']).astype(np.float32))
                T_ego2globals.append(self.construct_T_from_vector(
                    adj_mminfo['cams'][sensor]['ego2global_translation'],
                    adj_mminfo['cams'][sensor]['ego2global_rotation']).astype(np.float32))
        T_sensor2egos = np.stack(T_sensor2egos, axis=0)   
        T_ego2globals = np.stack(T_ego2globals, axis=0)
        
        # 3. K (b33)
        Ks = []
        Ks.append(_collect_sample_Ks(curr_mminfo))
        for adj_mminfo in adj_mminfos:
            Ks.append(_collect_sample_Ks(adj_mminfo))
        Ks = np.concatenate(Ks, axis=0)
        
        # 4. img_rot_aug (b33)
        img_rot_aug = np.eye(3, dtype=np.float32)[None, ...].repeat(len(imgs), axis=0)
        
        # 5. img_trans_aug (b3)
        img_trans_aug = np.zeros((len(imgs), 3), dtype=np.float32)
        
        img_inputs = (imgs, T_sensor2egos, T_ego2globals, Ks, img_rot_aug, img_trans_aug)
        
        return img_inputs
    
        
    def get_inputs(self, results, mode='gs_file'):
        """
        results['curr'], results['adjacent'][i] 都是一份mminfo
        """
        # Step 1 : init
        # assert mode in ['plain', 'post_pro', 'gs_map', 'gs_file']
        assert mode in ['gs_file']
        if mode == 'gs_file':
            _nvs_meta = dict(
                means3D=[], rgbs=[], scales=[], n_raw_pts=[],  # gaussians
                raw_cams=[], nvs_cams=[],                               # raw and nvs cams config (K, T_cam2ego, T_ego2cam, h, w)
                c2ws=[], fovxs=[], fovys=[], camera_args=[], crop_start=[], crop_end=[],                  # gaussian render parameters
                input_size=self.input_size,     # training size
                debug_imgs=[],                # for debug
                # raw inputs : raw_depths is lidar_depth or dense_depth
                raw_imgs=[], raw_depths=[], use_nvs_flag=True, use_lidar_depth=self.use_lidar_depth,
            )

        nvs_meta = {
            'curr': copy.deepcopy(_nvs_meta),
            'adjacent': [],
        }
        if self.sequential:
            nvs_meta['adjacent'] = [copy.deepcopy(_nvs_meta) for _ in range(len(results['adjacent']))]
        
        # curr_as_raw_flag = np.random.rand() < self.p_curr_as_raw
        # adj_as_raw_flag = np.random.rand() < self.p_adj_as_raw
        raw_flag = np.random.rand() < self.p_raw
        global_flip_flag = np.random.rand() < (self.global_flip_aug * 0.5)
        
        # Step 2 : load mminfo
        curr_mminfo = results['curr']
        adj_mminfos = results['adjacent']
        
        #! NOTE: BUG FOR WAYMO. In these scenes, articulated buses are annotated as two seperated car, the hinge are seen as background in Waymo
        #! so that in ego-centric gaussians contruction, lidar points in the hinge cannot be filter, then the meshes depth in these scene are too noisy for use. 
        if curr_mminfo['scene_token'] in ['0001', '0340', '0633']:
            raw_flag = True
        
        # if self.nvs_wo_cam_sync:
        if (not raw_flag) and self.nvs_wo_cam_sync:
            self.nvs_wo_cam_sync_adapt(curr_mminfo, adj_mminfos)
        
        # Step 3 : collect raw cams
        self.collect_raw_cams(curr_mminfo, nvs_meta['curr'])
        for i, adj_mminfo in enumerate(adj_mminfos):
            self.collect_raw_cams(adj_mminfo, nvs_meta['adjacent'][i])
                            
        # Step 4 : create new camera configurations
        if self.aug_ego2global and (not raw_flag):
            T_ego_perturbed = NuscenesCameraGroups._recompose_extrinsic(
                angles_deg=[
                    np.random.uniform(-self.aug_ego2global[0], self.aug_ego2global[0]),
                    np.random.uniform(-self.aug_ego2global[1], self.aug_ego2global[1]),
                    np.random.uniform(-self.aug_ego2global[2], self.aug_ego2global[2]),
                    ],
                t=np.zeros((3,))
            )
            # T_ego_perturbed = NuscenesCameraGroups._recompose_extrinsic(angles_deg=[0, 0, 0], t=[0, 0, 0])
        else:
            T_ego_perturbed = np.eye(4)
        self.collect_nvs_cameras(nvs_meta['curr'], T_ego_perturbed, raw_flag)
        for _meta in nvs_meta['adjacent']:
            if self.sync_adj_aug:
                for _key in ['nvs_cams', 'c2ws', 'fovxs', 'fovys', 'camera_args', 'crop_start', 'crop_end']:
                    _meta[_key] = nvs_meta['curr'][_key]
            else:
                self.collect_nvs_cameras(_meta, T_ego_perturbed, raw_flag)
            
        # Step 5 : create img_inputs placeholder
        # (b=n_cam*n_adj) 0: bchw img ; 1: b44 T_sensor2ego; 2: b44 T_ego2global; 3: b33 K; 4: b33 img_rot_aug; 5: b3 img_trans_aug;
        img_inputs, gt_depth = self.collect_img_inputs(nvs_meta, curr_mminfo, adj_mminfos, raw_flag)
        
        # Step 6 : process global flip augmentation (change annotations)
        # if global_flip_flag and (len(results['ann_infos'][0]) != 0):
        #     self.flip_annotations(results)
        if global_flip_flag:
            raise NotImplementedError
            if (len(results['ann_infos'][0]) != 0):
                self.flip_annotations(results)
            self.flip_ego_pose(img_inputs, raw_flag)
            
        # Step 7 : collect gaussians (we move here to adjust the lidar depth collection)
        if mode == 'gs_file':
            # only curr frame need depth to supervision
            self.collect_one_frame_gaussians_gsfile(curr_mminfo, nvs_meta['curr'], raw_flag, global_flip_flag, need_depth=True)
            for i, adj_mminfo in enumerate(adj_mminfos):
                self.collect_one_frame_gaussians_gsfile(adj_mminfo, nvs_meta['adjacent'][i], raw_flag, global_flip_flag)

        # Step 8*: raw style augmentation
        if raw_flag and (self.use_raw_aug) and (np.random.rand() < 1.0):    #and (not global_flip_flag) 
            img_inputs, nvs_meta = self.raw_style_augmentation(img_inputs, nvs_meta)

        # Step 9*: whether to filter fg depth only
        if self.fg_depth_only:
            nvs_meta['curr']['raw_depths'] = self.mask_fg_depth(nvs_meta['curr']['raw_depths'], curr_mminfo, img_inputs)

        # Step 10*: plain flip augmentation
        if self.plain_flip_aug:
            img_inputs = self.plain_flip_augmentation(img_inputs)

        return nvs_meta, img_inputs, gt_depth
    
 
    def __call__(self, results):
        if not self.test_mode:
            results['nvs_meta'], results['img_inputs'], results['gt_depth'] = self.get_inputs(results)
        else:
            results['img_inputs'] = self.get_inputs_test(results)
        return results
        
        

@PIPELINES.register_module()
class PrepareNVSMetaDataMix(PrepareNVSMetaData):
    def __init__(
        self,
        nuscenes_meta_data_root,
        lyft_meta_data_root,
        waymo_meta_data_root,
        sequential=False,
        # final size for training
        input_size=(256, 704),
        cam_names=None,
        # max pts to pad
        max_pts=3000000,
        # global flip augmentation
        global_flip_aug=True,
        plain_flip_aug=False,
        fb_flip_aug=False,
        # whether use raw augmentation policy (scale/rotation/flip) when use raw input
        use_raw_aug=False,
        aug_K_on_raw=None,    # [type, r1, r2], type=['range', 'ratio'], aug range: [f*r1, f*r2] focal augmentation cfg
        # whether to use raw images as input
        p_raw=0.,
        inpaint_ego_occ_train=False,
        # novel camera generation policy
        aug_ego2global=None,    # list, ego pose augmentation [rx, ry, rz] degrees, rx/y/z -> roll/pitch/yaw
        aug_cam2ego=None,       # list, mounting pose augmentation(by offset) [tx, ty, tz, rx, ry, rz] tz is a tuple (type, a, b)!
        aug_K=None,       # [type, r1, r2], type=['range', 'ratio'], aug range: [f*r1, f*r2]
        integrate_extrinsic_aug=True,
        # use lidar depth to supervise
        use_lidar_depth=False,
        fg_depth_only=False,
        
        # sync params
        nvs_wo_cam_sync=False,
        sync_adj_aug=False,
        
        # test mode
        test_mode=False,
        inpaint_ego_occ_test=False,
        crop_ego_occ_test=False,
        
        # customize augmentation
        custom_aug=None,
    ):
        self.nuscenes_meta_data_root = nuscenes_meta_data_root
        self.lyft_meta_data_root = lyft_meta_data_root
        self.waymo_meta_data_root = waymo_meta_data_root
        self.sequential = sequential
        self.input_size = input_size
        self.cam_names = cam_names
        
        self.test_mode = test_mode
        if test_mode:
            assert not(inpaint_ego_occ_test and crop_ego_occ_test)
            self.inpaint_ego_occ_test = inpaint_ego_occ_test
            self.crop_ego_occ_test = crop_ego_occ_test
            self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            return
            
        self.max_pts = max_pts
        
        self.global_flip_aug = global_flip_aug  # change results['ann_infos'], results['curr']['ann_infos'], ignore adj(we dont need)
        self.plain_flip_aug = plain_flip_aug
        self.fb_flip_aug = fb_flip_aug
        assert not (global_flip_aug and plain_flip_aug)
        
        self.use_raw_aug = use_raw_aug
        self.aug_ego2global = aug_ego2global
        self.aug_cam2ego = aug_cam2ego
        self.aug_K = aug_K
        self.aug_K_on_raw = aug_K_on_raw
        self.integrate_extrinsic_aug = integrate_extrinsic_aug
        if aug_ego2global is not None:
            assert len(aug_ego2global) == 3
        if aug_cam2ego is not None:
            assert len(aug_cam2ego) == 6
        if aug_K is not None:
            assert len(aug_K) == 3
            assert aug_K[0] in ['range', 'ratio']
        if aug_K_on_raw is not None:
            assert len(aug_K_on_raw) == 3
            assert aug_K_on_raw[0] in ['range', 'ratio']
        
        self.p_raw = p_raw
        self.inpaint_ego_occ_train = inpaint_ego_occ_train
        
        self.use_lidar_depth = use_lidar_depth
        self.fg_depth_only = fg_depth_only
        
        # placeholders
        self.means3D_placeholder = np.zeros((self.max_pts, 3), dtype=np.float32)
        self.rgbs_placeholder = np.zeros((self.max_pts, 3), dtype=np.float32)
        self.scales_placeholder = np.zeros((self.max_pts, 3), dtype=np.float32)
        self.raw_imgs_placeholder = np.zeros((len(self.cam_names), 3, self.input_size[0], self.input_size[1]), dtype=np.float32)
        self.raw_depths_placeholder = np.zeros((len(self.cam_names), self.input_size[0], self.input_size[1]), dtype=np.float32)
    
        # sync parameters
        self.nvs_wo_cam_sync = nvs_wo_cam_sync
        self.sync_adj_aug = sync_adj_aug
    
        # customize augmentaion
        self.custom_aug = custom_aug
        if custom_aug is not None:
            assert custom_aug in ['lyft', 'waymo', 'lyft2nuscenes']
            
    
    def get_inputs(self, results, mode='gs_file'):
        """
        results['curr'], results['adjacent'][i] 都是一份mminfo
        """
        # Step 1 : init
        # assert mode in ['plain', 'post_pro', 'gs_map', 'gs_file']
        assert mode in ['gs_file']
        if mode == 'gs_file':
            _nvs_meta = dict(
                means3D=[], rgbs=[], scales=[], n_raw_pts=[],  # gaussians
                raw_cams=[], nvs_cams=[],                               # raw and nvs cams config (K, T_cam2ego, T_ego2cam, h, w)
                c2ws=[], fovxs=[], fovys=[], camera_args=[], crop_start=[], crop_end=[],                  # gaussian render parameters
                input_size=self.input_size,     # training size
                debug_imgs=[],                # for debug
                # raw inputs : raw_depths is lidar_depth or dense_depth
                raw_imgs=[], raw_depths=[], use_nvs_flag=True, use_lidar_depth=self.use_lidar_depth,
            )

        nvs_meta = {
            'curr': copy.deepcopy(_nvs_meta),
            'adjacent': [],
        }
        if self.sequential:
            nvs_meta['adjacent'] = [copy.deepcopy(_nvs_meta) for _ in range(len(results['adjacent']))]
        
        raw_flag = np.random.rand() < self.p_raw
        global_flip_flag = np.random.rand() < (self.global_flip_aug * 0.5)
        
        # Step 2 : load mminfo
        curr_mminfo = results['curr']
        adj_mminfos = results['adjacent']
        
        #! NOTE:DEBUG FOR WAYMO
        if curr_mminfo['scene_token'] in ['0001', '0340', '0633']:
            raw_flag = True
        
        if (not raw_flag) and self.nvs_wo_cam_sync:
            self.nvs_wo_cam_sync_adapt(curr_mminfo, adj_mminfos)
        
        # Step 3 : collect raw cams
        self.collect_raw_cams(curr_mminfo, nvs_meta['curr'])
        for i, adj_mminfo in enumerate(adj_mminfos):
            self.collect_raw_cams(adj_mminfo, nvs_meta['adjacent'][i])
                            
        # Step 4 : create new camera configurations
        if self.aug_ego2global and (not raw_flag):
            T_ego_perturbed = NuscenesCameraGroups._recompose_extrinsic(
                angles_deg=[
                    np.random.uniform(-self.aug_ego2global[0], self.aug_ego2global[0]),
                    np.random.uniform(-self.aug_ego2global[1], self.aug_ego2global[1]),
                    np.random.uniform(-self.aug_ego2global[2], self.aug_ego2global[2]),
                    ],
                t=np.zeros((3,))
            )
            # T_ego_perturbed = NuscenesCameraGroups._recompose_extrinsic(angles_deg=[0, 0, 0], t=[0, 0, 0])
        else:
            T_ego_perturbed = np.eye(4)
        self.collect_nvs_cameras(nvs_meta['curr'], T_ego_perturbed, raw_flag)
        for _meta in nvs_meta['adjacent']:
            if self.sync_adj_aug:
                for _key in ['nvs_cams', 'c2ws', 'fovxs', 'fovys', 'camera_args', 'crop_start', 'crop_end']:
                    _meta[_key] = nvs_meta['curr'][_key]
            else:
                self.collect_nvs_cameras(_meta, T_ego_perturbed, raw_flag)
            
        # Step 5 : create img_inputs placeholder
        # (b=n_cam*n_adj) 0: bchw img ; 1: b44 T_sensor2ego; 2: b44 T_ego2global; 3: b33 K; 4: b33 img_rot_aug; 5: b3 img_trans_aug;
        img_inputs, gt_depth = self.collect_img_inputs(nvs_meta, curr_mminfo, adj_mminfos, raw_flag)
        
        # Step 6 : process global flip augmentation (change annotations)
        if global_flip_flag:
            raise NotImplementedError
            if (len(results['ann_infos'][0]) != 0):
                self.flip_annotations(results)
            self.flip_ego_pose(img_inputs, raw_flag)
            
        # Step 7 : collect gaussians (we move here to adjust the lidar depth collection)
        if mode == 'gs_file':
            # only curr frame need depth to supervision
            self.collect_one_frame_gaussians_gsfile(curr_mminfo, nvs_meta['curr'], raw_flag, global_flip_flag, need_depth=True)
            for i, adj_mminfo in enumerate(adj_mminfos):
                self.collect_one_frame_gaussians_gsfile(adj_mminfo, nvs_meta['adjacent'][i], raw_flag, global_flip_flag)

        # Step 8*: raw style augmentation
        if raw_flag and (self.use_raw_aug) and (np.random.rand() < 1.0):    #and (not global_flip_flag) 
            img_inputs, nvs_meta = self.raw_style_augmentation(img_inputs, nvs_meta, curr_mminfo['dataset'])

        # Step 9*: whether to filter fg depth only
        if self.fg_depth_only:
            nvs_meta['curr']['raw_depths'] = self.mask_fg_depth(nvs_meta['curr']['raw_depths'], curr_mminfo, img_inputs, curr_mminfo['dataset'])

        # Step 10*: plain flip augmentation
        if self.plain_flip_aug:
            img_inputs = self.plain_flip_augmentation(img_inputs)

        return nvs_meta, img_inputs, gt_depth
    
    
    def collect_nvs_cameras(self, data_dict, T_ego_perturbed, raw_flag):
        def _adjust_to_train_size(nvs_cams):
            for cam in nvs_cams.keys():
                # adjust to training size
                train_w = self.input_size[1]
                train_rs = train_w / nvs_cams[cam]['img_w']
                train_h = int(nvs_cams[cam]['img_h'] * train_rs)
                train_K = nvs_cams[cam]['K']
                train_K[:2, :] *= train_rs
        
                nvs_cams[cam]['K'] = train_K
                nvs_cams[cam]['img_w'] = train_w
                nvs_cams[cam]['img_h'] = train_h
            return nvs_cams
        
        nvs_cams = copy.deepcopy(data_dict['raw_cams'])
        
        # adjust to training size
        nvs_cams = _adjust_to_train_size(nvs_cams)
        
        cam_groups = NuscenesCameraGroups(nvs_cams, self.aug_ego2global, 
                                          self.aug_cam2ego, self.aug_K, 
                                          self.integrate_extrinsic_aug,
                                          custom_aug=self.custom_aug
                                          )
        if ((not self.aug_ego2global) and (not self.aug_cam2ego) and (not self.aug_K)) or raw_flag:
            c2ws, fovxs, fovys, camera_args, crop_start, crop_end = cam_groups.gen_raw_cam_for_gaussian()
        else:
            c2ws, fovxs, fovys, camera_args, crop_start, crop_end = cam_groups.gen_new_cam_for_gaussian(T_ego_perturbed)
            
        data_dict.update({
            # 'nvs_cams': nvs_cams,
            'nvs_cams': cam_groups.new_cams,
            'c2ws': c2ws, 'fovxs': fovxs, 'fovys': fovys, 'camera_args': camera_args, 'crop_start': crop_start, 'crop_end': crop_end,
        })
    
    
    def collect_one_frame_gaussians_gsfile(self, mminfo, data_dict, use_raw=False, use_global_flip=False, need_depth=False):
        sample_token = mminfo['token']
        dataset = mminfo['dataset']
        
        # collect gaussians/imgs
        if not use_raw:
            gs_file = os.path.join(getattr(self, f'{dataset}_meta_data_root'), 'gaussians', f'{sample_token}.npz')
            gs_data = np.load(gs_file)
            
            means3D = (gs_data['means3D']).astype(np.float32)
            rgbs = (gs_data['rgbs'] / 255.).astype(np.float32)
            scales = (gs_data['scales']).astype(np.float32)[:, None].repeat(3, axis=1)
            overlap_mask = gs_data['masks']
            lidar_pts = gs_data['lidar_pts'].astype(np.float32)
            
            if use_global_flip:
                means3D[:, 1] *= -1
                
            # pad
            raw_pts = len(means3D)
            n_pad = self.max_pts - raw_pts
            means3D = np.pad(means3D, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)
            rgbs = np.pad(rgbs, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)
            scales = np.pad(scales, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)
            
            # get lidar depths
            if need_depth:
                raw_depths = self.collect_lidar_depths(lidar_pts, mminfo, data_dict, use_global_flip)
            else:
                raw_depths = self.raw_depths_placeholder
            
            data_dict.update({
                'means3D': means3D, 'rgbs': rgbs, 'scales': scales, 'n_raw_pts': raw_pts,\
                # placeholder
                'raw_imgs': self.raw_imgs_placeholder, 'raw_depths': raw_depths,
                'use_nvs_flag': True,
            })
    
        else:
            raw_imgs = [None for _ in self.cam_names]
            raw_depths = [None for _ in self.cam_names]
            
            for i, sensor in enumerate(self.cam_names):
                img_file = mminfo['cams'][sensor]['data_path']
                if self.use_lidar_depth:
                    if dataset == 'waymo':
                        if sensor == 'CAM_BACK':
                            depth_file = '/data1/znkwong/Cross-Cam-Config-Generalization/BEVDet/data/ds_size/waymo_ghost_depth.png'
                        elif sensor == 'CAM_BACK_LEFT' or sensor == 'CAM_BACK_RIGHT':
                            depth_file = os.path.join(getattr(self, f'{dataset}_meta_data_root'), f'zoomout_rgbd_0.7', 'depth', sensor.replace('BACK', 'SIDE'), f'{sample_token}.png')
                        else:
                            depth_file = os.path.join(getattr(self, f'{dataset}_meta_data_root'), f'zoomout_rgbd_0.7', 'depth', sensor, f'{sample_token}.png')
                    else:
                        depth_file = os.path.join(getattr(self, f'{dataset}_meta_data_root'), 'depths', 'lidar_depths', sensor, f'{sample_token}.png')
                else:
                    depth_file = os.path.join(getattr(self, f'{dataset}_meta_data_root'), 'depths', 'dense_depth_SPNorm', sensor, f'{sample_token}.png')
                
                img = cv2.imread(img_file)[..., ::-1]
                if need_depth: depth = self.read_depth_map(depth_file)
                
                if dataset == 'lyft':
                    inpainted_img_file = os.path.join(getattr(self, f'{dataset}_meta_data_root'), 'inpainted', 'img', sensor, f'{sample_token}.png')
                    key_mask_file = os.path.join(getattr(self, f'{dataset}_meta_data_root'), 'masks', 'key_mask2d', sensor, f'{sample_token}.png')
                    inpainted_img = cv2.imread(inpainted_img_file)[..., ::-1]
                    key_mask = cv2.imread(key_mask_file)
                    
                    inpainted_mask = (key_mask[..., 2] == 250)
                    img[inpainted_mask] = inpainted_img[inpainted_mask]
                    
                h, w, _ = img.shape
                nh, nw = int(h*self.input_size[1]/w), self.input_size[1]
                img = cv2.resize(img, (nw, nh))
                if need_depth: depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
                
                img = img[nh-self.input_size[0]:, ...]
                if need_depth: depth = depth[nh-self.input_size[0]:, ...]
                
                if use_global_flip:
                    img = img[:, ::-1, :]
                    if need_depth: depth = depth[:, ::-1]
                    if 'LEFT' in sensor:
                        j = self.cam_names.index(sensor.replace('LEFT', 'RIGHT'))
                    elif 'RIGHT' in sensor:
                        j = self.cam_names.index(sensor.replace('RIGHT', 'LEFT'))
                    else:
                        # front / back
                        j = i
                    raw_imgs[j] = img
                    if need_depth: raw_depths[j] = depth
                else:
                    raw_imgs[i] = img
                    if need_depth: raw_depths[i] = depth
                
            raw_imgs = np.stack(raw_imgs).transpose(0, 3, 1, 2)
            if need_depth: 
                raw_depths = np.stack(raw_depths)
            else:
                raw_depths = self.raw_depths_placeholder
                    
            data_dict.update({
                'raw_imgs': (raw_imgs / 255.).astype(np.float32), 'raw_depths': raw_depths.astype(np.float32),
                # placeholder
                'means3D': self.means3D_placeholder, 'rgbs': self.rgbs_placeholder, 'scales': self.scales_placeholder, 'n_raw_pts': -1,
                'use_nvs_flag': False,
            })
            
            if dataset == 'waymo':
                zoomout_imgs = [None for _ in self.cam_names]
                for i, sensor in enumerate(self.cam_names):
                    if sensor == 'CAM_BACK':
                        zoomout_imgs[i] = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
                    else:
                        if sensor == 'CAM_BACK_LEFT' or sensor == 'CAM_BACK_RIGHT':
                            img_file = os.path.join(self.waymo_meta_data_root, f'zoomout_rgbd_0.7', 'img', sensor.replace('BACK', 'SIDE'), f'{sample_token}.jpg')
                        else:
                            img_file = os.path.join(self.waymo_meta_data_root, f'zoomout_rgbd_0.7', 'img', sensor, f'{sample_token}.jpg')
                        img = cv2.imread(img_file)[..., ::-1]
                        h, w, _ = img.shape
                        nh, nw = int(h*self.input_size[1]/w), self.input_size[1]
                        img = cv2.resize(img, (nw, nh))
                        img = img[nh-self.input_size[0]:, ...]
                        zoomout_imgs[i] = img
                zoomout_imgs = np.stack(zoomout_imgs).transpose(0, 3, 1, 2)
                data_dict.update({'zoomout_imgs': (zoomout_imgs / 255.).astype(np.float32)})
            
            # ! DEBUG
            # print(f'sample_token:{sample_token} raw_imgs.shape:{raw_imgs.shape} raw_depths.shape:{raw_depths.shape}')
        
        # NOTE: DEBUG
        # debug_imgs = {}
        # for sensor in self.cam_names:
        #     _dbimg = cv2.imread(mminfo['cams'][sensor]['data_path'])
        #     # _dbimg = _dbimg[-450:, :, :]
        #     _dbimg = _dbimg[-369:, :, :]
        #     debug_imgs[sensor] = _dbimg
        #     # debug_imgs[sensor] = cv2.imread(mminfo['cams'][sensor]['data_path'])
        # data_dict['debug_imgs'] = debug_imgs
        

    def raw_style_augmentation(self, img_inputs, nvs_meta, dataset):
        def _get_rot(h):
            return np.array([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])
            
        def _img_transform_core_opencv(img, post_rot, post_tran, crop, is_depth=False):
            if is_depth:
                _flag = cv2.INTER_NEAREST
            else:
                _flag = cv2.INTER_LINEAR
                img = img.transpose(1, 2, 0)
                
                
            img = cv2.warpAffine(img.astype(np.float32),
                                 np.concatenate([post_rot,
                                                 post_tran.reshape(2,1)],
                                                 axis=1),
                                 (crop[2]-crop[0], crop[3]-crop[1]),
                                 flags=_flag)
            
            if not is_depth:
                img = img.transpose(2, 0, 1)
            
            return img
            
        def _aug_K(img, K, ratio):
            # assert ratio > 1    # we only apply zoom-in aug in raw, for zoom-out, there will occur black edge
            
            fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            if len(img.shape) == 3:
                img = img.transpose(1, 2, 0)
                h, w, _ = img.shape
                _flag = cv2.INTER_LINEAR
            else:
                h, w = img.shape
                _flag = cv2.INTER_NEAREST
            
            # dist from edge to priciple point
            l, r, t, d = cu, w - cu, cv, h - cv
            if ratio > 1:
                crop = (int(cu - l/ratio), int(cv - t/ratio), int(cu + r/ratio), int(cv + d/ratio))
                img = img[crop[1]:crop[3], crop[0]:crop[2]]
                img = cv2.resize(img, (w, h), interpolation=_flag)
            elif ratio < 1:
                new_w, new_h = int(w * ratio), int(h * ratio)
                _u, _v = int(cu - l*ratio), int(cv - t*ratio)
                tmp = np.zeros_like(img)
                img = cv2.resize(img, (new_w, new_h), interpolation=_flag)      
                tmp[_v:_v+new_h, _u:_u+new_w] = img
                img = tmp
            
            if len(img.shape) == 3:
                img = img.transpose(2, 0, 1)
                
            return img
            
            
        n_cam = len(self.cam_names)
        n_adj = len(nvs_meta['adjacent'])        
        imgs, T_sensor2egos, T_ego2globals, Ks, img_rot_aug, img_trans_aug = img_inputs
            
        for i in range(n_cam):
            ratio = 1.
            
            # NOTE:focal augmentation on raw image
            # ! 如果进行了内参增强 那么原始增强就不进行resize增强了
            if self.aug_K_on_raw:
                if dataset == 'waymo':
                    ratio = np.random.uniform(ratio*0.7, ratio*1.0)
                else:
                    if self.aug_K_on_raw[0] == 'ratio':
                        ratio = np.random.uniform(ratio*self.aug_K_on_raw[1], ratio*self.aug_K_on_raw[2])
                    elif self.aug_K_on_raw[0] == 'range':
                        ratio = np.random.uniform(self.aug_K_on_raw[1], self.aug_K_on_raw[2]) / (Ks[i][0, 0] * ratio)
                    
            # NOTE:ratio<1时图像要缩小我们不进行这操作 ratio=1时没变化也不进行这操作
            if ratio != 1:
                if dataset == 'waymo':
                    zoomout_ratio = 0.7
                    zoomout_K = Ks[i].copy()
                    zoomout_K[:2, :2] *= zoomout_ratio
                    nvs_meta['curr']['raw_imgs'][i] = _aug_K(nvs_meta['curr']['raw_imgs'][i], Ks[i], ratio)
                    nvs_meta['curr']['zoomout_imgs'][i] = _aug_K(nvs_meta['curr']['zoomout_imgs'][i], zoomout_K, ratio/zoomout_ratio)
                    nvs_meta['curr']['raw_depths'][i] = _aug_K(nvs_meta['curr']['raw_depths'][i], zoomout_K, ratio/zoomout_ratio)
                    Ks[i][:2, :2] *= ratio
                    zoomout_mask = (nvs_meta['curr']['raw_imgs'][i]==0)
                    nvs_meta['curr']['raw_imgs'][i][zoomout_mask] = nvs_meta['curr']['zoomout_imgs'][i][zoomout_mask]
                    # del nvs_meta['curr']['zoomout_imgs'][i]
                    for j, _meta in enumerate(nvs_meta['adjacent']):
                        zoomout_K = Ks[n_cam*(j+1)+i].copy()
                        zoomout_K[:2, :2] *= zoomout_ratio
                        nvs_meta['adjacent'][j]['raw_imgs'][i] = _aug_K(nvs_meta['adjacent'][j]['raw_imgs'][i], Ks[n_cam*(j+1)+i], ratio)
                        nvs_meta['adjacent'][j]['zoomout_imgs'][i] = _aug_K(nvs_meta['adjacent'][j]['zoomout_imgs'][i], zoomout_K, ratio/zoomout_ratio)
                        nvs_meta['adjacent'][j]['raw_depths'][i] = _aug_K(nvs_meta['adjacent'][j]['raw_depths'][i], zoomout_K, ratio/zoomout_ratio)
                        Ks[n_cam*(j+1)+i][:2, :2] *= ratio
                        zoomout_mask = (nvs_meta['adjacent'][j]['raw_imgs'][i]==0)
                        nvs_meta['adjacent'][j]['raw_imgs'][i][zoomout_mask] = nvs_meta['adjacent'][j]['zoomout_imgs'][i][zoomout_mask]
                        # del nvs_meta['adjacent'][j]['zoomout_imgs'][i]
                    
                else:
                    nvs_meta['curr']['raw_imgs'][i] = _aug_K(nvs_meta['curr']['raw_imgs'][i], Ks[i], ratio)
                    nvs_meta['curr']['raw_depths'][i] = _aug_K(nvs_meta['curr']['raw_depths'][i], Ks[i], ratio)
                    Ks[i][:2, :2] *= ratio      
                    for j, _meta in enumerate(nvs_meta['adjacent']):
                        nvs_meta['adjacent'][j]['raw_imgs'][i] = _aug_K(nvs_meta['adjacent'][j]['raw_imgs'][i], Ks[n_cam*(j+1)+i], ratio)
                        nvs_meta['adjacent'][j]['raw_depths'][i] = _aug_K(nvs_meta['adjacent'][j]['raw_depths'][i], Ks[n_cam*(j+1)+i], ratio)
                        Ks[n_cam*(j+1)+i][:2, :2] *= ratio  
                
            # sample augmentation
            h, w = self.input_size
            if self.aug_K_on_raw:
                resize = 1.
            else:
                # resize = 1 + np.random.uniform(-0.06, 0.11)
                resize = 1 + np.random.uniform(-0.06, 0.11) / w * 1600
            new_h, new_w = int(h*resize), int(w*resize)
            resize_dims = (new_w, new_h)
            crop_h = new_h - h
            crop_w = int(np.random.uniform(0, max(0, new_w - w)))
            crop = (crop_w, crop_h, crop_w + w, crop_h + h)
            rotate = np.random.uniform(-5.4, 5.4)

            # post-homography transformation
            post_rot = np.eye(2)
            post_tran = np.zeros(2)
            post_rot *= resize
            post_tran -= np.array(crop[:2])

            A = _get_rot(rotate / 180 * np.pi)
            b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A @ (-b) + b
            post_rot = A @ (post_rot)
            post_tran = A @ (post_tran) + b
            
            # apply transformation
            nvs_meta['curr']['raw_imgs'][i] = _img_transform_core_opencv(
                nvs_meta['curr']['raw_imgs'][i],
                post_rot, post_tran, crop, is_depth=False)
            nvs_meta['curr']['raw_depths'][i] = _img_transform_core_opencv(
                nvs_meta['curr']['raw_depths'][i],
                post_rot, post_tran, crop, is_depth=True)
            img_rot_aug[i, :2, :2] = post_rot
            img_trans_aug[i, :2] = post_tran
            
            for j, _meta in enumerate(nvs_meta['adjacent']):
                nvs_meta['adjacent'][j]['raw_imgs'][i] = _img_transform_core_opencv(
                    nvs_meta['adjacent'][j]['raw_imgs'][i],
                    post_rot, post_tran, crop, is_depth=False)
                nvs_meta['adjacent'][j]['raw_depths'][i] = _img_transform_core_opencv(
                    nvs_meta['adjacent'][j]['raw_depths'][i],
                    post_rot, post_tran, crop, is_depth=True)
                img_rot_aug[n_cam*(j+1)+i, :2, :2] = post_rot
                img_trans_aug[n_cam*(j+1)+i, :2] = post_tran
            
        img_inputs = (imgs, T_sensor2egos, T_ego2globals, Ks, img_rot_aug, img_trans_aug)
        
        #NOTE: delete zoomout imgs
        if dataset == 'waymo':
            del nvs_meta['curr']['zoomout_imgs']
            for _meta in nvs_meta['adjacent']:
                del _meta['zoomout_imgs']
        
        return img_inputs, nvs_meta


    def mask_fg_depth(self, depths, mminfo, img_inputs, dataset):
        ego_boxes = np.array(mminfo['ann_infos'][0])
        cls_mask = np.array(mminfo['ann_infos'][1]) == 0  # car only for experiments
        if not (self.fg_depth_only == 'car'):
            cls_mask = np.array(mminfo['ann_infos'][1]) <= 2    # car, pedestrian, motorcycle
            # cls_mask = np.ones_like(cls_mask)   
        ego_boxes = ego_boxes[cls_mask]
        
        n_obj = len(ego_boxes)
        if n_obj == 0:
            return np.zeros_like(depths)
        
        # Step 1 : get object corners in local space
        objs_dims = ego_boxes[:, 3:6]   # dx, dy, dz
        
        local_corners = np.array([
            [ 0.5,  0.5,  0.5],
            [ 0.5, -0.5,  0.5],
            [-0.5, -0.5,  0.5],
            [-0.5,  0.5,  0.5],
            [ 0.5,  0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5,  0.5, -0.5],
        ])  # (8, 3)
        
        local_corners = local_corners[None, :, :] * objs_dims[:, None, :]    # (n_obj, 8, 3)
        
        # Step 2 : init transformation from local to cam, cam to img
        if dataset == 'nuscenes':
            _centers = ego_boxes[:, :3]
            _yaws = ego_boxes[:, 6]
            cos_yaw = np.cos(_yaws)
            sin_yaw = np.sin(_yaws)
            zeros = np.zeros_like(_yaws)
            ones = np.ones_like(_yaws)
            _R = np.stack([
                np.stack([cos_yaw, -sin_yaw, zeros], axis=1),
                np.stack([sin_yaw,  cos_yaw, zeros], axis=1),
                np.stack([zeros,    zeros,   ones],  axis=1),
            ], axis=1)  # (N, 3, 3)
            T_local2egos = np.eye(4)[None, :, :].repeat(n_obj, axis=0)
            T_local2egos[:, :3, :3] = _R
            T_local2egos[:, :3, 3] = _centers
        else:
            T_local2egos = mminfo['T_local2egos'][cls_mask]       # (n_obj, 4, 4)
        
        
        T_cam2egos = img_inputs[1][:len(self.cam_names)]
        T_ego2cams = np.linalg.inv(T_cam2egos)
        Ks = img_inputs[3][:len(self.cam_names)]
        T_img_aug = img_inputs[4][:len(self.cam_names)].copy()
        T_img_aug[:, :2, 2] = img_inputs[5][:len(self.cam_names)][:, :2].copy()
        T_cam2imgs = T_img_aug @ Ks
        T_cam2imgs_homo = np.eye(4)[None, ...].repeat(len(self.cam_names), axis=0)
        T_cam2imgs_homo[:, :3, :3] = T_cam2imgs
        
        # Step 3 : project each object corners to each image space and get fg mask
        depth_mask = []
        for i in range(len(self.cam_names)):
            _mask = np.zeros((self.input_size[0], self.input_size[1]), dtype=bool)
            
            if dataset == 'waymo' and (self.cam_names[i] == 'CAM_BACK'):
                depth_mask.append(_mask)
                continue
            
            T_local2imgs = T_cam2imgs_homo[i] @ T_ego2cams[i] @ T_local2egos
            for _corner, T_local2img in zip(local_corners, T_local2imgs):
                # get img corner
                _corner = (self.cart2homo(_corner) @ T_local2img.T)[:, :3]
                
                # scan each object
                _corner_mask = _corner[:, -1] > 0
                if np.all(~_corner_mask): continue
                
                _corner = _corner[_corner_mask]
                _corner = _corner[:, :2] / _corner[:, 2:3]  # corner uv
                
                _corner[:, 0] = np.clip(_corner[:, 0], 0, self.input_size[1])
                _corner[:, 1] = np.clip(_corner[:, 1], 0, self.input_size[0])
                _u1, _u2 = int(_corner[:, 0].min()), int(_corner[:, 0].max())
                _v1, _v2 = int(_corner[:, 1].min()), int(_corner[:, 1].max())
                
                _mask[_v1:_v2, _u1:_u2] = True
            depth_mask.append(_mask)
        depth_mask = np.stack(depth_mask, axis=0)
        
        depths[~depth_mask] = 0.
        
        return depths



def yaw_to_unit_vector(yaw_rad):
    """
    yaw角定义: 重力轴z轴 参考轴x轴 从重力轴的负方向（轴的正方向指向人眼）看，yaw沿着参考轴逆时针方向增加
    """
    x = np.cos(yaw_rad)
    y = np.sin(yaw_rad)
    z = np.zeros_like(yaw_rad)
    return np.stack([x, y, z], axis=-1)

def vector_to_yaw(vec):
    x = vec[:, 0]
    y = vec[:, 1]
    return np.arctan2(y, x)


class NuscenesCameraGroups:
    """
    Nuscenes Camera Groups Setting of One Frame
    """
    # T_cam2ego = T_cam2ego_aligned @ T_aligned : 分解成两次变换 第一次是对齐cam和ego的xyz轴 然后再平移旋转 这样观察平移旋转更直观
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
    
    
    def gen_new_cam_for_gaussian(self, T_ego_perturbed):
        n_cam = len(self.raw_cams)
        
        c2ws = np.zeros((n_cam, 4, 4), dtype=np.float32)
        cus = np.zeros((n_cam,), dtype=np.float32)
        cvs = np.zeros((n_cam,), dtype=np.float32)
        fus = np.zeros((n_cam,), dtype=np.float32)
        fvs = np.zeros((n_cam,), dtype=np.float32)
        imghs = np.zeros((n_cam,), dtype=np.int32)
        imgws = np.zeros((n_cam,), dtype=np.int32)
        
        for i, cam_name in enumerate(self.raw_cams.keys()):
            # extrinsic aug
            # for raw T_cam2ego, T_cam2ego = T_cam2ego_aligned @ T_aligned, T_aligned to align cam-ego axis, T_cam2ego_aligned trans position from ego-origin to cam-origin
            # for extrinsic_aug, T_cam2ego_new = T_cam2ego_aligned_new @ T_ego_perturbed @ T_aligned, 
            # where T_cam2ego_aligned_new define a new camera mounting position relate to ego, T_ego_perturbed define the car pose perturbation 
            # intrinsic aug: just adjust the focal 
            
            if not self.custom_aug: 
                c2ws[i] = self.cfg_func_extrinsic(cam_name, T_ego_perturbed).astype(np.float32)
            else:
                c2ws[i] = getattr(self, f'cfg_func_extrinsic_{self.custom_aug}_custom')(cam_name, T_ego_perturbed).astype(np.float32)
                            
            # intrinsic
            if not self.custom_aug:
                fu, fv, cu, cv, img_w, img_h = self.cfg_func_intrinsic(cam_name)
            else:
                fu, fv, cu, cv, img_w, img_h = getattr(self, f'cfg_func_intrinsic_{self.custom_aug}_custom')(cam_name)
                            
            fus[i] = fu
            fvs[i] = fv
            cus[i] = cu
            cvs[i] = cv
            imghs[i] = img_h
            imgws[i] = img_w
            
        # cropping to adapt gaussian render kernel
        fovxs, fovys, resolution, crop_start, crop_end = self.get_fov_and_cropping(cus, cvs, fus, fvs, imghs, imgws)
        camera_args = {
            'resolution': resolution,
            'znear': 0.1,
            'zfar': 1000.0,
        }
        
        return c2ws, fovxs, fovys, camera_args, crop_start, crop_end
            
            
    def gen_raw_cam_for_gaussian(self):
        n_cam = len(self.raw_cams)
        
        c2ws = np.zeros((n_cam, 4, 4), dtype=np.float32)
        cus = np.zeros((n_cam,), dtype=np.float32)
        cvs = np.zeros((n_cam,), dtype=np.float32)
        fus = np.zeros((n_cam,), dtype=np.float32)
        fvs = np.zeros((n_cam,), dtype=np.float32)
        imghs = np.zeros((n_cam,), dtype=np.int32)
        imgws = np.zeros((n_cam,), dtype=np.int32)
        
        for i, cam_name in enumerate(self.raw_cams.keys()):
            c2ws[i] = np.linalg.inv(self.raw_cams[cam_name]['T_ego2cam']).astype(np.float32)
            fus[i] = self.raw_cams[cam_name]['fu']
            fvs[i] = self.raw_cams[cam_name]['fv']
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
    
    
    def cfg_func_extrinsic(self, cam_name, T_ego_perturbed):
        # extrinsic aug
        # for raw T_cam2ego, T_cam2ego = T_cam2ego_aligned @ T_aligned, T_aligned to align cam-ego axis, T_cam2ego_aligned trans position from ego-origin to cam-origin
        # for extrinsic_aug, T_cam2ego_new = T_cam2ego_aligned_new @ T_ego_perturbed @ T_aligned, 
        # where T_cam2ego_aligned_new define a new camera mounting position relate to ego, T_ego_perturbed define the car pose perturbation 

        T_cam2ego = self.raw_cams[cam_name]['T_cam2ego']

        T_cam2ego_aligned = T_cam2ego @ self.T_aligned_inv
        if self.aug_cam2ego:
            angles_deg, t = self._decompose_extrinsic(T_cam2ego_aligned)
            
            raw_rz = angles_deg[2]      # NOTE: for waymo side camera adjustment
            raw_tx, raw_ty = t[0], t[1]
            
            for i in range(3):
                # rotation aug
                if type(self.aug_cam2ego[3+i]) in [tuple, list]:
                    if self.aug_cam2ego[3+i][0] == 'ratio':
                        angles_deg[i] += np.random.uniform(self.aug_cam2ego[3+i][1], self.aug_cam2ego[3+i][2])
                    elif self.aug_cam2ego[3+i][0] == 'range':
                        angles_deg[i] = np.random.uniform(self.aug_cam2ego[3+i][1], self.aug_cam2ego[3+i][2])
                    else:
                        raise TypeError
                else:
                    angles_deg[i] += np.random.uniform(-self.aug_cam2ego[3+i], self.aug_cam2ego[3+i])
                
                # trans aug
                if type(self.aug_cam2ego[i]) in [tuple, list]:
                    # for tz(camera height) augmentation
                    if self.aug_cam2ego[i][0] == 'ratio': 
                        t[i] += np.random.uniform(self.aug_cam2ego[i][1], self.aug_cam2ego[i][2])
                    elif self.aug_cam2ego[i][0] == 'range':
                        t[i] = np.random.uniform(self.aug_cam2ego[i][1], self.aug_cam2ego[i][2])
                    else:
                        raise TypeError
                else:
                    t[i] += np.random.uniform(-self.aug_cam2ego[i], self.aug_cam2ego[i])
            
            # NOTE: for waymo side camera adjustment
            if cam_name == 'CAM_SIDE_LEFT':
                angles_deg[2] = np.minimum(angles_deg[2], raw_rz)
                t[0] = np.maximum(t[0], raw_tx)
                t[1] = np.minimum(t[1], raw_ty)
            elif cam_name == 'CAM_SIDE_RIGHT':
                angles_deg[2] = np.maximum(angles_deg[2], raw_rz)
                t[0] = np.maximum(t[0], raw_tx)
                t[1] = np.maximum(t[1], raw_ty)
            
            T_cam2ego_aligned_new = self._recompose_extrinsic(angles_deg, t)
        else:
            T_cam2ego_aligned_new = T_cam2ego_aligned
        T_cam2ego_new = T_ego_perturbed @ T_cam2ego_aligned_new @ self.T_aligned
        T_ego2cam_new = np.linalg.inv(T_cam2ego_new)
        
        self.new_cams[cam_name]['T_ego2cam'] = T_ego2cam_new.astype(np.float32)
        self.new_cams[cam_name]['T_cam2ego'] = T_cam2ego_new.astype(np.float32)

        return T_cam2ego_new  
          
        
    def cfg_func_intrinsic(self, cam_name):
        # load raw cam intrinsic
        raw_fu = self.raw_cams[cam_name]['fu']
        raw_fv = self.raw_cams[cam_name]['fv']
        raw_cu = self.raw_cams[cam_name]['cu']
        raw_cv = self.raw_cams[cam_name]['cv']
        raw_img_w = self.raw_cams[cam_name]['img_w']
        raw_img_h = self.raw_cams[cam_name]['img_h']
        raw_fovx = self.raw_cams[cam_name]['fovx_degree']
        raw_fovy = self.raw_cams[cam_name]['fovy_degree']
        
        if self.aug_K:
            new_img_w = raw_img_w
            new_img_h = raw_img_h
            if self.aug_K[0] == 'ratio':
                new_fu = np.random.uniform(raw_fu*self.aug_K[1], raw_fu*self.aug_K[2])
            elif self.aug_K[0] == 'range':
                new_fu = np.random.uniform(self.aug_K[1], self.aug_K[2])
            new_fv = new_fu
            new_cu = raw_img_w/2 + np.random.uniform(-abs(raw_cu-raw_img_w/2), abs(raw_cu-raw_img_w/2))
            new_cv = raw_img_h/2 + np.random.uniform(-abs(raw_cv-raw_img_h/2), abs(raw_cv-raw_img_h/2))
            new_fovx = 2 * np.arctan(new_img_w / 2 / new_fu) / np.pi * 180
            new_fovy = 2 * np.arctan(new_img_h / 2 / new_fv) / np.pi * 180
        else:
            new_img_w, new_img_h, new_fu, new_fv, new_cu, new_cv, new_fovx, new_fovy = \
                raw_img_w, raw_img_h, raw_fu, raw_fv, raw_cu, raw_cv, raw_fovx, raw_fovy
            
        self.new_cams[cam_name]['K'] = np.array([[new_fu, 0, new_cu],
                                       [0, new_fv, new_cv],
                                       [0, 0, 1]], dtype=np.float32)
        self.new_cams[cam_name]['fu'], self.new_cams[cam_name]['fv'], \
            self.new_cams[cam_name]['cu'], self.new_cams[cam_name]['cv'] = \
                new_fu, new_fv, new_cu, new_cv
        self.new_cams[cam_name]['img_w'], self.new_cams[cam_name]['img_h'] = \
            new_img_w, new_img_h
        self.new_cams[cam_name]['fovx_degree'], self.new_cams[cam_name]['fovy_degree'] = \
            new_fovx, new_fovy
            
        return new_fu, new_fv, new_cu, new_cv, new_img_w, new_img_h
        
        
    #! customize augmentaion for lyft and waymo
    def cfg_func_extrinsic_lyft_custom(self, cam_name, T_ego_perturbed):
        T_cam2ego = self.raw_cams[cam_name]['T_cam2ego']

        T_cam2ego_aligned = T_cam2ego @ self.T_aligned_inv
        if self.aug_cam2ego:
            angles_deg, t = self._decompose_extrinsic(T_cam2ego_aligned)
            
            t[2] = np.random.uniform(1.5, 1.8)  
            
            if cam_name == 'CAM_FRONT':
                t[0] = np.random.uniform(1.5, 1.7)
            elif cam_name == 'CAM_FRONT_LEFT':
                angles_deg[2] = np.random.uniform(45, 65)
                t[0] = np.random.uniform(1.25, 1.55)
                t[1] = np.random.uniform(0.3, 0.5)
            elif cam_name == 'CAM_FRONT_RIGHT':
                angles_deg[2] = np.random.uniform(-65, -45)
                t[0] = np.random.uniform(1.25, 1.55)
                t[1] = np.random.uniform(-0.5, -0.3)
            elif cam_name == 'CAM_BACK':
                t[0] = np.random.uniform(0, 0.25)
            elif cam_name == 'CAM_BACK_LEFT':
                angles_deg[2] = np.random.uniform(100, 120)
                t[1] = np.random.uniform(0.3, 0.5)
            elif cam_name == 'CAM_BACK_RIGHT':
                angles_deg[2] = np.random.uniform(-120, -100)
                t[1] = np.random.uniform(-0.5, -0.3)
                
            T_cam2ego_aligned_new = self._recompose_extrinsic(angles_deg, t)
        else:
            T_cam2ego_aligned_new = T_cam2ego_aligned
        # T_cam2ego_new = T_cam2ego_aligned_new @ T_ego_perturbed @ self.T_aligned  
        T_cam2ego_new = T_ego_perturbed @ T_cam2ego_aligned_new @ self.T_aligned
        T_ego2cam_new = np.linalg.inv(T_cam2ego_new)
        
        self.new_cams[cam_name]['T_ego2cam'] = T_ego2cam_new.astype(np.float32)
        self.new_cams[cam_name]['T_cam2ego'] = T_cam2ego_new.astype(np.float32)

        return T_cam2ego_new  
          
        
    def cfg_func_intrinsic_lyft_custom(self, cam_name):
        # load raw cam intrinsic
        raw_fu = self.raw_cams[cam_name]['fu']
        raw_fv = self.raw_cams[cam_name]['fv']
        raw_cu = self.raw_cams[cam_name]['cu']
        raw_cv = self.raw_cams[cam_name]['cv']
        raw_img_w = self.raw_cams[cam_name]['img_w']
        raw_img_h = self.raw_cams[cam_name]['img_h']
        raw_fovx = self.raw_cams[cam_name]['fovx_degree']
        raw_fovy = self.raw_cams[cam_name]['fovy_degree']
        
        if self.aug_K:
            new_img_w = raw_img_w
            new_img_h = raw_img_h
            
            if cam_name == 'CAM_BACK':
                new_fu = np.random.uniform(350, 560)
            else:
                new_fu = np.random.uniform(450, 560)
                            
            new_fv = new_fu
            new_cu = raw_img_w/2 + np.random.uniform(-abs(raw_cu-raw_img_w/2), abs(raw_cu-raw_img_w/2))
            new_cv = raw_img_h/2 + np.random.uniform(-abs(raw_cv-raw_img_h/2), abs(raw_cv-raw_img_h/2))
            new_fovx = 2 * np.arctan(new_img_w / 2 / new_fu) / np.pi * 180
            new_fovy = 2 * np.arctan(new_img_h / 2 / new_fv) / np.pi * 180
        else:
            new_img_w, new_img_h, new_fu, new_fv, new_cu, new_cv, new_fovx, new_fovy = \
                raw_img_w, raw_img_h, raw_fu, raw_fv, raw_cu, raw_cv, raw_fovx, raw_fovy
            
        self.new_cams[cam_name]['K'] = np.array([[new_fu, 0, new_cu],
                                       [0, new_fv, new_cv],
                                       [0, 0, 1]], dtype=np.float32)
        self.new_cams[cam_name]['fu'], self.new_cams[cam_name]['fv'], \
            self.new_cams[cam_name]['cu'], self.new_cams[cam_name]['cv'] = \
                new_fu, new_fv, new_cu, new_cv
        self.new_cams[cam_name]['img_w'], self.new_cams[cam_name]['img_h'] = \
            new_img_w, new_img_h
        self.new_cams[cam_name]['fovx_degree'], self.new_cams[cam_name]['fovy_degree'] = \
            new_fovx, new_fovy
            
        return new_fu, new_fv, new_cu, new_cv, new_img_w, new_img_h
    
    
    
    def cfg_func_extrinsic_waymo_custom(self, cam_name, T_ego_perturbed):
        T_cam2ego = self.raw_cams[cam_name]['T_cam2ego']

        T_cam2ego_aligned = T_cam2ego @ self.T_aligned_inv
        if self.aug_cam2ego:
            angles_deg, t = self._decompose_extrinsic(T_cam2ego_aligned)
            
            t[2] = np.random.uniform(1.5, 2.1)
            
            if cam_name == 'CAM_FRONT':
                t[0] = np.random.uniform(1.5, 1.7)
            elif cam_name == 'CAM_FRONT_LEFT':
                angles_deg[2] = np.random.uniform(45, 65)
                # t[0] = np.random.uniform(1.5, 1.55)
                t[0] = np.random.uniform(1.0, 1.55)
                t[1] = np.random.uniform(0.1, 0.5)
            elif cam_name == 'CAM_FRONT_RIGHT':
                angles_deg[2] = np.random.uniform(-65, -45)
                # t[0] = np.random.uniform(1.5, 1.55)
                t[0] = np.random.uniform(1.0, 1.55)
                t[1] = np.random.uniform(-0.5, -0.1)
            elif cam_name == 'CAM_SIDE_LEFT':
                angles_deg[2] = np.random.uniform(60, 80)
                t[0] = np.random.uniform(0.5, 0.6)
                t[1] = np.random.uniform(0.1, 0.5)
            elif cam_name == 'CAM_SIDE_RIGHT':
                angles_deg[2] = np.random.uniform(-80, -60)
                t[0] = np.random.uniform(-0.6, -0.5)
                t[1] = np.random.uniform(-0.5, -0.1)
                
                
            T_cam2ego_aligned_new = self._recompose_extrinsic(angles_deg, t)
        else:
            T_cam2ego_aligned_new = T_cam2ego_aligned
        # T_cam2ego_new = T_cam2ego_aligned_new @ T_ego_perturbed @ self.T_aligned  
        T_cam2ego_new = T_ego_perturbed @ T_cam2ego_aligned_new @ self.T_aligned
        T_ego2cam_new = np.linalg.inv(T_cam2ego_new)
        
        self.new_cams[cam_name]['T_ego2cam'] = T_ego2cam_new.astype(np.float32)
        self.new_cams[cam_name]['T_cam2ego'] = T_cam2ego_new.astype(np.float32)

        return T_cam2ego_new  
          
        
    def cfg_func_intrinsic_waymo_custom(self, cam_name):
        # load raw cam intrinsic
        # waymo special intrinsic: 
        # front/front_left/front_right : raw_img_h=469, raw_cv=raw_img_h/2
        # side_left/side_right : raw_img_h = 324, raw_cv=90=324-469/2
        raw_fu = self.raw_cams[cam_name]['fu']
        raw_fv = self.raw_cams[cam_name]['fv']
        raw_cu = self.raw_cams[cam_name]['cu']
        raw_cv = self.raw_cams[cam_name]['cv']
        raw_img_w = self.raw_cams[cam_name]['img_w']
        raw_img_h = self.raw_cams[cam_name]['img_h']
        raw_fovx = self.raw_cams[cam_name]['fovx_degree']
        raw_fovy = self.raw_cams[cam_name]['fovy_degree']
        
        if self.aug_K:
            new_img_w = raw_img_w
            new_img_h = raw_img_h
            
            # new_fu = np.random.uniform(350, 560)
            if cam_name in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
                if np.random.rand() < 1/6:
                    new_fu = np.random.uniform(350, 550)
                else:
                    new_fu = np.random.uniform(500, 700)
            else:
                new_fu = np.random.uniform(500, 700)
                
                            
            new_fv = new_fu
            new_cu = raw_img_w/2 + np.random.uniform(-abs(raw_cu-raw_img_w/2), abs(raw_cu-raw_img_w/2))
            if 'SIDE' in cam_name:
                new_cv = 90 + np.random.uniform(-5, 5)
            else:
                new_cv = raw_img_h/2 + np.random.uniform(-abs(raw_cv-raw_img_h/2), abs(raw_cv-raw_img_h/2))
            new_fovx = 2 * np.arctan(new_img_w / 2 / new_fu) / np.pi * 180
            new_fovy = 2 * np.arctan(new_img_h / 2 / new_fv) / np.pi * 180
        else:
            new_img_w, new_img_h, new_fu, new_fv, new_cu, new_cv, new_fovx, new_fovy = \
                raw_img_w, raw_img_h, raw_fu, raw_fv, raw_cu, raw_cv, raw_fovx, raw_fovy
            
        self.new_cams[cam_name]['K'] = np.array([[new_fu, 0, new_cu],
                                       [0, new_fv, new_cv],
                                       [0, 0, 1]], dtype=np.float32)
        self.new_cams[cam_name]['fu'], self.new_cams[cam_name]['fv'], \
            self.new_cams[cam_name]['cu'], self.new_cams[cam_name]['cv'] = \
                new_fu, new_fv, new_cu, new_cv
        self.new_cams[cam_name]['img_w'], self.new_cams[cam_name]['img_h'] = \
            new_img_w, new_img_h
        self.new_cams[cam_name]['fovx_degree'], self.new_cams[cam_name]['fovy_degree'] = \
            new_fovx, new_fovy
            
        return new_fu, new_fv, new_cu, new_cv, new_img_w, new_img_h
    
    
