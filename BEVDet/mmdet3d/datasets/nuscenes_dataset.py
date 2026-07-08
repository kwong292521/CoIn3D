# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
import json
import math
import os
import torch

import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
        img_info_prototype (str, optional): Type of img information.
            Based on 'img_info_prototype', the dataset will prepare the image
            data info in the type of 'mmcv' for official image infos,
            'bevdet' for BEVDet, and 'bevdet4d' for BEVDet4D.
            Defaults to 'mmcv'.
        multi_adj_frame_id_cfg (tuple[int]): Define the selected index of
            reference adjcacent frames.
        ego_cam (str): Specify the ego coordinate relative to a specified
            camera by its name defined in NuScenes.
            Defaults to None, which use the mean of all cameras.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 multi_adj_frame_id_cfg=None,
                 ego_cam='CAM_FRONT',
                 stereo=False,
                 nvs_mode=False,
                 random_adj_skip=None
                 ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            nvs_mode=nvs_mode
            )

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.ego_cam = ego_cam
        self.stereo = stereo
        
        self.random_adj_skip = random_adj_skip

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']
        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(
                        cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                            shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos
            else:
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
        return input_dict

    def get_adj_info(self, info, index):
        info_adj_list = []
        if (self.random_adj_skip is None) or self.test_mode:        
            adj_id_list = list(range(*self.multi_adj_frame_id_cfg))         
        else:
            assert type(self.random_adj_skip) is int and self.random_adj_skip > 0
            assert len(list(range(*self.multi_adj_frame_id_cfg))) == 1      
            _start, _end, _step = self.multi_adj_frame_id_cfg
            _offset = np.random.randint(0, self.random_adj_skip)
            adj_id_list = list(range(_start+_offset, _end+_offset, _step))
            
        if self.stereo:
            assert self.multi_adj_frame_id_cfg[0] == 1
            assert self.multi_adj_frame_id_cfg[2] == 1
            adj_id_list.append(self.multi_adj_frame_id_cfg[1])
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)
            if not self.data_infos[select_id]['scene_token'] == info[
                    'scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes = det['boxes_3d'].tensor.numpy()
            scores = det['scores_3d'].numpy()
            labels = det['labels_3d'].numpy()
            sample_token = self.data_infos[sample_id]['token']

            trans = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_translation']
            rot = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_rotation']
            rot = pyquaternion.Quaternion(rot)
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                #! NOTE: for no velocity in task
                if len(box_vel) == 0:
                    box_vel = [0, 0]
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                if np.sqrt(nusc_box.velocity[0]**2 +
                           nusc_box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    velocity=nusc_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox',
                         cde_setting=None
                         ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            
            # ! for lyft
            'v1.01-trainval': 'val',
            
            # ! for waymo
            'v1.4.0-trainval': 'val'
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
            cde_setting=cde_setting
            )
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 cde_setting=None
                 ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name], cde_setting=cde_setting)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, cde_setting=cde_setting)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)
            
            
            
    def format_results_to_markdown(self, json_file_path, caronly=False):
        json_data = read_json(json_file_path)
        if caronly:
            markdown_output = json_to_markdown_caronly(json_data)
        else:
            markdown_output = json_to_markdown(json_data)
        
        # Save the markdown output to a file
        with open(os.path.join(os.path.dirname(json_file_path), "metrics_table.md"), "w") as md_file:
            md_file.write(markdown_output)

        print("Markdown file has been generated successfully.")
        
    def format_results_to_markdown_all(self, json_file_path):
        json_data = read_json(json_file_path)
        
        markdown_output = json_to_markdown(json_data)
        markdown_output_car = json_to_markdown_caronly(json_data)
    
        # Save the markdown output to a file
        with open(os.path.join(os.path.dirname(json_file_path), "metrics_table.md"), "w") as md_file:
            md_file.write(markdown_output)
        with open(os.path.join(os.path.dirname(json_file_path), "metrics_table_car.md"), "w") as md_file:
            md_file.write(markdown_output_car)

        print("Markdown file has been generated successfully.")

    def convert_outputs_to_general_3cls(self, outputs, src_dataset='nuscenes', dst_dataset='nuscenes'):
        assert src_dataset in ['nuscenes', 'lyft', 'waymo']
        general_3cls = ['car', 'pedestrian', 'motorcycle']
        
        # NOTE: build-in mapping
        nusc_mapping = {
            'car': 'car', 'truck': 'car', 'trailer' : 'car', 'bus' : 'car', 'construction_vehicle' : 'car',
            'bicycle' : 'motorcycle', 'motorcycle' : 'motorcycle', 
            'pedestrian' : 'pedestrian', 
            'traffic_cone' : 'barrier', 'barrier' : 'barrier'
        
        }
        nusc_raw_cls = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
        nusc_new_cls = ['car', 'pedestrian', 'motorcycle', 'barrier']
        
        lyft_mapping = {
            'car' : 'car', 'truck' : 'car', 'bus' : 'car', 'emergency_vehicle' : 'car', 'other_vehicle' : 'car',
            'motorcycle' : 'motorcycle', 'bicycle' : 'motorcycle',
            'pedestrian' : 'pedestrian', 
            'animal' : 'barrier'
        }
        lyft_raw_cls = ['car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle', 'motorcycle', 'bicycle', 'pedestrian', 'animal']
        lyft_new_cls = ['car', 'pedestrian', 'motorcycle', 'barrier']
        
        waymo_mapping = {
            'VEHICLE' : 'car', 
            'PEDESTRIAN' : 'pedestrian', 
            'CYCLIST' : 'motorcycle'
        }
        waymo_raw_cls = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        waymo_new_cls = ['car', 'pedestrian', 'motorcycle', 'barrier']
        
        # ! convert
        if src_dataset == 'nuscenes':
            for output in outputs:
                for i in range(len(output['pts_bbox']['labels_3d'])):
                    label = output['pts_bbox']['labels_3d'][i]
                    output['pts_bbox']['labels_3d'][i] = nusc_new_cls.index(nusc_mapping[nusc_raw_cls[label]])
                    
        elif src_dataset == 'lyft':
            for output in outputs:
                for i in range(len(output['pts_bbox']['labels_3d'])):
                    label = output['pts_bbox']['labels_3d'][i]
                    output['pts_bbox']['labels_3d'][i] = lyft_new_cls.index(lyft_mapping[lyft_raw_cls[label]])
                    
        elif src_dataset == 'waymo':
            for output in outputs:
                for i in range(len(output['pts_bbox']['labels_3d'])):
                    label = output['pts_bbox']['labels_3d'][i]
                    output['pts_bbox']['labels_3d'][i] = waymo_new_cls.index(waymo_mapping[waymo_raw_cls[label]])
        
        return outputs


    def align_outputs_to_raw_cls(self, outputs, src_dataset='nuscenes', dst_dataset='nuscenes'):
        """
        hack implementation for nuscenes <-> lyft raw cls convertion
        """
        assert (src_dataset == 'nuscenes' and dst_dataset == 'lyft') or (dst_dataset == 'nuscenes' and src_dataset == 'lyft')
        
        # NOTE: build-in mapping
        comman_raw_cls = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian']     # set bicycle as a outlier holder
        
        nusc_mapping = {
            'car': 'car', 'truck': 'truck', 'trailer' : 'truck', 'bus' : 'bus', 'construction_vehicle' : 'truck',
            'bicycle' : 'motorcycle', 'motorcycle' : 'motorcycle', 
            'pedestrian' : 'pedestrian', 
            'traffic_cone' : 'bicycle', 'barrier' : 'bicycle'
        
        }
        nusc_raw_cls = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
        
        lyft_mapping = {
            'car' : 'car', 'truck' : 'truck', 'bus' : 'bus', 'emergency_vehicle' : 'truck', 'other_vehicle' : 'truck',
            'motorcycle' : 'motorcycle', 'bicycle' : 'motorcycle',
            'pedestrian' : 'pedestrian', 
            'animal' : 'bicycle'
        }
        lyft_raw_cls = ['car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle', 'motorcycle', 'bicycle', 'pedestrian', 'animal']
        
        # ! convert
        if src_dataset == 'nuscenes':
            for output in outputs:
                for i in range(len(output['pts_bbox']['labels_3d'])):
                    label = output['pts_bbox']['labels_3d'][i]
                    output['pts_bbox']['labels_3d'][i] = lyft_raw_cls.index(nusc_mapping[nusc_raw_cls[label]])
                    
        elif src_dataset == 'lyft':
            for output in outputs:
                for i in range(len(output['pts_bbox']['labels_3d'])):
                    label = output['pts_bbox']['labels_3d'][i]
                    output['pts_bbox']['labels_3d'][i] = nusc_raw_cls.index(lyft_mapping[lyft_raw_cls[label]])
        
        return outputs


def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list


# Function to convert JSON data to markdown table
def json_to_markdown(json_data):
    markdown_output = []

    # Convert Label APS
    markdown_output.append("**Label APS**")
    markdown_output.append("| Label             | 0.5      | 1.0      | 2.0      | 4.0      |")
    markdown_output.append("|-------------------|----------|----------|----------|----------|")
    for label, aps in json_data["label_aps"].items():
        markdown_output.append(f"| {label:<17} | {aps['0.5']: .3f}  | {aps['1.0']: .3f}  | {aps['2.0']: .3f}  | {aps['4.0']: .3f}  |")
    markdown_output.append("")  # Blank line for separation

    # Convert Mean Dist APS
    markdown_output.append("**Mean Dist APS**")
    markdown_output.append("| Label             | Mean Dist AP |")
    markdown_output.append("|-------------------|--------------|")
    mAP = 0
    count = 0
    for label, mean_dist_ap in json_data["mean_dist_aps"].items():
        markdown_output.append(f"| {label:<17} | {mean_dist_ap: .3f}        |")
        mAP += mean_dist_ap
        count += 1
    markdown_output.append("")  # Blank line for separation

    mAP /= count
    markdown_output.append("**mAP**")
    markdown_output.append("| mAP |")
    markdown_output.append("|-----|")
    markdown_output.append(f"| {mAP: .3f} |")
    markdown_output.append("")  # Blank line for separation

    # Convert TP Errors
    markdown_output.append("**TP Errors (Translational, Scale, Orientation, Velocity, Attribute)**")
    markdown_output.append("| Label             | Trans Err | Scale Err | Orient Err | Vel Err | Attr Err |")
    markdown_output.append("|-------------------|-----------|-----------|------------|---------|----------|")
    for label, errors in json_data["label_tp_errors"].items():
        markdown_output.append(f"| {label:<17} | {errors['trans_err']: .3f}     | {errors['scale_err']: .3f}     | {errors['orient_err']: .3f}      | {errors['vel_err']: .3f}   | {errors['attr_err']: .3f}    |")
    markdown_output.append("")  # Blank line for separation
    
    # Convert Mean TP Errors
    sum_trans_err = 0
    sum_scale_err = 0
    sum_orient_err = 0
    sum_vel_err = 0
    sum_attr_err = 0
    count_trans = count_scale = count_orient = count_vel = count_attr = 0
    
    for errors in json_data["label_tp_errors"].values():
        if not math.isnan(errors['trans_err']):
            sum_trans_err += errors['trans_err']
            count_trans += 1
        if not math.isnan(errors['scale_err']):
            sum_scale_err += errors['scale_err']
            count_scale += 1
        if not math.isnan(errors['orient_err']):
            sum_orient_err += errors['orient_err']
            count_orient += 1
        if not math.isnan(errors['vel_err']):
            sum_vel_err += errors['vel_err']
            count_vel += 1
        if not math.isnan(errors['attr_err']):
            sum_attr_err += errors['attr_err']
            count_attr += 1

    avg_trans_err = sum_trans_err / count_trans if count_trans > 0 else float('nan')
    avg_scale_err = sum_scale_err / count_scale if count_scale > 0 else float('nan')
    avg_orient_err = sum_orient_err / count_orient if count_orient > 0 else float('nan')
    avg_vel_err = sum_vel_err / count_vel if count_vel > 0 else float('nan')
    avg_attr_err = sum_attr_err / count_attr if count_attr > 0 else float('nan')

    markdown_output.append("**Average TP Errors Across All Labels (excluding NaNs)**")
    markdown_output.append("| Trans Err | Scale Err | Orient Err | Vel Err | Attr Err |")
    markdown_output.append("|-----------|-----------|------------|---------|----------|")
    markdown_output.append(f"| {avg_trans_err: .3f}     | {avg_scale_err: .3f}     | {avg_orient_err: .3f}      | {avg_vel_err: .3f}   | {avg_attr_err: .3f}    |")
    markdown_output.append("")  # Blank line for separation
    

    # Convert TP Scores
    markdown_output.append("**TP Scores (Translational, Scale, Orientation, Velocity, Attribute)**")
    markdown_output.append("| Metric            | TP Score   |")
    markdown_output.append("|-------------------|------------|")
    tp_scores = json_data["tp_scores"]
    for metric, score in tp_scores.items():
        markdown_output.append(f"| {metric:<17} | {score: .3f}      |")
    markdown_output.append("")  # Blank line for separation

    # Convert ND Score
    markdown_output.append("**ND Score**")
    markdown_output.append("| ND Score  |")
    markdown_output.append("|-----------|")
    markdown_output.append(f"| {json_data['nd_score']: .3f}     |")
    markdown_output.append("")  # Blank line for separation

    # Convert Evaluation Time
    markdown_output.append("**Evaluation Time**")
    markdown_output.append("| Evaluation Time (s) |")
    markdown_output.append("|---------------------|")
    markdown_output.append(f"| {json_data['eval_time']: .3f}              |")
    
    return "\n".join(markdown_output)


def json_to_markdown_caronly(json_data):
    markdown_output = []

    # Convert Label APS
    markdown_output.append("**Label APS (Car Only)**")
    markdown_output.append("| Label | 0.5 | 1.0 | 2.0 | 4.0 |")
    markdown_output.append("|-------|------|------|------|------|")
    car_aps = json_data["label_aps"].get("car")
    if car_aps:
        markdown_output.append(f"| car   | {car_aps['0.5']: .3f} | {car_aps['1.0']: .3f} | {car_aps['2.0']: .3f} | {car_aps['4.0']: .3f} |")
    markdown_output.append("")

    # mAP is just for car
    car_md_ap = json_data["mean_dist_aps"].get("car", float('nan'))
    markdown_output.append("**mAP (Car Only)**")
    markdown_output.append("| mAP |")
    markdown_output.append("|-----|")
    markdown_output.append(f"| {car_md_ap: .3f} |")
    markdown_output.append("")

    # Convert TP Errors
    markdown_output.append("**TP Errors (Car Only)**")
    markdown_output.append("| Label | Trans Err | Scale Err | Orient Err | Vel Err | Attr Err |")
    markdown_output.append("|--------|-----------|-----------|------------|---------|----------|")
    car_errors = json_data["label_tp_errors"].get("car")
    if car_errors:
        markdown_output.append(f"| car    | {car_errors['trans_err']: .3f}     | {car_errors['scale_err']: .3f}     | {car_errors['orient_err']: .3f}      | {car_errors['vel_err']: .3f}   | {car_errors['attr_err']: .3f}    |")
    markdown_output.append("")

    # NDS*
    _tp_score = 0
    for key in ["trans_err", "scale_err", "orient_err"]:
        _tp_score += (1 - min(1.0, car_errors[key]))
    nds_star_car = (1 / 6) * (3 * car_md_ap + _tp_score)

    markdown_output.append("**Domain-Generalized NDS\\***")
    markdown_output.append("| Metric | Value |")
    markdown_output.append("|--------|--------|")
    markdown_output.append(f"| NDS*   | {nds_star_car: .3f} |")
    markdown_output.append("")

    return "\n".join(markdown_output)


# Read JSON data from file
def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
