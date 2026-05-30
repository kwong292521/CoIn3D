import os
import numpy as np
import torch
import cv2
import tqdm
import pickle
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from common_utils import *
from __PATHS__ import *
from typing import List, Tuple, Optional
from dataclasses import dataclass
from easydict import EasyDict
from collections import defaultdict
from scipy.spatial.transform import Slerp, Rotation
from scipy.linalg import logm, expm
from typing import Optional


# ! some constants
LIDAR_FEAT = {
    'nuscenes': 5,
    'lyft': 5,
    'waymo': 6
}


# ! common utils
def construct_T_from_vector(translation_vector, rotation_vector):
    assert len(translation_vector) == 3
    assert len(rotation_vector) == 4
    
    T = np.eye(4)
    R = Quaternion(*rotation_vector).rotation_matrix
    T[:3, :3] = R
    T[:3, 3] = translation_vector
    return T
    

def cart2homo(pts):
    assert pts.shape[-1] == 3
    return np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)

def cart2homo_cuda(pts):
    assert pts.shape[-1] == 3
    return torch.cat([pts, torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)], axis=1)

def get_box_size(box2d):
    return (box2d[2] - box2d[0]) * (box2d[3] - box2d[1])


def get_pts_2d_to_3d(pts_2d, pts_depth, K):
    u_pts, v_pts = pts_2d[:, 0], pts_2d[:, 1]
    z = pts_depth
    
    cu, cv, fu, fv = K[0, 2], K[1, 2], K[0, 0], K[1, 1]
    
    x = (u_pts - cu) * z / fu
    y = (v_pts - cv) * z / fv
    
    pts_3d = np.stack((x, y, z), axis=1)
    return pts_3d


def get_pts2d3d_from_dense_depth(depth, K, mask):
    v_fg, u_fg = np.where(mask)
    
    cu = K[0, 2]
    cv = K[1, 2]
    fu = K[0, 0]
    fv = K[1, 1]
    z = depth[v_fg, u_fg]
    
    x = (u_fg - cu) * z / fu
    y = (v_fg - cv) * z / fv
    
    # x_offset = -calib.P2[0, 3] / fu
    # x += x_offset
    
    pts_3d = np.stack((x, y, z), axis=1)
    pts_2d = np.stack((u_fg, v_fg), axis=1)
    
    return pts_2d, pts_3d


def get_pts2d3d_from_dense_depth_cuda(depth, K, mask):
    v_fg, u_fg = torch.where(mask)
    
    cu = K[0, 2]
    cv = K[1, 2]
    fu = K[0, 0]
    fv = K[1, 1]
    z = depth[v_fg, u_fg]
    
    x = (u_fg - cu) * z / fu
    y = (v_fg - cv) * z / fv
    
    pts_3d = torch.stack((x, y, z), axis=1)
    pts_2d = torch.stack((u_fg, v_fg), axis=1)
    
    return pts_2d, pts_3d


def get_pts2d3d_from_sparse_depth(depth, K):
    mask = (depth != 0)
    v_fg, u_fg = np.where(mask)
    
    cu = K[0, 2]
    cv = K[1, 2]
    fu = K[0, 0]
    fv = K[1, 1]
    z = depth[v_fg, u_fg]
    
    x = (u_fg - cu) * z / fu
    y = (v_fg - cv) * z / fv
    
    # x_offset = -calib.P2[0, 3] / fu
    # x += x_offset
    
    pts_3d = np.stack((x, y, z), axis=1)
    pts_2d = np.stack((u_fg, v_fg), axis=1)
    
    return pts_2d, pts_3d


def save_texture_point_cloud_to_ply(geo, tex, filename):
    """
    保存带颜色的点云到PLY文件
    :param geo: (N,3) 顶点坐标
    :param tex: (N,3) 顶点颜色（RGB，范围0-255）
    """
    # tex = (tex * 255).astype(np.uint8)  # 确保颜色在0-255范围内
    with open(filename, 'wb') as f:
        # 写入PLY头
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {geo.shape[0]}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(b"end_header\n")
        
        # 写入二进制数据
        for i in range(geo.shape[0]):
            f.write(geo[i].astype(np.float32).tobytes())
            f.write(tex[i].tobytes())


def save_texture_point_cloud_to_ply_with_aux(geo, tex, filename, boxes=None, lidar_pts=None):
    """
    保存带颜色的点云和3D box的离散化点到PLY文件
    :param geo: (N,3) 顶点坐标
    :param tex: (N,3) 顶点颜色（RGB，范围0-255）
    :param boxes: 每个box的8个顶点坐标，形状为(M, 8, 3)
    :param lidar_pts: (N,3) Lidar点坐标
    """
    box_color = (255, 0, 0)  # 红色
    lidar_color = (0, 255, 0)  # 绿色

    # 生成3D box的边缘点
    if boxes is not None:
        box_points = []
        box_colors = []
        num_samples = 100  # 每条边插入的点数
        for box in boxes:
            # 连接每个box的8个顶点，形成12条边
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
                (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
                (0, 4), (1, 5), (2, 6), (3, 7)   # 连接上下面的边
            ]
            for start, end in edges:
                start_point = box[start]
                end_point = box[end]
                for i in range(num_samples):
                    t = i / (num_samples - 1)  # 线性插值因子
                    point = (1 - t) * start_point + t * end_point
                    box_points.append(point)
                    box_colors.append(box_color)  # 每个边的颜色为红色

        # 将3D box的边缘点添加到点云数据中
        geo = np.vstack([geo, np.array(box_points)])
        tex = np.vstack([tex, np.array(box_colors).astype(np.uint8)]) 
    
    # 将Lidar点添加到点云数据中
    if lidar_pts is not None:
        geo = np.vstack([geo, lidar_pts])
        tex = np.vstack([tex, np.tile(lidar_color, (lidar_pts.shape[0], 1)).astype(np.uint8)])  # Lidar点颜色为绿色
    
    # 保存到PLY文件
    with open(filename, 'wb') as f:
        # 写入PLY头
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {geo.shape[0]}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(b"end_header\n")

        # 写入点云和3D box的边缘点
        for i in range(geo.shape[0]):
            f.write(geo[i].astype(np.float32).tobytes())
            f.write(tex[i].tobytes())




def load_texture_point_cloud_from_ply(filename):
    with open(filename, 'rb') as f:
        # 跳过PLY头
        while True:
            line = f.readline().decode('utf-8').strip()
            if line == "end_header":
                break
        
        # 定义结构化数据类型
        dtype = np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
        ])
        
        # 读取数据
        data = np.fromfile(f, dtype=dtype)
        geo = np.column_stack((data['x'], data['y'], data['z']))
        tex = np.column_stack((data['r'], data['g'], data['b'])) 
        
    return geo, tex


def read_depth_map(depth_map_path):

    depth_image = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
    depth_map = depth_image / 256.0

    # Discard depths less than 10cm from the camera
    depth_map[depth_map < 0.1] = 0.0

    return depth_map.astype(np.float32)


def save_depth_map(save_path, depth_map,
                   version='cv2', png_compression=3):
    """Saves depth map to disk as uint16 png

    Args:
        save_path: path to save depth map
        depth_map: depth map numpy array [h w]
        version: 'cv2' or 'pypng'
        png_compression: Only when version is 'cv2', sets png compression level.
            A lower value is faster with larger output,
            a higher value is slower with smaller output.
    """

    # Convert depth map to a uint16 png
    depth_image = (depth_map * 256.0).astype(np.uint16)

    if version == 'cv2':
        ret = cv2.imwrite(save_path, depth_image, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])

        if not ret:
            raise RuntimeError('Could not save depth map')
    else:
        raise ValueError('Invalid version', version)


def q2degrees(q):
    assert type(q) == list
    q = Quaternion(*q)
    yaw, pitch, roll = q.yaw_pitch_roll
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


# ! nusc devkit aux 
def load_dataset_devkit(dataset, split, use_cache=True):
    assert dataset in ['nuscenes', 'lyft', 'waymo']
    assert split in ['train', 'val']
    
    print(f'loading devkit {dataset} {split}...')
    
    _mapping = {
        'nuscenes': {
            'mminfo_train': NUSC_MMINFO_TRAIN,
            'mminfo_val': NUSC_MMINFO_VAL,
            'devkit': NUSC_DEVKIT,
            'root': NUSC_ROOT_DIR
        },
        'lyft': {
            'mminfo_train': LYFT_MMINFO_TRAIN,
            'mminfo_val': LYFT_MMINFO_VAL,
            'devkit': LYFT_DEVKIT,
            'root': LYFT_ROOT_DIR
        },
        'waymo': {
            'mminfo_train': WAYMO_MMINFO_TRAIN,
            'mminfo_val': WAYMO_MMINFO_VAL,
            'devkit': WAYMO_DEVKIT,
            'root': WAYMO_ROOT_DIR
        }
    }
    
    assert os.path.exists(_mapping[dataset][f'mminfo_{split}'])
    with open(_mapping[dataset][f'mminfo_{split}'], 'rb') as f:
        mminfo = pickle.load(f)
    
    if use_cache:
        if os.path.exists(_mapping[dataset]['devkit']):
            with open(_mapping[dataset]['devkit'], 'rb') as f:
                devkit = pickle.load(f)
        else:
            devkit = NuScenes(
                version=mminfo['metadata']['version'],
                dataroot=_mapping[dataset]['root'],
                verbose=True
            )
            with open(_mapping[dataset]['devkit'], 'wb') as f:
                pickle.dump(devkit, f)
    else:
        devkit = NuScenes(
            version=mminfo['metadata']['version'],
            dataroot=_mapping[dataset]['root'],
            verbose=True
        )


    out_dir_mapping = {
        'nuscenes': NUSC_OUT_DIR,
        'lyft': LYFT_OUT_DIR,
        'waymo': WAYMO_OUT_DIR
    }
    out_dir = out_dir_mapping[dataset]

    print('devkit loaded...')

    return devkit, mminfo, out_dir


def get_sample_tokens_from_scene(nusc, scene_token):
    scene = nusc.get('scene', scene_token)

    sample_tokens = []
    current_sample_token = scene['first_sample_token']
    while current_sample_token:
        sample_tokens.append(current_sample_token)
        current_sample = nusc.get('sample', current_sample_token)
        current_sample_token = current_sample['next']  # 跳到下一帧

    return sample_tokens


def get_all_lidar_data_tokens(nusc, scene_token, no_key_frame=False):
    scene_rec = nusc.get("scene", scene_token)
    
    start_sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
    sd_rec = nusc.get("sample_data", start_sample_rec["data"]['LIDAR_TOP'])
    
    tokens = []
    
    cur_sd_rec = sd_rec
    
    while True:
        is_key_frame = cur_sd_rec['is_key_frame']
        if no_key_frame and is_key_frame:
            if cur_sd_rec["next"] == "": break
            cur_sd_rec = nusc.get("sample_data", cur_sd_rec["next"])
            continue

        tokens.append(cur_sd_rec['token'])
        
        if cur_sd_rec["next"] == "": break
        cur_sd_rec = nusc.get("sample_data", cur_sd_rec["next"])
        
    return tokens
        
        

def get_all_lidar_sweeps(nusc, scene_token, no_key_frame=True):
    scene_rec = nusc.get("scene", scene_token)
    
    start_sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
    sd_rec = nusc.get("sample_data", start_sample_rec["data"]['LIDAR_TOP'])
    
    lidar_data = {
        'tokens': [],
        'timestamps': [],
        'is_key_frame': [],
        'T_lidar2egos': [],
        'T_ego2globals': [],
        
        # raw type data
        'lidar2ego_translations': [],
        'lidar2ego_rotations': [],
        'ego2global_translations': [],
        'ego2global_rotations': []   
    }
    
    cur_sd_rec = sd_rec
    
    while True:
        is_key_frame = cur_sd_rec['is_key_frame']
        if no_key_frame and is_key_frame:
            if cur_sd_rec["next"] == "": break
            cur_sd_rec = nusc.get("sample_data", cur_sd_rec["next"])
            continue
        
        lidar_data['tokens'].append(cur_sd_rec['token'])
        lidar_data['timestamps'].append(cur_sd_rec['timestamp'])
        lidar_data['is_key_frame'].append(cur_sd_rec['is_key_frame'])
        
        lidar2ego_token = cur_sd_rec['calibrated_sensor_token']
        lidar2ego_pose = nusc.get('calibrated_sensor', lidar2ego_token)
        T_lidar2ego = construct_T_from_vector(lidar2ego_pose['translation'], lidar2ego_pose['rotation'])
        lidar_data['T_lidar2egos'].append(T_lidar2ego)
        lidar_data['lidar2ego_translations'].append(lidar2ego_pose['translation'])
        lidar_data['lidar2ego_rotations'].append(lidar2ego_pose['rotation'])
    
        ego2world_token = cur_sd_rec['ego_pose_token']
        ego2world_pose = nusc.get('ego_pose', ego2world_token)
        T_ego2world = construct_T_from_vector(ego2world_pose['translation'], ego2world_pose['rotation'])
        lidar_data['T_ego2globals'].append(T_ego2world)
        lidar_data['ego2global_translations'].append(ego2world_pose['translation'])
        lidar_data['ego2global_rotations'].append(ego2world_pose['rotation'])
        
        if cur_sd_rec["next"] == "": break
        cur_sd_rec = nusc.get("sample_data", cur_sd_rec["next"])

    return lidar_data



def get_sensor_data_tokens_from_sample(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    sensor_data_tokens = {key: sample['data'][key] for key in sample['data'].keys()}
    return sensor_data_tokens


def retrieve_scene_token_from_filename(nusc, filename):
    """
    根据传感器数据文件名（如某张图像）追溯其所在的 scene，并返回 scene token。
    使用 NuScenes 内置的 get 和查询方法来简化实现。

    :param nusc: 已加载的 NuScenes 数据集实例
    :param filename: 传感器数据文件名，例如 "n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915245012465.jpg"
    :return: scene token（如果找到），否则返回 None
    """
    # 1. 查找与 filename 匹配的 sample_data token
    sample_data_token = None
    for sample_data in nusc.sample_data:
        if filename in sample_data['filename']:  # 文件名匹配
            sample_data_token = sample_data['token']
            break
    
    if not sample_data_token:
        print(f"未找到文件 {filename} 对应的 sample_data。")
        return None
    
    # 2. 根据 sample_data token 获取 sample 和 scene
    sample_data = nusc.get('sample_data', sample_data_token)
    sample_token = sample_data['sample_token']
    
    # 3. 使用 sample_token 查找对应的 scene
    sample = nusc.get('sample', sample_token)
    scene_token = sample['scene_token']
    
    # 4. 获取对应的 scene 信息
    scene = nusc.get('scene', scene_token)
    return scene_token


def get_sd_token_ego2global(nusc, sample_data_token):
    """
    返回某一帧的 ego 坐标系到 NuScenes 世界坐标系的 4×4 变换矩阵
    """
    sd = nusc.get('sample_data', sample_data_token)
    ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
    
    q = Quaternion(ego_pose['rotation'])  # w, x, y, z
    t = np.array(ego_pose['translation'])  # x, y, z
    
    # 构造 4x4 齐次变换矩阵
    T_ego2global = q.transformation_matrix  # 默认返回旋转部分
    T_ego2global[0:3, 3] = t  # 设置平移部分

    return T_ego2global, ego_pose


def get_sensor_ego2global(nusc, sample_token, camera_name):
    sample = nusc.get('sample', sample_token)
    camera_data_token = sample['data'][camera_name]
    camera_data = nusc.get('sample_data', camera_data_token)
    ego_pose = nusc.get('ego_pose', camera_data['ego_pose_token'])
    
    return ego_pose['translation'], ego_pose['rotation']


def identify_token_type(nusc, token):
    """
    判断给定 token 属于 NuScenes 中的哪个表格（如 sample, sample_data, ego_pose 等）
    """
    table_names = nusc.table_names
    for table_name in table_names:
        table = getattr(nusc, table_name, None)
        if table is not None:
            for record in table:
                if record['token'] == token:
                    return table_name, record 

    return None, None


# ! sweep boxes interpolate utils
@dataclass
class BBoxState:
    """表示BEV视角下的物体状态（仅位置和朝向）"""
    x: float    # 中心点x坐标 (m)
    y: float    # 中心点y坐标 (m)
    z: float    # 中心点z坐标 (m)
    yaw: float  # 朝向角 (弧度)
    timestamp: float  # 时间戳 (s)

class SweepBoxEstimator:
    def __init__(self, states: List[BBoxState]):
        """
        初始化基于CTRV模型的插值器
        NOTE: 输入的box的xyz yaw应该是在同一个坐标系下的(我们用global坐标系)
        
        参数:
            states: 按时间戳排序的BBoxState列表
        """
        self.states = sorted(states, key=lambda s: s.timestamp)
        self._validate_states()

    def __call__(self, timestamps: List[float]) -> List[BBoxState]:
        """
        插值指定时间戳的状态
        
        参数:
            timestamps: 目标时间戳列表（需在输入状态时间范围内）
        
        返回:
            插值后的状态列表（仅含x,y,z,yaw）
        """
        results = []
        for t in sorted(timestamps):
            prev, next = self._find_nearest_states(t)
            if prev is None and next is None:
                raise ValueError(f"时间戳 {t} 超出输入状态范围")
            
            if prev is None:  # 早于第一个状态 -> 用第一个状态
                results.append(next)
            elif next is None:  # 晚于最后一个状态 -> 用最后一个状态
                results.append(prev)
            else:  # 正常插值
                results.append(self._ctrv_interpolation(prev, next, t))
        return results

    def _validate_states(self):
        """检查输入状态是否有效"""
        if len(self.states) < 2:
            raise ValueError("至少需要两个状态点用于插值")
        if any(s1.timestamp >= s2.timestamp for s1, s2 in zip(self.states[:-1], self.states[1:])):
            raise ValueError("时间戳必须严格递增")

    def _find_nearest_states(self, timestamp: float) -> tuple:
        """找到给定时间戳最近的前后状态"""
        if timestamp <= self.states[0].timestamp:
            return None, self.states[0]
        elif timestamp >= self.states[-1].timestamp:
            return self.states[-1], None
        
        for prev, next in zip(self.states[:-1], self.states[1:]):
            if prev.timestamp <= timestamp <= next.timestamp:
                return prev, next
        return None, None


    def _ctrv_interpolation(self, prev: BBoxState, next: BBoxState, t: float) -> BBoxState:
        delta_t_total = next.timestamp - prev.timestamp
        t_rel = t - prev.timestamp
        
        # 1. 计算实际速度方向（忽略输入的yaw，直接由位移推导）
        dx = next.x - prev.x
        dy = next.y - prev.y
        v_mag = np.hypot(dx, dy) / delta_t_total  # 速度大小
        actual_yaw = np.arctan2(dy, dx)           # 实际运动方向
        
        # 2. 计算转向率（基于实际运动方向的变化）
        delta_yaw = (next.yaw - prev.yaw + np.pi) % (2 * np.pi) - np.pi
        omega = delta_yaw / delta_t_total if abs(delta_yaw) > 1e-6 else 0
        
        # 3. 插值位置（沿实际运动方向）
        if abs(omega) < 1e-6:  # 直线运动
            x = prev.x + (dx / delta_t_total) * t_rel
            y = prev.y + (dy / delta_t_total) * t_rel
        else:  # 转弯运动（以实际yaw为基准）
            current_yaw = actual_yaw + omega * t_rel
            radius = v_mag / omega  # 转弯半径
            x = prev.x + radius * (np.sin(current_yaw) - np.sin(actual_yaw))
            y = prev.y - radius * (np.cos(current_yaw) - np.cos(actual_yaw))
        
        # 4. 插值朝向和高度
        yaw = prev.yaw + omega * t_rel
        z = prev.z + (next.z - prev.z) * (t_rel / delta_t_total)
        
        return BBoxState(x, y, z, yaw, t)



def sweep_velocity_estimator(
    keyframe_timestamps: List[float], 
    keyframe_velocities: List,  # 每个元素是(vx, vy)
    sweep_timestamps: List[float]
) -> List:
    """
    基于恒定加速度模型的速度插值
    
    参数:
        keyframe_timestamps: 关键帧时间戳列表 (严格递增)
        keyframe_velocities: 对应关键帧的速度列表 [(vx1, vy1), (vx2, vy2), ...]
        sweep_timestamps: 需要插值的时间戳列表
    
    返回:
        插值后的速度列表 [(vx_interp, vy_interp), ...]
    """
    # 输入校验
    if len(keyframe_timestamps) != len(keyframe_velocities):
        raise ValueError("时间戳和速度列表长度必须相同")
    if len(keyframe_timestamps) < 2:
        raise ValueError("至少需要两个关键帧用于插值")
    if not all(t1 < t2 for t1, t2 in zip(keyframe_timestamps[:-1], keyframe_timestamps[1:])):
        raise ValueError("关键帧时间戳必须严格递增")
    
    # 转换为NumPy数组便于计算
    ts_kf = np.array(keyframe_timestamps)
    vx_kf = np.array([v[0] for v in keyframe_velocities])
    vy_kf = np.array([v[1] for v in keyframe_velocities])
    ts_sw = np.array(sweep_timestamps)
    
    # 初始化输出
    interp_velocities = []
    
    for t in ts_sw:
        # 找到最近的左右关键帧
        idx = np.searchsorted(ts_kf, t, side='right') - 1
        
        if idx < 0:  # 早于第一个关键帧 -> 用第一个关键帧速度
            interp_velocities.append((vx_kf[0], vy_kf[0]))
        elif idx >= len(ts_kf) - 1:  # 晚于最后一个关键帧 -> 用最后一个关键帧速度
            interp_velocities.append((vx_kf[-1], vy_kf[-1]))
        else:  # 正常插值
            t_prev, t_next = ts_kf[idx], ts_kf[idx+1]
            dt_total = t_next - t_prev
            dt = t - t_prev
            
            # 计算加速度
            ax = (vx_kf[idx+1] - vx_kf[idx]) / dt_total
            ay = (vy_kf[idx+1] - vy_kf[idx]) / dt_total
            
            # 恒定加速度模型
            vx = vx_kf[idx] + ax * dt
            vy = vy_kf[idx] + ay * dt
            interp_velocities.append([float(vx), float(vy)])
    
    return interp_velocities




# ! objects class and utils
class MMInstances:
    """
    collect each instancs information
    obj pts model, obj rendering mask, and so on
    we finally export it as a pkl file for save and loading
    """
    def __init__(self):
        pass
        
    
    
    


class MMSceneObjects:
    """
    Collect all objects for the given scene_token
    """
    def __init__(self, devkit, mminfo, scene_token, dataset_name, use_sweeps=True):
        assert dataset_name in ['nuscenes', 'waymo', 'lyft']
        self.dataset_name = dataset_name
        
        mminfo_token2idx = {_info['token']: idx for idx, _info in enumerate(mminfo)}
        self.sample_tokens = get_sample_tokens_from_scene(devkit, scene_token)
        self.scene_mminfos = {}
        for sample_token in self.sample_tokens:
            self.scene_mminfos[sample_token] = mminfo[mminfo_token2idx[sample_token]]
        
        self.all_lidar_data_tokens = get_all_lidar_data_tokens(devkit, scene_token)
        
        if dataset_name == 'nuscenes' and use_sweeps:
            # lyft / waymo is without sweeps (each frame is key frame)
            self.all_lidar_sweeps = get_all_lidar_sweeps(devkit, scene_token, no_key_frame=True)

            
        # init objects in each frame
        self.lidar_objs = []
        self.cam_objs = []
        for sample_token in self.sample_tokens:
            # if sample_token == '6c67092c8464af4f86e5927e8abda9a3ef6cb5b00222901d3dc25d9df2906abb':
            #     print('debug')
            self.lidar_objs.append(
                MMSampleLidarObjects(self.scene_mminfos[sample_token]))
            self.cam_objs.append(
                MMSampleCameraObjects(self.scene_mminfos[sample_token]))
        self.lidar_timestamps = [_obj.lidar_timestamp for _obj in self.lidar_objs]
        
        # init instance in this scene
        self.instance_occ = self.get_scene_instances(self.lidar_objs)   # 'instance_token': (frame_idx, obj_idx)
        
        # init instances static/dynamic flag
        self.instance_static = self.get_instance_static()  
        
        # interpolate sweeps boxes
        if dataset_name == 'nuscenes' and use_sweeps:
            self.interpolate_lidar_sweep_boxes()
            self.lidar_timestamps = np.array(self.lidar_timestamps + [_obj.lidar_timestamp for _obj in self.lidar_sweeps_objs])
            self.sample_tokens = self.sample_tokens + self.all_lidar_sweeps['tokens']
            self.lidar_objs = self.lidar_objs + self.lidar_sweeps_objs
            # sorted according to timestamps
            sorted_idx = np.argsort(self.lidar_timestamps)
            self.lidar_timestamps = np.array(self.lidar_timestamps)[sorted_idx]
            self.sample_tokens = np.array(self.sample_tokens)[sorted_idx]
            self.lidar_objs = [self.lidar_objs[i] for i in sorted_idx]
                
    
    def get_instance_static(self):
        instance_static = {}
        for i, sample_token in enumerate(self.sample_tokens):
            mmobjs = self.lidar_objs[i]
            for j, instance_token in enumerate(mmobjs.lidar_instance_tokens):
                if instance_token not in instance_static.keys():
                    instance_static[instance_token] = []
                instance_static[instance_token].append(mmobjs.lidar_boxes_global[j])
        
        for k, v in instance_static.items():
            # _xyz = np.array(v)[:, :3]
            # _xyz -= _xyz[0]
            # _xyz = np.sum(np.abs(_xyz), axis=1)
            # instance_static[k] = (_xyz.max() < 1e-2)       # NOTE: xyz_global一致看作static
            
            _xy = np.array(v)[:, :2]
            _xy -= _xy[0]
            _xy = np.sum(np.abs(_xy), axis=1)
            instance_static[k] = (_xy.max() < 0.5)       # NOTE: xyz_global一致看作static

        return instance_static
    
    
    def get_scene_instances(self, lidar_objs):
        instance_tokens = [obj.lidar_instance_tokens for obj in lidar_objs]
        _max_objs = max([len(item) for item in instance_tokens])
        _max_frame = len(lidar_objs)
        instance_occ_map = np.empty((_max_frame, _max_objs), dtype=object)
        uni_tokens = []
        for i, _tokens in enumerate(instance_tokens):
            for j, _token in enumerate(_tokens):
                instance_occ_map[i, j] = _token
                uni_tokens.append(_token)
                
        uni_tokens = np.unique(uni_tokens)
        instance_occ_list = {}
        for _token in uni_tokens:
            instance_occ_list[_token] = np.argwhere(instance_occ_map == _token)
        
        return instance_occ_list
        
        
    def interpolate_lidar_sweep_boxes(self):
        """
        插值得到两个key_frame中间的boxes
        P.S.:yaw角定义: 重力轴z轴 参考轴x轴 从重力轴的负方向（轴的正方向指向人眼）看，yaw沿着参考轴逆时针方向增加 即x->y
        插值方式: 短时间内恒定转向率+恒定加速度假设
        """
        sweep_tokens = self.all_lidar_sweeps['tokens']
        sweep_timestamps = np.array(self.all_lidar_sweeps['timestamps']) / 1e+6
        
        key_tokens = self.sample_tokens
        key_timestamps = np.array(self.lidar_timestamps)
        
        instance_boxes = {}
        instance_velocity = {}
        for k, v in self.instance_occ.items():
            instance_boxes[k] = [
                self.lidar_objs[v[i, 0]].lidar_boxes_global[v[i, 1]] for i in range(len(v))]
            instance_velocity[k] = [
                self.lidar_objs[v[i, 0]].boxes_velocity[v[i, 1]] for i in range(len(v))]
        
        instance_sweep_boxes = {}
        instance_sweep_veloccities = {}
        instance_sweep_tokens = {}
        instance_sweep_timestamps = {}
        
        for instance_token in self.instance_occ.keys():
            key_boxes = instance_boxes[instance_token]
            key_velocity = instance_velocity[instance_token]
            key_frame_idx = self.instance_occ[instance_token][:, 0]
            
            n_key_obj = len(key_boxes)
            
            if n_key_obj < 2:
                instance_sweep_boxes[instance_token] = []
                instance_sweep_veloccities[instance_token] = []
                instance_sweep_tokens[instance_token] = []
                instance_sweep_timestamps[instance_token] = []
                continue
            
            _boxes = []
            for i in range(n_key_obj):
                _boxes.append(BBoxState(
                    x=key_boxes[i][0], y=key_boxes[i][1], z=key_boxes[i][2], yaw=key_boxes[i][-1],
                    timestamp=key_timestamps[key_frame_idx[i]]))
            
            # only care sweep between key frames
            _mask = (sweep_timestamps > key_timestamps[key_frame_idx[0]]) & (sweep_timestamps < key_timestamps[key_frame_idx[-1]])
            _sweep_tokens = np.array(sweep_tokens)[_mask]
            _sweep_timestamps = sweep_timestamps[_mask]
            
            sweep_boxes_estimator = SweepBoxEstimator(states=_boxes)
            _sweep_boxes = sweep_boxes_estimator(_sweep_timestamps)
            _sweep_velocities = sweep_velocity_estimator(
                keyframe_timestamps=[key_timestamps[key_frame_idx[i]] for i in range(n_key_obj)],
                keyframe_velocities=key_velocity,
                sweep_timestamps=_sweep_timestamps
            )
            
            _l, _w, _h = key_boxes[0][3], key_boxes[0][4], key_boxes[0][5]
            sweep_boxes = []
            for _box in _sweep_boxes:
                sweep_boxes.append(
                    [_box.x, _box.y, _box.z, _l, _w, _h, _box.yaw])
            
            instance_sweep_boxes[instance_token] = sweep_boxes
            instance_sweep_veloccities[instance_token] = _sweep_velocities
            instance_sweep_tokens[instance_token] = _sweep_tokens
            instance_sweep_timestamps[instance_token] = _sweep_timestamps
            
        # create self.lidar_objs like self.lidar_sweeps_objs
        self.lidar_sweeps_objs = []
        for _sweep_idx, _sweep_token in enumerate(sweep_tokens):
            mminfo = {
                'token': _sweep_token,
                
                'gt_boxes_global': [],
                'instance_tokens': [],
                'timestamp': self.all_lidar_sweeps['timestamps'][_sweep_idx],
                'gt_velocity': [],
                
                'lidar2ego_translation': self.all_lidar_sweeps['lidar2ego_translations'][_sweep_idx],
                'lidar2ego_rotation': self.all_lidar_sweeps['lidar2ego_rotations'][_sweep_idx],
                'ego2global_translation': self.all_lidar_sweeps['ego2global_translations'][_sweep_idx],
                'ego2global_rotation': self.all_lidar_sweeps['ego2global_rotations'][_sweep_idx],
            }
        
            for instance_token in instance_sweep_boxes.keys():
                for i, _token in enumerate(instance_sweep_tokens[instance_token]):
                    if _token == _sweep_token:
                        mminfo['gt_boxes_global'].append(instance_sweep_boxes[instance_token][i])
                        mminfo['instance_tokens'].append(instance_token)
                        mminfo['gt_velocity'].append(instance_sweep_veloccities[instance_token][i])
        
            mminfo['gt_boxes_global'] = np.array(mminfo['gt_boxes_global'])
            mminfo['gt_velocity'] = np.array(mminfo['gt_velocity'])
            
            self.lidar_sweeps_objs.append(MMSampleLidarObjects(mminfo))
        
            
        
class MMSampleLidarObjects:
    """
    Collect all objects for the given sample lidar (即某个标注sample下所有cam标注帧的物体类)
    """
    def __init__(self, mminfo):
        self.mminfo = mminfo
        
        #==== lidar extrinsic
        self.T_lidar2ego = construct_T_from_vector(
            mminfo['lidar2ego_translation'],
            mminfo['lidar2ego_rotation'])
        self.T_ego2lidar = np.linalg.inv(self.T_lidar2ego)
        
        self.T_ego2global = construct_T_from_vector(
            mminfo['ego2global_translation'],
            mminfo['ego2global_rotation'])
        self.T_global2ego = np.linalg.inv(self.T_ego2global)
        
        self.T_lidar2global = self.T_ego2global @ self.T_lidar2ego
        self.T_global2lidar = self.T_ego2lidar @ self.T_global2ego
        
        
        #==== init lidar objects for all ====#
        self.sample_token = mminfo['token']
        self.lidar_instance_tokens = mminfo['instance_tokens']
        self.lidar_timestamp = mminfo['timestamp'] / 1e+6
        
        self.boxes_velocity = mminfo['gt_velocity']       # NOTE: 速度是定义在global坐标系的
        # self.boxes_static = np.linalg.norm(self.boxes_velocity, axis=1) < 1e-6
        
        if 'gt_boxes' in mminfo.keys():
            assert 'gt_boxes_global' not in mminfo.keys()
            self.lidar_boxes = mminfo['gt_boxes']        
            self.lidar_boxes_global = self.trans_boxes_lidar_global(lidar_boxes=self.lidar_boxes)
        elif 'gt_boxes_global' in mminfo.keys():
            assert 'gt_boxes' not in mminfo.keys()
            self.lidar_boxes_global = mminfo['gt_boxes_global']
            self.lidar_boxes = self.trans_boxes_lidar_global(global_boxes=self.lidar_boxes_global)
        
        #==== lidar box corners
        lidar_corners, local_corners, T_local2lidars = self.lidarbox3d_to_corners(self.lidar_boxes)
        self.lidar_corners = lidar_corners
        self.local_corners = local_corners
        self.T_local2lidars = T_local2lidars    # (N, 4, 4)
        self.T_lidar2locals = np.linalg.inv(T_local2lidars)  # (N, 4, 4)
        
        self.T_local2globals = self.T_lidar2global @ self.T_local2lidars
        self.T_global2locals = self.T_lidar2locals @ self.T_global2lidar
            
            
    def trans_boxes_lidar_global(self, lidar_boxes=None, global_boxes=None):
        def yaw_to_unit_vector(yaw_rad):
            x = np.cos(yaw_rad)
            y = np.sin(yaw_rad)
            z = np.zeros_like(yaw_rad)
            return np.stack([x, y, z], axis=-1)
        def vector_to_yaw(vec):
            x = vec[:, 0]
            y = vec[:, 1]
            return np.arctan2(y, x)
        
        if lidar_boxes is not None:
            assert global_boxes is None
            _mode = 'lidar2global'
        elif global_boxes is not None:
            assert lidar_boxes is None
            _mode = 'global2lidar'
        
        if _mode == 'lidar2global':
            xyz_lidar = lidar_boxes[:, :3]
            size_lidar = lidar_boxes[:, 3:6]
            yaw_lidar = lidar_boxes[:, 6]
            
            R, t = self.T_lidar2global[:3, :3], self.T_lidar2global[:3, 3]
            xyz_global = (xyz_lidar @ R.T) + t
            
            yaw_vector_lidar = yaw_to_unit_vector(yaw_lidar)
            yaw_vector_global = yaw_vector_lidar @ R.T
            yaw_global = vector_to_yaw(yaw_vector_global)

            return np.concatenate([xyz_global, size_lidar, yaw_global[:, None]], axis=1)
        
        elif _mode == 'global2lidar':
            xyz_global = global_boxes[:, :3]
            size_global = global_boxes[:, 3:6]
            yaw_global = global_boxes[:, 6]
            
            R, t = self.T_global2lidar[:3, :3], self.T_global2lidar[:3, 3]
            xyz_lidar = (xyz_global @ R.T) + t

            yaw_vector_global = yaw_to_unit_vector(yaw_global)
            yaw_vector_lidar = yaw_vector_global @ R.T
            yaw_lidar = vector_to_yaw(yaw_vector_lidar)
            return np.concatenate([xyz_lidar, size_global, yaw_lidar[:, None]], axis=1)
        
        
    def lidarbox3d_to_corners(self, lidar_boxes):
        """
        将 [x, y, z, dx(l), dy(w), dz(h), yaw] 的 3D box 转换为 8 个角点
        
        前上面（顺时针）：0,1,2,3
        后下面（顺时针）：4,5,6,7
        """
        N = lidar_boxes.shape[0]
        centers = lidar_boxes[:, :3]   # (N, 3)
        dims = lidar_boxes[:, 3:6]     # (N, 3): dx, dy, dz
        yaws = lidar_boxes[:, 6]       # (N,)

        # 8 个局部角点坐标（同上，局部坐标系中固定）
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

        # 广播复制：(N, 8, 3)
        local_corners = local_corners[None, :, :] * dims[:, None, :]  # (N, 8, 3)
        
        # 构建旋转矩阵 R_z (yaw 绕 z 轴)
        cos_yaw = np.cos(yaws)
        sin_yaw = np.sin(yaws)

        zeros = np.zeros_like(yaws)
        ones = np.ones_like(yaws)

        R = np.stack([
            np.stack([cos_yaw, -sin_yaw, zeros], axis=1),
            np.stack([sin_yaw,  cos_yaw, zeros], axis=1),
            np.stack([zeros,    zeros,   ones],  axis=1),
        ], axis=1)  # (N, 3, 3)

        # 旋转角点
        rotated_corners = np.matmul(local_corners, R.transpose(0, 2, 1))  # (N, 8, 3)

        # 平移到全局位置
        lidar_corners = rotated_corners + centers[:, None, :]  # (N, 8, 3)
        
        T_local2lidars = np.eye(4)[None, :, :].repeat(N, axis=0)
        T_local2lidars[:, :3, :3] = R
        T_local2lidars[:, :3, 3] = centers

        return lidar_corners, local_corners, T_local2lidars
        
    
    def get_lidar_pts_in_boxes(self, pts_3d, bev_box=False, _res=0., _res_inbottom_only=False):
        """
        params:
        pts_3d: (n, 3) in lidar coord
        rets:
        instance_mask: (n, 1) 背景点标注-1 然后0~k-1标注每一个物体的点mask 默认每个物体的点是不重叠的
        """
        instance_mask = np.ones((pts_3d.shape[0], ), dtype=np.int32) * -1
        obj_local_pts = [None for _ in range(len(self.lidar_boxes))]
        
        for i in range(len(self.lidar_boxes)):
            # step 1: 生成box的8个角点
            _corners = self.lidar_corners[i]
            x_max, x_min = np.max(_corners[:,0]), np.min(_corners[:,0])
            y_max, y_min = np.max(_corners[:,1]), np.min(_corners[:,1])
            
            # step 2: 找到box的更大的包围框，截取这一部分点云作为初筛结果
            roi_mask = (
                (pts_3d[:, 0] < x_max) &
                (pts_3d[:, 0] > x_min) &
                (pts_3d[:, 1] < y_max) &
                (pts_3d[:, 1] > y_min))
            roi_points = pts_3d[roi_mask]
            roi_indices = np.where(roi_mask)[0]
            
            # step 3: 转换为局部坐标表示
            T_lidar2local = self.T_lidar2locals[i]
            roi_points_hom = np.concatenate([roi_points, np.ones((roi_points.shape[0], 1))], axis=1)
            roi_points_loc = (roi_points_hom @ T_lidar2local.T)[:, 0:3]
            
            # step 4: 通过物体的尺寸筛选得到在box内的点
            l, w, h = self.lidar_boxes[i, 3], self.lidar_boxes[i, 4], self.lidar_boxes[i, 5]
            if not _res_inbottom_only:
                if bev_box:
                    inbox_mask = (
                        (roi_points_loc[:,0] < l/2 - _res) &
                        (roi_points_loc[:,0] > -l/2 + _res) &
                        (roi_points_loc[:,1] < w/2 - _res) &
                        (roi_points_loc[:,1] > -w/2 + _res))
                else:
                    inbox_mask = (
                        (roi_points_loc[:,0] < l/2 - _res) &
                        (roi_points_loc[:,0] > -l/2 + _res) &
                        (roi_points_loc[:,2] < h/2 - _res) &
                        (roi_points_loc[:,2] > -h/2 + _res) &
                        (roi_points_loc[:,1] < w/2 - _res) &
                        (roi_points_loc[:,1] > -w/2 + _res))
            else:    
                if bev_box:
                    inbox_mask = (
                        (roi_points_loc[:,0] < l/2) &
                        (roi_points_loc[:,0] > -l/2) &
                        (roi_points_loc[:,1] < w/2) &
                        (roi_points_loc[:,1] > -w/2))
                else:
                    inbox_mask = (
                        (roi_points_loc[:,0] < l/2) &
                        (roi_points_loc[:,0] > -l/2) &
                        (roi_points_loc[:,2] < h/2) &
                        (roi_points_loc[:,2] > -h/2 + _res) &
                        (roi_points_loc[:,1] < w/2) &
                        (roi_points_loc[:,1] > -w/2))
            instance_indices = roi_indices[inbox_mask]
            
            # step 5: 标记哪些点是属于该物体的 & 提取box内的点
            instance_mask[instance_indices] = i
            obj_local_pts[i] = roi_points_loc[inbox_mask]
            
        return instance_mask, obj_local_pts

                
    
        

class MMSampleCameraObjects(MMSampleLidarObjects):
    """
    Collect all objects for the given sample cameras (即某个标注sample下所有cam标注帧的物体类)
    """
    def __init__(self, mminfo):
        super().__init__(mminfo=mminfo)
        
        #==== init camera objects ====#
        self.cam_objs = {}
        for sensor in mminfo['cams'].keys():
            ### get camera extrinsics
            # cam <-> lidar
            cam2lidar_rotation = np.array(mminfo['cams'][sensor]['sensor2lidar_rotation'])
            cam2lidar_translation = np.array(mminfo['cams'][sensor]['sensor2lidar_translation'])
            T_cam2lidar = np.eye(4)
            T_cam2lidar[:3, :3] = cam2lidar_rotation
            T_cam2lidar[:3, 3] = cam2lidar_translation
            T_lidar2cam = np.linalg.inv(T_cam2lidar)
            
            # cam <-> ego
            T_cam2ego = construct_T_from_vector(
                mminfo['cams'][sensor]['sensor2ego_translation'], 
                mminfo['cams'][sensor]['sensor2ego_rotation'])
            T_ego2cam = np.linalg.inv(T_cam2ego)
            
            # cam <-> global
            T_cam2global = self.T_ego2global @ T_cam2ego
            T_global2cam = np.linalg.inv(T_cam2global)
            
            # cam <-> each obj's local coord
            T_local2cams = np.matmul(T_lidar2cam, self.T_local2lidars) # (N, 4, 4)
            T_cam2locals = np.linalg.inv(T_local2cams) # (N, 4, 4)

            ### get camera intrinsics
            img_file = mminfo['cams'][sensor]['data_path']
            # img = cv2.imread(img_file)
            # cam_h, cam_w, _  = img.shape
            cam_h, cam_w = mminfo['cams'][sensor]['img_h'], mminfo['cams'][sensor]['img_w']
            
            K = np.array(mminfo['cams'][sensor]['cam_intrinsic'])
            T_cam2img = np.zeros((3, 4))
            T_cam2img[:3, :3] = K
            T_lidar2img = T_cam2img @ T_lidar2cam
            
            ### filter objs in this camera
            incam_mask, img_corners = self.filter_boxes_incam(T_lidar2cam, T_lidar2img, cam_h, cam_w)
            _lidar_boxes = self.lidar_boxes[incam_mask]
            _lidar_corners = self.lidar_corners[incam_mask]
            img_corners = img_corners[incam_mask]
            _T_local2lidars = self.T_local2lidars[incam_mask]
            _T_lidar2locals = self.T_lidar2locals[incam_mask]
            T_local2cams = T_local2cams[incam_mask]
            T_cam2locals = T_cam2locals[incam_mask]
            _instance_tokens = [item for i, item in enumerate(self.lidar_instance_tokens) if incam_mask[i]]
            
            _tmp = np.concatenate([_lidar_corners.reshape(-1, 3), np.ones((_lidar_corners.shape[0]*8, 1))], axis=1)
            cam_corners = (_tmp @ T_lidar2cam.T)[:, :3].reshape(-1, 8, 3)
            
            ### get box2d
            proj_box2d_raw, proj_box2d = self.get_box2d(img_corners, cam_h, cam_w)
            
            self.cam_objs[sensor] = {
                'T_cam2lidar': T_cam2lidar,
                'T_lidar2cam': T_lidar2cam,
                'T_cam2ego': T_cam2ego,
                'T_ego2cam': T_ego2cam,
                'T_cam2global': T_cam2global,
                'T_global2cam': T_global2cam,
                'T_local2cams': T_local2cams,
                'T_cam2locals': T_cam2locals,
                
                'K': K,
                'T_cam2img': T_cam2img,
                'T_lidar2img': T_lidar2img,
                'cam_h': cam_h,
                'cam_w': cam_w,

                'incam_mask': incam_mask,
                'lidar_boxes': _lidar_boxes,
                'lidar_corners': _lidar_corners,
                'img_corners': img_corners,
                'T_local2lidars': _T_local2lidars,
                'T_lidar2locals': _T_lidar2locals,
                'instance_tokens': _instance_tokens,
                'cam_corners': cam_corners,
                
                'proj_box2d_raw': proj_box2d_raw,
                'proj_box2d': proj_box2d
            }
            
        self.cam_objs = EasyDict(self.cam_objs)

        
    def filter_boxes_incam(self, T_lidar2cam, T_lidar2img, cam_h, cam_w):
        n_objs = self.lidar_corners.shape[0]
        _lidar_corners = self.lidar_corners.reshape(-1, 3)
        
        _lidar_center = self.lidar_boxes[:, :3]
        _lidar_center = np.concatenate([_lidar_center, np.ones((_lidar_center.shape[0], 1))], axis=1)
        _cam_center = (_lidar_center @ T_lidar2cam.T)[:, :3]
        
        _lidar_corners = np.concatenate([_lidar_corners, np.ones((_lidar_corners.shape[0], 1))], axis=1)
        img_corners = _lidar_corners @ T_lidar2img.T              # (N*8, 3)
        corners_depth = img_corners[:, 2]
        img_corners = img_corners[:, :2] / img_corners[:, 2:]   # (N*8, 2)
        
        # ! corners which depth<0 need to filter out
        img_corners[(corners_depth <= 0) & (np.repeat(_cam_center[:, 0], 8) <= 0)] = -1
        img_corners[(corners_depth <= 0) & (np.repeat(_cam_center[:, 0], 8) > 0)] = 999999
        
        # depth_mask = (corners_depth > 0.1).reshape(n_objs, 8)
        # depth_mask = np.all(depth_mask, axis=1)
        
        mask = (img_corners[:, 0] >= 0) & (img_corners[:, 0] < cam_w) & \
               (img_corners[:, 1] >= 0) & (img_corners[:, 1] < cam_h)
        mask = mask.reshape(n_objs, 8)
        mask = np.any(mask, axis=1)
        
        # mask = mask & (_cam_center[:, 2] > 0)
        # mask[~depth_mask] = False
        
        return mask, img_corners.reshape(n_objs, 8, 2)
    
    
    def get_box2d(self, img_corners, cam_h, cam_w):
        proj_box2d_raw = np.zeros((len(img_corners), 4))
        for i in range(len(img_corners)):
            _corners = img_corners[i]
            proj_box2d_raw[i, 0] = np.min(_corners[:, 0])
            proj_box2d_raw[i, 1] = np.min(_corners[:, 1])
            proj_box2d_raw[i, 2] = np.max(_corners[:, 0])
            proj_box2d_raw[i, 3] = np.max(_corners[:, 1])
            
        proj_box2d = proj_box2d_raw.copy()
        proj_box2d[:, 0] = np.maximum(0, proj_box2d[:, 0])
        proj_box2d[:, 1] = np.maximum(0, proj_box2d[:, 1])
        proj_box2d[:, 2] = np.minimum(cam_w-1, proj_box2d[:, 2])
        proj_box2d[:, 3] = np.minimum(cam_h-1, proj_box2d[:, 3])
        proj_box2d = proj_box2d.astype(np.int32)
        
        return proj_box2d_raw, proj_box2d
    
    