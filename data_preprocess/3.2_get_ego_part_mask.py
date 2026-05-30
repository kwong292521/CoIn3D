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

from scipy.ndimage import label as label_connected_components
import segment_anything
from segment_anything import sam_model_registry, SamPredictor




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


def load_sam(device):
    sam_ckpt = '/data0/znkwong/data_process/raw_data_process/sam_vit_h_4b8939.pth'
    model_type = 'vit_h'

    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    return predictor



def get_egopart_mask_nusc(dataset, split, scene_token, devkit, mminfo, out_dir, predictor):
    print(f'getting egopart mask {dataset} => {split} => {scene_token}...')
    
    mminfo_token2idx = {_info['token']: idx for idx, _info in enumerate(mminfo)}
    sample_tokens = get_sample_tokens_from_scene(devkit, scene_token)
    
    # NOTE: we only need to segment the first CAM_BACK img, because the carback area is consistant in all frames
    sample_token = sample_tokens[0]
    _info = mminfo[mminfo_token2idx[sample_token]]
    
    img_file = _info['cams']['CAM_BACK']['data_path']
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w, _ = img.shape
    
    predictor.set_image(img)
    
    pts_2d = np.array([[w//2, h-20]])
    
    # construct point input
    input_points = torch.tensor(pts_2d, device=predictor.device)
    transformed_points = predictor.transform.apply_coords_torch(input_points, img.shape[:2]).unsqueeze(0)
    pts_label = torch.ones((1, transformed_points.shape[1]), dtype=torch.int32, device=predictor.device)

    mask, _, _ = predictor.predict_torch(
                point_coords=transformed_points,
                point_labels=pts_label,
                multimask_output=False,
            )
    mask = mask.cpu().numpy().squeeze(1).squeeze(0)
        
    # post process
    mask = postprocess_mask(mask, dilate_iter=1, kernel_size=5)
            
    # save results
    result_mask_file = os.path.join(out_dir, 'masks', 'ego_area_mask', 'CAM_BACK.png')
    os.makedirs(os.path.dirname(result_mask_file), exist_ok=True)
    cv2.imwrite(result_mask_file, mask.astype(np.uint8) * 255)

    


def get_egopart_mask_lyft_large_overlap_fleet(dataset, split, scene_token, devkit, mminfo, out_dir, predictor):
    print(f'getting egopart mask {dataset} => {split} => {scene_token}...')
    
    mminfo_token2idx = {_info['token']: idx for idx, _info in enumerate(mminfo)}
    sample_tokens = get_sample_tokens_from_scene(devkit, scene_token)
    
    sample_token = sample_tokens[0]
    _info = mminfo[mminfo_token2idx[sample_token]]
    
    for sensor in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        img_file = _info['cams'][sensor]['data_path']
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
    
        predictor.set_image(img)
                    
        if sensor == 'CAM_FRONT':
            pts_2d = np.array([[w//2, h-60],
                               [w//2, h-1]])
        elif sensor == 'CAM_BACK':
            pts_2d = np.array([[w//2, h-60],
                               [w//2, h-240],
                               [int(w/4*1), h-60],
                               [int(w/4*3), h-60]])
        elif sensor == 'CAM_FRONT_LEFT':
            pts_2d = np.array([[int(w/4*3), h-20]])
        elif sensor == 'CAM_FRONT_RIGHT':
            pts_2d = np.array([[int(w/4*1), h-20]])
        elif sensor == 'CAM_BACK_LEFT':
            pts_2d = np.array([[int(w/3*1), h-20],
                               [int(w/6*1), h-50]])
        elif sensor == 'CAM_BACK_RIGHT':
            pts_2d = np.array([[int(w/3*2), h-20],
                               [int(w/6*5), h-50]])
            
        # construct point input
        input_points = torch.tensor(pts_2d, device=predictor.device)
        transformed_points = predictor.transform.apply_coords_torch(input_points, img.shape[:2]).unsqueeze(0)
        pts_label = torch.ones((1, transformed_points.shape[1]), dtype=torch.int32, device=predictor.device)

        mask, _, _ = predictor.predict_torch(
                    point_coords=transformed_points,
                    point_labels=pts_label,
                    multimask_output=False,
                )
        mask = mask.cpu().numpy().squeeze(1).squeeze(0)
        
        # post process
        mask = postprocess_mask(mask, dilate_iter=1, kernel_size=5)

        # save results
        result_mask_file = os.path.join(out_dir, 'masks', 'ego_area_mask', f'v1_{sensor}.png')
        os.makedirs(os.path.dirname(result_mask_file), exist_ok=True)
        cv2.imwrite(result_mask_file, mask.astype(np.uint8) * 255)
        
    
def get_egopart_mask_lyft_small_overlap_fleet(dataset, split, scene_token, devkit, mminfo, out_dir, predictor):
    print(f'getting egopart mask {dataset} => {split} => {scene_token}...')
    
    mminfo_token2idx = {_info['token']: idx for idx, _info in enumerate(mminfo)}
    sample_tokens = get_sample_tokens_from_scene(devkit, scene_token)
    
    sample_token = sample_tokens[0]
    _info = mminfo[mminfo_token2idx[sample_token]]
    
    for sensor in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        img_file = _info['cams'][sensor]['data_path']
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
    
        predictor.set_image(img)
                    
        if sensor == 'CAM_FRONT':
            pts_2d = np.array([[w//2, h-60],
                               [w//2, h-1],
                               [10, h-60],
                               [10, h-1],
                               [w-10, h-60],
                               [w-10, h-1],
                               [w//2, h-120]])
        elif sensor == 'CAM_BACK':
            pts_2d = np.array([[w//2, h-60],
                            #    [w//2, h-240],
                            #    [int(w/4*1), h-60],
                            #    [int(w/4*3), h-60],
                               [w-10, h-200],
                               [w-10, h-60],
                               [10, h-200],
                               [10, h-60],
                               [w//2, h-250],
                               [w//2-100, h-250],
                               [w//2+100, h+250]])
        elif sensor == 'CAM_FRONT_LEFT':
            pts_2d = np.array([[int(w/4*3), h-20]])
        elif sensor == 'CAM_FRONT_RIGHT':
            pts_2d = np.array([[int(w/4*1), h-20]])
        elif sensor == 'CAM_BACK_LEFT':
            pts_2d = np.array([[int(w/3*1), h-20],
                               [int(w/6*1), h-50]])
        elif sensor == 'CAM_BACK_RIGHT':
            pts_2d = np.array([[int(w/3*2), h-20],
                               [int(w/6*5), h-50]])
            
        # construct point input
        input_points = torch.tensor(pts_2d, device=predictor.device)
        transformed_points = predictor.transform.apply_coords_torch(input_points, img.shape[:2]).unsqueeze(0)
        pts_label = torch.ones((1, transformed_points.shape[1]), dtype=torch.int32, device=predictor.device)

        mask, _, _ = predictor.predict_torch(
                    point_coords=transformed_points,
                    point_labels=pts_label,
                    multimask_output=False,
                )
        mask = mask.cpu().numpy().squeeze(1).squeeze(0)
        
        # post process
        mask = postprocess_mask(mask, dilate_iter=1, kernel_size=11)

        # save results
        result_mask_file = os.path.join(out_dir, 'masks', 'ego_area_mask', f'v2_{sensor}.png')
        os.makedirs(os.path.dirname(result_mask_file), exist_ok=True)
        cv2.imwrite(result_mask_file, mask.astype(np.uint8) * 255)


if __name__ == '__main__':
    device = 'cuda'
    sam_predictor = load_sam(device)
    
    # ! nusc
    dataset, split = 'nuscenes', 'train'
    devkit, mminfo, out_dir = load_dataset_devkit(dataset, split)
    _metadata = mminfo['metadata']
    mminfo = mminfo['infos']
    
    # nusc_scene_token = '2dc89e0f9c6c4dbab908d341fab020c6'
    nusc_sample_token = '0a59e631ee26413ca3c34e89bf60dc82'
    nusc_scene_token = devkit.get('sample', nusc_sample_token)['scene_token']
    
    get_egopart_mask_nusc(dataset, split, nusc_scene_token, devkit, mminfo, out_dir, sam_predictor)
    
    # # # ! lyft
    # dataset, split = 'lyft', 'train'
    # devkit, mminfo, out_dir = load_dataset_devkit(dataset, split)
    # _metadata = mminfo['metadata']
    # mminfo = mminfo['infos']
    
    # # hw = 1080 1920
    # lyft_scene_token = 'da4ed9e02f64c544f4f1f10c6738216dcb0e6b0d50952e158e5589854af9f100'
    # get_egopart_mask_lyft_large_overlap_fleet(dataset, split, lyft_scene_token, devkit, mminfo, out_dir, sam_predictor)
        
    # # hw = 1080 1224
    # lyft_sample_token = '0e96bcd36cf6d74ea0c8e27e5003e21d5aa83a335c4da02019c093a8947498e4'
    # lyft_scene_token = devkit.get('sample', lyft_sample_token)['scene_token']
    # get_egopart_mask_lyft_small_overlap_fleet(dataset, split, lyft_scene_token, devkit, mminfo, out_dir, sam_predictor)
    