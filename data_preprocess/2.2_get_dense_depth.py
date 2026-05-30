import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  
import numpy as np
import cv2
import tqdm
import pickle
import math
import argparse

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from common_utils import *

from __PATHS__ import *

from G2_MonoDepth.src.networks import UNet
from SPNet.src.networks import V2Net
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset

# turn fast mode on
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# DC_MODEL = 'G2'
DC_MODEL = 'SPNorm'
# DOWNSAMPLE_RATIO = 1.5
# DOWNSAMPLE_RATIO = 2
DOWNSAMPLE_RATIO = 1    # ! use ds_size

def load_G2():
    model_dir = '/data1/znkwong/Cam-params-free-Project/view-transformation-module/nuscenes_demo/G2_MonoDepth/epoch_100.pth'
    G2 = UNet(rezero=True).cuda().eval()
    G2.load_state_dict(torch.load(model_dir)['network'])
    G2 = G2.cuda()
    G2.eval()

    return G2


def load_SPNorm_Large():
    model_dir = '/data1/znkwong/Cam-params-free-Project/view-transformation-module/mm_all_demo/SPNet/Large_300.pth'
    dims = [192, 384, 768, 1536]  # dimensions
    depths = [3, 3, 27, 3]  # block number
    dp_rate = 0.2
    norm_type = 'CNX'
    
    SPNorm = V2Net(dims, depths, dp_rate, norm_type).cuda().eval()
    SPNorm.load_state_dict(torch.load(model_dir)['network'])
    
    return SPNorm


class DCDataset(Dataset):
    def __init__(self, mminfo, mminfo_token2idx, sample_tokens, out_dir, downsample_ratio=1.):
        self.all_sample_tokens = []
        self.all_sensors = []
        self.all_sparse_depth_files = []
        self.all_img_files = []
        self.all_result_files = []
        
        for sample_token in sample_tokens:
            _info = mminfo[mminfo_token2idx[sample_token]]
            for sensor in _info['cams'].keys():
                sparse_depth_file = os.path.join(out_dir, 'depths', 'mesh_depth', sensor, f'{sample_token}.png')
                img_file = _info['cams'][sensor]['data_path']
                result_file = os.path.join(out_dir, 'depths', f'dense_depth_{DC_MODEL}', sensor, f'{sample_token}.png')
                os.makedirs(os.path.dirname(result_file), exist_ok=True)
                
                self.all_sample_tokens.append(sample_token)
                self.all_sensors.append(sensor)
                self.all_sparse_depth_files.append(sparse_depth_file)
                self.all_img_files.append(img_file)
                self.all_result_files.append(result_file)        
                
        self.downsample_ratio = downsample_ratio


    def __len__(self):
        return len(self.all_sample_tokens)


    def __getitem__(self, idx):
        # load data
        img = cv2.imread(self.all_img_files[idx])
        sparse_depth = read_depth_map(self.all_sparse_depth_files[idx])
        raw_sparse_depth = sparse_depth.copy()                
                
        # resize to 32×(2n)
        raw_h, raw_w = img.shape[:2]
        _scale = 32*2
        new_h, new_w = int(raw_h//self.downsample_ratio//_scale*_scale), int(raw_w//self.downsample_ratio//_scale*_scale)
        img = cv2.resize(img, (new_w, new_h))
        sparse_depth = cv2.resize(sparse_depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # pre-process
        sparse_depth = (torch.tensor(sparse_depth / 255)).unsqueeze(0).float()#.cuda()
        img = (torch.tensor(img / 255)).permute(2, 0, 1).float()#.cuda()
        hole = torch.ones_like(sparse_depth)
        hole[sparse_depth == 0] = 0.
        
        return img, sparse_depth, hole, raw_sparse_depth, raw_h, raw_w, self.all_result_files[idx]


def run_one_scene(dataset, split, scene_token, devkit, mminfo, out_dir, model):
    mminfo_token2idx = {_info['token']: idx for idx, _info in enumerate(mminfo)}
    sample_tokens = get_sample_tokens_from_scene(devkit, scene_token)
    
    dc_dataset = DCDataset(mminfo, mminfo_token2idx, sample_tokens, out_dir, DOWNSAMPLE_RATIO)
    
    all_done = [os.path.exists(_file) for _file in dc_dataset.all_result_files]
    if np.all(all_done):
        return
    
    dc_dataloader = DataLoader(
        dc_dataset,
        batch_size=1,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )
    
    # ! begin to generate dense depth by depth completion model
    pb = tqdm.tqdm(total=len(dc_dataloader), leave=True, desc=f'generating {DC_MODEL} dense depths {dataset} => {split} => {scene_token}...')
    for i, batch_data in enumerate(dc_dataloader):
        img, sparse_depth, hole, raw_sparse_depth, raw_h, raw_w, result_file = batch_data
        img = img.cuda()
        sparse_depth = sparse_depth.cuda()
        hole = hole.cuda()
        
        raw_sparse_depth = raw_sparse_depth.squeeze(0).numpy()
        raw_h, raw_w = int(raw_h), int(raw_w)
        result_file = result_file[0]

        # forward
        dense_depth = model(img, sparse_depth, hole)
        dense_depth = dense_depth.squeeze().detach().cpu().numpy()
        dense_depth = np.clip(dense_depth, 0, 1)
        dense_depth *= 255

        # resize back
        dense_depth = cv2.resize(dense_depth, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
        
        # keep the value which sparse depth hold invariant
        mask = raw_sparse_depth!=0
        dense_depth[mask] = raw_sparse_depth[mask]
        
        save_depth_map(result_file, dense_depth)
            
        pb.update()
    pb.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta Data Construction Pipeline.")
    parser.add_argument('--dataset', choices=['nuscenes', 'lyft', 'waymo'], default='nuscenes', help="Dataset to construct.")
    parser.add_argument('--split', choices=['train', 'val'], default='train', help="Dataset to construct.")
    args = parser.parse_args()
    
    dataset = args.dataset
    split = args.split
    device = 'cuda'
    
    devkit, mminfo, out_dir = load_dataset_devkit(dataset, split)
    _metadata = mminfo['metadata']
    mminfo = mminfo['infos']
    
    if DC_MODEL == 'G2':
        model = load_G2()
    elif DC_MODEL == 'SPNorm':
        model = load_SPNorm_Large()
    
    all_scene_tokens = get_scenes_from_mminfo(mminfo)
    
    n_scenes = len(all_scene_tokens)
    
    pb = tqdm.tqdm(total=len(all_scene_tokens), leave=True, desc=f'generating dense depths : {dataset} => {split}')
    for i, scene_token in enumerate(all_scene_tokens):
        run_one_scene(dataset, split, scene_token, devkit, mminfo, out_dir, model)
        pb.update()
    pb.close()
    
