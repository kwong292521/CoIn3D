import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
zits_path = os.path.join(project_root, 'ZITS-PlusPlus')
if zits_path not in sys.path:
    sys.path.insert(0, zits_path)

import numpy as np
import torch
import cv2
import tqdm
import pickle
import math
import shutil
import argparse

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from common_utils import *
from __PATHS__ import *
from pathlib import Path

# zits import
import argparse
import torch.nn.functional as FF
import torch.nn.parallel
from torch.utils.data.dataloader import DataLoader

import utils
from base.parse_config import ConfigParser
from dataset.dataloader_ours import InpaintingDataset
from dnnlib.util import get_obj_by_name
from inpainting_metric import get_inpainting_metrics
from trainers.nms_temp import get_nms as get_np_nms
from trainers.pl_trainers import wf_inference_test


def create_zits_model(args, config):
    # build models architecture, then print to console
    structure_upsample = get_obj_by_name(config['structure_upsample_class'])()

    edgeline_tsr = get_obj_by_name(config['edgeline_tsr_class'])()
    grad_tsr = get_obj_by_name(config['grad_tsr_class'])()
    ftr = get_obj_by_name(config['g_class'])(config=config['g_args'])
    D = get_obj_by_name(config['d_class'])(config=config['d_args'])

    if 'PLTrainer' not in config.config or config['PLTrainer'] is None:
        config.config['PLTrainer'] = 'trainers.pl_trainers.FinetunePLTrainer'

    model = get_obj_by_name(config['PLTrainer'])(structure_upsample, edgeline_tsr, grad_tsr, ftr, D, config,
                                                 'ckpts/' + args.exp_name, use_ema=args.use_ema, dynamic_size=args.dynamic_size, test_only=True)

    if args.use_ema:
        model.reset_ema()

    if args.ckpt_resume:
        print("Loading checkpoint: {} ...".format(args.ckpt_resume))
        checkpoint = torch.load(args.ckpt_resume, map_location='cpu')
        utils.torch_init_model(model, checkpoint, key='state_dict')

    if hasattr(model, "wf"):
        model.wf.load_state_dict(torch.load(args.wf_ckpt, map_location='cpu')['model'])

    model.cuda()

    if args.use_ema:
        model.ftr_ema.eval()
    else:
        model.ftr.eval()
        
    return model


def run_one_scene(dataset_name, split, scene_token, devkit, mminfo, out_dir, args, config, model):
    print(f'running {dataset_name} => {split} => {scene_token} fg obj inpainting...')    
    
    mm_scene_objs = MMSceneObjects(devkit, mminfo, scene_token, dataset_name, use_sweeps=False)
    mminfo_token2idx = {_info['token']: idx for idx, _info in enumerate(mminfo)}
    sample_tokens = get_sample_tokens_from_scene(devkit, scene_token)

    # collect input
    print('[0] collecting input ... ')
    img_file_list = []
    egopart_img_file_list = []
    key_mask_file_list = []
    sample_token_list = []
    sensor_list = []
    cam_names = None
    for frame_idx, sample_token in enumerate(sample_tokens):
        _info = mminfo[mminfo_token2idx[sample_token]]
        if cam_names is None: cam_names = list(_info['cams'].keys())
        for sensor in _info['cams'].keys():    
            img_file = _info['cams'][sensor]['data_path']
            egopart_img_file = os.path.join(out_dir, 'inpainted', 'img', sensor, f'{sample_token}.png')
            key_mask_file = os.path.join(out_dir, 'masks', 'key_mask2d', sensor, f'{sample_token}.png')
            
            img_file_list.append(img_file)
            egopart_img_file_list.append(egopart_img_file)
            key_mask_file_list.append(key_mask_file)
            sample_token_list.append(sample_token)
            sensor_list.append(sensor)
            
            
    # create dataset
    print('[1] creating dataset ... ')
    dataset = InpaintingDataset(test_size=args.test_size, 
                                use_gradient=config['g_args']['use_gradient'], 
                                img_file_list=img_file_list,
                                egopart_img_file_list=egopart_img_file_list,
                                key_mask_file_list=key_mask_file_list,
                                sample_token_list=sample_token_list, 
                                sensor_list=sensor_list,
                                )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # running inpainting
    print('[2] running inpainting ...')
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            batch['size_ratio'] = -1
            batch['H'] = -1
            for k in batch:
                if type(batch[k]) is torch.Tensor:
                    batch[k] = batch[k].cuda()

            # load line
            batch['line_256'] = wf_inference_test(model.wf, batch['img_512'], h=256, w=256, masks=batch['mask_512'],
                                                  valid_th=0.85, mask_th=0.85, obj_remove=args.obj_removal)
            imgh = batch['imgh'][0].item()
            imgw = batch['imgw'][0].item()

            # inapint prior
            edge_pred, line_pred = model.edgeline_tsr.forward(batch['img_256'], batch['line_256'], masks=batch['mask_256'])
            line_pred = batch['line_256'] * (1 - batch['mask_256']) + line_pred * batch['mask_256']

            edge_pred = edge_pred.detach()
            line_pred = line_pred.detach()

            current_size = 256
            if current_size != min(imgh, imgw):
                while current_size * 2 <= max(imgh, imgw):
                    # nms for HR
                    line_pred = model.structure_upsample(line_pred)[0]
                    edge_pred_nms = get_np_nms(edge_pred, binary_threshold=args.binary_threshold)
                    edge_pred_nms = model.structure_upsample(edge_pred_nms)[0]
                    edge_pred_nms = torch.sigmoid((edge_pred_nms + 2) * 2)
                    line_pred = torch.sigmoid((line_pred + 2) * 2)
                    current_size *= 2

                edge_pred_nms = FF.interpolate(edge_pred_nms, size=(imgh, imgw), mode='bilinear', align_corners=False)
                edge_pred = FF.interpolate(edge_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)
                edge_pred[edge_pred >= 0.25] = edge_pred_nms[edge_pred >= 0.25]
                line_pred = FF.interpolate(line_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)
            else:
                edge_pred = FF.interpolate(edge_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)
                line_pred = FF.interpolate(line_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)

                if config['g_args']['use_gradient'] is True:
                    gradientx, gradienty = model.grad_tsr.forward(batch['img_256'], batch['gradientx'], batch['gradienty'], masks=batch['mask_256'])
                    gradientx = batch['gradientx'] * (1 - batch['mask_256']) + gradientx * batch['mask_256']
                    gradienty = batch['gradienty'] * (1 - batch['mask_256']) + gradienty * batch['mask_256']
                    gradientx = FF.interpolate(gradientx, size=(imgh, imgw), mode='bilinear')
                    gradientx = gradientx * batch['mask'] + batch['gradientx_hr'] * (1 - batch['mask'])

                    gradienty = FF.interpolate(gradienty, size=(imgh, imgw), mode='bilinear')
                    gradienty = gradienty * batch['mask'] + batch['gradienty_hr'] * (1 - batch['mask'])

                    batch['gradientx'] = gradientx.detach()
                    batch['gradienty'] = gradienty.detach()

            batch['edge'] = edge_pred.detach()
            batch['line'] = line_pred.detach()

            if args.use_ema:
                gen_ema_img, _ = model.run_G_ema(batch)
            else:
                gen_ema_img, _ = model.run_G(batch)
            gen_ema_img = torch.clamp(gen_ema_img, -1, 1)
            gen_ema_img = (gen_ema_img + 1) / 2
            gen_ema_img = gen_ema_img * 255.0
            gen_ema_img = gen_ema_img.permute(0, 2, 3, 1).int().cpu().numpy()
            
            
            # post-process to save inpainted image
            inpainted_img = gen_ema_img[0][..., ::-1].astype(np.uint8)        # (512, 512, 3) @ BGR
            inpainted_img = cv2.resize(inpainted_img, (int(batch['img_hw_raw'][1]), int(batch['img_hw_raw'][0])))
                            
            # save inpainted results
            sample_token, sensor = batch['sample_token'][0], batch['sensor'][0]
            saved_inpainted_file = os.path.join(out_dir, 'inpainted', 'img', sensor, f'{sample_token}.png')
            
            key_mask = batch['key_mask'][0].cpu().numpy()
            
            saved_inpainted = inpainted_img.copy()
            _m = (key_mask[..., 2] != 0)
            saved_inpainted[~_m] = 0
            
            os.makedirs(os.path.dirname(saved_inpainted_file), exist_ok=True)
            cv2.imwrite(saved_inpainted_file, saved_inpainted)
    
    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--config', type=str, help='config file path', 
                      default="./ZITS-PlusPlus/configs/config_zitspp_finetune.yml")
    args.add_argument('--exp_name', type=str, help='method name', 
                      default='model_512')
    args.add_argument('--dynamic_size', action='store_true', help='Whether to finetune in dynamic size?')
    args.add_argument('--use_ema', help='Whether to use ema?', default=True)
    args.add_argument('--ckpt_resume', type=str, help='PL path to restore', 
                      default='./ZITS-PlusPlus/ckpts/model_512/models/last.ckpt')
    args.add_argument('--wf_ckpt', type=str, help='Line detector weights', 
                      default='./ZITS-PlusPlus/ckpts/best_lsm_hawp.pth')
    args.add_argument('--test_size', type=int, help='Test image size', 
                      default=512)
    args.add_argument('--binary_threshold', type=int, default=50, help='binary_threshold for E-NMS (from 0 to 255)')
    args.add_argument('--obj_removal', help='obj_removal', default=True)
    
    # NOTE: main parameters
    args.add_argument('--dataset', choices=['nuscenes', 'lyft', 'waymo'], default='nuscenes', help="Dataset to construct.")
    args.add_argument('--split', choices=['train', 'val'], default='train', help="Dataset to construct.")

    args = args.parse_args()
    args.resume = None
    config = ConfigParser.from_args(args, mkdir=False)
    SEED = 123456
    torch.manual_seed(SEED)
    num_gpus = torch.cuda.device_count()
    args.num_gpus = num_gpus
    
    model = create_zits_model(args, config)
    
    dataset = args.dataset
    split = args.split
    
    devkit, mminfo, out_dir = load_dataset_devkit(dataset, split)
    _metadata = mminfo['metadata']
    mminfo = mminfo['infos']
    
    all_scene_tokens = get_scenes_from_mminfo(mminfo)
    
    n_scenes = len(all_scene_tokens)
    
    pb = tqdm.tqdm(total=len(all_scene_tokens), leave=True, desc=f'inpainting foreground by zits++ : {dataset} => {split}')
    for i, scene_token in enumerate(all_scene_tokens):
        
        run_one_scene(dataset, split, scene_token, devkit, mminfo, out_dir, args, config, model)
        
        torch.cuda.empty_cache()
        pb.update()
    pb.close()
    
    