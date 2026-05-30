INPUT_SIZE = (320, 704)
MAX_PTS = 4500000

_base_ = ['../../_base_/datasets/nusc_format_multi_datasets/nus-3d.py', '../../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'pedestrian', 'motorcycle'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': INPUT_SIZE,
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
    
    'use_pmd': False,
    'p_pmd': 0.5,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}

# Prior Map
prior_config = {
    'all_maps': ['hori_fov', 'vert_fov', 'ground_depth', 'plucker_ray', 'plucker_moment', 'focal', 'inv_focal', 'gd_pointmap', 'gd_grad'],
    'map2use': ['inv_focal', 'ground_depth', 'gd_grad', 'plucker_ray', 'plucker_moment'],
    'feat2modulate': {
        'after_backbone': True,
        'after_neck': True,
    },
    'feat2cat': {
        'img_input': True,
        'after_backbone': True,
        'after_neck': True,
    },
    'prior_channel': 0,
    'depth_norm_factor': 25.,
    'focal_norm_factor': 500.,
    'square_focal': True,
    'plucker_normed': False,
    'plucker_fb_equal': False,       
    'plucker_fb_flip': True,
    'plucker_moment_origin': 'cam_in_ego',
    'use_unpert_ego': False,
    #! feature modulate
    'feat_modulate':{
        'type': 'PM_InvFMul_GDGGPRAdd'
    }, 
}
prior_config['map_info'] = {}
sidx = 0
for _map in prior_config['map2use']:
    if _map in ['plucker_ray', 'plucker_moment']:
        prior_config['prior_channel'] += 3
        prior_config['map_info'][_map] = (sidx, 3)
        sidx += 3
    elif _map in ['gd_pointmap']:
        prior_config['prior_channel'] += 2
        prior_config['map_info'][_map] = (sidx, 2)
        sidx += 2
    else:
        prior_config['prior_channel'] += 1
        prior_config['map_info'][_map] = (sidx, 1)
        sidx += 1


voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80         

multi_adj_frame_id_cfg = (1, 1+1, 1)        

model = dict(
    type='BEVDepth4D',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    prior_cfg=prior_config,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        # in_channels=[1024, 2048],
        in_channels=[1024+prior_config['prior_channel'], 2048+prior_config['prior_channel']] \
            if prior_config['feat2cat']['after_backbone'] else [1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        # in_channels=512,
        in_channels=512+prior_config['prior_channel'] if prior_config['feat2cat']['after_neck'] else 512,
        out_channels=numC_Trans,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96, block_cam_aware=True),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[2,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        # tasks=[
        #     dict(num_class=10, class_names=['car', 'truck',
        #                                     'construction_vehicle',
        #                                     'bus', 'trailer',
        #                                     'barrier',
        #                                     'motorcycle', 'bicycle',
        #                                     'pedestrian', 'traffic_cone']),
        # ],    #! raw nuscenes cls
        tasks=[
            dict(num_class=3, class_names=['car', 'pedestrian', 'motorcycle']),
        ],      #! general 3cls
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), 
            # vel=(2, 2)
            ),     
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            # code_size=9
            code_size=7
            ),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=6.),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=1.5),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            # code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=500,

            # Scale-NMS 
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    )
)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
nuscenes_data_root = 'data/nuscenes/'
lyft_data_root = 'data/lyft/'
waymo_data_root = 'data/waymo/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)



train_pipeline = [
    dict(
        type='PrepareNVSMetaDataMix',
        nuscenes_meta_data_root=nuscenes_data_root + 'meta_data',
        lyft_meta_data_root=lyft_data_root + 'meta_data',
        waymo_meta_data_root=waymo_data_root + 'meta_data',
        sequential=True,
        # final size for training
        input_size=data_config['input_size'],
        # cam names
        cam_names=data_config['cams'],
        # max gaussians to pad
        max_pts=MAX_PTS,
        # global flip augmentation
        global_flip_aug=False,
        plain_flip_aug=True,
        # whether use raw augmentation policy (scale/rotation/flip) when use raw input
        use_raw_aug=True,
        # whether to use raw images as input
        p_raw=0.5,
        # nvs camera generation policy
        aug_ego2global=[2, 2, 0],      # [rx, ry, rz] degrees, rx/y/z -> roll/pitch/yaw
        aug_cam2ego=[0.2, 0.2, ('range', 1.5, 2.2), 0, 0, 20],       # [tx, ty, tz, rx, ry, rz] 
        # aug_K=['ratio', 0.7, 1.4],   
        aug_K=['range', 385, 770],   
        aug_K_on_raw=['ratio', 0.94, 1.4],
        # depth supervision
        use_lidar_depth=True,
        fg_depth_only=True,
        # !NEW sync parameters
        nvs_wo_cam_sync=True,
        sync_adj_aug=True,
        ),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                                'gt_depth', 'nvs_meta'])
]


# raw test pipeline
test_pipeline = [
    # dict(type='PrepareImageInputs', data_config=data_config, sequential=True, K_adapt=True),        # raw load img
    dict(           # our load img
        type='PrepareNVSMetaData',
        meta_data_root=data_root + 'meta_data',
        sequential=True,
        # final size for training
        input_size=data_config['input_size'],
        # cam names
        cam_names=data_config['cams'],
        test_mode=True
    ),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug',             
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=5,
    #     file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',   
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=[
                # 'points', 
                'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    nvs_mode=True,
    pipeline=test_pipeline,
    # ann_file=data_root + 'bevdetv3-nuscenes_infos_val_general_3cls.pkl')
    ann_file=data_root + 'bevdetv3-nuscenes_infos_val_general_3cls_sync+raw.pkl')


# ! Plain dataset(nocbgs) for debug
data = dict(
    samples_per_gpu=10,         
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file='/data1/znkwong/Cross-Cam-Config-Generalization/BEVDet/data/ds_size/bevdetv3-mix_infos_train_general_3cls_sync+raw.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        nvs_mode=True,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)
# ! End of Plain dataset

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[20,])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_epoch=2,
    ),
    dict(
        type='NovelViewSynthesisHook',
        max_pts=MAX_PTS,
        use_placeholder=True,
        use_pmd=data_config['use_pmd'],
        p_pmd=data_config['p_pmd'],
        # priority='VERY_HIGH',
        priority=0,
    ),
]

workflow = [('train', 1)]

# fp16 = dict(loss_scale='dynamic')
