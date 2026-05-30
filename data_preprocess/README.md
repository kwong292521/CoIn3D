# Data Preparation

We transform data format of Waymo and Lyft to Nuscenes. All image in the three dataset are downsampled to width=800. For Waymo, we downsample the temporal resolution from 10Hz to 2.5Hz for training (validation split keep 10Hz).

## Quick Start
We have provided the processed data for Nuscenes, Lyft, and Waymo in [ModelScope](https://modelscope.cn/datasets/znkwong/CoIn3D_Data). Download to `CoIn3D/BEVDet/data`.

### Data-structure
```bash
CoIn3D/BEVDet/data
├── bevdetv3-mix_infos_train_general_3cls_sync+raw.pkl                              
├── waymo_ghost_depth.png                                                           
├── waymo_ghost_img.jpg                                                             
├── nuscenes
│   ├── bevdetv3-nuscenes_infos_train_general_3cls_sync+raw.pkl                     
│   ├── bevdetv3-nuscenes_infos_val_general_3cls_sync+raw.pkl                       
│   ├── bevdetv3-nuscenes_infos_val_sync+raw.pkl                                    
│   ├── maps                                                                        
│   ├── samples                                                                     
│   │   └── [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT, LIDAR_TOP]
│   ├── v1.0-trainval                                                               
│   └── meta_data 
│       ├── meshes                                                                  
│       ├── depths                                                                  
│       │   ├── dense_depth_SPNorm                                      
│       │   └── lidar_depths
│       ├── gaussians                                                               
│       ├── inpainted                                                              
│       │   ├── blind_area_pts
│       │   ├── depth
│       │   └── img
│       ├── masks                                                                   
│       │   ├── key_mask2d
│       │   └── obj_render_mask
│       └── objs_model                                                             
├── lyft
│   ├── lyft_infos_train_bevdet_format_general_3cls_sync+raw.pkl
│   ├── lyft_infos_val_bevdet_format_general_3cls_sync+raw.pkl
│   ├── lyft_infos_train_bevdet_format_sync+raw.pkl
│   ├── lyft_infos_val_bevdet_format_sync+raw.pkl
│   ├── maps
│   ├── samples
│   │   └── [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT, LIDAR_TOP]
│   ├── v1.01-trainval
│   └── meta_data
│       ├── meshes                                                                 
│       ├── depths
│       │   ├── dense_depth_SPNorm
│       │   └── lidar_depths
│       ├── gaussians
│       ├── inpainted
│       │   ├── blind_area_pts
│       │   ├── depth
│       │   └── img
│       ├── masks
│       │   ├── key_mask2d
│       │   └── obj_render_mask
│       └── objs_model
├── waymo
│   ├── waymo_infos_train_bevdet_format_general_3cls_ds@4.pkl
│   ├── waymo_infos_val_bevdet_format_general_3cls.pkl
│   ├── maps
│   ├── samples
│   │   └── [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_SIDE_LEFT, CAM_SIDE_RIGHT, LIDAR_TOP]
│   ├── v1.4.0-trainval
│   └── meta_data
│       ├── meshes                                                                 
│       ├── depths
│       │   ├── dense_depth_SPNorm
│       │   └── lidar_depths
│       ├── gaussians
│       ├── inpainted
│       │   ├── blind_area_pts
│       │   ├── depth
│       │   └── img
│       ├── masks
│       │   ├── key_mask2d
│       │   └── obj_render_mask
│       ├── objs_model
│       └── zoomout_rgbd_0.7                                                        
│           ├── depth
│           └── img
CoIn3D/BEVDet/ckpts
├── mix.pth
├── nuscenes.pth
├── lyft.pth
├── waymo.pth
```

### Class names mapping
```bash
dataset     unified classes in nuscenes format / paper          raw classes
nuscenes    car / car                                           car, truck, trailer, bus, construction_vehicle
            pedestrian / pedestrian                             pedestrian
            motorcycle / two-wheel                              bicycle, motorcycle
lyft        car / car                                           car, truck, bus, emergency_vehicle, other_vehicle
            pedestrian / pedestrian                             pedestrian
            motorcycle / two-wheel                              bicycle, motorcycle
waymo       car / car                                           VEHICLE
            pedestrian / pedestrian                             PEDESTRIAN
            motorcycle / two-wheel                              CYCLIST
```

## Build meta-data from scratch

We provide the resorted codes with seperated implementation for each step in each python file named `1_....py, 2_....py, 3....py, ...` in order. 

[TODO] Release the codes of coverting lyft and waymo to nuscenes format

### Key Dependencies
```bash
nuscenes-devkit(../BEVDet/third_party/nuscenes-devkit)

open3d
trimesh
vdbfusion
pymeshfix
pytorch3d
segment_anything (Use to extract ego-part mask)

requirements in ZITS-PlusPlus (Optional if use ZITS-PlusPlus for RGB inpainting)
```

### Ego-Centric Gaussians Construction

**Step 1: Mesh Reconstruction**
```bash
python 1.1_construct_meshes.py
python 1.2_fix_mesh_to_watertight.py
```

**Step 2: Depth Rendering**
```bash
python 2.1_render_mesh_to_depth.py
python 2.2_get_dense_depth.py
```

**Step 3: Assets Texulization**
```bash
python 3.1_construct_instance_model.py
python 3.2_get_ego_part_mask.py
python 3.3_get_key_masks.py 
python 3.4_inpainted_ego_and_blind_area.py 

# Optional for unseen-background in ego-view
(Optional) python 3.5_inpainted_objs_zits.py 
(Optional) python 3.6_inpainted_obj_depth.py
```

**Step 4: Gaussians Construction**
```bash
python 4_construct_gaussians.py
```

**Step 5: Construct some auxilary data for training**
```bash
python 5.1_get_lidar_depths.py (to replace loading lidar and render depth at each iteration for BEVDepth)
python 5.2_pregen_waymo_zoomout_rgbd.py (to pre-generate NVS-Images at raw extrinsics for Waymo focal augmentation)
```

[TODO] Release the multi-process codes for meta-data construction

