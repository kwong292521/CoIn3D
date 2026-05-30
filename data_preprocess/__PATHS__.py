import os

# ! nuscenes(trainval) paths
NUSC_ROOT_DIR = 'data/nuscenes'
NUSC_MMINFO_TRAIN = os.path.join(NUSC_ROOT_DIR, 'bevdetv3-nuscenes_infos_train_sync.pkl')
NUSC_MMINFO_VAL = os.path.join(NUSC_ROOT_DIR, 'bevdetv3-nuscenes_infos_val_sync.pkl')
# NUSC_DEVKIT = './devkits/devkit_nusc.pkl'

# ! Lyft(trainval) paths
LYFT_ROOT_DIR = 'data/lyft'
LYFT_MMINFO_TRAIN = os.path.join(LYFT_ROOT_DIR, 'lyft_infos_train_bevdet_format_sync.pkl')
LYFT_MMINFO_VAL = os.path.join(LYFT_ROOT_DIR, 'lyft_infos_val_bevdet_format_sync.pkl')
# LYFT_DEVKIT = './devkits/devkit_lyft.pkl'

# ! Waymo(trainval) paths
WAYMO_ROOT_DIR = 'data/waymo'
WAYMO_MMINFO_TRAIN = os.path.join(WAYMO_ROOT_DIR, 'waymo_infos_train_bevdet_format.pkl')
WAYMO_MMINFO_VAL = os.path.join(WAYMO_ROOT_DIR, 'waymo_infos_val_bevdet_format.pkl')
# WAYMO_DEVKIT = './devkits/devkit_waymo.pkl'

# ! outputs paths
NUSC_OUT_DIR = os.path.join(NUSC_ROOT_DIR, 'meta_data')
LYFT_OUT_DIR = os.path.join(LYFT_ROOT_DIR, 'meta_data')
WAYMO_OUT_DIR = os.path.join(WAYMO_ROOT_DIR, 'meta_data')


