from src.datasets.sequence_dataset import SequenceDataset
from src.utils.train_utils import ConfigManager

config = ConfigManager().load_config('configs/train_config.yaml')


data_config = config['data']
seq_config = data_config['sequence_config']


test_dataset = SequenceDataset(
            data_root=data_config['data_root'],
            bbox_root=data_config.get('bbox_root'),
            split=data_config['val_split'],
            max_points=seq_config['max_points'],
            normalize_coords=seq_config['normalize_coords'],
            grid_size=seq_config['grid_size'],
            num_frames=seq_config['num_frames'],
            frame_interval=seq_config['frame_interval'],
            center_sampling_strategy=seq_config['center_sampling_strategy'],
            ensure_full_sequence=seq_config['ensure_full_sequence'],
            seed=seq_config['seed']
        )



import os
import numpy as np
import torch


# By default, resolve paths relative to this script so the package is relocatable.
BASE_DIR = 'src/models/point_models/pointnet2_msg_pkg'
CONFIG = {
    'cfg': os.path.join(BASE_DIR, 'config_merged.py'),
    'fallback_cfg': "src/models/point_models/pointnet2_msg_pkg/config_merged.py",
    'ckpt': os.path.join(BASE_DIR, 'model.pth'),
    'device': 'cuda:0',
}



# Resolve checkpoint default if not provided
if not CONFIG.get('ckpt') or not os.path.exists(CONFIG['ckpt']):
    raise FileNotFoundError('Checkpoint not found at: ' + str(CONFIG.get('ckpt')))

# Try to import init_model from mmdet3d
try:
    from mmdet3d.apis import init_model
except Exception as e:
    raise ImportError('mmdet3d must be installed in the runtime environment: ' + str(e))

# # Choose config path: merged in package or fallback to repo config
cfg_path = CONFIG.get('cfg')

print(f"Initializing model with cfg={cfg_path} ckpt={CONFIG['ckpt']} on device={CONFIG['device']}")
model = init_model(cfg_path, CONFIG['ckpt'], device=CONFIG['device'])
model.eval()


sample=test_dataset[0]
frame_idx = 0

# Extract coord and color tensors for the chosen frame
coords_list = sample.get('sequence_coord', [])
colors_list = sample.get('sequence_color', [])


coord = coords_list[frame_idx]
color = colors_list[frame_idx]


pc6 = torch.cat([coord, color], dim=1)
pc6 = pc6.to(CONFIG['device'])

    
points_list = [pc6]
points_stack = torch.stack(points_list)  # [B, N, C]

with torch.no_grad():
    out = model.extract_feat(points_stack)
    print('model extracted features')




