#!/usr/bin/env python3
"""Simple encoder-only inference script for PointNet2-MSG.

This script loads the model (config + checkpoint), extracts the backbone
encoder, and runs a forward pass to produce point embeddings. It is
designed to be stand-alone inside the package and accept either an input
point cloud file or generate random points for testing.

Notes:
- The script still requires PyTorch, mmcv (or mmcv-full), and mmdet3d to be
  installed in the runtime environment. It does not require the whole repo
  to be present, but the config file should be the merged config named
  `config_merged.py` inside the package. If that file is not present the
  script will fall back to the original repo config path.
"""
import os
import numpy as np
import torch


def load_points(path):
    path = str(path)
    if path.endswith('.npy'):
        pts = np.load(path)
    elif path.endswith('.bin'):
        data = np.fromfile(path, dtype=np.float32)
        if data.size % 6 == 0:
            pts = data.reshape(-1, 6)
        elif data.size % 4 == 0:
            pts = data.reshape(-1, 4)[:, :3]
        elif data.size % 3 == 0:
            pts = data.reshape(-1, 3)
        else:
            raise ValueError('Unsupported .bin layout; please provide .npy or .bin with 3/4/6 floats per point')
    else:
        raise ValueError('Unsupported point cloud format; use .npy or .bin')
    return pts


def ensure_channels(pts, channels=6):
    """Ensure points array has shape (B, N, C). Accepts Nx3, Nx6 or BxNxC."""
    pts = np.asarray(pts)
    if pts.ndim == 2:
        # NxC -> 1xNxC
        pts = pts[np.newaxis, ...]
    elif pts.ndim == 3:
        pass
    else:
        raise ValueError('Unsupported input shape for points: ' + str(pts.shape))

    B, N, C = pts.shape
    if C == channels:
        return pts
    if C < channels:
        # pad with zeros for missing channels (e.g., xyz -> xyz000)
        pad = np.zeros((B, N, channels - C), dtype=pts.dtype)
        pts = np.concatenate([pts, pad], axis=2)
        return pts
    if C > channels:
        # truncate extra channels
        return pts[:, :, :channels]


# --- Inline configuration (edit these values to change behavior) ---
# By default, resolve paths relative to this script so the package is relocatable.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = {
    # prefer the merged config shipped with the package
    'cfg': os.path.join(BASE_DIR, 'config_merged.py'),
    # Fallback repo config path (only used when running inside repo)
    'fallback_cfg': "packaged_models/pointnet2_msg_pkg/pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class_20210514_144009-24477ab1.pth",
    # default checkpoint inside the package release
    'ckpt': os.path.join(BASE_DIR, 'model.pth'),
    # input options: set 'random' True to generate random points; otherwise set 'input' to file path
    'random': True,
    'input': None,
    'num_points': 8192,
    'channels': 6,
    'batch': 1,
    'device': 'cuda:0',
    'out': None,  # optional output .npy path to save embeddings (relative to BASE_DIR if provided)
}


def main():
    # Resolve checkpoint default if not provided
    if not CONFIG.get('ckpt') or not os.path.exists(CONFIG['ckpt']):
        raise FileNotFoundError('Checkpoint not found at: ' + str(CONFIG.get('ckpt')))

    # Try to import init_model from mmdet3d
    try:
        from mmdet3d.apis import init_model
    except Exception as e:
        raise ImportError('mmdet3d must be installed in the runtime environment: ' + str(e))

    # Choose config path: merged in package or fallback to repo config
    cfg_path = CONFIG.get('cfg')
    if not os.path.exists(cfg_path):
        cfg_path = CONFIG.get('fallback_cfg')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError('Config not found in package or repo. Checked: ' + str(CONFIG.get('cfg')) + ' and ' + str(CONFIG.get('fallback_cfg')))

    print(f"Initializing model with cfg={cfg_path} ckpt={CONFIG['ckpt']} on device={CONFIG['device']}")
    model = init_model(cfg_path, CONFIG['ckpt'], device=CONFIG['device'])
    model.eval()

    # prepare points
    if CONFIG.get('random'):
        pts = np.random.rand(CONFIG['batch'], CONFIG['num_points'], CONFIG['channels']).astype(np.float32)
    else:
        if not CONFIG.get('input'):
            raise ValueError('Either CONFIG["input"] must be set or CONFIG["random"] must be True')
        pts = load_points(CONFIG['input'])
        pts = ensure_channels(pts, channels=CONFIG['channels'])

    # move to torch and device
    device = torch.device(CONFIG['device'])
    pts_t = torch.from_numpy(pts).to(device)

    # run backbone/encoder only to extract embeddings
    backbone = getattr(model, 'backbone', None)
    if backbone is None:
        raise RuntimeError('Loaded model has no `backbone` attribute')

    with torch.no_grad():
        out = backbone(pts_t)

    # normalize result and print summary
    print('Encoder output type:', type(out))
    if isinstance(out, dict):
        for k, v in out.items():
            try:
                print(f'  {k}: type={type(v)}, shape={tuple(v.shape)}')
            except Exception:
                print(f'  {k}: (unprintable)')
        # choose first tensor-like entry to save
        first = next((v for v in out.values() if hasattr(v, 'cpu')), None)
        if first is not None and CONFIG.get('out'):
            emb = first.cpu().numpy()
            np.save(CONFIG['out'], emb)
            print('Saved embeddings to', CONFIG['out'])
    elif isinstance(out, (list, tuple)):
        for i, v in enumerate(out):
            print(f'  [{i}]: type={type(v)}, shape={getattr(v, "shape", None)}')
        if CONFIG.get('out'):
            emb = out[0].cpu().numpy()
            np.save(CONFIG['out'], emb)
            print('Saved embeddings to', CONFIG['out'])
    else:
        print('  scalar/unknown output:', out)
        if CONFIG.get('out') and hasattr(out, 'cpu'):
            np.save(CONFIG['out'], out.cpu().numpy())


if __name__ == '__main__':
    main()
