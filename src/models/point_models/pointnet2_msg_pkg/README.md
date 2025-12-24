# PointNet2 MSG packaged model

This folder contains a small scaffold to create a portable package for the
PointNet2-MSG segmentation model.

Files
- `config.py` - copy of the original repo config (may still reference `_base_` files)
- `merge_cfg.py` - helper that uses `mmcv.Config` to dump a merged, self-contained config
- `inference.py` - a minimal inference wrapper that loads a merged config and checkpoint and runs inference on a `.npy` / `.bin` point cloud
- `package_create.sh` - script that collects files, copies the checkpoint (if present at the expected path), produces `config_merged.py`, and bundles a `pointnet2_msg_pkg.tar.gz`
- `requirements.txt` - minimal list of runtime dependencies (edit with `pip freeze` for exact versions)

How to build the distributable package (run inside this repo):

1. Ensure you have Python 3 and the local repo. From the repo root run:

```bash
bash packaged_models/pointnet2_msg_pkg/package_create.sh
```

2. The script will attempt to copy the checkpoint from
   `checkpoints/pointnet2_msg_xyz-only_16x2_cosine_250e_scannet_seg-3d-20class_20210514_143838-b4a3cf89.pth`.
   If your checkpoint is in a different location, copy it into
   `packaged_models/pointnet2_msg_release/model.pth` before or after running the script.

3. The resulting `pointnet2_msg_pkg.tar.gz` contains:
   - `model.pth` (checkpoint)
   - `config_merged.py` (self-contained config)
   - `inference.py` (run inference)
   - `requirements.txt`

How to run on the target machine

1. Extract the archive and create a venv:

```bash
tar -xzvf pointnet2_msg_pkg.tar.gz
cd pointnet2_msg_release
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run inference (example with a saved NumPy point cloud `sample_pts.npy`):

```bash
python inference.py --cfg config_merged.py --ckpt model.pth --input sample_pts.npy --device cpu
```

Notes and caveats
- Some installations (mmcv-full, matching CUDA/cuDNN and PyTorch) are sensitive to exact versions. For easiest reproduction, replace `requirements.txt` with a `pip freeze` output from the source environment.
- If `inference.py` fails with model forward signature issues, open the model with `init_model` in an interactive session and inspect `model.forward` / `model.simple_test` for the expected input format. The wrapper assumes `inference_detector(model, points)` works.
- If you need a dependency-free runtime, consider exporting the model to TorchScript/ONNX — this requires additional work to make the forward traceable.
