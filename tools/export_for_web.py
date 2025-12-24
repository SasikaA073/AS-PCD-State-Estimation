import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.datasets.unified_dataset import UnifiedDatasetForOurs
from src.tracking.kalman_filter import KalmanFilter
from src.tracking.extended_kalman_filter import ExtendedKalmanFilter

# --- Helper Functions ---
def get_bbox_center(bbox_dict):
    """Same helper as in run_tracking.py"""
    if not bbox_dict: return None
    for key, val in bbox_dict.items():
        if 'min_corner' in val and 'max_corner' in val:
             return (np.array(val['min_corner']) + np.array(val['max_corner'])) / 2.0
        if 'center' in val: return np.array(val['center'])
        if 'min' in val and 'max' in val:
            return (np.array(val['min']) + np.array(val['max'])) / 2.0
    if 'min_corner' in bbox_dict and 'max_corner' in bbox_dict:
         return (np.array(bbox_dict['min_corner']) + np.array(bbox_dict['max_corner'])) / 2.0
    return None

def cartesian_to_spherical(xyz):
    """Same helper as in run_tracking_ekf.py"""
    x, y, z = xyz
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arcsin(z / rho) if rho > 1e-6 else 0
    return np.array([rho, theta, phi])

def export_point_cloud(sample, output_dir):
    """
    Export a representative frame's Point Cloud (Scene + Human) to JSON.
    Downsamples to max 10k points for web performance.
    """
    # Use middle frame
    seq_coords = sample['sequence_coord']
    frame_idx = len(seq_coords) // 2
    
    points = seq_coords[frame_idx]
    if hasattr(points, 'numpy'): points = points.numpy()
    
    # Check if color exists (sequence_feat usually has [xyz, rgb, normals])
    # UnifiedDatasetForOurs: feat = [coord_norm, color, normal] -> 3+3+3 = 9
    try:
        feats = sample['sequence_feat'][frame_idx]
        if hasattr(feats, 'numpy'): feats = feats.numpy()
        # Cols 3,4,5 are RGB
        colors = feats[:, 3:6]
    except:
        colors = np.ones_like(points) * 0.5 # Grey fallback
        
    # Downsample
    N = len(points)
    target_N = 5000 # Web friendly
    if N > target_N:
        indices = np.random.choice(N, target_N, replace=False)
        points = points[indices]
        colors = colors[indices]
        
    data = {
        "points": points.tolist(),
        "colors": colors.tolist()
    }
    
    with open(os.path.join(output_dir, "point_cloud.json"), 'w') as f:
        json.dump(data, f)
    print(f"  [x] Exported Point Cloud ({len(points)} pts)")

def generate_gifs(sample, output_dir, kf, ekf):
    """
    Generates Top-View GIFs for KF and EKF.
    """
    frames_bboxes = sample['sequence_human_bboxes']
    valid_frames = []
    
    for bbox in frames_bboxes:
        c = get_bbox_center(bbox)
        if c is not None: valid_frames.append(c)
        
    if not valid_frames: return

    # --- Run Filters ---
    gt = np.array(valid_frames)
    
    # KF Run
    kf_ests = []
    if len(valid_frames) > 0: kf.x[:3] = valid_frames[0].reshape(3,1)
    for pos in valid_frames:
        noise = np.random.normal(0, 0.2, 3)
        kf.predict()
        est = kf.update(pos + noise)
        kf_ests.append(est.flatten()[:3])
    kf_ests = np.array(kf_ests)

    # EKF Run
    ekf_ests = []
    # EKF Init
    if len(valid_frames) > 0:
        first_sph = cartesian_to_spherical(valid_frames[0])
        n = first_sph + np.random.normal(0,1,3)*np.array([0.2, 0.05, 0.05])
        r,t,p = n
        ix = r*np.cos(p)*np.cos(t); iy = r*np.cos(p)*np.sin(t); iz = r*np.sin(p)
        ekf.x[:3] = np.array([ix,iy,iz]).reshape(3,1)
        
    for pos in valid_frames:
        sph = cartesian_to_spherical(pos)
        noise = np.random.normal(0,1,3)*np.array([0.2, 0.05, 0.05])
        ekf.predict()
        est = ekf.update(sph + noise)
        ekf_ests.append(est.flatten()[:3])
    ekf_ests = np.array(ekf_ests)
    
    # --- Helper to plot GIF ---
    def save_gif(dataset_name, trajectory, title, filename):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=90, azim=-90) # Top View
        
        ax.set_xlim(np.min(gt[:,0]), np.max(gt[:,0]))
        ax.set_ylim(np.min(gt[:,1]), np.max(gt[:,1]))
        ax.set_zlim(np.min(gt[:,2]), np.max(gt[:,2]))
        ax.set_title(title)
        
        line_gt, = ax.plot([], [], [], 'g-', label='GT')
        line_est, = ax.plot([], [], [], 'b-', label='Est')
        point, = ax.plot([], [], [], 'ro')
        
        def update(i):
            line_gt.set_data(gt[:i,0], gt[:i,1])
            line_gt.set_3d_properties(gt[:i,2])
            
            line_est.set_data(trajectory[:i,0], trajectory[:i,1])
            line_est.set_3d_properties(trajectory[:i,2])
            
            point.set_data(trajectory[i:i+1,0], trajectory[i:i+1,1])
            point.set_3d_properties(trajectory[i:i+1,2])
            return line_gt, line_est, point
            
        anim = FuncAnimation(fig, update, frames=len(gt), interval=100)
        anim.save(os.path.join(output_dir, filename), writer='pillow')
        plt.close(fig)
        print(f"  [x] Saved {filename}")

    save_gif('KF', kf_ests, "KF Estimate (Top View)", "KF_estimate_top.gif")
    save_gif('EKF', ekf_ests, "EKF Estimate (Top View)", "EKF_estimate_top.gif")
    
    # --- Export Trajectory JSON ---
    traj_data = {
        "ground_truth": gt.tolist(),
        "kf_estimate": kf_ests.tolist(),
        "ekf_estimate": ekf_ests.tolist()
    }
    with open(os.path.join(output_dir, "trajectory.json"), 'w') as f:
        json.dump(traj_data, f)
    print("  [x] Exported Trajectory JSON")

def main():
    # Setup
    ours_data_path = os.path.join(PROJECT_ROOT, "data/v3-infingen-scene-mdm-human")
    ours_bbox_path = ours_data_path + "-bboxes"
    web_data_root = os.path.join(PROJECT_ROOT, "docs/public/data")
    
    print("Loading Dataset...")
    dataset = UnifiedDatasetForOurs(
        data_root=ours_data_path,
        split="train",
        split_ratio=(1,0,0),
        bbox_root=ours_bbox_path,
        num_frames=-1,
        frame_interval=1,
        seed=42
    )
    
    # Process 'walking' sequences
    count = 0
    manifest = []
    
    # Limit to 3 sequences for demo
    limit = 3 
    
    for i in range(len(dataset)):
        if count >= limit: break
        try:
            sample = dataset[i]
            if sample['action_info']['action_label'] != 'walking': continue
            
            seq_name = sample['sequence_name']
            print(f"Processing {seq_name}...")
            
            seq_dir = os.path.join(web_data_root, seq_name)
            os.makedirs(seq_dir, exist_ok=True)
            
            # Init Filters
            kf = KalmanFilter(dt=1.0)
            ekf = ExtendedKalmanFilter(dt=1.0, process_noise_std=0.2, measurement_noise_std=0.1)
            ekf.R = np.diag([0.2**2, 0.05**2, 0.05**2])
            
            # Generate Assets
            export_point_cloud(sample, seq_dir)
            generate_gifs(sample, seq_dir, kf, ekf)
            
            manifest.append({
                "id": seq_name,
                "label": f"{seq_name} (Walking)",
                "path": f"/data/{seq_name}"
            })
            count += 1
            
        except Exception as e:
            print(f"Skipping {i}: {e}")
            continue
            
    # Save Manifest
    with open(os.path.join(web_data_root, "manifest.json"), 'w') as f:
        json.dump(manifest, f)
    print("Done! Web data generated.")

if __name__ == "__main__":
    main()
