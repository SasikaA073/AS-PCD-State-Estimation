import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.datasets.unified_dataset import UnifiedDatasetForOurs
from src.tracking.kalman_filter import KalmanFilter

def get_bbox_center(bbox_dict):
    """
    Extracts the center (x, y, z) from the bounding box dictionary.
    """
    if not bbox_dict:
        return None
    
    # Check for keys like 'male', 'female' or others
    for key, val in bbox_dict.items():
        # logic for current structure
        if 'min_corner' in val and 'max_corner' in val:
             c_min = np.array(val['min_corner'])
             c_max = np.array(val['max_corner'])
             return (c_min + c_max) / 2.0
             
        # Previous logic backups
        if 'center' in val:
            return np.array(val['center'])
        if 'min' in val and 'max' in val:
            c_min = np.array(val['min'])
            c_max = np.array(val['max'])
            return (c_min + c_max) / 2.0
            
    # Fallback if top-level has keys
    if 'min_corner' in bbox_dict and 'max_corner' in bbox_dict:
         c_min = np.array(bbox_dict['min_corner'])
         c_max = np.array(bbox_dict['max_corner'])
         return (c_min + c_max) / 2.0
        
    return None

def main():
    # --- 1. Load Dataset ---
    ours_data_path = os.path.join(PROJECT_ROOT, "data/v3-infingen-scene-mdm-human")
    ours_bbox_path = ours_data_path + "-bboxes"
    
    print("Initializing dataset...")
    # Using user's configuration
    dataset = UnifiedDatasetForOurs(
        data_root=ours_data_path,
        split="train",
        split_ratio=(1, 0, 0),
        bbox_root=ours_bbox_path,
        num_frames=-1,
        frame_interval=1,
        seed=42,
    )
    
    # --- 2. Filter for 'walking' ---
    print("Filtering for 'walking' sequences...")
    walking_samples = []
    limit = min(200, len(dataset))
    
    for i in range(limit):
        try:
            s = dataset[i]
            if s['action_info']['action_label'] == 'walking':
                walking_samples.append(s)
        except:
            continue
            
    if not walking_samples:
        print("No walking samples found.")
        return

    sample = walking_samples[0]
    print(f"Tracking Sequence: {sample['sequence_name']}")
    
    # --- 3. Run Tracking ---
    kf = KalmanFilter(dt=1.0, process_noise_std=0.05, measurement_noise_std=0.5)
    
    gt_positions = []
    measurements = []
    estimates = []
    
    frames = sample['sequence_human_bboxes']
    
    # Helper to clean invalid frames if bbox missing
    valid_frames = []
    
    for i, bbox_dict in enumerate(frames):
        center = get_bbox_center(bbox_dict)
        if center is None:
            # If no GT, we can't test accuracy, but in real tracking we'd predict only
            continue
        
        valid_frames.append(center)

    print(f"Running KF on {len(valid_frames)} valid frames...")
    
    # Initialize KF state with first measurement to help it converge faster
    if len(valid_frames) > 0:
        first_meas = valid_frames[0]
        kf.x[:3] = first_meas.reshape(3, 1)

    for i, true_pos in enumerate(valid_frames):
        # 3.1 Generate Noisy Measurement
        noise = np.random.normal(0, 0.2, 3) # 0.2m std dev noise
        measured_pos = true_pos + noise
        
        # 3.2 Kalman Filter Step
        kf.predict()
        est_pos = kf.update(measured_pos)
        
        # 3.3 Store
        gt_positions.append(true_pos)
        measurements.append(measured_pos)
        estimates.append(est_pos.flatten()[:3])
        
    # --- 4. Analysis & Plotting ---
    gt_positions = np.array(gt_positions)
    measurements = np.array(measurements)
    estimates = np.array(estimates)
    
    # Calculate RMSE
    rmse_meas = np.sqrt(np.mean((measurements - gt_positions)**2))
    rmse_est = np.sqrt(np.mean((estimates - gt_positions)**2))
    
    print(f"\n--- Results ---")
    print(f"Measurement RMSE: {rmse_meas:.4f} m")
    print(f"KF Estimate RMSE: {rmse_est:.4f} m")
    print(f"Improvement: {(rmse_meas - rmse_est):.4f} m")
    
    # Plot 3D Trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(gt_positions[:,0], gt_positions[:,1], gt_positions[:,2], 'g-', linewidth=2, label='Ground Truth')
    ax.scatter(measurements[:,0], measurements[:,1], measurements[:,2], c='r', s=10, alpha=0.5, label='Noisy Measurements')
    ax.plot(estimates[:,0], estimates[:,1], estimates[:,2], 'b-', linewidth=2, label='KF Estimate')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f"3D Human Tracking (Kalman Filter)\nSeq: {sample['sequence_name']}")
    
    # Set camera angle
    ax.view_init(elev=30, azim=-60)
    
    # Create Output Directory
    room_type = sample.get('room_type', 'unknown')
    scene_id = sample.get('scene_id', 'unknown')
    output_dir = os.path.join(PROJECT_ROOT, "output", f"{room_type}_{scene_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "tracking_result_plot.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
