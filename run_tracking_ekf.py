import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.datasets.unified_dataset import UnifiedDatasetForOurs
from src.tracking.extended_kalman_filter import ExtendedKalmanFilter

def get_bbox_center(bbox_dict):
    """
    Extracts the center (x, y, z) from the bounding box dictionary.
    Handles {'gender': {'min_corner': ..., 'max_corner': ...}}
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

def cartesian_to_spherical(xyz):
    """
    Convert (x, y, z) -> (rho, theta, phi)
    """
    x, y, z = xyz
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    # Check division by zero or small rho
    if rho < 1e-6:
        phi = 0
    else:
        phi = np.arcsin(z / rho)
    return np.array([rho, theta, phi])

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
    
    # --- 3. Run EKF Tracking ---
    # Sensor Noise Params (Simulated Radar)
    std_range = 0.2     # meters
    std_azimuth = 0.05  # radians (~3 degrees)
    std_elev = 0.05     # radians
    
    ekf = ExtendedKalmanFilter(
        dt=1.0, 
        process_noise_std=0.2,  # Tuned for balance 
        measurement_noise_std=0.1 
    )
    # Update EKF's R matrix: Trust the range less (high noise) but maybe angles are okay?
    # Actually, let's match R to the simulated noise exactly for optimal filtering
    ekf.R = np.diag([std_range**2, std_azimuth**2, std_elev**2])
    
    gt_positions = []
    measurements_cartesian = [] # For visualization only
    estimates = []
    
    frames = sample['sequence_human_bboxes']
    valid_frames = []
    
    for i, bbox_dict in enumerate(frames):
        center = get_bbox_center(bbox_dict)
        if center is None: continue
        valid_frames.append(center)

    print(f"Running EKF on {len(valid_frames)} valid frames...")
    
    # Initialize EKF
    if len(valid_frames) > 0:
        # Use first measurement to initialize state
        first_cart = valid_frames[0]
        # More realistic: initialize from the first NOISY measurement to simulate real scenario
        # But for stability, let's use the first spherical measurement converted back
        first_sph = cartesian_to_spherical(first_cart)
        # Add noise to start
        first_sph_noisy = first_sph + np.random.normal(0, 1, 3) * np.array([std_range, std_azimuth, std_elev])
        
        # Convert back to Cartesian for initial state
        r, t, p = first_sph_noisy
        ix = r * np.cos(p) * np.cos(t)
        iy = r * np.cos(p) * np.sin(t)
        iz = r * np.sin(p)
        ekf.x[:3] = np.array([ix, iy, iz]).reshape(3, 1)

    for i, true_pos in enumerate(valid_frames):
        # 3.1 Simulate Non-linear Measurement (Radar)
        # Convert True Cart -> True Spherical
        true_sph = cartesian_to_spherical(true_pos)
        
        # Add Noise
        noise = np.random.normal(0, 1, 3) * np.array([std_range, std_azimuth, std_elev])
        meas_sph = true_sph + noise
        
        # 3.2 EKF Step
        ekf.predict()
        est_state = ekf.update(meas_sph) # Est is [x, y, z, vx, vy, vz]
        
        # 3.3 Store Data
        gt_positions.append(true_pos)
        estimates.append(est_state.flatten()[:3])
        
        # Visualize Measurement (Convert back to Cartesian to see how noisy it was)
        r, t, p = meas_sph
        mx = r * np.cos(p) * np.cos(t)
        my = r * np.cos(p) * np.sin(t)
        mz = r * np.sin(p)
        measurements_cartesian.append([mx, my, mz])
        
    # --- 4. Analysis & Plotting ---
    gt_positions = np.array(gt_positions)
    estimates = np.array(estimates)
    measurements_cartesian = np.array(measurements_cartesian)
    
    # Calculate RMSE
    rmse_est = np.sqrt(np.mean((estimates - gt_positions)**2))
    rmse_meas = np.sqrt(np.mean((measurements_cartesian - gt_positions)**2))

    print(f"\n--- Results ---")
    print(f"Simulated Sensor RMSE: {rmse_meas:.4f} m (XYZ approx)")
    print(f"EKF Estimate RMSE:     {rmse_est:.4f} m")
    print(f"Improvement:           {(rmse_meas - rmse_est):.4f} m")
    
    # Plot 3D Trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(gt_positions[:,0], gt_positions[:,1], gt_positions[:,2], 'g-', linewidth=2, label='Ground Truth')
    ax.scatter(measurements_cartesian[:,0], measurements_cartesian[:,1], measurements_cartesian[:,2], c='r', s=10, alpha=0.3, label='Radar Measurements (Simulated)')
    ax.plot(estimates[:,0], estimates[:,1], estimates[:,2], 'b-', linewidth=2, label='EKF Estimate')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f"Extended Kalman Filter Tracking (Radar Model)\nSeq: {sample['sequence_name']}")
    
    # Set camera angle
    ax.view_init(elev=30, azim=-60)
    
    # Create Output Directory
    room_type = sample.get('room_type', 'unknown')
    scene_id = sample.get('scene_id', 'unknown')
    output_dir = os.path.join(PROJECT_ROOT, "output", f"{room_type}_{scene_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "ekf_tracking_result.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
