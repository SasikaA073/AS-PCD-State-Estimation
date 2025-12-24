import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

VIEWS = {'top_view':{
    'camera_elevation':0, 'camera_azimuth': 90
}}
# Configuration
CAMERA_ELEVATION = VIEWS['top_view']['camera_elevation']
CAMERA_AZIMUTH = VIEWS['top_view']['camera_azimuth']
SHOW_BOUNDING_BOX = False # Set to True to show red bounding box around human

# Determine the project root (one level up from this script)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.datasets.unified_dataset import UnifiedDatasetForOurs

def get_bbox_corners(bbox_dict):
    if not bbox_dict:
        return None
        
    # Find the human key (e.g., 'male', 'female')
    target_info = None
    for k, v in bbox_dict.items():
        if 'min_corner' in v and 'max_corner' in v:
            target_info = v
            break
    
    if not target_info:
        return None
        
    min_c = np.array(target_info['min_corner'])
    max_c = np.array(target_info['max_corner'])
    
    # Define 8 corners
    corners = np.array([
        [min_c[0], min_c[1], min_c[2]],
        [max_c[0], min_c[1], min_c[2]],
        [max_c[0], max_c[1], min_c[2]],
        [min_c[0], max_c[1], min_c[2]],
        [min_c[0], min_c[1], max_c[2]],
        [max_c[0], min_c[1], max_c[2]],
        [max_c[0], max_c[1], max_c[2]],
        [min_c[0], max_c[1], max_c[2]]
    ])
    return corners

def generate_gif(i, sample_scene, output_dir):
    scene_id = sample_scene.get('scene_id', 'unknown')
    sequence_name = sample_scene.get('sequence_name', 'unknown')
    print(f"Processing sequence: {sequence_name} (Scene: {scene_id})")

    # Animation Setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Unpack sequence data
    seq_coords = sample_scene['sequence_coord'] 
    seq_human_bboxes = sample_scene['sequence_human_bboxes']
    
    # Convert tensors to numpy if needed
    frames = []
    for coord in seq_coords:
        if hasattr(coord, 'numpy'):
            frames.append(coord.numpy())
        else:
            frames.append(coord)
            
    if not frames:
        print("Error: No frames found.")
        plt.close(fig)
        return

    # Set plot limits based on the first frame
    first_frame = frames[0]
    ax.set_xlim(np.min(first_frame[:, 0]), np.max(first_frame[:, 0]))
    ax.set_ylim(np.min(first_frame[:, 1]), np.max(first_frame[:, 1]))
    ax.set_zlim(np.min(first_frame[:, 2]), np.max(first_frame[:, 2]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Title format: scene_id_sequence_name (sequence_id was requested but likely refers to sequence_name)
    ax.set_title(f"{scene_id}_{sequence_name}")
    
    # Set camera angle
    ax.view_init(elev=CAMERA_ELEVATION, azim=CAMERA_AZIMUTH)

    scat = ax.scatter([], [], [], s=1, c='blue', alpha=0.5)
    bbox_lines = [ax.plot([], [], [], 'r-')[0] for _ in range(12)]

    # Prepare Output Paths
    base_name = f"{i}_{scene_id}_{sequence_name}_preview_top_view"
    save_filename = f"{base_name}.gif"
    save_path = os.path.join(output_dir, save_filename)
    
    frames_dir = os.path.join(output_dir, base_name)
    os.makedirs(frames_dir, exist_ok=True)

    def update(frame_idx):
        # Update Point Cloud
        points = frames[frame_idx]
        scat._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        
        # Update Bounding Box
        if SHOW_BOUNDING_BOX:
            bbox_dict = seq_human_bboxes[frame_idx]
            corners = get_bbox_corners(bbox_dict)
            
            if corners is not None:
                lines_idx = [
                    [0, 1], [1, 2], [2, 3], [3, 0], # Bottom
                    [4, 5], [5, 6], [6, 7], [7, 4], # Top
                    [0, 4], [1, 5], [2, 6], [3, 7]  # Verticals
                ]
                
                for line_i, (start, end) in enumerate(lines_idx):
                    line_x = [corners[start, 0], corners[end, 0]]
                    line_y = [corners[start, 1], corners[end, 1]]
                    line_z = [corners[start, 2], corners[end, 2]]
                    bbox_lines[line_i].set_data(line_x, line_y)
                    bbox_lines[line_i].set_3d_properties(line_z)
            else:
                 for line in bbox_lines:
                    line.set_data([], [])
                    line.set_3d_properties([])
        else:
             for line in bbox_lines:
                line.set_data([], [])
                line.set_3d_properties([])

        # Update Title with Frame ID
        ax.set_title(f"{scene_id}_{sequence_name}\nFrame: {frame_idx}/{len(frames)}")
        
        # Save Frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:03d}.png")
        fig.savefig(frame_path)
        
        return scat, *bbox_lines

    # Create Animation
    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
    
    # Save GIF
    print(f"Saving animation to {save_path}...")
    anim.save(save_path, writer='pillow')
    plt.close(fig)
    print("Done.")

def main():
    # --- 1. Load Dataset ---
    ours_data_path = os.path.join(PROJECT_ROOT, "data/v3-infingen-scene-mdm-human")
    ours_bbox_path = ours_data_path + "-bboxes"
    
    print(f"Loading dataset from: {ours_data_path}")
    
    train_dataset_ours = UnifiedDatasetForOurs(
        data_root=ours_data_path,
        split="train",
        split_ratio=(1, 0, 0),
        bbox_root=ours_bbox_path,
        num_frames=-1,      
        frame_interval=1,   
        seed=42,
    )
    
    # --- 2. Filter sequences ---
    print("Filtering dataset...")
    target_samples = []
    
    # Limit search as per user previous changes (keeping min(10, ...) logic)
    limit = min(10, len(train_dataset_ours))
    
    for i in range(limit):
        try:
            sample = train_dataset_ours[i]
            action_label = sample['action_info']['action_label']
            
            if action_label in ['walking', 'walking_and_turning']:
                target_samples.append(sample)
                print(i, sample['room_type'], sample['sequence_name'], sample['action_prompt'])
      
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue

    if not target_samples:
        print("No matches found in the search limit!")
        return

    print(f"Found {len(target_samples)} sequences to process.")

    # --- 3. Process All ---
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    for i, sample in enumerate(target_samples):
        print(f"--- Processing {i+1}/{len(target_samples)} ---")
        generate_gif(i, sample, output_dir)

if __name__ == "__main__":
    main()
