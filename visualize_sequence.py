import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Determine the project root (one level up from this script)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.datasets.unified_dataset import UnifiedDatasetForOurs

def main():
    # --- 1. Load Dataset ---
    ours_data_path = os.path.join(PROJECT_ROOT, "data/v3-infingen-scene-mdm-human")
    ours_bbox_path = ours_data_path + "-bboxes"
    
    print(f"Loading dataset from: {ours_data_path}")
    
    # Using user's configuration
    train_dataset_ours = UnifiedDatasetForOurs(
        data_root=ours_data_path,
        split="train",
        split_ratio=(1, 0, 0),
        bbox_root=ours_bbox_path,
        num_frames=-1,      # Include ALL frames
        frame_interval=1,   # No skipping
        seed=42,
    )
    
    # --- 2. Filter for "walking" (User's Logic) ---
    print("Filtering dataset for 'walking' sequences...")
    data = {}
    data['walking'] = []
    
    # Limit search to first 200 samples as per user snippet, or dataset length
    limit = min(200, len(train_dataset_ours))
    
    for i in range(limit):
        try:
            sample = train_dataset_ours[i]
            action_label = sample['action_info']['action_label']
            
            if action_label == 'walking':
                data['walking'].append(sample)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue

    if not data['walking']:
        print("No 'walking' sequences found in the first 200 samples!")
        return

    # Select the first walking sequence
    sample_scene = data['walking'][0]
    print(f"Selected sequence: {sample_scene['sequence_name']}")
    print(f"Number of frames: {len(sample_scene['sequence_coord'])}")

    # --- 3. Animation Setup ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Unpack sequence data
    seq_coords = sample_scene['sequence_coord'] # List of Tensors [N, 3]
    seq_human_bboxes = sample_scene['sequence_human_bboxes'] # List of Dicts
    
    # Convert tensors to numpy if needed
    frames = []
    for coord in seq_coords:
        if hasattr(coord, 'numpy'):
            frames.append(coord.numpy())
        else:
            frames.append(coord)
            
    # Set plot limits based on the first frame (approximate)
    first_frame = frames[0]
    ax.set_xlim(np.min(first_frame[:, 0]), np.max(first_frame[:, 0]))
    ax.set_ylim(np.min(first_frame[:, 1]), np.max(first_frame[:, 1]))
    ax.set_zlim(np.min(first_frame[:, 2]), np.max(first_frame[:, 2]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Sequence: {sample_scene['sequence_name']} (Walking)")
    
    # Set camera angle (Elevation 30 deg, Azimuth -60 deg)
    ax.view_init(elev=30, azim=-60)

    scat = ax.scatter([], [], [], s=1, c='blue', alpha=0.5)
    bbox_lines = [ax.plot([], [], [], 'r-')[0] for _ in range(12)] # 12 lines for a 3D box

    def get_bbox_corners(bbox_dict):
        # Assuming bbox format: center, sizes, or min/max
        # Based on file inspection, we need to handle the specific format.
        # Fallback: if 'min' and 'max' keys exist
        if not bbox_dict:
            return None
            
        # Inspect keys to determine format (simplification for this visualization)
        # Try finding 'human_0' or similar key
        target_key = None
        for k in bbox_dict.keys():
            if 'human' in k.lower():
                target_key = k
                break
        
        if not target_key:
            return None
            
        box = bbox_dict[target_key]
        # Assuming box has min/max or similar. 
        # For now, let's print the structure if we fail, but try standard center/extent
        # Let's try to just plot the center for tracking verification if box is complex
        if 'center' in box:
            c = box['center']
            return np.array([c]) # Just return center point for simplicity if box is complex
            
        return None

    def update(frame_idx):
        # Update Point Cloud
        points = frames[frame_idx]
        scat._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        
        # Update Bounding Box (Simplification: Just draw center if available)
        bbox_dict = seq_human_bboxes[frame_idx]
        # Logic to extract center/box would go here. 
        # For this check, we mainly want to ensure the loop runs.
        
        ax.set_title(f"Frame {frame_idx}/{len(frames)}")
        return scat,

    print("Starting animation...")
    # Showing 1 frame per 100ms
    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
    
    # Create Output Directory
    room_type = sample_scene.get('room_type', 'unknown')
    scene_id = sample_scene.get('scene_id', 'unknown')
    output_dir = os.path.join(PROJECT_ROOT, "output", f"{room_type}_{scene_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Note: plt.show() might block, so we save a gif
    save_path = os.path.join(output_dir, "walking_sequence_preview.gif")
    print(f"Saving animation to {save_path}...")
    anim.save(save_path, writer='pillow')
    print("Done!")

if __name__ == "__main__":
    main()
