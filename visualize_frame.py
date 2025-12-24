import open3d as o3d
import os
import glob

# Configuration
HUMAN_ONLY = True  # Set to True to visualize only the human, False for human + scene
DRAW_BOUNDING_BOX = False # Set to True to draw a bounding box around the human
BBOX_LINE_WIDTH = 5.0 # Line width for the bounding box

# Paths
BASE_DIR = "/Users/sasika/my_projects/Autonomous Project/data/v3-infingen-scene-mdm-human/Office/Office_001/Seed1"
SEQUENCE_DIR = os.path.join(BASE_DIR, "sequence205")
SCENE_PCD_PATH = os.path.join(BASE_DIR, "scene_point_cloud", "scene.pcd")

def main():
    # 1. Count frames
    frame_files = sorted(glob.glob(os.path.join(SEQUENCE_DIR, "frame_*.pcd")))
    num_frames = len(frame_files)
    
    if num_frames == 0:
        print(f"Error: No frame files found in {SEQUENCE_DIR}")
        return

    print(f"Found {num_frames} frames in sequence1.")

    # 2. Get user input
    while True:
        try:
            frame_idx = int(input(f"Enter frame number (0 to {num_frames - 1}): "))
            if 0 <= frame_idx < num_frames:
                break
            else:
                print(f"Invalid frame number. Please enter a value between 0 and {num_frames - 1}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # 3. Load Point Clouds
    geometries = []

    # Load Human PCD
    # Construct filename like frame_000.pcd, frame_010.pcd, etc.
    # We should use the actual filename from the sorted list to be safe, or format the string.
    # The file listing showed 'frame_000.pcd', 'frame_100.pcd', so it seems to be 3 digits zero padded.
    human_pcd_path = os.path.join(SEQUENCE_DIR, f"frame_{frame_idx:03d}.pcd")
    
    if not os.path.exists(human_pcd_path):
         print(f"Error: Frame file does not exist: {human_pcd_path}")
         return

    print(f"Loading human point cloud: {human_pcd_path}")
    human_pcd = o3d.io.read_point_cloud(human_pcd_path)
    
    # Optional: Paint human red to distinguish it? User didn't ask, but it's helpful.
    # Let's keep it simple for now as Open3D usually renders with colors if present or default color.
    # human_pcd.paint_uniform_color([1, 0, 0]) 
    geometries.append(human_pcd)

    # 3.5 Draw Bounding Box
    if DRAW_BOUNDING_BOX:
        print("Adding red bounding box for human.")
        bbox = human_pcd.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0) # Red
        geometries.append(bbox)

    # Load Scene PCD if needed
    if not HUMAN_ONLY:
        if os.path.exists(SCENE_PCD_PATH):
            print(f"Loading scene point cloud: {SCENE_PCD_PATH}")
            scene_pcd = o3d.io.read_point_cloud(SCENE_PCD_PATH)
            # Optional: Paint scene gray?
            # scene_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            geometries.append(scene_pcd)
        else:
            print(f"Warning: Scene point cloud not found at {SCENE_PCD_PATH}")

    # 4. Visualize
    print("Starting visualization...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Frame {frame_idx} (Human Only: {HUMAN_ONLY}, BBox: {DRAW_BOUNDING_BOX})",
                      width=1280, height=720)
    
    for geom in geometries:
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.line_width = BBOX_LINE_WIDTH
    # opt.point_size = 2.0 # Optional: make points slightly bigger too if desired

    vis.run()
    vis.destroy_window()
    print("Visualization closed.")

if __name__ == "__main__":
    main()
