"""
Data Utilities for Sonata Custom Dataset

This module contains reusable data processing functions for combining scenes,
humans, label mapping, and other data transformations that can be used across
different parts of the project.

Key Functions:
- combine_scene_and_human: Combine scene and human point clouds
- convert_labels_to_indices: Convert string labels to class indices
- apply_augmentation: Data augmentation functions
- normalize_point_cloud: Point cloud normalization
- load_npz_data: Safe NPZ file loading with corruption handling
"""

import logging
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

# Setup logging
logger = logging.getLogger(__name__)

# Custom dataset class labels (21 classes total)
CUSTOM_CLASSES = [
    "Art", "Bathtub", "Bed", "Cabinet", "Ceiling", "Chair", "Desk", "Door", 
    "Floor", "Lamp", "Light", "Mirror", "Pillow", "Plant", "Shelf", "Sink", 
    "Table", "Toilet", "Wall", "Window", "female"
]

# Create label to index mapping (0-20)
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CUSTOM_CLASSES)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

# Color map for visualization (21 colors)
CUSTOM_COLOR_MAP = {
    0: (174, 199, 232),   # Art
    1: (82, 84, 163),     # Bathtub  
    2: (31, 119, 180),    # Bed
    3: (152, 223, 138),   # Cabinet
    4: (188, 189, 34),    # Ceiling
    5: (255, 187, 120),   # Chair
    6: (23, 190, 207),    # Desk
    7: (255, 152, 150),   # Door
    8: (152, 223, 138),   # Floor
    9: (219, 219, 141),   # Lamp
    10: (140, 86, 75),    # Light
    11: (196, 156, 148),  # Mirror
    12: (197, 176, 213),  # Pillow
    13: (148, 103, 189),  # Plant
    14: (44, 160, 44),    # Shelf
    15: (112, 128, 144),  # Sink
    16: (140, 86, 75),    # Table
    17: (227, 119, 194),  # Toilet
    18: (174, 199, 232),  # Wall
    19: (214, 39, 40),    # Window
    20: (255, 127, 14),   # female
}

CUSTOM_COLORS = [CUSTOM_COLOR_MAP[i] for i in range(21)]


def load_npz_data(
    npz_path: Path,
    corrupted_files: Optional[Set[str]] = None,
    total_files_attempted: Optional[List[int]] = None,
    corruption_threshold: float = 0.1
) -> Optional[Dict]:
    """
    Load data from NPZ file with corruption handling.
    
    Args:
        npz_path: Path to the NPZ file
        corrupted_files: Set to track corrupted files (modified in-place)
        total_files_attempted: List to track total attempts (modified in-place)
        corruption_threshold: Threshold for warning about corruption rate
        
    Returns:
        Dictionary with loaded data or None if corrupted
    """
    if total_files_attempted is not None:
        total_files_attempted[0] += 1
    
    try:
        data = np.load(npz_path)
        
        # Check for required keys
        required_keys = ['points', 'colors', 'normals']
        label_keys = ['label', 'labels']  # Accept either 'label' or 'labels'
        
        missing_keys = [key for key in required_keys if key not in data.files]
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}")
        
        # Find label key
        label_key = None
        for key in label_keys:
            if key in data.files:
                label_key = key
                break
        
        if label_key is None:
            raise KeyError(f"No label key found. Available keys: {data.files}")
        
        # Convert to the expected format
        result = {
            'points': data['points'].astype(np.float32),
            'colors': data['colors'].astype(np.float32),
            'normals': data['normals'].astype(np.float32),
            'labels': data[label_key]
        }
        
        # Basic validation
        n_points = len(result['points'])
        if n_points == 0:
            raise ValueError("Empty point cloud")
        
        # Check array shapes
        if (result['colors'].shape[0] != n_points or 
            result['normals'].shape[0] != n_points or 
            len(result['labels']) != n_points):
            raise ValueError(f"Mismatched array lengths: points={n_points}, "
                           f"colors={result['colors'].shape[0]}, "
                           f"normals={result['normals'].shape[0]}, "
                           f"labels={len(result['labels'])}")
        
        return result
        
    except Exception as e:
        if corrupted_files is not None:
            corrupted_files.add(str(npz_path))
        
        logger.warning(f"Skipping corrupted file {npz_path}: {str(e)}")
        
        # Check if corruption rate is too high
        if corrupted_files is not None and total_files_attempted is not None:
            corruption_rate = len(corrupted_files) / total_files_attempted[0]
            if corruption_rate > corruption_threshold:
                logger.error(
                    f"HIGH CORRUPTION RATE DETECTED: {corruption_rate:.2%} "
                    f"({len(corrupted_files)}/{total_files_attempted[0]} files corrupted). "
                    f"This exceeds the threshold of {corruption_threshold:.1%}. "
                    f"Please check your data quality!"
                )
        
        # Return None to indicate corruption
        return None


def normalize_label_name(label: str) -> str:
    """
    Normalize label names by removing instance IDs.
    
    Examples:
        Window_1 -> Window
        Cabinet_2 -> Cabinet
        female -> female (unchanged)
    
    Args:
        label: Raw label string
        
    Returns:
        Normalized label string (preserves original case)
    """
    # Remove instance IDs (underscore followed by digits)
    import re
    normalized = re.sub(r'_\d+$', '', str(label).strip())
    return normalized  # Keep original case


def convert_labels_to_indices(labels: np.ndarray) -> np.ndarray:
    """
    Convert string labels to class indices with normalization.
    
    Args:
        labels: Array of string labels
        
    Returns:
        Array of class indices (0-20)
    """
    normalized_labels = [normalize_label_name(label) for label in labels]
    label_indices = []
    
    # Create case-insensitive lookup
    case_insensitive_mapping = {k.lower(): v for k, v in LABEL_TO_IDX.items()}
    
    for norm_label in normalized_labels:
        # Try exact match first
        if norm_label in LABEL_TO_IDX:
            label_indices.append(LABEL_TO_IDX[norm_label])
        # Try case-insensitive match
        elif norm_label.lower() in case_insensitive_mapping:
            label_indices.append(case_insensitive_mapping[norm_label.lower()])
        else:
            # For unknown labels, default to class 0 (Art)
            logger.debug(f"Unknown label '{norm_label}', using default class 0 (Art)")
            label_indices.append(0)
    
    return np.array(label_indices, dtype=np.int64)


def combine_scene_and_human(scene_data: Dict, human_data: Dict) -> Dict:
    """
    Combine scene point cloud with human points.
    
    Args:
        scene_data: Dictionary with scene point cloud data
        human_data: Dictionary with human point cloud data
        
    Returns:
        Combined point cloud dictionary
    """
    # Concatenate all arrays
    combined_points = np.concatenate([scene_data['points'], human_data['points']], axis=0)
    combined_colors = np.concatenate([scene_data['colors'], human_data['colors']], axis=0)
    combined_normals = np.concatenate([scene_data['normals'], human_data['normals']], axis=0)
    combined_labels = np.concatenate([scene_data['labels'], human_data['labels']], axis=0)
    
    return {
        'points': combined_points,
        'colors': combined_colors,
        'normals': combined_normals,
        'labels': combined_labels
    }


def apply_augmentation(data: Dict, enable: bool = True) -> Dict:
    """
    Apply data augmentation to point cloud.
    
    Args:
        data: Point cloud data dictionary
        enable: Whether to apply augmentation
        
    Returns:
        Augmented point cloud data
    """
    if not enable:
        return data
    
    points = data['points'].copy()
    colors = data['colors'].copy()
    normals = data['normals'].copy()
    
    # Random rotation around Z-axis
    if random.random() < 0.5:
        angle = random.uniform(-np.pi, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        points = points @ rot_matrix.T
        normals = normals @ rot_matrix.T
    
    # Random scaling
    if random.random() < 0.3:
        scale = random.uniform(0.9, 1.1)
        points = points * scale
    
    # Random jitter
    if random.random() < 0.5:
        jitter = np.random.normal(0, 0.01, points.shape)
        points = points + jitter
    
    # Color jitter
    if random.random() < 0.5:
        color_jitter = np.random.normal(0, 0.05, colors.shape)
        colors = np.clip(colors + color_jitter, 0, 1)
    
    return {
        'points': points,
        'colors': colors,
        'normals': normals,
        'labels': data['labels']
    }


def normalize_point_cloud(data: Dict, enable: bool = True) -> Dict:
    """
    Normalize point cloud coordinates.
    
    Args:
        data: Point cloud data dictionary
        enable: Whether to apply normalization
        
    Returns:
        Normalized point cloud data
    """
    if not enable:
        return data
    
    points = data['points'].copy()
    
    # Center the point cloud
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Scale to unit sphere
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    
    return {
        'points': points,
        'colors': data['colors'],
        'normals': data['normals'],
        'labels': data['labels']
    }


def get_class_weights(dataset) -> torch.Tensor:
    """
    Calculate class weights for balanced training.
    
    Args:
        dataset: Dataset object with samples containing 'segment' field
        
    Returns:
        Tensor of class weights
    """
    logger.info("Calculating class weights...")
    
    class_counts = np.zeros(21)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        labels = sample['segment']
        
        for label in labels:
            if 0 <= label < 21:
                class_counts[label] += 1
    
    # Calculate inverse frequency weights
    total_samples = np.sum(class_counts)
    class_weights = total_samples / (21 * class_counts + 1e-6)  # Add small epsilon to avoid division by zero
    
    logger.info("Class distribution:")
    for i, (class_name, count, weight) in enumerate(zip(CUSTOM_CLASSES, class_counts, class_weights)):
        logger.info(f"  {i:2d} {class_name:12s}: {int(count):8d} samples (weight: {weight:.3f})")
    
    return torch.tensor(class_weights, dtype=torch.float32)


def calculate_corruption_stats(corrupted_files: Set[str], total_files_attempted: int) -> Dict:
    """
    Calculate statistics about corrupted files.
    
    Args:
        corrupted_files: Set of corrupted file paths
        total_files_attempted: Total number of files attempted to load
        
    Returns:
        Dictionary with corruption statistics
    """
    if total_files_attempted == 0:
        return {
            'corruption_rate': 0.0,
            'corrupted_files': 0,
            'total_files': 0,
            'corrupted_file_list': []
        }
    
    corruption_rate = len(corrupted_files) / total_files_attempted
    
    return {
        'corruption_rate': corruption_rate,
        'corrupted_files': len(corrupted_files),
        'total_files': total_files_attempted,
        'corrupted_file_list': list(corrupted_files)
    }


def print_corruption_report(corrupted_files: Set[str], total_files_attempted: int, corruption_threshold: float = 0.1):
    """
    Print a detailed corruption report.
    
    Args:
        corrupted_files: Set of corrupted file paths
        total_files_attempted: Total number of files attempted to load
        corruption_threshold: Threshold for warning about high corruption rate
    """
    stats = calculate_corruption_stats(corrupted_files, total_files_attempted)
    
    print("=" * 60)
    print("DATASET CORRUPTION REPORT")
    print("=" * 60)
    print(f"Total files attempted: {stats['total_files']}")
    print(f"Corrupted files: {stats['corrupted_files']}")
    print(f"Corruption rate: {stats['corruption_rate']:.2%}")
    
    if stats['corrupted_files'] > 0:
        print(f"\nCorrupted files list:")
        for i, filepath in enumerate(stats['corrupted_file_list'][:10]):  # Show first 10
            print(f"  {i+1}. {filepath}")
        
        if len(stats['corrupted_file_list']) > 10:
            remaining = len(stats['corrupted_file_list']) - 10
            print(f"  ... and {remaining} more files")
        
        if stats['corruption_rate'] > corruption_threshold:
            print(f"\n⚠️  WARNING: Corruption rate ({stats['corruption_rate']:.2%}) exceeds threshold ({corruption_threshold:.1%})!")
            print("   Consider checking your data quality or increasing the threshold.")
    else:
        print("\n✅ No corrupted files detected!")
    
    print("=" * 60)


# PCD file utilities
def load_pcd_safe(pcd_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Safely load a PCD file and convert to numpy arrays.
    
    Args:
        pcd_path: Path to PCD file
        
    Returns:
        Dictionary with 'points', 'colors', 'normals', 'labels' or None if failed
    """
    try:
        # First try to load with Open3D (for basic geometry)
        import open3d as o3d
        
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        
        if len(pcd.points) == 0:
            logger.warning(f"Empty point cloud: {pcd_path}")
            return None
        
        # Extract points
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # Handle colors
        if pcd.has_colors():
            colors = np.asarray(pcd.colors, dtype=np.float32)
        else:
            colors = np.ones((len(points), 3), dtype=np.float32) * 0.5  # Gray default
        
        # Handle normals
        if pcd.has_normals():
            normals = np.asarray(pcd.normals, dtype=np.float32)
        else:
            # Estimate normals
            pcd.estimate_normals()
            normals = np.asarray(pcd.normals, dtype=np.float32)
        
        # Normalize colors to [0,1] if needed
        if colors.max() > 1.0:
            colors = colors / 255.0
        
        # Extract point labels by parsing the PCD file manually
        # Open3D doesn't support custom fields like 'label', so we read manually
        with open(pcd_path, 'r') as f:
            lines = f.readlines()
        
        # Find FIELDS line and data start
        fields_line = None
        data_start = -1
        for i, line in enumerate(lines):
            if line.startswith('FIELDS'):
                fields_line = line.strip()
            elif line.startswith('DATA'):
                data_start = i + 1
                break
        
        if not fields_line or 'label' not in fields_line or data_start < 0:
            raise ValueError(f"PCD file {pcd_path} missing required 'label' field")
        
        # Parse field positions
        fields = fields_line.split()[1:]  # Skip 'FIELDS'
        label_idx = fields.index('label')
        
        # Extract raw labels from data lines
        raw_labels = []
        for line in lines[data_start:]:
            if line.strip():  # Skip empty lines
                parts = line.strip().split()
                if len(parts) > label_idx:
                    raw_labels.append(parts[label_idx])
        
        if len(raw_labels) != len(points):
            raise ValueError(f"Label count mismatch in {pcd_path}: {len(raw_labels)} vs {len(points)}")
        
        # Convert labels using normalization and mapping
        labels = convert_labels_to_indices(np.array(raw_labels))
        
        result = {
            'points': points,
            'colors': colors,
            'normals': normals,
            'labels': labels
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error loading PCD {pcd_path}: {e}")
        return None


def validate_pcd_data(pcd_data: Dict[str, np.ndarray], min_points: int = 100) -> bool:
    """Validate PCD data quality."""
    if pcd_data is None:
        return False
    
    # Check minimum points
    if len(pcd_data['points']) < min_points:
        return False
    
    # Check for invalid values
    for key in ['points', 'colors', 'normals']:
        if key in pcd_data:
            arr = pcd_data[key]
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                return False
    
    return True


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for DataLoader."""
    # Use Sonata's collate function
    import sonata
    return sonata.data.collate_fn(batch)