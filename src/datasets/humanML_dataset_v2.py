"""
UnifiedDataset (Refactored for Sequence-based Loading)

This dataset implementation:
- Traverses the root to find action segments (`report.json`).
- `__init__` discovers all action segments, builds an index where EACH sample is one
  action segment (containing N frame paths).
- `__getitem__` loads ALL frames for a single action segment, applies augmentations
  (if configured), and returns a dictionary representing the full sequence.
- This simplifies the collate function and aligns better with the pipeline's
  expected input structure.
- USES A CUSTOM PURE-PYTHON PCD PARSER TO AVOID OPEN3D SEGFAULTS.
"""

import json
import logging
import random
import os
import math # <<< Added for rotation
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import re

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# --- Copied from data_utils to make this file self-contained ---
CUSTOM_CLASSES = [
    "Art", "Bathtub", "Bed", "Cabinet", "Ceiling", "Chair", "Desk", "Door",
    "Floor", "Lamp", "Light", "Mirror", "Pillow", "Plant", "Shelf", "Sink",
    "Table", "Toilet", "Wall", "Window", "female"
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CUSTOM_CLASSES)}
# --- End of copied data ---


class UnifiedDatasetForHumanML3D(Dataset):
    """
    Unified dataset that yields a full action segment (sequence of frames) per sample.
    Uses a pure-Python PCD loader to ensure multiprocessing safety.
    
    Includes text and rotation augmentations for the training split.
    """

    def __init__(self, data_root: str,
                 split: str = "train",
                 split_ratio: Tuple[float, float] = None,  # Deprecated
                 bbox_root: Optional[str] = None,
                 num_frames: int = 11,
                 frame_interval: int = 5,
                 seed: int = 42,
                 grid_size: float = 0.05,
                 scene_cache_size: int = 8,
                 augment: bool = True,
                 human_only: bool = True,
                 scene_downsample_ratio: float = 0.7): # <<< Downsample scene to prevent int32 overflow
        """
        Create a dataset instance by discovering data from a root directory.
        Split is now determined by folder structure (e.g., data_root/train/Env1).
        
        Args:
            data_root: Root directory of the dataset.
            split: "train", "val", or "test".
            bbox_root: Optional path to bounding box annotations.
            num_frames: Number of frames to sample per sequence.
            frame_interval: Interval (not used by default, sampling is linspace).
            seed: Random seed.
            grid_size: Voxel grid size for downstream processing.
            scene_cache_size: Size of the LRU cache for scene PCDs.
            augment: If True, apply augmentations to the 'train' split.
            human_only: If True, return only human points; if False, return human+scene points.
            scene_downsample_ratio: Ratio to downsample scene point clouds (0.1 = keep 10%).
        """
        self.data_root = Path(data_root)
        self.bbox_root = Path(bbox_root) if bbox_root else None
        self.num_frames_config = int(num_frames)
        self.frame_interval = int(frame_interval)
        self.seed = seed
        self.split = split.lower()

        # <<< 2. Enable augmentation only for train split and if flag is True
        self.augment = (self.split == 'train') and augment
        if self.augment:
            print("Augmentations (rotation, text) are ENABLED for the 'train' split.")
        
        # <<< 3. Store human_only option
        self.human_only = human_only
        self.scene_downsample_ratio = scene_downsample_ratio
        print(f"human_only set to {self.human_only}")
        if self.human_only:
            print("Dataset mode: HUMAN ONLY (scene points excluded)")
        else:
            print(f"Dataset mode: HUMAN + SCENE (combined points, scene downsampled to {scene_downsample_ratio*100:.1f}%)")
        
        # Cache for scene data (if it were used)
        self._scene_cache: Optional[OrderedDict] = None
        self._scene_cache_size = int(scene_cache_size)

        # Grid size
        self.grid_size = torch.tensor([grid_size, grid_size, grid_size], dtype=torch.float32)

        # --- Data Discovery ---
        logger.info(f"Building dataset index for split='{split}' from root: {data_root}")
        self.samples = self._discover_action_segments()
        logger.info(f"Dataset for split='{split}' initialized with {len(self.samples)} samples (action segments).")
        
        # --- Pre-load scene point clouds if not human_only ---
        self.scene_data_cache = {}  # Maps scene_pcd_file path to loaded scene data
        if not self.human_only:
            logger.info(f"Pre-loading scene point clouds (downsampled to {scene_downsample_ratio*100:.1f}%)...")
            unique_scene_files = set(s.get('scene_pcd_file') for s in self.samples if s.get('scene_pcd_file'))
            for scene_file in unique_scene_files:
                scene_data = self._load_pcd_custom(str(scene_file))
                if self._validate_pcd_data(scene_data, min_points=1000):
                    # Downsample scene to reduce memory and prevent int32 overflow
                    num_points = len(scene_data['points'])
                    num_keep = max(int(num_points * scene_downsample_ratio), 1000)  # Keep at least 1000 points
                    indices = np.random.choice(num_points, num_keep, replace=False)
                    scene_data['points'] = scene_data['points'][indices]
                    scene_data['colors'] = scene_data['colors'][indices]
                    scene_data['normals'] = scene_data['normals'][indices]
                    scene_data['labels'] = scene_data['labels'][indices]
                    
                    self.scene_data_cache[str(scene_file)] = scene_data
                    logger.info(f"Loaded scene: {scene_file} ({num_points} -> {num_keep} points)")
                else:
                    logger.warning(f"Invalid scene data: {scene_file}")
            logger.info(f"Pre-loaded {len(self.scene_data_cache)} unique scene point clouds.")

    def _discover_action_segments(self) -> List[Dict]:
        """
        Walks the data root and builds a lightweight index of all action segments.
        """
        root = self.data_root
        all_samples: List[Dict] = []
        
        split_folder = root / self.split
        print(f'looking for data in {split_folder}')

        env_folders = [d for d in split_folder.iterdir() if d.is_dir() and d.name.startswith('Env')]
        
        if env_folders:
            search_dirs = env_folders
            logger.info(f"Found {len(env_folders)} environment folders: {[d.name for d in env_folders]}")
        else:
            search_dirs = [split_folder]
            logger.info(f"No Env folders found. Searching directly in: {split_folder}")
        
        print(f'found data in {split_folder}')
        
        for search_dir in search_dirs:
            for seq_dir in sorted(search_dir.iterdir()):
                if not seq_dir.is_dir() or not seq_dir.name.startswith('sequence'):
                    continue
                
                sequence_name = seq_dir.name
                sequence_key = sequence_name
                
                report_path = seq_dir / 'report.json'
                if not report_path.exists():
                    logger.debug(f"No report.json at {report_path}; skipping sequence")
                    continue
                
                try:
                    with open(report_path, 'r') as rf:
                        report = json.load(rf)
                except Exception as e:
                    logger.warning(f"Failed to load report.json at {report_path}: {e}")
                    continue
                
                description = report.get('description', '')
                if not description:
                    logger.debug(f"No description in {report_path}; skipping sequence")
                    continue
                
                description_sentences = [s.strip() for s in description.split('.') if s.strip()]
                if not description_sentences:
                    logger.debug(f"Empty description after split in {report_path}; skipping sequence")
                    continue
                
                frame_files = sorted(seq_dir.glob('frame_*.pcd'))
                if not frame_files:
                    logger.debug(f"No frame_*.pcd files in {seq_dir}; skipping sequence")
                    continue
                
                frame_numbers = []
                all_frame_paths = []
                for ff in frame_files:
                    try:
                        frame_num = int(ff.stem.split('_')[1])
                        frame_numbers.append(frame_num)
                        all_frame_paths.append(ff)
                    except (IndexError, ValueError):
                        logger.warning(f"Could not parse frame number from {ff.name}")
                
                if not frame_numbers:
                    logger.debug(f"No valid frame numbers in {seq_dir}; skipping sequence")
                    continue
                
                if self.num_frames_config <= 0:
                    logger.debug(f"num_frames_config <= 0; skipping sequence {sequence_name}")
                    continue
                
                total_frames = len(frame_numbers)
                if total_frames <= self.num_frames_config:
                    chosen_indices = list(range(total_frames))
                else:
                    chosen_indices = np.linspace(0, total_frames - 1, num=self.num_frames_config, dtype=int).tolist()
                
                sampled_files = [all_frame_paths[i] for i in chosen_indices]
                sampled_frame_numbers = [frame_numbers[i] for i in chosen_indices]
                
                middle_index = len(sampled_files) // 2
                middle_frame_file_path = sampled_files[middle_index]
                try:
                    sample_file_name = str(middle_frame_file_path.relative_to(root))
                except ValueError:
                    sample_file_name = str(middle_frame_file_path)
                
                # Search for scene PCD file - same level as sequence folders
                scene_pcd_file = None
                scene_pcd_path = search_dir / 'scene_point_cloud' / 'scene.pcd'
                
                if scene_pcd_path.exists():
                    scene_pcd_file = scene_pcd_path
                    logger.debug(f"Found scene PCD at {scene_pcd_file}")
                else:
                    logger.debug(f"No scene PCD found at {scene_pcd_path} for sequence {sequence_name}")
                
                action_info = {
                    'action_label': 'human_motion',
                    'action_prompt': description_sentences[0],  # Default
                    'all_descriptions': description_sentences,  # Store all
                    'start_frame': min(sampled_frame_numbers),
                    'end_frame': max(sampled_frame_numbers),
                    'duration': len(sampled_frame_numbers),
                    'frame_numbers': sampled_frame_numbers,
                }
                
                sample = {
                    'human_pcd_files': sampled_files,
                    'frame_numbers': sampled_frame_numbers,
                    'scene_pcd_file': scene_pcd_file,
                    'json_file': report_path,
                    'file_name': sample_file_name,
                    'room_type': None,
                    'scene_id': None,
                    'seed_name': None,
                    'sequence_name': sequence_name,
                    'sequence_key': sequence_key,
                    'metadata': report,
                    'action_info': action_info,
                }
                all_samples.append(sample)
                logger.debug(f"Added sequence {sequence_name} with {len(sampled_files)} sampled frames")

        return all_samples

    # <<< START NEW/MODIFIED HELPERS (REPLACING data_utils) >>>

    def _normalize_label_name(self, label: str) -> str:
        """Normalize label names by removing instance IDs."""
        return re.sub(r'_\d+$', '', str(label).strip())

    def _convert_labels_to_indices(self, labels: np.ndarray) -> np.ndarray:
        """Convert string labels to class indices with normalization."""
        case_insensitive_mapping = {k.lower(): v for k, v in LABEL_TO_IDX.items()}
        
        label_indices = []
        for raw_label in labels:
            norm_label = self._normalize_label_name(raw_label)
            
            if norm_label in LABEL_TO_IDX:
                label_indices.append(LABEL_TO_IDX[norm_label])
            elif norm_label.lower() in case_insensitive_mapping:
                label_indices.append(case_insensitive_mapping[norm_label.lower()])
            else:
                logger.debug(f"Unknown label '{norm_label}', using default class 0 (Art)")
                label_indices.append(0)  # Default to "Art"
        
        return np.array(label_indices, dtype=np.int64)

    def _validate_pcd_data(self, pcd_data: Dict[str, np.ndarray], min_points: int = 100) -> bool:
        """Validate PCD data quality."""
        if pcd_data is None:
            return False
        if len(pcd_data['points']) < min_points:
            return False
        for key in ['points', 'colors', 'normals']:
            if key in pcd_data:
                arr = pcd_data[key]
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    return False
        return True

    def _combine_scene_and_human(self, scene_data: Dict, human_data: Dict) -> Dict:
        """Combine scene point cloud with human points."""
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

    def _load_pcd_custom(self, pcd_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Safely load an ASCII PCD file using pure Python.
        Handles packed 'rgb' fields and separate 'r','g','b' fields.
        """
        try:
            with open(pcd_path, 'r') as f:
                lines = f.readlines()

            header = {}
            data_start_line = -1
            for i, line in enumerate(lines):
                if line.startswith('DATA'):
                    if 'ascii' not in line:
                        raise ValueError(f"Only ASCII PCD files are supported. {pcd_path} is not ASCII.")
                    data_start_line = i + 1
                    break
                
                parts = line.strip().split()
                if len(parts) > 1:
                    header[parts[0]] = parts[1:]
            
            if data_start_line == -1 or 'FIELDS' not in header:
                raise ValueError("Invalid PCD header")

            fields = header['FIELDS']
            
            try:
                idx_x = fields.index('x')
                idx_y = fields.index('y')
                idx_z = fields.index('z')
                idx_label = fields.index('label')
            except ValueError as e:
                raise ValueError(f"Missing required field in {pcd_path}: {e}")

            # --- Color field handling ---
            idx_r, idx_g, idx_b, idx_rgb = -1, -1, -1, -1
            try:
                idx_r = fields.index('r')
                idx_g = fields.index('g')
                idx_b = fields.index('b')
                has_colors = True
                is_packed_rgb = False
                logger.debug("Found r, g, b fields")
            except ValueError:
                try:
                    idx_rgb = fields.index('rgb')
                    has_colors = True
                    is_packed_rgb = True
                    logger.debug("Found packed rgb field")
                except ValueError:
                    has_colors = False
                    is_packed_rgb = False
                    logger.debug("No color fields found")
            
            # --- Normal field handling ---
            idx_nx, idx_ny, idx_nz = -1, -1, -1
            try:
                idx_nx = fields.index('normal_x')
                idx_ny = fields.index('normal_y')
                idx_nz = fields.index('normal_z')
                has_normals = True
            except ValueError:
                has_normals = False
            
            points, colors, normals, labels = [], [], [], []
            
            for line in lines[data_start_line:]:
                parts = line.strip().split()
                if not parts:
                    continue
                
                try:
                    points.append([float(parts[idx_x]), float(parts[idx_y]), float(parts[idx_z])])
                    labels.append(parts[idx_label])

                    if has_colors:
                        if is_packed_rgb:
                            packed_rgb = int(parts[idx_rgb])
                            r = (packed_rgb >> 16) & 255
                            g = (packed_rgb >> 8) & 255
                            b = packed_rgb & 255
                            colors.append([r, g, b])
                        else:
                            colors.append([float(parts[idx_r]), float(parts[idx_g]), float(parts[idx_b])])
                    
                    if has_normals:
                        normals.append([float(parts[idx_nx]), float(parts[idx_ny]), float(parts[idx_nz])])
                
                except (IndexError, ValueError) as e:
                    logger.warning(f"Skipping malformed line in {pcd_path}: {e}")
                    continue

            points_np = np.array(points, dtype=np.float32)
            n_points = len(points_np)

            if n_points == 0:
                raise ValueError("Empty point cloud")

            if has_colors:
                colors_np = np.array(colors, dtype=np.float32)
                if colors_np.max() > 1.0:
                    colors_np /= 255.0
            else:
                colors_np = np.ones((n_points, 3), dtype=np.float32) * 0.5
            
            if has_normals:
                normals_np = np.array(normals, dtype=np.float32)
            else:
                normals_np = np.zeros((n_points, 3), dtype=np.float32)

            labels_np = self._convert_labels_to_indices(np.array(labels))

            return {
                'points': points_np,
                'colors': colors_np,
                'normals': normals_np,
                'labels': labels_np
            }

        except Exception as e:
            # logger.error(f"Error loading custom PCD {pcd_path}: {e}")
            return None

    # <<< END NEW HELPERS >>>


    def _load_combined_pcd(self, frame_sample_info: Dict) -> Optional[Dict]:
        """Loads a single human point cloud for one frame (scene will be combined later)."""
        human_path = str(frame_sample_info['human_pcd_file'])
        human_data = self._load_pcd_custom(human_path)
        if not self._validate_pcd_data(human_data, min_points=10):
            # logger.warning(f"Invalid human PCD data at {human_path}")
            return None

        # Only load human data here, scene will be combined after augmentation
        combined = human_data

        if self.bbox_root is not None:
            frame_number = frame_sample_info['action_info'].get('frame_number')
            combined['human_bboxes'] = self._load_human_bboxes(frame_sample_info, frame_number) if frame_number is not None else {}
            combined['scene_bboxes'] = self._load_scene_bboxes(frame_sample_info)
        else:
            combined['human_bboxes'] = {}
            combined['scene_bboxes'] = {}

        return combined

    def _load_human_bboxes(self, sample_info: Dict, frame_number: int) -> Dict:
        """Load human bboxes for a specific frame."""
        if self.bbox_root is None:
            return {}
        bbox_path = (self.bbox_root / sample_info['sequence_name'] / f"frame_{frame_number:03d}.json")
        if not bbox_path.exists():
            return {}
        try:
            with open(bbox_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load human bboxes from {bbox_path}: {e}")
            return {}

    def _load_scene_bboxes(self, sample_info: Dict) -> Dict:
        """Load scene bboxes. Returns empty dict as scene data not used."""
        return {}

    def _normalize_coordinates(self, coord: torch.Tensor) -> torch.Tensor:
        """Deprecated: Normalization is now done globally in __getitem__."""
        logger.warning("Called _normalize_coordinates, which is deprecated.")
        coord_min = coord.min(dim=0, keepdim=True)[0]
        coord_max = coord.max(dim=0, keepdim=True)[0]
        coord_range = coord_max - coord_min
        coord_range = torch.where(coord_range == 0, torch.ones_like(coord_range), coord_range)
        normalized_coord = (coord - coord_min) / coord_range
        return normalized_coord

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one full action segment (sequence of frames) with augmentations.
        Scene is loaded once, augmented once, then combined with augmented human frames.
        """
        sample_info = self.samples[idx]
        human_files = sample_info.get('human_pcd_files', [])
        frame_numbers = sample_info.get('frame_numbers', [])

        # --- 3. Random Text Augmentation ---
        all_descriptions = sample_info.get('action_info', {}).get('all_descriptions', [''])
        if not all_descriptions:
            all_descriptions = ['']  # Fallback
        action_prompt = random.choice(all_descriptions)
        # --- End Text Augmentation ---

        # --- Load and augment scene ONCE if needed ---
        scene_coord_augmented = None
        scene_feat_augmented = None
        if not self.human_only:
            scene_pcd_file = sample_info.get('scene_pcd_file')
            if scene_pcd_file and str(scene_pcd_file) in self.scene_data_cache:
                scene_data = self.scene_data_cache[str(scene_pcd_file)]
                scene_coord = torch.from_numpy(scene_data['points'].astype(np.float32).copy())
                scene_color = torch.from_numpy(scene_data['colors'].astype(np.float32).copy())
                scene_normal = torch.from_numpy(scene_data['normals'].astype(np.float32).copy())
                
                # Augment scene if training
                if self.augment:
                    k = random.choice([1, 2, 3])
                    theta = k * (math.pi / 2)
                    c, s = math.cos(theta), math.sin(theta)
                    rot_matrix = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
                    
                    # Rotate scene coords
                    xz_coords = scene_coord[:, [0, 2]]
                    rotated_xz = xz_coords @ rot_matrix.T
                    scene_coord[:, 0] = rotated_xz[:, 0]
                    scene_coord[:, 2] = rotated_xz[:, 1]
                    
                    # Rotate scene normals
                    xz_normals = scene_normal[:, [0, 2]]
                    rotated_xz_n = xz_normals @ rot_matrix.T
                    scene_normal[:, 0] = rotated_xz_n[:, 0]
                    scene_normal[:, 2] = rotated_xz_n[:, 1]
                
                scene_coord_augmented = scene_coord
                scene_feat_augmented = torch.cat([scene_color, scene_normal], dim=1)

        sequence_coords_raw = []
        sequence_feats_raw = [] # Will hold [color, normal]
        sequence_human_bboxes = []
        scene_bboxes = {}

        # --- 1. Load all human frames (unnormalized) ---
        for i, hf_path in enumerate(human_files):
            frame_num = frame_numbers[i]
            per_frame_info = {
                'human_pcd_file': hf_path,
                'action_info': {'frame_number': frame_num},
                'sequence_name': sample_info.get('sequence_name'),
                'scene_id': sample_info.get('scene_id'),
            }

            combined = self._load_combined_pcd(per_frame_info)
            if combined is None:
                logger.debug(f"Skipping invalid frame {hf_path} in sample idx={idx}")
                continue

            coord = torch.from_numpy(combined['points'].astype(np.float32).copy())
            color = torch.from_numpy(combined['colors'].astype(np.float32).copy())
            normal = torch.from_numpy(combined['normals'].astype(np.float32).copy())
            
            sequence_coords_raw.append(coord)
            sequence_feats_raw.append(torch.cat([color, normal], dim=1)) # [N, 6]
            
            sequence_human_bboxes.append(combined.get('human_bboxes', {}))
            if not scene_bboxes:
                scene_bboxes = combined.get('scene_bboxes', {})

        if len(sequence_coords_raw) == 0:
            raise RuntimeError(f"Failed to load any valid frames for sample idx={idx}, sequence_key={sample_info.get('sequence_key')}")

        # --- 4. Rotation Augmentation Block (Human frames only) ---
        if self.augment:
            # Use same rotation as scene (already applied above)
            k = random.choice([1, 2, 3])
            theta = k * (math.pi / 2)
            
            c, s = math.cos(theta), math.sin(theta)
            rot_matrix = torch.tensor([[c, -s],
                                        [s,  c]], dtype=torch.float32)
            
            for i in range(len(sequence_coords_raw)):
                coords_frame = sequence_coords_raw[i] # [N, 3]
                # Feats are [color, normal], so normals are at [:, 3:6]
                normals_frame = sequence_feats_raw[i][:, 3:6] # [N, 3]
                
                # Rotate Coords (x, z are at indices 0, 2)
                xz_coords = coords_frame[:, [0, 2]]
                rotated_xz_coords = xz_coords @ rot_matrix.T
                coords_frame[:, 0] = rotated_xz_coords[:, 0]
                coords_frame[:, 2] = rotated_xz_coords[:, 1]
                
                # Rotate Normals (nx, nz are at indices 0, 2)
                xz_normals = normals_frame[:, [0, 2]]
                rotated_xz_normals = xz_normals @ rot_matrix.T
                normals_frame[:, 0] = rotated_xz_normals[:, 0]
                normals_frame[:, 2] = rotated_xz_normals[:, 1]
        
        # --- End Augmentation Block ---
        
        # --- Combine scene with human frames (after augmentation) ---
        if scene_coord_augmented is not None:
            for i in range(len(sequence_coords_raw)):
                sequence_coords_raw[i] = torch.cat([scene_coord_augmented, sequence_coords_raw[i]], dim=0)
                sequence_feats_raw[i] = torch.cat([scene_feat_augmented, sequence_feats_raw[i]], dim=0)

        # --- 6. Global Normalization (after augmentation and combination) ---
        all_coords_tensor = torch.cat(sequence_coords_raw, dim=0)
        coord_min = all_coords_tensor.min(dim=0, keepdim=True)[0]
        coord_max = all_coords_tensor.max(dim=0, keepdim=True)[0]
        coord_range = coord_max - coord_min
        coord_range = torch.where(coord_range == 0, torch.ones_like(coord_range), coord_range)
        
        final_sequence_coords = []
        final_sequence_feats = []

        for i in range(len(sequence_coords_raw)):
            # Normalize coords using the global min/max
            norm_coord = (sequence_coords_raw[i] - coord_min) / coord_range
            final_sequence_coords.append(norm_coord)
            
            # Combine [norm_coord, color, normal]
            final_feat = torch.cat([norm_coord, sequence_feats_raw[i]], dim=1)
            final_sequence_feats.append(final_feat)
        # --- End Normalization ---

        sample_out = {
            'sequence_coord': final_sequence_coords, # List of [N, 3]
            'sequence_feat': final_sequence_feats,  # List of [N, 9]
            'grid_size': self.grid_size,
            'action_prompt': action_prompt, # Use the augmented prompt
            'num_frames_per_sample': len(final_sequence_coords),
            
            # metadata
            'file_name': sample_info.get('file_name'),
            'room_type': sample_info.get('room_type'),
            'scene_id': sample_info.get('scene_id'),
            'sequence_name': sample_info.get('sequence_name'),
            'sequence_frame_numbers': frame_numbers,
            'sequence_human_bboxes': sequence_human_bboxes,
            'scene_bboxes': scene_bboxes,
        }

        return sample_out

    @staticmethod
    def collate_fn(batch_samples: List[Dict]) -> Dict:
        """
        A simple collate_fn that batches pre-built sequences.
        """
        batch_samples = [s for s in batch_samples if s is not None]

        if not batch_samples:
            return {
                'batch_size': 0,
                'num_frames_per_sample': torch.tensor([], dtype=torch.int64),
                'sequence_coord': [],
                'sequence_feat': [],
                'grid_size': torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32),
                'action_prompts': [],
                'metadata': {}
            }

        batch_size = len(batch_samples)
        grid_size = batch_samples[0].get('grid_size')

        sequence_coord = [s['sequence_coord'] for s in batch_samples]
        sequence_feat = [s['sequence_feat'] for s in batch_samples]
        action_prompts = [s['action_prompt'] for s in batch_samples] # <<< BUG FIX
        # The key in sample_out is 'action_prompt' (singular)
        action_prompts = [s['action_prompt'] for s in batch_samples] 
        num_frames_per_sample = [s['num_frames_per_sample'] for s in batch_samples]
        
        metadata = {
            'file_name': [s.get('file_name') for s in batch_samples],
            'scene_id': [s.get('scene_id') for s in batch_samples],
            'sequence_name': [s.get('sequence_name') for s in batch_samples],
            'sequence_frame_numbers': [s.get('sequence_frame_numbers') for s in batch_samples],
            'sequence_human_bboxes': [s.get('sequence_human_bboxes') for s in batch_samples],
            'scene_bboxes': [s.get('scene_bboxes') for s in batch_samples],
        }

        batch = {
            'batch_size': batch_size,
            'num_frames_per_sample': torch.tensor(num_frames_per_sample, dtype=torch.int64),
            'sequence_coord': sequence_coord,
            'sequence_feat': sequence_feat,
            'grid_size': grid_size,
            'action_prompts': action_prompts, # This is now correct
            'metadata': metadata
        }

        return batch