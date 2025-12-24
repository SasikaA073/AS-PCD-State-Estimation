"""
UnifiedDataset (Refactored for Sequence-based Loading)

This dataset implementation:
- Traverses the root to find action segments (`report.json`).
- `__init__` discovers all action segments, performs a train/val split, and
  builds an index where EACH sample is one action segment (containing N frame paths). 
- `__getitem__` loads ALL frames for a single action segment and returns a
  dictionary representing the full sequence.
- This simplifies the collate function and aligns better with the pipeline's
  expected input structure.
- Includes a per-process LRU cache for scene point clouds,
  initialized safely for multiprocessing.
- USES A CUSTOM PURE-PYTHON PCD PARSER TO AVOID OPEN3D SEGFAULTS.
"""

import json
import logging
import random
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import re  # <<< Added for label normalization

import numpy as np
import torch
from torch.utils.data import Dataset

# NOTE: We no longer import from src.utils.data_utils
# from src.utils.data_utils import load_pcd_safe, validate_pcd_data, combine_scene_and_human

logger = logging.getLogger(__name__)

# --- Copied from data_utils to make this file self-contained ---
CUSTOM_CLASSES = [
    "Art", "Bathtub", "Bed", "Cabinet", "Ceiling", "Chair", "Desk", "Door", 
    "Floor", "Lamp", "Light", "Mirror", "Pillow", "Plant", "Shelf", "Sink", 
    "Table", "Toilet", "Wall", "Window", "female"
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CUSTOM_CLASSES)}
# --- End of copied data ---


class UnifiedDatasetForOurs(Dataset):
    """
    Unified dataset that yields a full action segment (sequence of frames) per sample.
    Uses a pure-Python PCD loader to ensure multiprocessing safety.
    """

    def __init__(self, data_root: str,
                 split: str = "train",
                 split_ratio: Tuple[float, float] = (0.7, 0.3),
                 bbox_root: Optional[str] = None,
                 num_frames: int = 1,
                 frame_interval: int = 1,
                 seed: int = 42,
                 grid_size: float = 0.05,
                 scene_cache_size: int = 8):
        """
        Create a dataset instance by discovering and splitting data from a root directory.
        """
        self.data_root = Path(data_root)
        self.bbox_root = Path(bbox_root) if bbox_root else None
        self.num_frames_config = int(num_frames) 
        self.frame_interval = int(frame_interval)
        self.seed = seed
        self.split_ratio = split_ratio

        # <<< FIX FOR SEGFAULT (Part 1) >>>
        # Initialize cache as None. It will be created safely by each worker.
        self._scene_cache: Optional[OrderedDict] = None
        self._scene_cache_size = int(scene_cache_size)
        
        # Grid size used by Sonata / downstream model (tensor expected in batch)
        self.grid_size = torch.tensor([grid_size, grid_size, grid_size], dtype=torch.float32)

        # --- Data Discovery and Splitting ---
        logger.info(f"Building dataset index for split='{split}' from root: {data_root}")
        all_samples = self._discover_action_segments()
        logger.info(f"Discovered {len(all_samples)} total action segments.")

        # Group by sequence for splitting
        seq_groups = {}
        for s in all_samples:
            seq_groups.setdefault(s['sequence_key'], []).append(s)

        keys = list(seq_groups.keys())
        rng = random.Random(self.seed)
        rng.shuffle(keys)
        
        total = len(keys)
        t_end = int(total * self.split_ratio[0])
        
        train_keys = keys[:t_end]
        val_keys = keys[t_end:]

        # Select the keys for the requested split
        if split.lower() == "train":
            target_keys = train_keys
            logger.info(f"Using {len(target_keys)} sequences for 'train' split.")
        elif split.lower() in ("val", "validation", "test"):
            target_keys = val_keys
            logger.info(f"Using {len(target_keys)} sequences for 'val' split.")
        elif split.lower() == "all":
            target_keys = keys
            logger.info(f"Using all {len(target_keys)} sequences (split='all').")
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'all'.")

        # Flatten the list of segments for the chosen split
        self.samples = [sample for k in target_keys for sample in seq_groups.get(k, [])]
        
        logger.info(f"Dataset for split='{split}' initialized with {len(self.samples)} samples (action segments).")

    def _discover_action_segments(self) -> List[Dict]:
        """
        Walks the data root and builds a lightweight index of all action segments.
        """
        root = self.data_root
        all_samples: List[Dict] = []
        
        for room_dir in sorted(root.iterdir()):
            if not room_dir.is_dir():
                continue
            room_type = room_dir.name

            for scene_dir in sorted(room_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                scene_id = scene_dir.name

                # <<< FIX FOR UnboundLocalError >>>
                for seed_dir in sorted(scene_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith('Seed'):
                        continue
                    seed_name = seed_dir.name

                    scene_description_path = seed_dir / 'scene_description.json'
                    scene_pcd = seed_dir / 'scene_point_cloud' / 'scene.pcd'
                    if not scene_pcd.exists():
                        logger.debug(f"No scene.pcd at {scene_pcd}; skipping seed")
                        continue

                    for seq_dir in sorted(seed_dir.iterdir()):
                        if not seq_dir.is_dir() or not seq_dir.name.startswith('sequence'):
                            continue
                        sequence_key = str(seq_dir.relative_to(root))

                        report_path = seq_dir / 'report.json'
                        report = {}
                        if report_path.exists():
                            try:
                                with open(report_path, 'r') as rf:
                                    report = json.load(rf)
                            except Exception as e:
                                logger.warning(f"Failed to read report {report_path}: {e}")

                        prompt_segments = report.get('prompt_segments', [])
                        if not prompt_segments:
                            logger.debug(f"No prompt_segments in {report_path}; skipping sequence {sequence_key}")
                            continue

                        for segment in prompt_segments:
                            start = segment.get('start_frame')
                            end = segment.get('end_frame')
                            if start is None or end is None:
                                continue
                            start_i = int(start)
                            end_i = int(end)
                            if end_i < start_i:
                                continue

                            # 1. Determine frame numbers
                            # 1. Determine frame numbers
                            full_frames = list(range(start_i, end_i + 1))

                            # Apply frame interval
                            step = self.frame_interval if self.frame_interval > 0 else 1
                            chosen = full_frames[::step]
                            
                            # Apply num_frames limit if positive
                            # If self.num_frames_config <= 0, we take all selected frames.
                            if self.num_frames_config > 0:
                                if len(chosen) > self.num_frames_config:
                                    # Uniform sampling to exactly num_frames_config
                                    indices = np.linspace(0, len(chosen) - 1, self.num_frames_config, dtype=int)
                                    chosen = [chosen[i] for i in indices]
                            
                            # 2. Build list of existing pcd files
                            human_files = []
                            existing_frames = []
                            for frame_num in chosen:
                                human_pcd = seq_dir / f"frame_{frame_num:03d}.pcd"
                                if human_pcd.exists():
                                    human_files.append(human_pcd)
                                    existing_frames.append(frame_num)
                                else:
                                    logger.debug(f"Missing human frame {human_pcd}; skipping")

                            if len(human_files) == 0:
                                logger.debug(f"No valid frames for segment {segment.get('prompt')}; skipping")
                                continue

                            # 3. Get middle frame's path for unique ID
                            middle_index = len(human_files) // 2
                            middle_frame_file_path = human_files[middle_index]
                            try:
                                sample_file_name = str(middle_frame_file_path.relative_to(root))
                            except ValueError:
                                sample_file_name = str(middle_frame_file_path)

                            # 4. Create ONE sample for the WHOLE segment
                            action_info = {
                                'action_label': segment.get('label', 'unknown'),
                                'action_prompt': segment.get('prompt', ''),
                                'start_frame': start_i,
                                'end_frame': end_i,
                                'duration': segment.get('num_frames', None),
                                'frame_numbers': existing_frames,
                            }

                            sample = {
                                'human_pcd_files': human_files,
                                'frame_numbers': existing_frames,
                                'scene_pcd_file': scene_pcd,
                                'json_file': report_path if report_path.exists() else None,
                                'file_name': sample_file_name,
                                'room_type': room_type,
                                'scene_id': scene_id,
                                'seed_name': seed_name,
                                'sequence_name': seq_dir.name,
                                'sequence_key': sequence_key,
                                'metadata': report,
                                'action_info': action_info,
                            }
                            all_samples.append(sample)

        return all_samples

    # <<< START NEW/MODIFIED HELPERS (REPLACING data_utils) >>>

    def _normalize_label_name(self, label: str) -> str:
        """
        Normalize label names by removing instance IDs.
        e.g., "Window_1" -> "Window"
        """
        return re.sub(r'_\d+$', '', str(label).strip())

    def _convert_labels_to_indices(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert string labels to class indices with normalization.
        """
        # Create case-insensitive lookup
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
                label_indices.append(0) # Default to "Art"
        
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
        """
        Combine scene point cloud with human points.
        """
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
        This avoids open3d segfaults in multiprocessing.
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
            
            # Find column indices
            try:
                idx_x = fields.index('x')
                idx_y = fields.index('y')
                idx_z = fields.index('z')
                idx_label = fields.index('label')
            except ValueError as e:
                raise ValueError(f"Missing required field in {pcd_path}: {e}")

            # Optional fields
            try: idx_r = fields.index('r')
            except ValueError: idx_r = -1
            try: idx_g = fields.index('g')
            except ValueError: idx_g = -1
            try: idx_b = fields.index('b')
            except ValueError: idx_b = -1
            
            try: idx_nx = fields.index('normal_x')
            except ValueError: idx_nx = -1
            try: idx_ny = fields.index('normal_y')
            except ValueError: idx_ny = -1
            try: idx_nz = fields.index('normal_z')
            except ValueError: idx_nz = -1
            
            has_normals = idx_nx != -1 and idx_ny != -1 and idx_nz != -1
            has_colors = idx_r != -1 and idx_g != -1 and idx_b != -1

            points, colors, normals, labels = [], [], [], []
            
            for line in lines[data_start_line:]:
                parts = line.strip().split()
                if not parts:
                    continue
                
                try:
                    points.append([float(parts[idx_x]), float(parts[idx_y]), float(parts[idx_z])])
                    labels.append(parts[idx_label])

                    if has_colors:
                        colors.append([float(parts[idx_r]), float(parts[idx_g]), float(parts[idx_b])])
                    
                    if has_normals:
                        normals.append([float(parts[idx_nx]), float(parts[idx_ny]), float(parts[idx_nz])])
                
                except (IndexError, ValueError):
                    logger.warning(f"Skipping malformed line in {pcd_path}")
                    continue

            points_np = np.array(points, dtype=np.float32)
            n_points = len(points_np)

            # Handle colors: fallback to gray if not present
            if has_colors:
                colors_np = np.array(colors, dtype=np.float32)
                if colors_np.max() > 1.0: # Normalize if in 0-255 range
                    colors_np /= 255.0
            else:
                colors_np = np.ones((n_points, 3), dtype=np.float32) * 0.5
            
            # Handle normals: fallback to zeros if not present
            if has_normals:
                normals_np = np.array(normals, dtype=np.float32)
            else:
                normals_np = np.zeros((n_points, 3), dtype=np.float32)

            # Convert string labels to indices
            labels_np = self._convert_labels_to_indices(np.array(labels))

            if n_points == 0:
                raise ValueError("Empty point cloud")

            return {
                'points': points_np,
                'colors': colors_np,
                'normals': normals_np,
                'labels': labels_np
            }

        except Exception as e:
            logger.error(f"Error loading custom PCD {pcd_path}: {e}")
            return None

    # <<< END NEW HELPERS >>>


    def _load_combined_pcd(self, frame_sample_info: Dict) -> Optional[Dict]:
        """
        Loads a single combined (scene + human) point cloud for one frame.
        """
        # <<< FIX FOR SEGFAULT (Part 2) >>>
        # Safely initialize the cache *per-process* if it doesn't exist.
        if self._scene_cache is None:
            self._scene_cache = OrderedDict()
            logger.info(f"Initialized scene cache for process {os.getpid()}.")

        # Get scene data from cache
        scene_path = str(frame_sample_info['scene_pcd_file'])
        scene_data = self._scene_cache.get(scene_path)
        
        if scene_data is None:
            # Load using our new O3D-free function
            scene_data = self._load_pcd_custom(scene_path) 
            if not self._validate_pcd_data(scene_data, min_points=100):
                logger.warning(f"Invalid scene PCD data at {scene_path}")
                return None
            
            # Manage cache size (LRU)
            if len(self._scene_cache) >= self._scene_cache_size:
                self._scene_cache.popitem(last=False) # Remove oldest item
            self._scene_cache[scene_path] = scene_data
            logger.debug(f"Cached scene: {scene_path}. Cache size: {len(self._scene_cache)}")
        
        # Load human frame (also using O3D-free function)
        human_path = str(frame_sample_info['human_pcd_file'])
        human_data = self._load_pcd_custom(human_path)
        if not self._validate_pcd_data(human_data, min_points=10):
            logger.warning(f"Invalid human PCD data at {human_path}")
            return None

        # Combine scene and human
        combined = self._combine_scene_and_human(scene_data, human_data)
        #temporay overide loading scene data. load human only
        # combined = human_data 

        # Attach bbox info
        if self.bbox_root is not None:
            frame_number = frame_sample_info['action_info'].get('frame_number')
            combined['human_bboxes'] = self._load_human_bboxes(frame_sample_info, frame_number) if frame_number is not None else {}
            combined['scene_bboxes'] = self._load_scene_bboxes(frame_sample_info)
        else:
            combined['human_bboxes'] = {}
            combined['scene_bboxes'] = {}

        return combined

    def _load_human_bboxes(self, sample_info: Dict, frame_number: int) -> Dict:
        if self.bbox_root is None:
            return {}
        bbox_path = (self.bbox_root / sample_info['room_type'] / sample_info['scene_id'] / sample_info['seed_name'] / sample_info['sequence_name'] / f"frame_{frame_number:03d}.json")
        if not bbox_path.exists():
            return {}
        try:
            with open(bbox_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load human bboxes from {bbox_path}: {e}")
            return {}

    def _load_scene_bboxes(self, sample_info: Dict) -> Dict:
        if self.bbox_root is None:
            return {}
        bbox_path = (self.bbox_root / sample_info['room_type'] / sample_info['scene_id'] / sample_info['seed_name'] / 'scene_point_cloud' / 'scene.json')
        if not bbox_path.exists():
            return {}
        try:
            with open(bbox_path, 'r') as f:
                scene_bboxes = json.load(f)
            # filter floor/ceiling/wall
            filtered = {}
            for obj, bbox in scene_bboxes.items():
                n = obj.lower().split('_')[0]
                if n not in ['floor', 'ceiling', 'wall']:
                    filtered[obj] = bbox
            return filtered
        except Exception as e:
            logger.warning(f"Failed to load scene bboxes from {bbox_path}: {e}")
            return {}

    def _normalize_coordinates(self, coord: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to [0, 1] range like SequenceDataset."""
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
        Returns one full action segment (sequence of frames).
        """
        sample_info = self.samples[idx]
        human_files = sample_info.get('human_pcd_files', [])
        frame_numbers = sample_info.get('frame_numbers', [])

        sequence_coords = []
        sequence_feats = []
        sequence_human_bboxes = []
        scene_bboxes = {}

        for i, hf_path in enumerate(human_files):
            frame_num = frame_numbers[i]
            per_frame_info = {
                'human_pcd_file': hf_path,
                'scene_pcd_file': sample_info.get('scene_pcd_file'),
                'room_type': sample_info.get('room_type'),
                'scene_id': sample_info.get('scene_id'),
                'seed_name': sample_info.get('seed_name'),
                'sequence_name': sample_info.get('sequence_name'),
                'action_info': {'frame_number': frame_num}
            }

            combined = self._load_combined_pcd(per_frame_info)
            if combined is None:
                logger.debug(f"Skipping invalid frame {hf_path} in sample idx={idx}")
                continue

            # <<< FIX FOR SEGFAULT (Part 3) >>>
            # Add .copy() to from_numpy to prevent memory sharing issues
            coord = torch.from_numpy(combined['points'].astype(np.float32).copy())
            color = torch.from_numpy(combined['colors'].astype(np.float32).copy())
            
            if 'normals' in combined and combined['normals'] is not None:
                normal = torch.from_numpy(combined['normals'].astype(np.float32).copy())
            else:
                normal = torch.zeros_like(coord)
            
            coord_norm = self._normalize_coordinates(coord)
            feat = torch.cat([coord_norm, color, normal], dim=1) # Shape [N, 9]

            sequence_coords.append(coord_norm)
            sequence_feats.append(feat)
            sequence_human_bboxes.append(combined.get('human_bboxes', {}))
            if not scene_bboxes:
                scene_bboxes = combined.get('scene_bboxes', {})

        if len(sequence_coords) == 0:
            raise RuntimeError(f"Failed to load any valid frames for sample idx={idx}, sequence_key={sample_info.get('sequence_key')}")

        sample_out = {
            'sequence_coord': sequence_coords,
            'sequence_feat': sequence_feats,
            'grid_size': self.grid_size,
            'action_prompt': sample_info.get('action_info', {}).get('action_prompt', ''),
            
            'num_frames_per_sample': len(sequence_coords),
            
            # metadata
            'file_name': sample_info.get('file_name'),
            'room_type': sample_info.get('room_type'),
            'scene_id': sample_info.get('scene_id'),
            'sequence_name': sample_info.get('sequence_name'),
            'sequence_frame_numbers': frame_numbers,
            'sequence_human_bboxes': sequence_human_bboxes,
            'scene_bboxes': scene_bboxes,
            'action_info': sample_info.get('action_info', {}),
        }

        return sample_out

    @staticmethod
    def collate_fn(batch_samples: List[Dict]) -> Dict:
        """
        A simple collate_fn that batches pre-built sequences.
        """
        # Filter out potential None values if __getitem__ fails softly
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
        grid_size = batch_samples[0].get('grid_size', torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32))

        sequence_coord = [s['sequence_coord'] for s in batch_samples]
        sequence_feat = [s['sequence_feat'] for s in batch_samples]
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
            'action_prompts': action_prompts,
            'metadata': metadata
        }

        return batch