"""
4D Text Grounding Pipeline V1

Complete end-to-end pipeline for 4D text grounding that combines:
- Point cloud feature extraction (Sonata)
- Temporal sequence modeling (TemporalTransformer) 
- Text encoding (CLIP)
- Feature aggregation and alignment

This pipeline can be used for both training and inference.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import logging

# Import model components
import src.models.point_models.sonata as sonata
from ..text_models import create_text_encoder
from ..temporal_models import TemporalTransformer, TEMPORAL_TRANSFORMER_CONFIGS
from ...utils.aggregate import MeanMaxAggregator

logger = logging.getLogger(__name__)


class Pipeline4DGrounding(nn.Module):
    """
    Complete 4D Text Grounding Pipeline.
    
    This pipeline processes temporal point cloud sequences and text descriptions
    to produce aligned feature representations for 4D text grounding tasks.
    
    Architecture:
    Input Batch → Point Feature Extraction → Frame Aggregation → 
    Temporal Modeling → Text Encoding → Output Features
    """
    
    def __init__(self,
                 # Point cloud model config
                 point_model_name: str = "sonata",
                 point_model_repo: str = "facebook/sonata",
                 
                 # Text encoder config
                 text_encoder_type: str = "clip",
                 text_model_name: str = "ViT-B/32",
                 
                 # Temporal transformer config
                 temporal_config: str = "base",  # "base", "large", "video_focused"
                 
                 # Device config
                 device: str = "cuda",
                 # Point feature dimension (embedding dim from point model)
                 point_feature_dim: int = 512,
                 # Aggregated feature dimension (output of aggregator, input to temporal)
                 aggregated_feature_dim: int = 1024,
                 
                 # Training config
                 freeze_point_model: bool = True,
                 freeze_text_encoder: bool = True,
                 use_frame_aggregation: bool = True):
        """
        Initialize the 4D Text Grounding Pipeline.
        
        Args:
            point_model_name: Name of point cloud model
            point_model_repo: Repository for point cloud model
            text_encoder_type: Type of text encoder
            text_model_name: Name of text model
            temporal_config: Configuration for temporal transformer
            device: Device to run models on
            freeze_point_model: Whether to freeze point cloud model
            freeze_text_encoder: Whether to freeze text encoder
        """
        super().__init__()
        
        self.device = device
        self.freeze_point_model = freeze_point_model
        self.freeze_text_encoder = freeze_text_encoder
        self.point_feature_dim = point_feature_dim
        self.aggregated_feature_dim = aggregated_feature_dim
        self.use_frame_aggregation = use_frame_aggregation

        # Store config for later use
        self.config = {
            'point_model_name': point_model_name,
            'point_model_repo': point_model_repo,
            'text_encoder_type': text_encoder_type,
            'text_model_name': text_model_name,
            'temporal_config': temporal_config,
            'device': device,
            'freeze_point_model': freeze_point_model,
            'freeze_text_encoder': freeze_text_encoder,
            'point_feature_dim': point_feature_dim,
            'aggregated_feature_dim': aggregated_feature_dim,
            'use_frame_aggregation': use_frame_aggregation
        }

        # Initialize models
        self._setup_models()

        logger.info(f"Pipeline4DGrounding initialized with config: {self.config}")
    
    def _setup_models(self):
        """Setup all model components."""
        
        #################################################################
        # 1. Point Cloud Model (Sonata)
        #################################################################
        logger.info(f"Loading point cloud model: {self.config['point_model_name']}")
        self.point_model = sonata.load(
            self.config['point_model_name'], 
            repo_id=self.config['point_model_repo']
        ).to(self.device)
        
        if self.freeze_point_model:
            self.point_model.eval()
            for param in self.point_model.parameters():
                param.requires_grad = False
            logger.info("Point cloud model frozen")
        
        #################################################################
        # 2. Text Encoder (CLIP)
        #################################################################
        logger.info(f"Loading text encoder: {self.config['text_encoder_type']}")
        
        self.text_encoder = create_text_encoder(
            encoder_type=self.config['text_encoder_type'],
            model_name=self.config['text_model_name'],
            device=self.device,
            freeze_encoder=self.freeze_text_encoder,
            add_projection=False
        )
        
        #################################################################
        # 3. Feature Aggregator
        #################################################################
        if self.use_frame_aggregation:
            self.frame_aggregator = MeanMaxAggregator(
                input_dim=self.point_feature_dim
            ).to(self.device)
            self.aggregated_feature_dim = self.frame_aggregator.output_dim
        else:
            self.frame_aggregator = None

        #################################################################
        # 4. Temporal Transformer
        #################################################################
        temporal_cfg = TEMPORAL_TRANSFORMER_CONFIGS[self.config['temporal_config']].copy()
        if self.use_frame_aggregation:
            temporal_cfg['input_dim'] = self.aggregated_feature_dim
        else:
            temporal_cfg['input_dim'] = self.point_feature_dim
        self.temporal_transformer = TemporalTransformer(**temporal_cfg).to(self.device)

        logger.info("All models loaded successfully")
    
    
    
    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete pipeline.
        
        Args:
            batch: Batch data from SequenceDataset containing:
                - sequence_coord: List of coordinate sequences
                - sequence_feat: List of feature sequences  
                - action_prompts: List of text descriptions
                - grid_size: Grid size for voxelization
                - batch_size: Number of sequences
                - num_frames_per_sample: Frames per sequence
        
        Returns:
            Tuple of (temporal_features, text_features):
                - temporal_features: [batch_size, d_model] temporal representations
                - text_features: [batch_size, text_dim] text representations
        """
       
        batch_size = batch['batch_size']
        frames_per_sample = batch['num_frames_per_sample']
        
        #################################################################
        # EFFICIENT POINT FEATURE EXTRACTION
        #################################################################
        # Flatten all sequences and frames into single batch
        all_coords, all_feats = [], []
        sequence_frame_mapping = []  # Track (seq_idx, frame_idx) for each flattened frame
        
        for seq_idx in range(batch_size):
            coord_seq = batch['sequence_coord'][seq_idx]
            feat_seq = batch['sequence_feat'][seq_idx]
            num_frames = frames_per_sample[seq_idx].item()
            
            for frame_idx in range(num_frames):
                all_coords.append(coord_seq[frame_idx])
                all_feats.append(feat_seq[frame_idx])
                sequence_frame_mapping.append((seq_idx, frame_idx))
        
        # Single batched forward pass through Sonata
        batched_coords = torch.cat(all_coords, dim=0)
        batched_feats = torch.cat(all_feats, dim=0)
        frame_offsets = torch.cumsum(torch.tensor([c.shape[0] for c in all_coords]), dim=0)
        
        point_data = {
            'coord': batched_coords,
            'feat': batched_feats,
            'grid_size': batch['grid_size'],
            'offset': frame_offsets
        }
        
        # Move to GPU and extract features
        point = sonata.structure.Point(point_data)
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].to(self.device, non_blocking=True)
        
        # Extract point features
        with torch.set_grad_enabled(not self.freeze_point_model):
            output = self.point_model(point)
            point_features = output['feat']  # [total_points, 512]
            batch_indices = output['batch']  # [total_points]
        
    # ...existing code...
        
        if self.use_frame_aggregation:
            #################################################################
            # FRAME-LEVEL AGGREGATION
            #################################################################
            aggregated_frames = []
            # Aggregate each frame separately
            for frame_idx, (seq_idx, f_idx) in enumerate(sequence_frame_mapping):
                frame_mask = (batch_indices == frame_idx)
                frame_points = point_features[frame_mask]
                if frame_points.shape[0] > 0:
                    # Aggregate frame points to fixed size
                    agg_features = self.frame_aggregator(
                        frame_points, 
                        torch.zeros(frame_points.shape[0], dtype=torch.long, device=self.device)
                    )
                    aggregated_frames.append((seq_idx, f_idx, agg_features.squeeze(0)))
            # Group aggregated frames back into sequences
            sequence_features = [[] for _ in range(batch_size)]
            for seq_idx, frame_idx, frame_feat in aggregated_frames:
                sequence_features[seq_idx].append((frame_idx, frame_feat))
            # Sort frames within each sequence and stack
            batch_sequences = []
            for seq_idx in range(batch_size):
                # Sort by frame index and stack
                sorted_frames = sorted(sequence_features[seq_idx], key=lambda x: x[0])
                frame_tensors = [frame_feat for _, frame_feat in sorted_frames]
                if frame_tensors:
                    sequence_tensor = torch.stack(frame_tensors, dim=0)  # [num_frames, 1024]
                    batch_sequences.append(sequence_tensor)
            if not batch_sequences:
                raise RuntimeError("No valid sequences found in batch")
            batch_sequences = torch.stack(batch_sequences, dim=0)  # [batch_size, frames, 1024]
        else:
            #################################################################
            # NO FRAME-LEVEL AGGREGATION: pass all tokens to temporal model
            #################################################################
            # Group all frame points for each sequence
            sequence_features = [[] for _ in range(batch_size)]
            for idx, (seq_idx, frame_idx) in enumerate(sequence_frame_mapping):
                frame_mask = (batch_indices == idx)
                frame_points = point_features[frame_mask]
                if frame_points.shape[0] > 0:
                    sequence_features[seq_idx].append(frame_points)
            # For each sequence, concatenate all frame points (tokens)
            batch_sequences = []
            for seq in sequence_features:
                if seq:
                    tokens = torch.cat(seq, dim=0)  # [num_tokens, feature_dim]
                    batch_sequences.append(tokens)
            if not batch_sequences:
                raise RuntimeError("No valid sequences found in batch")
            # batch_sequences is now a list of [num_tokens, 1024] tensors, one per sequence
        
        #################################################################
        # TEMPORAL TRANSFORMER
        #################################################################
        
        if self.use_frame_aggregation:
            temporal_features = self.temporal_transformer(batch_sequences)  # [batch_size, frames, 1024] -> [batch_size, 512]
        else:
            # Pass a list of [num_tokens, 1024] tensors to the temporal transformer
            temporal_features = self.temporal_transformer(batch_sequences)  # [batch_size, total_tokens, 1024] -> [batch_size, 512]
        
        #################################################################
        # TEXT FEATURES
        #################################################################
        action_prompts = batch['action_prompts']
        text_features = self.text_encoder.encode_text(action_prompts)  # [batch_size, 512]
        
        return temporal_features, text_features
    
    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        
        # Keep frozen models in eval mode
        if self.freeze_point_model:
            self.point_model.eval()
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        
        return self
    
    def get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return self.config.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about model components."""
        info = {
            'point_model': {
                'type': self.config['point_model_name'],
                'frozen': self.freeze_point_model,
                'parameters': sum(p.numel() for p in self.point_model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.point_model.parameters() if p.requires_grad)
            },
            'text_encoder': {
                'type': self.config['text_encoder_type'],
                'model': self.config['text_model_name'],
                'frozen': self.freeze_text_encoder,
                'embedding_dim': self.text_encoder.get_embedding_dim()
            }
        }
        
        if self.frame_aggregator is not None:
            info['frame_aggregator'] = {
                'type': 'MeanMaxAggregator',
                'parameters': sum(p.numel() for p in self.frame_aggregator.parameters())
            }
        
        if self.temporal_transformer is not None:
            info['temporal_transformer'] = {
                'config': self.config['temporal_config'],
                'parameters': sum(p.numel() for p in self.temporal_transformer.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.temporal_transformer.parameters() if p.requires_grad)
            }
        
        return info


def create_pipeline(config: Optional[Dict[str, Any]] = None) -> Pipeline4DGrounding:
    """
    Factory function to create Pipeline4DGrounding.
    
    Args:
        config: Configuration dictionary. If None, uses default config.
        
    Returns:
        Initialized pipeline
    """
    if config is None:
        config = {}
    
    return Pipeline4DGrounding(**config)