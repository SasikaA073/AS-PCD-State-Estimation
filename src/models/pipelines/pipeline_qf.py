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

from transformers import BertConfig
from src.models.temporal_models.q_former import BertModel as QFormerModel
from transformers import BertTokenizer, BertModel

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
                 text_emb_dim: int = 512,
                 # Optional legacy temporal configuration name (kept for backward compatibility)
                 temporal_config: Optional[str] = None,
                 
                 # Device config
                 device: str = "cuda",
                 # Point feature dimension (embedding dim from point model)
                 point_feature_dim: int = 512,
                 # Aggregated feature dimension (output of aggregator, input to temporal)
                 aggregated_feature_dim: int = 1024,

                 max_frame_tokens: int = 8,
                 num_sequence_frames: int = 11,
                 num_query_tokens: int = 32,
                 
                 qformer_dim: int = 768,
                 qformer_layers: int = 6,
                 qformer_heads: int = 12,
                 qformer_pooling: str = 'mean', # other options 'mean', 'first'

                 add_spatial_pos_embd: bool = False,
                 
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
        self.max_tokens_per_frame = max_frame_tokens
        self.num_frames_per_sequence = num_sequence_frames
        self.text_emb_dim = text_emb_dim
        self.num_query_tokens = num_query_tokens
        self.qformer_dim = qformer_dim
        self.qformer_layers = qformer_layers
        self.qformer_heads = qformer_heads
        self.qformer_pooling = qformer_pooling
        self.text_encoder_type = text_encoder_type
        self.add_spatial_pos_embd = add_spatial_pos_embd


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
        logger.info(f"Loading text encoder: {self.text_encoder_type}")

        if self.text_encoder_type.lower() == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.text_encoder  = BertModel.from_pretrained('bert-base-uncased')
            self.text_encoder.to(self.device)
            self.text_encoder.eval()
        
        else:
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
        # 4. Q former initialization
        #################################################################
        
        
        

        # Q-Former's internal dimension (can be different from encoder_dim)
        # QFORMER_DIM = 768 
        # NUM_QFORMER_LAYERS = 6
        # NUM_ATTENTION_HEADS = 12 # 768 / 64 = 12

        

        qf_config = BertConfig(
            # --- Standard BERT/Q-Former Params ---
            hidden_size=self.qformer_dim,
            num_hidden_layers=self.qformer_layers,
            num_attention_heads=self.qformer_heads,
            intermediate_size=self.qformer_dim * 4, # Standard 4x expansion
            add_cross_attention=True,
            cross_attention_freq=2,
            encoder_width=self.point_feature_dim,  # ADD THIS: width of the encoder features
        )
        self.qf_model = QFormerModel(qf_config,
                             num_frames = self.num_frames_per_sequence,
                             tokens_per_frame = self.max_tokens_per_frame).to(self.device)
        
        # Query tokens: [1, NUM_QUERY_TOKENS, QFORMER_DIM] - will be expanded to batch size
        # Create on the correct device from the start
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_query_tokens, self.qformer_dim, device=self.device)
        )
        
        #################################################################
        # 5. Text Projection layer
        #################################################################
        self.text_projection = nn.Linear(
            qf_config.hidden_size, self.text_emb_dim
        ).to(self.device)

        logger.info("All models loaded successfully")

        #################################################################
        # 6. Spatial embedding generation
        #################################################################
        if self.add_spatial_pos_embd:
            # create an linear layer that takes in 3D coords and outputs self.point_feature_dim
            self.spatial_pos_embd_gen = nn.Linear(3, self.point_feature_dim).to(self.device)
    
    
    
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
       
        batch_size = len(batch['sequence_coord'])
        frames_per_sample = batch['num_frames_per_sample']
        #################################################################
        # EFFICIENT POINT FEATURE EXTRACTION
        #################################################################
        # Flatten all sequences and frames into single batch
        all_coords, all_feats = [], []
        sequence_frame_mapping = []  # Track (seq_idx, frame_idx) for each flattened frame
        # collect centroids per-sequence if requested
        all_centroids = [] if self.add_spatial_pos_embd else None

        for seq_idx in range(batch_size):
            coord_seq = batch['sequence_coord'][seq_idx]
            if self.add_spatial_pos_embd:
                # get centroid of each frame to a tensor of shape [num_frames, 3]
                # Assuming num_frames is 11 for this to work
                centroids = torch.stack([coords.mean(dim=0) for coords in coord_seq])
                all_centroids.append(centroids)
            feat_seq = batch['sequence_feat'][seq_idx]
            num_frames = frames_per_sample[seq_idx].item()
            
            for frame_idx in range(num_frames):
                all_coords.append(coord_seq[frame_idx])
                all_feats.append(feat_seq[frame_idx])
                sequence_frame_mapping.append((seq_idx, frame_idx))
        
        if self.add_spatial_pos_embd and all_centroids:
            #print number of centroids collected per batch
            for i, cent in enumerate(all_centroids):
                logger.info(f"Sequence {i} - Collected {cent.shape[0]} centroids")
            batch_centroids = torch.stack(all_centroids, dim=0).to(self.device)
        
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
        
        #################################################################
        # Temporal Model - Q former
        #################################################################
        
        # Q former pre - processing --> output: [batch_size, frames, tokens_per_frame, feature_dim]
        # Group features by sequence and frame
        sequence_frames_features = [[] for _ in range(batch_size)]
        for flat_frame_idx, (seq_idx, f_idx) in enumerate(sequence_frame_mapping):
            frame_mask = (batch_indices == flat_frame_idx)
            frame_points = point_features[frame_mask]
            # We keep empty frames (shape[0] == 0) to correctly find padded frames later
            # Store (original_frame_index, frame_point_features)
            sequence_frames_features[seq_idx].append((f_idx, frame_points))


        # Sort frames within each sequence by frame index
        for seq_idx in range(batch_size):
            sequence_frames_features[seq_idx].sort(key=lambda x: x[0])


        # Define padding/truncation limits
        B = batch_size
        # Assume constant number of frames per sequence from the batch
        T = self.num_frames_per_sequence
        N_max = self.max_tokens_per_frame
        D_point = self.point_feature_dim


        #################################################################
        # Padding for a tensor of shape [B, T, N_max, D_point]
        #################################################################
        
        # 4D tensor for hierarchical processing: [B, T, N_max, D_point]
        batch_token_tensor = torch.zeros(
            (B, T, N_max, D_point),
            dtype=point_features.dtype,
            device=self.device
        )
        # Token padding mask: [B, T, N_max], True for padded tokens
        batch_token_mask = torch.ones(
            (B, T, N_max),
            dtype=torch.bool,
            device=self.device
        )
        
        for seq_idx, frames in enumerate(sequence_frames_features):
            
            for frame_idx_in_seq, (original_f_idx, frame_tokens) in enumerate(frames):
                # Truncate frames if this sequence is longer than T
                    
                num_real_tokens = frame_tokens.shape[0]

                if num_real_tokens > N_max:
                    logger.warning(
                        f"Frame {original_f_idx} in seq {seq_idx} truncated: "
                        f"{num_real_tokens} tokens > max {N_max}. Consider increasing max_frame_tokens."
                    )
                    frame_tokens = frame_tokens[:N_max]
                    num_real_tokens = N_max
                
                # Insert tokens into 4D tensor
                batch_token_tensor[seq_idx, frame_idx_in_seq, :num_real_tokens] = frame_tokens
                # Mark real tokens
                batch_token_mask[seq_idx, frame_idx_in_seq, :num_real_tokens] = False

                # warn if the padding ratio is higher than 0.75
                # padding_ratio = 1.0 - (num_real_tokens / N_max)
                # if padding_ratio > 0.75:
                #     logger.warning(
                #         f"Frame {original_f_idx} in seq {seq_idx} has high padding ratio: "
                #         f"{padding_ratio:.2f}. Consider decreasing max_frame_tokens."
                #     )

        sequence_features = batch_token_tensor # [B, T, N_max, D_point]

        #################################################################
        # Spatial embedding generation
        #################################################################
        if self.add_spatial_pos_embd:
            # flatten batch_centroids [B,num_frames, 3] --> [B * num_frames, 3]
            batch_centroids = batch_centroids.view(-1, 3)  # [B * num_frames, 3]
            # pass through linear layer
            spatial_pos_embd = self.spatial_pos_embd_gen(batch_centroids).to(self.device) # [B * num_frames, point_feature_dim]
            # split them back to [B, num_frames, point_feature_dim]
            spatial_pos_embd = spatial_pos_embd.view(batch_size,
                                self.num_frames_per_sequence,
                                self.point_feature_dim)  # [B, num_frames, point_feature_dim]
            # expand to [B, num_frames, N_max, point_feature_dim]. copy the frame token to all tokens in the frame
            spatial_pos_embd = spatial_pos_embd.unsqueeze(2).expand(-1, -1, N_max, -1)  # [B, num_frames, N_max, point_feature_dim]
            # add to sequence_features
            sequence_features = sequence_features + spatial_pos_embd
        #################################################################
        # Q former forward pass
        #################################################################
        
        # Expand query tokens to match batch size
        # Note: expand() preserves the device of the original tensor
        batch_query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Ensure all inputs are on the same device
        if batch_query_tokens.device != sequence_features.device:
            batch_query_tokens = batch_query_tokens.to(sequence_features.device)
        
        qf_output = self.qf_model(
            query_embeds=batch_query_tokens,
            encoder_hidden_states=sequence_features,
            encoder_attention_mask=None,
            return_dict=True
        )

        temporal_features = qf_output.last_hidden_state  # [B, NUM_QUERY_TOKENS, QFORMER_DIM]

        #################################################################
        # Text Projection
        #################################################################

        if self.qformer_pooling == 'mean':
            #mean pool feature
            pooled_temp_features = temporal_features.mean(dim=1)  # [B, QFORMER_DIM]
        elif self.qformer_pooling == 'first':
            #first token feature
            pooled_temp_features = temporal_features[:, 0]  # [B, QFORMER_DIM]
        else:
            print(f'Pooling method {self.qformer_pooling} not implemented. falling back to mean pool')
            pooled_temp_features = temporal_features.mean(dim=1)  # [B, QFORMER_DIM]


        proj_temp_features = self.text_projection(pooled_temp_features)
   
            

        #################################################################
        # TEXT FEATURES
        #################################################################
        
        action_prompts = batch['action_prompts']
        if self.text_encoder_type.lower() == 'bert':
            # Tokenize input texts
            inputs = self.tokenizer(action_prompts, padding='longest', return_tensors="pt").to(self.device)
            outputs = self.text_encoder(**inputs)
            last_hidden_state = outputs.last_hidden_state

            # The [CLS] token is always the first token (at index 0)
            # We slice the tensor to get the [CLS] token for *every* sentence in the batch
            # Slicing [:, 0] means:
            #   :   -> Get all items in the batch
            #   0   -> Get the token at index 0 (which is [CLS])
            cls_tokens = last_hidden_state[:, 0]
            text_features = cls_tokens  # [batch_size, hidden_size]
        else:
            text_features = self.text_encoder.encode_text(action_prompts)  # [batch_size, 512]
        
        return proj_temp_features, text_features
    
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
                'embedding_dim': self.text_emb_dim
            }
        }
        
        if self.frame_aggregator is not None:
            info['frame_aggregator'] = {
                'type': 'MeanMaxAggregator',
                'parameters': sum(p.numel() for p in self.frame_aggregator.parameters())
            }
        
        if self.qf_model is not None:
            info['temporal_transformer'] = {
                'config': None,
                'parameters': sum(p.numel() for p in self.qf_model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.qf_model.parameters() if p.requires_grad)
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