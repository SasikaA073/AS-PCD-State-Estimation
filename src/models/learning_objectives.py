"""
Learning Objectives for 4D Text Grounding Pipeline

This module provides text-temporal alignment loss for training 4D text grounding models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneTextMatchingLoss(nn.Module):
    """
    Scene-Text Matching Loss for global alignment (CrossEntropyLoss version).
    Predicts if scene and text are matched using a small MLP.
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 256, device: str = "cpu"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # 2 output logits for match/mismatch
        ).to(device)
        self.device = device

    def forward(self, temporal_features, text_features, labels):
        # Concatenate features
        fused = torch.cat([temporal_features, text_features], dim=1)
        logits = self.mlp(fused)  # [batch_size, 2]
        loss = F.cross_entropy(logits, labels)
        return loss, logits

class TextActionAlignment(nn.Module):
    """
    Text-Temporal Alignment Loss using contrastive learning.
    
    Aligns temporal point cloud features with text descriptions using InfoNCE loss.
    This encourages the model to learn joint embeddings where matching text-temporal
    pairs are closer than non-matching pairs.
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 normalize: bool = True):
        """
        Args:
            temperature: Temperature parameter for contrastive loss
            normalize: Whether to L2 normalize features before computing similarity
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        
    def forward(self, 
                temporal_features: torch.Tensor,
                text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute text-temporal alignment loss.
        
        Args:
            temporal_features: [batch_size, d_model] temporal representations
            text_features: [batch_size, text_dim] text representations
            
        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = temporal_features.shape[0]
        
        # Ensure both tensors have the same dtype (convert to float32)
        temporal_features = temporal_features.float()
        text_features = text_features.float()
        
        # # Ensure features have same dimension
        # if temporal_features.shape[1] != text_features.shape[1]:
        #     print('using loss level projection')
        #     # Project text features to match temporal feature dimension
        #     if not hasattr(self, 'text_projection'):
        #         self.text_projection = nn.Linear(
        #             text_features.shape[1], 
        #             temporal_features.shape[1]
        #         ).to(text_features.device)
        #     text_features = self.text_projection(text_features)
        
        # L2 normalize features if specified
        if self.normalize:
            temporal_features = F.normalize(temporal_features, dim=1)
            text_features = F.normalize(text_features, dim=1)
        
        # Compute similarity matrix
        # [batch_size, batch_size] where (i,j) = similarity between temporal_i and text_j
        logits = temporal_features @ text_features.T / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=logits.device)
        
        # Symmetric contrastive loss
        # Loss from temporal->text direction
        loss_t2v = F.cross_entropy(logits, labels)
        
        # Loss from text->temporal direction  
        loss_v2t = F.cross_entropy(logits.T, labels)
        
        # Average both directions
        total_loss = (loss_t2v + loss_v2t) / 2
        
        return total_loss
    
    def compute_similarities(self,
                           temporal_features: torch.Tensor,
                           text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix for analysis (without loss computation).
        
        Args:
            temporal_features: [batch_size, d_model] temporal representations
            text_features: [batch_size, text_dim] text representations
            
        Returns:
            similarities: [batch_size, batch_size] similarity matrix
        """
        # Ensure both tensors have the same dtype (convert to float32)
        temporal_features = temporal_features.float()
        text_features = text_features.float()
        
        # Project text features if needed
        if temporal_features.shape[1] != text_features.shape[1]:
            if hasattr(self, 'text_projection'):
                text_features = self.text_projection(text_features)
        
        # Normalize if specified
        if self.normalize:
            temporal_features = F.normalize(temporal_features, dim=1)
            text_features = F.normalize(text_features, dim=1)
        
        # Compute similarity matrix
        similarities = temporal_features @ text_features.T
        
        return similarities