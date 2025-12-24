"""
Feature Aggregation Modules for Point Cloud Processing

This module provides various feature aggregation techniques for converting
point-wise features into sample-level representations. Useful for classification,
regression, and other downstream tasks.

Supported methods:
- Mean+Max pooling: Simple but effective statistical pooling
- Transformer-based: Learnable aggregation using attention mechanisms
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Tuple, Optional

__all__ = ['MeanMaxAggregator', 'TransformerAggregator', 'FeatureAggregator']


class MeanMaxAggregator(nn.Module):
    """
    Mean + Max pooling aggregation.
    
    Computes both mean and max pooling across the point dimension and 
    concatenates them to create a 2x feature dimension representation.
    Simple but effective for many tasks.
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize Mean+Max aggregator.
        
        Args:
            input_dim: Input feature dimension (e.g., 512)
        """
        super(MeanMaxAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim * 2  # Mean + Max = 2x input dim
    
    def forward(self, features: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Aggregate point features using mean+max pooling.
        
        Args:
            features: [N, input_dim] point-wise features
            batch_indices: [N] which sample each point belongs to
        
        Returns:
            aggregated_features: [batch_size, output_dim] aggregated features
        """
        batch_size = batch_indices.max().item() + 1
        aggregated_features = []
        
        for sample_idx in range(batch_size):
            # Find all points that belong to this sample
            sample_mask = (batch_indices == sample_idx)
            sample_features = features[sample_mask]  # [num_points_in_sample, input_dim]
            
            if len(sample_features) == 0:
                # Handle edge case: no points in sample
                mean_feat = torch.zeros(self.input_dim, device=features.device)
                max_feat = torch.zeros(self.input_dim, device=features.device)
            else:
                # Mean + Max pooling: [input_dim] + [input_dim] = [2*input_dim]
                mean_feat = sample_features.mean(dim=0)
                max_feat = sample_features.max(dim=0)[0]
            
            aggregated_feat = torch.cat([mean_feat, max_feat], dim=0)
            aggregated_features.append(aggregated_feat)
        
        # Stack to create batch: [batch_size, 2*input_dim]
        return torch.stack(aggregated_features, dim=0)


class TransformerAggregator(nn.Module):
    """
    Transformer-based feature aggregation using CLS token.
    
    Uses a learnable CLS token and transformer encoder to aggregate
    point features through attention mechanisms. More sophisticated
    than pooling methods but requires more computation.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_layers: int = 2, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        max_sequence_length: int = 1000
    ):
        """
        Initialize Transformer aggregator.
        
        Args:
            input_dim: Input feature dimension (e.g., 512)
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_sequence_length: Maximum number of points to process
        """
        super(TransformerAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim  # Same as input dimension
        self.max_sequence_length = max_sequence_length
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, input_dim))
        
        # Positional encoding (simple learned embeddings)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_sequence_length + 1, input_dim))
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize transformer parameters."""
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
    
    def to(self, device):
        """Move model to device and ensure all parameters are on the same device."""
        super().to(device)
        # Explicitly move learnable parameters
        self.cls_token = self.cls_token.to(device)
        self.pos_embedding = self.pos_embedding.to(device)
        return self
    
    def _aggregate_single_sample(self, sample_features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features for a single sample using transformer.
        
        Args:
            sample_features: [num_points, input_dim] features for one sample
        
        Returns:
            aggregated_feat: [input_dim] aggregated feature vector
        """
        num_points = sample_features.shape[0]
        
        # Truncate if sequence is too long
        if num_points > self.max_sequence_length:
            sample_features = sample_features[:self.max_sequence_length]
            num_points = self.max_sequence_length
        
        # Add CLS token: [1, input_dim] + [num_points, input_dim] = [num_points+1, input_dim]
        # Ensure CLS token is on the same device as sample_features
        cls_token = self.cls_token.expand(1, -1).to(sample_features.device)
        tokens = torch.cat([cls_token, sample_features], dim=0)  # [num_points+1, input_dim]
        
        # Add positional encoding
        seq_len = tokens.shape[0]
        pos_emb = self.pos_embedding[:, :seq_len, :].to(sample_features.device)
        tokens = tokens.unsqueeze(0) + pos_emb  # [1, seq_len, input_dim]
        
        # Apply transformer
        transformed = self.transformer(tokens)  # [1, seq_len, input_dim]
        
        # Extract CLS token and apply layer norm
        cls_output = transformed[0, 0, :]  # [input_dim]
        cls_output = self.layer_norm(cls_output)
        
        return cls_output
    
    def forward(self, features: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Aggregate point features using transformer with CLS token.
        
        Args:
            features: [N, input_dim] point-wise features
            batch_indices: [N] which sample each point belongs to
        
        Returns:
            aggregated_features: [batch_size, input_dim] aggregated features
        """
        batch_size = batch_indices.max().item() + 1
        aggregated_features = []
        
        for sample_idx in range(batch_size):
            # Find all points that belong to this sample
            sample_mask = (batch_indices == sample_idx)
            sample_features = features[sample_mask]  # [num_points_in_sample, input_dim]
            
            if len(sample_features) == 0:
                # Handle edge case: no points in sample
                aggregated_feat = torch.zeros(self.input_dim, device=features.device)
            else:
                aggregated_feat = self._aggregate_single_sample(sample_features)
            
            aggregated_features.append(aggregated_feat)
        
        # Stack to create batch: [batch_size, input_dim]
        return torch.stack(aggregated_features, dim=0)


class FeatureAggregator(nn.Module):
    """
    Unified feature aggregator that supports multiple aggregation methods.
    
    A convenient wrapper that allows switching between different aggregation
    techniques using a single interface.
    """
    
    def __init__(
        self, 
        input_dim: int,
        method: str = 'mean_max',
        **kwargs
    ):
        """
        Initialize feature aggregator.
        
        Args:
            input_dim: Input feature dimension
            method: Aggregation method ('mean_max' or 'transformer')
            **kwargs: Additional arguments for specific aggregators
        """
        super(FeatureAggregator, self).__init__()
        self.input_dim = input_dim
        self.method = method
        
        if method == 'mean_max':
            self.aggregator = MeanMaxAggregator(input_dim)
        elif method == 'transformer':
            # Extract transformer-specific arguments
            num_layers = kwargs.get('num_layers', 2)
            num_heads = kwargs.get('num_heads', 8)
            dropout = kwargs.get('dropout', 0.1)
            max_sequence_length = kwargs.get('max_sequence_length', 1000)
            
            self.aggregator = TransformerAggregator(
                input_dim=input_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_sequence_length=max_sequence_length
            )
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        self.output_dim = self.aggregator.output_dim
    
    def forward(self, features: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features using the specified method.
        
        Args:
            features: [N, input_dim] point-wise features
            batch_indices: [N] which sample each point belongs to
        
        Returns:
            aggregated_features: [batch_size, output_dim] aggregated features
        """
        return self.aggregator(features, batch_indices)
    
    def get_output_dim(self) -> int:
        """Get the output feature dimension."""
        return self.output_dim


def create_aggregator(
    input_dim: int,
    method: str = 'mean_max',
    **kwargs
) -> FeatureAggregator:
    """
    Factory function to create feature aggregators.
    
    Args:
        input_dim: Input feature dimension
        method: Aggregation method ('mean_max' or 'transformer')
        **kwargs: Additional arguments for specific aggregators
    
    Returns:
        FeatureAggregator instance
    
    Examples:
        # Create mean+max aggregator
        aggregator = create_aggregator(512, 'mean_max')
        
        # Create transformer aggregator
        aggregator = create_aggregator(
            512, 'transformer', 
            num_layers=3, num_heads=8, dropout=0.1
        )
    """
    return FeatureAggregator(input_dim, method, **kwargs)


# Utility functions for calculating output dimensions
def get_aggregated_dim(input_dim: int, method: str) -> int:
    """
    Get the output dimension for a given aggregation method.
    
    Args:
        input_dim: Input feature dimension
        method: Aggregation method
    
    Returns:
        Output dimension after aggregation
    """
    if method == 'mean_max':
        return input_dim * 2
    elif method == 'transformer':
        return input_dim
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def list_available_methods() -> list:
    """Return list of available aggregation methods."""
    return ['mean_max', 'transformer']