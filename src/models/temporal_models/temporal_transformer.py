"""
Temporal Transformer for Sequential Point Cloud Processing

Inspired by SOTA video understanding models:
- TimeSformer (Facebook AI): Divided space-time attention for video understanding
- ViViT (Google): Video Vision Transformer with factorized attention
- X3D (Facebook AI): Efficient video networks with temporal modeling
- Video Swin Transformer: Hierarchical temporal modeling
- MVD (Multi-View Depth): Temporal consistency in 3D

This module implements temporal transformers specifically designed for
processing sequences of aggregated point cloud features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math


class PositionalEncoding3D(nn.Module):
    """
    3D Positional Encoding for temporal sequences.
    Combines temporal and spatial positional information.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_seq_len = max_seq_len
        self._make_pe(max_seq_len)

    def _make_pe(self, seq_len):
        pe = torch.zeros(seq_len, self.d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x with positional encoding: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        if seq_len > self.max_seq_len or not hasattr(self, 'pe') or self.pe.size(1) < seq_len:
            # Dynamically expand positional encoding
            self._make_pe(seq_len)
            self.max_seq_len = seq_len
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)


class TemporalMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention specialized for temporal modeling.
    Inspired by TimeSformer's divided space-time attention.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: Optional attention mask
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        return self.out_proj(attn_output)


class TemporalTransformerBlock(nn.Module):
    """
    Single Temporal Transformer Block with temporal-specific optimizations.
    Inspired by ViViT and Video Swin Transformer architectures.
    """
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        
        # Temporal attention
        self.temporal_attn = TemporalMultiHeadAttention(d_model, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional attention mask
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Temporal self-attention with residual connection
        attn_output = self.temporal_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection  
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for processing sequences of aggregated point cloud features.
    
    Inspired by SOTA video understanding models:
    - Uses temporal positional encoding for sequence modeling
    - Specialized attention mechanisms for temporal relationships
    - Flexible pooling strategies for sequence-to-fixed-size mapping
    
    Designed to process sequences like: [batch_size, num_frames, feature_dim]
    where each frame represents aggregated point cloud features.
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 max_seq_len: int = 32,
                 dropout: float = 0.1,
                 pooling_strategy: str = "cls",  # "cls", "mean", "last", "attention"
                 activation: str = "gelu",
                 final_proj_dim: Optional[int] = None):
        """
        Args:
            input_dim: Input feature dimension (e.g., 1024 from MeanMaxAggregator)
            d_model: Model dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feed-forward network dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            pooling_strategy: How to aggregate sequence into fixed-size representation
            activation: Activation function type
        """
        super().__init__()
        
        # keep original d_model for reference, but allow overriding internal token
        # dimension when final_proj_dim is provided. If final_proj_dim is given,
        # we use it as the transformer token dimension and build all temporal
        # layers around that size so the projection is applied to all tokens at
        # the beginning (as requested).
        self.d_model = d_model
        self.pooling_strategy = pooling_strategy

        # Internal token dimension used by transformer layers. If final_proj_dim
        # is provided, use it as the token dim; otherwise use d_model.
        self.model_dim = final_proj_dim if final_proj_dim is not None else d_model

        # Input projection: project input features directly to the transformer token dim
        self.input_projection = nn.Linear(input_dim, self.model_dim)

        # Add CLS token for classification-style pooling
        if pooling_strategy == "cls":
            # CLS token must match the internal token dimension
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.model_dim))

        # Positional encoding (uses internal token dim)
        self.pos_encoding = PositionalEncoding3D(self.model_dim, max_seq_len, dropout)

        # Transformer layers (built with the internal token dim)
        self.transformer_layers = nn.ModuleList([
            TemporalTransformerBlock(
                d_model=self.model_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            ) for _ in range(num_layers)
        ])

        # Attention pooling layer (if using attention pooling)
        if pooling_strategy == "attention":
            # MultiheadAttention expects embed_dim == self.model_dim
            self.attention_pool = nn.MultiheadAttention(self.model_dim, num_heads, dropout=dropout)
            self.attention_query = nn.Parameter(torch.randn(1, 1, self.model_dim))

    # Note: no output projection is applied here — tokens are left in the
    # internal transformer dimension (`self.model_dim`). The pooled output
    # will be returned in that same dimension.
        # keep final_proj_dim for external reference, but we don't create an
        # extra final MLP since projection (when requested) is handled up-front
        self.final_proj_dim = final_proj_dim
        self.final_mlp = None

        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for variable-length sequences.
        
        Args:
            x: [batch_size, max_seq_len, d_model]
            lengths: [batch_size] - actual sequence lengths
        Returns:
            mask: [batch_size, max_seq_len] - 1 for valid positions, 0 for padding
        """
        batch_size, max_len = x.size(0), x.size(1)
        mask = torch.arange(max_len, device=x.device).expand(
            batch_size, max_len) < lengths.unsqueeze(1)
        return mask
    
    def forward(self, 
                x: Union[torch.Tensor, list],
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through temporal transformer.
        
        Args:
            x: [batch_size, seq_len, input_dim] or list of [seq_len, input_dim] tensors
            lengths: [batch_size] - actual sequence lengths (optional)
        Returns:
            output: [batch_size, d_model] - aggregated temporal representation
        """
        if isinstance(x, list):
            # Variable-length input: process each sequence independently
            outputs = []
            for seq in x:
                seq = self.input_projection(seq)  # [seq_len, d_model]
                # Add batch dimension
                seq = seq.unsqueeze(0)  # [1, seq_len, d_model]
                # Add CLS token if needed
                if self.pooling_strategy == "cls":
                    cls_token = self.cls_token.expand(1, -1, -1)
                    seq = torch.cat([cls_token, seq], dim=1)
                # Positional encoding
                seq = self.pos_encoding(seq)
                # No padding mask needed (single sequence)
                for layer in self.transformer_layers:
                    seq = layer(seq)
                # Pooling
                if self.pooling_strategy == "cls":
                    out = seq[:, 0, :]
                elif self.pooling_strategy == "mean":
                    out = seq.mean(dim=1)
                elif self.pooling_strategy == "last":
                    out = seq[:, -1, :]
                elif self.pooling_strategy == "attention":
                    query = self.attention_query.expand(1, -1, -1)
                    attn_output, _ = self.attention_pool(
                        query.transpose(0, 1), 
                        seq.transpose(0, 1), 
                        seq.transpose(0, 1)
                    )
                    out = attn_output.transpose(0, 1).squeeze(1)
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
                # no output projection; out is already in internal token dim
                # Apply optional final MLP projection
                if self.final_mlp is not None:
                    out = self.final_mlp(out)
                outputs.append(out.squeeze(0))
            return torch.stack(outputs, dim=0)
        else:
            # Standard batch tensor input
            batch_size, seq_len, _ = x.size()
            x = self.input_projection(x)  # [batch_size, seq_len, d_model]
            if self.pooling_strategy == "cls":
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, seq_len+1, d_model]
                seq_len += 1
            x = self.pos_encoding(x)
            attn_mask = None
            if lengths is not None:
                if self.pooling_strategy == "cls":
                    lengths = lengths + 1
                padding_mask = self.create_padding_mask(x, lengths)
                attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            for layer in self.transformer_layers:
                x = layer(x, attn_mask)
            if self.pooling_strategy == "cls":
                output = x[:, 0, :]
            elif self.pooling_strategy == "mean":
                if lengths is not None:
                    mask = self.create_padding_mask(x, lengths).unsqueeze(-1)
                    x_masked = x * mask
                    output = x_masked.sum(dim=1) / lengths.unsqueeze(-1).float()
                else:
                    output = x.mean(dim=1)
            elif self.pooling_strategy == "last":
                if lengths is not None:
                    idx = (lengths - 1).clamp(min=0)
                    output = x[torch.arange(batch_size), idx]
                else:
                    output = x[:, -1, :]
            elif self.pooling_strategy == "attention":
                query = self.attention_query.expand(batch_size, -1, -1)
                attn_output, _ = self.attention_pool(
                    query.transpose(0, 1), 
                    x.transpose(0, 1), 
                    x.transpose(0, 1)
                )
                output = attn_output.transpose(0, 1).squeeze(1)
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            # no output projection; output is already in internal token dim
            # Apply optional final MLP projection
            if self.final_mlp is not None:
                output = self.final_mlp(output)
            return output
    
    def get_temporal_attention_weights(self, 
                                     x: torch.Tensor,
                                     layer_idx: int = -1) -> torch.Tensor:
        """
        Extract attention weights from a specific layer for visualization.
        
        Args:
            x: [batch_size, seq_len, input_dim]
            layer_idx: Which layer to extract weights from (-1 for last layer)
        Returns:
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # This would require modifying the forward pass to return attention weights
        # Implementation depends on specific visualization needs
        pass


class SpatioTemporalTransformer(nn.Module):
    """
    Spatio-Temporal Transformer that combines spatial and temporal modeling.
    Inspired by TimeSformer's divided space-time attention approach.
    
    This can be used for more advanced temporal modeling that considers
    both spatial relationships within frames and temporal relationships across frames.
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 max_seq_len: int = 32,
                 dropout: float = 0.1,
                 pooling_strategy: str = "cls",
                 activation: str = "gelu"):
        super().__init__()
        
        # For now, this is a placeholder for future development
        # Could implement divided space-time attention or other advanced architectures
        self.temporal_transformer = TemporalTransformer(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pooling_strategy=pooling_strategy,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.temporal_transformer(x, lengths)


def create_temporal_transformer(config: dict) -> TemporalTransformer:
    """
    Factory function to create temporal transformer from configuration.
    
    Args:
        config: Dictionary with transformer configuration
    Returns:
        TemporalTransformer instance
    """
    return TemporalTransformer(**config)


# Predefined configurations inspired by SOTA models
TEMPORAL_TRANSFORMER_CONFIGS = {
    "small": {
        "d_model": 512,
        "num_heads": 4,
        "num_layers": 3,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "pooling_strategy": "cls",
        "final_proj_dim": None
    },
    "base": {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "pooling_strategy": "cls",
        "final_proj_dim": None
    },
    "large": {
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 12,
        "dim_feedforward": 3072,
        "dropout": 0.1,
        "pooling_strategy": "cls",
        "final_proj_dim": None
    },
    "1024_text": {
        "d_model": 512,
        "num_heads": 4,
        "num_layers": 3,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "pooling_strategy": "cls",
        "final_proj_dim": 1024
    }
}