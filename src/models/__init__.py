"""
Models for 4D Text Grounding Pipeline.

This package provides various model components including text encoders,
point cloud models, and fusion architectures.
"""

from .text_models import (
    CLIPTextEncoder,
    HuggingFaceCLIPTextEncoder,
    create_text_encoder,
    get_available_clip_models,
    estimate_text_encoding_time
)

__all__ = [
    # Text models
    'CLIPTextEncoder',
    'HuggingFaceCLIPTextEncoder',
    'create_text_encoder', 
    'get_available_clip_models',
    'estimate_text_encoding_time'
]