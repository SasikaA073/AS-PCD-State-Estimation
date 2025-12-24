"""
Text Models for 4D Text Grounding Pipeline.

This package provides text encoding functionality using various pretrained models
for converting text descriptions into embeddings suitable for multimodal learning.
"""

from .clip_encoder import (
    CLIPTextEncoder,
    HuggingFaceCLIPTextEncoder,
    create_text_encoder,
    get_available_clip_models,
    estimate_text_encoding_time
)

__all__ = [
    'CLIPTextEncoder',
    'HuggingFaceCLIPTextEncoder', 
    'create_text_encoder',
    'get_available_clip_models',
    'estimate_text_encoding_time'
]