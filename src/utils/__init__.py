"""
Utility modules for the 4D Text Grounding Pipeline.

This package provides various utility functions and classes for data processing,
feature aggregation, and other common operations.
"""

from .data_utils import (
    load_pcd_safe, 
    validate_pcd_data, 
    combine_scene_and_human,
    convert_labels_to_indices,
    normalize_label_name,
    CUSTOM_CLASSES
)

from .aggregate import (
    MeanMaxAggregator,
    TransformerAggregator, 
    FeatureAggregator,
    create_aggregator,
    get_aggregated_dim,
    list_available_methods
)

__all__ = [
    # Data utilities
    'load_pcd_safe',
    'validate_pcd_data', 
    'combine_scene_and_human',
    'convert_labels_to_indices',
    'normalize_label_name',
    'CUSTOM_CLASSES',
    
    # Feature aggregation
    'MeanMaxAggregator',
    'TransformerAggregator',
    'FeatureAggregator',
    'create_aggregator',
    'get_aggregated_dim',
    'list_available_methods'
]