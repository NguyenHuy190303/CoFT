# Models package initialization
"""
CoFT Models Package
==================

This package contains all model definitions for the CoFT project:
- TC: Temporal Contrastive model  
- base_Model: Base model architecture
- FrequencyModel: Frequency domain model for CoFT
- FrequencyContrastive: Frequency contrastive learning
- loss: Loss functions
"""

# Import commonly used models for convenience
from .TC import TC
from .model import base_Model
from .loss import NTXentLoss, SupConLoss

__all__ = [
    'TC',
    'base_Model', 
    'NTXentLoss',
    'SupConLoss'
] 