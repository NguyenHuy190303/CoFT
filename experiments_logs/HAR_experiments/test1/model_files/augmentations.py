import numpy as np
import torch
import sys
import os

# Add InfoTS to path for imports
infots_path = os.path.join(os.path.dirname(__file__), '..', 'InfoTS')
if infots_path not in sys.path:
    sys.path.append(infots_path)

# Try to import InfoTS augmentations
try:
    from InfoTS.models.augmentations import AutoAUG
    from InfoTS.models.augclass import *
    INFOTS_AVAILABLE = True
except ImportError:
    INFOTS_AVAILABLE = False

def DataTransform(sample, config):
    """
    Apply data augmentations for self-supervised learning.
    
    Args:
        sample: Input data samples
        config: Configuration object with augmentation settings
        
    Returns:
        tuple: (weak_aug, strong_aug) - two augmented versions
    """
    # Check if InfoTS augmentation is enabled and available
    use_infots = getattr(config.augmentation, 'use_infots_augmentation', False)
    
    if use_infots and INFOTS_AVAILABLE:
        # Use InfoTS augmentations
        return _apply_infots_augmentation(sample, config)
    else:
        # Use original CoFT augmentations (default)
        weak_aug = weak_augmentation(sample, config)
        strong_aug = strong_augmentation(sample, config) 
        return weak_aug, strong_aug

def _apply_infots_augmentation(sample, config):
    """Apply InfoTS augmentations using AutoAUG."""
    try:
        # Get InfoTS parameters from config
        aug_p1 = getattr(config.augmentation, 'infots_aug_p1', 0.7)
        aug_p2 = getattr(config.augmentation, 'infots_aug_p2', 0.0)
        used_augs = getattr(config.augmentation, 'infots_used_augs', None)
        temperature = getattr(config.augmentation, 'infots_temperature', 1.0)
        
        # Convert to tensor if needed
        if isinstance(sample, np.ndarray):
            sample_tensor = torch.from_numpy(sample).float()
        else:
            sample_tensor = sample.float()
        
        # Initialize AutoAUG
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        infots_aug = AutoAUG(aug_p1=aug_p1, aug_p2=aug_p2, used_augs=used_augs, device=device)
        
        # Apply InfoTS augmentation
        aug1, aug2 = infots_aug((sample_tensor, temperature))
        
        # Convert back to numpy if original was numpy
        if isinstance(sample, np.ndarray):
            aug1 = aug1.cpu().detach().numpy()
            aug2 = aug2.cpu().detach().numpy()
        
        return aug1, aug2
        
    except Exception as e:
        print(f"⚠️  InfoTS augmentation failed: {e}")
        print("   Falling back to CoFT augmentations")
        # Fallback to original CoFT augmentations
        weak_aug = weak_augmentation(sample, config)
        strong_aug = strong_augmentation(sample, config) 
        return weak_aug, strong_aug

def weak_augmentation(x, config):
    """
    Weak augmentation according to paper:
    - Scaling ratio: 2
    - Jitter: [0, 0.1] after normalization
    """
    # Apply scaling first
    scaled = scaling(x, sigma=config.augmentation.jitter_scale_ratio)
    
    # Apply weak jitter [0, 0.1]
    jitter_strength = np.random.uniform(0.0, 0.1)
    weak_jittered = jitter(scaled, sigma=jitter_strength)
    
    return weak_jittered

def strong_augmentation(x, config):
    """
    Strong augmentation according to paper:
    - Permutation with dataset-specific max_segments
    - Jitter: [0.1, 1] after normalization  
    """
    # Apply permutation first
    permuted = permutation(x, max_segments=config.augmentation.max_seg)
    
    # Apply strong jitter [0.1, 1]
    jitter_strength = np.random.uniform(0.1, 1.0)
    strong_jittered = jitter(permuted, sigma=jitter_strength)
    
    return strong_jittered

def jitter(x, sigma=0.8):
    """Add Gaussian noise with specified sigma"""
    # https://arxiv.org/pdf/1706.00527.pdf
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    jittered = x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    return torch.from_numpy(jittered.astype(np.float32))

def scaling(x, sigma=2.0):
    """
    Apply scaling augmentation according to paper.
    Paper: scaling ratio = 2
    """
    # https://arxiv.org/pdf/1706.00527.pdf
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
        
    # Use loc=2.0 as specified in paper for scaling ratio
    factor = np.random.normal(loc=sigma, scale=0.1, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    result = np.concatenate((ai), axis=1)
    return torch.from_numpy(result.astype(np.float32))

def permutation(x, max_segments=5, seg_mode="random"):
    """
    Apply permutation augmentation with dataset-specific max_segments.
    Paper: M=12 (Epilepsy), M=20 (Sleep-EDF), M=10 (others)
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
        
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments + 1, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 1, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            
            # Fix: Convert splits to list and shuffle, then concatenate
            splits_list = [split for split in splits]
            np.random.shuffle(splits_list)
            warp = np.concatenate(splits_list).ravel()
            ret[i] = pat[:, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret.astype(np.float32))