import numpy as np
import torch
import sys
import os

# Legacy InfoTS Integration: The following code for InfoTS integration has been disabled
# and is kept for historical reference. It is not active in the current baseline.
# --- BEGIN LEGACY CODE ---
#
# # Add InfoTS to path for imports
# infots_path = os.path.join(os.path.dirname(__file__), '..', 'InfoTS')
# if infots_path not in sys.path:
#     sys.path.append(infots_path)
#
# # Try to import InfoTS augmentations
# try:
#     from InfoTS.models.augmentations import AutoAUG
#     from InfoTS.models.augclass import *
#     INFOTS_AVAILABLE = True
# except ImportError:
#     INFOTS_AVAILABLE = False
# --- END LEGACY CODE ---

def a_normalize(x, mean=0, std=1):
    return (x - mean) / std

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    return np.multiply(x, factor[:, np.newaxis, :])

def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[:, warp]
        else:
            ret[i] = pat
    return ret

def weak_augmentation(sample, config):
    """
    Weak augmentation according to paper:
    - Scaling ratio: 2
    - Jitter: [0, 0.1] after normalization
    """
    return scaling(sample, config.augmentation.jitter_scale_ratio)

def strong_augmentation(sample, config):
    """
    Strong augmentation according to paper:
    - Permutation with dataset-specific max_segments
    - Jitter: [0.1, 1] after normalization  
    """
    return jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

def DataTransform_TD(sample, config):
    """
    Apply transformations to the time domain data.
    """
    # --- BEGIN LEGACY CODE ---
    # The InfoTS check was formerly here. It has been removed.
    # --- END LEGACY CODE ---
    
    # Baseline CoFT augmentations
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    return weak_aug, strong_aug

# --- BEGIN LEGACY CODE ---
# def _apply_infots_augmentation(sample, config):
#     """Apply InfoTS augmentations using AutoAUG."""
#     try:
#         # Get InfoTS parameters from config
#         aug_p1 = getattr(config.augmentation, 'infots_aug_p1', 0.7)
#         aug_p2 = getattr(config.augmentation, 'infots_aug_p2', 0.0)
#         used_augs = getattr(config.augmentation, 'infots_used_augs', None)
#         temperature = getattr(config.augmentation, 'infots_temperature', 1.0)
#
#         # Ensure sample is a torch tensor on the correct device
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         sample_tensor = torch.from_numpy(sample).float().to(device)
#
#         # Initialize InfoTS AutoAUG
#         infots_aug = AutoAUG(aug_p1=aug_p1, aug_p2=aug_p2, used_augs=used_augs, device=device)
#
#         # Apply InfoTS augmentation
#         aug1, aug2 = infots_aug((sample_tensor, temperature))
#
#         # Return augmented data as numpy arrays
#         return aug1.cpu().numpy(), aug2.cpu().numpy()
#
#     except Exception as e:
#         print(f"⚠️  InfoTS augmentation failed: {e}")
#         # Fallback to baseline augmentations
#         weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
#         strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
#         return weak_aug, strong_aug
# --- END LEGACY CODE ---

def DataTransform_FD(sample, config):
    """
    Apply transformations to the frequency domain data.
    """
    # Baseline CoFT augmentations
    aug = jitter(sample, config.augmentation.jitter_ratio_FD)
    return aug, aug