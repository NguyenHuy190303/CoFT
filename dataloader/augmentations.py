import numpy as np
import torch
import sys
import os

# InfoTS-inspired augmentations for CoFT mode
INFOTS_AVAILABLE = True  # Always available since we implement it here

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
            
            # Fix: Convert splits to list and handle variable sizes
            splits_list = [split for split in splits]
            np.random.shuffle(splits_list)  # Shuffle the segments
            warp = np.concatenate(splits_list)
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

# InfoTS-inspired augmentations
def infots_cutout(x, perc=0.1):
    """InfoTS-style cutout augmentation"""
    seq_len = x.shape[2]
    new_x = x.copy()
    win_len = int(perc * seq_len)
    start = np.random.randint(0, seq_len - win_len - 1)
    end = start + win_len
    start = max(0, start)
    end = min(end, seq_len)
    new_x[:, :, start:end] = 0.0
    return new_x

def infots_window_slice(x, reduce_ratio=0.5):
    """InfoTS-style window slice augmentation"""
    target_len = int(np.ceil(reduce_ratio * x.shape[2]))
    if target_len >= x.shape[2]:
        return x
    
    # Different slice positions for each sample
    starts = np.random.randint(0, x.shape[2] - target_len, size=x.shape[0])
    ends = starts + target_len
    
    ret = np.zeros_like(x)
    for i in range(x.shape[0]):
        # Simple linear interpolation for resizing
        sliced = x[i, :, starts[i]:ends[i]]
        # Resize back to original length using numpy's interpolation
        indices = np.linspace(0, sliced.shape[1] - 1, x.shape[2])
        for ch in range(x.shape[1]):
            ret[i, ch, :] = np.interp(indices, np.arange(sliced.shape[1]), sliced[ch, :])
    
    return ret

def infots_subsequence(x):
    """InfoTS-style subsequence augmentation"""
    seq_len = x.shape[2]
    crop_l = np.random.randint(low=2, high=seq_len + 1)
    new_x = x.copy()
    start = np.random.randint(seq_len - crop_l + 1)
    end = start + crop_l
    start = max(0, start)
    end = min(end, seq_len)
    new_x[:, :, :start] = 0.0
    new_x[:, :, end:] = 0.0
    return new_x

def _apply_infots_augmentation(sample, config):
    """Apply InfoTS-inspired augmentations for CoFT mode"""
    try:
        # Get InfoTS parameters from config
        aug_p1 = getattr(config.augmentation, 'infots_aug_p1', 0.7)
        aug_p2 = getattr(config.augmentation, 'infots_aug_p2', 0.7)
        
        print(f"🎨 Applying InfoTS augmentations with p1={aug_p1}, p2={aug_p2}")
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(sample):
            sample_np = sample.cpu().numpy()
        else:
            sample_np = sample
        
        # Apply first augmentation with probability aug_p1
        if np.random.random() < aug_p1:
            # Randomly choose one of the InfoTS-style augmentations
            aug_choice = np.random.choice(['cutout', 'window_slice', 'subsequence', 'jitter', 'scaling'])
            
            if aug_choice == 'cutout':
                aug1 = infots_cutout(sample_np, perc=0.1)
            elif aug_choice == 'window_slice':
                aug1 = infots_window_slice(sample_np, reduce_ratio=0.5)
            elif aug_choice == 'subsequence':
                aug1 = infots_subsequence(sample_np)
            elif aug_choice == 'jitter':
                aug1 = jitter(sample_np, sigma=0.3)
            else:  # scaling
                aug1 = scaling(sample_np, sigma=0.5)
        else:
            aug1 = sample_np.copy()
        
        # Apply second augmentation with probability aug_p2
        if np.random.random() < aug_p2:
            # Use a different augmentation for second view
            aug_choice = np.random.choice(['jitter', 'scaling', 'permutation'])
            
            if aug_choice == 'jitter':
                aug2 = jitter(sample_np, sigma=0.2)
            elif aug_choice == 'scaling':
                aug2 = scaling(sample_np, sigma=0.3)
            else:  # permutation
                aug2 = permutation(sample_np, max_segments=config.augmentation.max_seg)
        else:
            aug2 = sample_np.copy()

        return aug1, aug2

    except Exception as e:
        print(f"⚠️  InfoTS augmentation failed: {e}")
        print("🔄 Falling back to CoFT baseline augmentations")
        
        # Convert tensor to numpy for fallback
        if torch.is_tensor(sample):
            sample_np = sample.cpu().numpy()
        else:
            sample_np = sample
            
        # Fallback to baseline augmentations
        weak_aug = scaling(sample_np, config.augmentation.jitter_scale_ratio)
        strong_aug = jitter(permutation(sample_np, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
        return weak_aug, strong_aug

def DataTransform_TD(sample, config, enable_coft=False):
    """
    Apply transformations to the time domain data.
    
    Args:
        sample: Time domain sample data
        config: Configuration object
        enable_coft: Boolean indicating if CoFT mode is enabled
    
    Returns:
        tuple: (weak_aug, strong_aug) for normal mode or InfoTS augmentations for CoFT mode
    """
    
    # CoFT mode: Use InfoTS-inspired augmentations as default
    if enable_coft:
        print("🎨 CoFT Mode: Using InfoTS-inspired augmentations")
        return _apply_infots_augmentation(sample, config)
    
    # Normal mode: Use traditional strong/weak augmentations
    else:
        print("📊 Normal Mode: Using strong/weak augmentations")
        weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
        strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
        return weak_aug, strong_aug

def DataTransform_FD(sample, config):
    """
    Apply transformations to the frequency domain data.
    """
    # Baseline CoFT augmentations for frequency domain
    aug = jitter(sample, config.augmentation.jitter_ratio_FD)
    return aug, aug