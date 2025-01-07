import numpy as np
import torch

def DataTransform(sample, config):
    weak_aug = scaling(sample, sigma=config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    return weak_aug, strong_aug

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    x = x.cpu().numpy()
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)

def permutation(x, max_segments=5, seg_mode="random"):
    """Randomly permutes segments of the input."""
    orig_steps = np.arange(x.shape[2])
    x = x.cpu().numpy()
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
            splits = [np.expand_dims(s, axis=0) for s in splits]  # Ensure homogeneous shape
            perm = np.random.permutation(len(splits))
            warp = np.concatenate([splits[j] for j in perm], axis=1).ravel()
            ret[i] = pat[:, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)