# CoFT Architecture and Methodology

**Author**: Leo  
**Date**: June 27, 2025  
**Status**: Final Architecture Documentation  

## Overview

CoFT (Co-training with Frequency and Temporal domains) is a semi-supervised learning framework that enhances time series classification by jointly learning from both temporal and frequency domain representations. The architecture leverages cross-domain co-training to improve performance in low-label scenarios.

## Architecture Components

### 1. Dual-Branch Architecture

```
Input Time Series
      |
      ├─────────────────────────┐
      ↓                         ↓
Temporal Branch            Frequency Branch
      |                         |
      |                    FFT Transform
      |                         |
  Conv1D Encoder           Conv1D Encoder
      |                         |
  Attention Layer         Attention Layer
      |                         |
  Temporal Features       Frequency Features
      |                         |
      └─────────┬───────────────┘
                |
         Co-training Module
                |
         Final Predictions
```

### 2. Key Components

#### 2.1 Temporal Branch
- **Base Model**: Convolutional encoder with attention mechanism
- **Input**: Raw time series data
- **Output**: Temporal feature representations and predictions

#### 2.2 Frequency Branch
- **Transform**: Real FFT with magnitude and phase separation
- **Architecture**: Parallel convolutional encoder
- **Output**: Frequency domain features and predictions

#### 2.3 Co-training Module
- **Cross-domain Pseudo-labeling**: High-confidence predictions from one domain guide the other
- **Consistency Regularization**: Ensures alignment between domain representations
- **Ensemble Strategy**: Combines predictions from both branches

### 3. Loss Function Design

```python
Total Loss = λ_temporal × L_temporal + 
             λ_frequency × L_frequency + 
             λ_cotraining × L_cotraining + 
             λ_consistency × L_consistency
```

Where:
- **L_temporal**: Temporal domain contrastive/supervised loss
- **L_frequency**: Frequency domain contrastive/supervised loss
- **L_cotraining**: Cross-domain co-training loss
- **L_consistency**: Cross-domain consistency regularization

## Training Pipeline

### Phase 1: Self-Supervised Pre-training
1. **Contrastive Learning**: Learn representations without labels
2. **Augmentation Strategy**: Domain-specific augmentations
3. **Objective**: Maximize agreement between augmented views

### Phase 2: Semi-Supervised Fine-tuning
1. **Limited Labels**: Use 1%, 5%, or 75% of labeled data
2. **Co-training**: Leverage unlabeled data through pseudo-labeling
3. **Progressive Learning**: Gradual confidence threshold adjustment

### Phase 3: Pseudo-label Generation & Refinement
1. **High-confidence Selection**: θ = 0.95 threshold
2. **Cross-validation**: Between temporal and frequency predictions
3. **Iterative Refinement**: Through SupCon training

## Key Innovations

### 1. Ultra-low Co-training Weight
- **Discovery**: λ_cotraining = 0.0001 optimal
- **Rationale**: Prevents label confusion in supervised modes
- **Impact**: +4% accuracy improvement on HAR dataset

### 2. Frequency Domain Processing
```python
# Real FFT with magnitude/phase separation
freq_data = torch.fft.rfft(temporal_data, dim=-1)
magnitude = torch.abs(freq_data)
phase = torch.angle(freq_data)
freq_input = torch.stack([magnitude, phase], dim=1)
```

### 3. Adaptive Ensemble Strategy
- **Low co-training weight**: simple_average optimal
- **High co-training weight**: temporal_only preferred
- **Threshold**: Flip occurs at λ_ct ≈ 0.003-0.005

## Implementation Details

### 1. Data Augmentation
- **Temporal**: Jitter, scaling, window slicing
- **Frequency**: Magnitude perturbation, phase shifting
- **Cross-domain**: Synchronized augmentations

### 2. Model Configuration
```python
# Encoder architecture
conv_channels = [32, 64, 128]
kernel_sizes = [8, 5, 3]
attention_heads = 4
hidden_dim = 128
```

### 3. Training Configuration
```python
# Optimization settings
optimizer = "Adam"
learning_rate = 3e-4
batch_size = 128
epochs = 40 (self-supervised) + 40 (fine-tuning)
```

## Experimental Protocol

### 1. Dataset Preparation
- **Splits**: Train/Validation/Test
- **Label Subsets**: 1%, 5%, 75% for semi-supervised
- **Preprocessing**: Normalization, windowing

### 2. Evaluation Metrics
- **Primary**: Accuracy, Macro F1-score
- **Secondary**: Per-class precision/recall
- **Statistical**: Mean ± std over 5 seeds

### 3. Baseline Comparison
- **CA-TCC**: Contrastive learning baseline
- **Ablations**: Temporal-only, Frequency-only
- **Fair Comparison**: Same data splits and seeds

## Computational Efficiency

### 1. Memory Optimization
- Mixed precision training (FP16)
- Gradient accumulation for large models
- Dynamic batch sizing based on GPU memory

### 2. Training Time
- Self-supervised: ~30 minutes (HAR dataset)
- Fine-tuning: ~15 minutes per percentage
- Total pipeline: ~2 hours for complete run

### 3. Inference Speed
- Single sample: <10ms
- Batch (128): ~50ms
- Real-time capable for most applications

## Conclusion

CoFT's architecture successfully combines temporal and frequency domain learning through a carefully designed co-training framework. The key to its success lies in:
1. Balanced cross-domain interaction (ultra-low co-training weight)
2. Complementary domain representations
3. Robust semi-supervised learning pipeline

This architecture provides a strong foundation for time series classification in low-label scenarios. 