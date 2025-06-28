# CoFT Implementation Guide

**Author**: Leo  
**Date**: June 27, 2025  
**Status**: Final Implementation Documentation  

## Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda env create -f coft_env.yaml
conda activate coft

# Or manual installation
pip install torch==2.4.1 numpy scikit-learn matplotlib
```

### 2. Running the Full Pipeline
```bash
# Run complete CoFT pipeline with 1% labels
python main.py --training_mode full_run --selected_dataset HAR \
               --label_percentage 1 --enable_coft

# Run baseline CA-TCC for comparison
python main.py --training_mode full_run --selected_dataset HAR \
               --label_percentage 1
```

## Detailed Implementation

### 1. Data Preparation

#### Dataset Structure
```
data/
├── HAR/
│   ├── train_1p.pt      # 1% labeled training data
│   ├── train_5p.pt      # 5% labeled training data
│   ├── train_75p.pt     # 75% labeled training data
│   ├── val.pt           # Validation data
│   └── test.pt          # Test data
├── sleep/
├── Epilepsy/
└── pFD/
```

#### Data Format
```python
# Each .pt file contains a dictionary:
{
    'samples': torch.Tensor,  # Shape: (N, channels, length)
    'labels': torch.Tensor    # Shape: (N,)
}
```

### 2. Training Pipeline Stages

The full pipeline consists of 6 stages:

```python
TRAINING_PIPELINE = [
    "self_supervised",           # Stage 1: Contrastive pre-training
    "train_linear_1p",          # Stage 2: Linear evaluation
    "ft_1p",                    # Stage 3: Fine-tuning
    "gen_pseudo_labels",        # Stage 4: Pseudo-label generation
    "SupCon",                   # Stage 5: Supervised contrastive
    "train_linear_SupCon_1p"    # Stage 6: Final linear evaluation
]
```

### 3. Key Command Line Arguments

```bash
# Essential arguments
--training_mode        # Training stage or "full_run" for complete pipeline
--selected_dataset     # Dataset choice: HAR, sleep, Epilepsy, pFD
--label_percentage     # Percentage of labels: 1, 5, or 75
--enable_coft         # Enable CoFT architecture (flag)
--seed                # Random seed for reproducibility (default: 0)

# Memory optimization (for limited GPU memory)
--memory_efficient    # Enable memory optimizations
--reduced_batch_size  # Override batch size (e.g., 32, 64)
--mixed_precision     # Enable FP16 training
--gradient_accumulation # Steps for gradient accumulation
```

### 4. Core Implementation Files

#### 4.1 Model Architecture
```python
# models/frequency_model.py
class FrequencyModel(nn.Module):
    """Frequency domain encoder"""
    
# models/frequency_contrastive.py  
class FrequencyContrastive(nn.Module):
    """Frequency domain contrastive learning"""
    
# models/coft_cotraining.py
class CoTrainingModule(nn.Module):
    """Cross-domain co-training logic"""
    
# models/coft_loss.py
class CoFTHybridLoss(nn.Module):
    """Combined loss function with optimal weights"""
```

#### 4.2 Training Logic
```python
# trainer/trainer_coft.py
class CoFTTrainer:
    """Main training loop for CoFT"""
    
# trainer/trainer_baseline.py
class Trainer:
    """Baseline CA-TCC trainer"""
```

### 5. Configuration Files

#### Dataset Configurations
```python
# config_files/HAR_Configs.py
class Config:
    # Model parameters
    input_channels = 9
    final_out_channels = 128
    num_classes = 6
    
    # Training parameters
    batch_size = 128
    lr = 3e-4
    
    # Augmentation settings
    augmentation = AugmentationConfig()
```

### 6. Running Experiments

#### Single Mode Execution
```bash
# Self-supervised pre-training only
python main.py --training_mode self_supervised --selected_dataset HAR \
               --enable_coft

# Fine-tuning with 5% labels
python main.py --training_mode ft_5p --selected_dataset sleep \
               --enable_coft
```

#### Batch Experiments
```bash
# Run experiments across all datasets and label percentages
for dataset in HAR sleep Epilepsy; do
    for label_pct in 1 5; do
        python main.py --training_mode full_run \
                      --selected_dataset $dataset \
                      --label_percentage $label_pct \
                      --enable_coft
    done
done
```

### 7. Output Structure

```
experiments_logs/
└── HAR_experiments/
    └── test1/
        ├── self_supervised_seed_0/
        │   ├── logs_*.log
        │   └── saved_models/
        ├── train_linear_1p_seed_0/
        ├── ft_1p_seed_0/
        └── ...
```

### 8. Evaluation and Metrics

#### Accessing Results
```python
# Results are logged in experiment directories
# Look for "TEST Performance" in log files

# Example log output:
# TEST Performance: Acc=0.8134 | F1=0.8013
```

#### Extracting Metrics Programmatically
```python
import re

def extract_metrics(log_file):
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract accuracy
    acc_match = re.search(r'Acc=(\d+\.\d+)', content)
    accuracy = float(acc_match.group(1)) if acc_match else None
    
    # Extract F1-score
    f1_match = re.search(r'F1=(\d+\.\d+)', content)
    f1_score = float(f1_match.group(1)) if f1_match else None
    
    return accuracy, f1_score
```

### 9. Reproducing Paper Results

#### For HAR 1% Results
```bash
# CoFT: 81.34% accuracy
python main.py --training_mode full_run --selected_dataset HAR \
               --label_percentage 1 --enable_coft --seed 0

# Baseline: 77.3% accuracy  
python main.py --training_mode full_run --selected_dataset HAR \
               --label_percentage 1 --seed 0
```

#### For Multi-Seed Evaluation
```bash
# Run with 5 seeds and compute mean/std
for seed in 0 1 2 3 4; do
    python main.py --training_mode full_run --selected_dataset HAR \
                   --label_percentage 1 --enable_coft --seed $seed
done
```

### 10. Troubleshooting

#### Common Issues

1. **Out of Memory**
   ```bash
   # Use memory optimization flags
   python main.py ... --memory_efficient --reduced_batch_size 32 \
                     --mixed_precision --gradient_accumulation 4
   ```

2. **File Not Found**
   - Check data file naming: should be `train_1p.pt` not `train_1perc.pt`
   - Ensure data preprocessing has been run

3. **Module Import Errors**
   - Verify all files in models/ and trainer/ directories
   - Check Python path includes project root

## Advanced Usage

### Custom Hyperparameters

Edit `models/coft_loss.py` to modify loss weights:
```python
self.lambda_cotraining = 0.0001    # Optimal for HAR
self.lambda_consistency = 0.15     # May need tuning for other datasets
```

### Adding New Datasets

1. Create config file in `config_files/`
2. Preprocess data to match expected format
3. Place processed files in `data/<dataset_name>/`
4. Add dataset handling in `main.py`

## Performance Tips

1. **GPU Optimization**: Enable TF32 for RTX 30/40 series
2. **Batch Size**: Larger batches generally improve accuracy
3. **Learning Rate**: Scale with effective batch size
4. **Seeds**: Use at least 5 for reliable results

## Citation

If you use this code, please cite:
```bibtex
@article{coft2025,
  title={CoFT: Co-training with Frequency and Temporal Domains 
         for Semi-Supervised Time Series Classification},
  author={Leo},
  year={2025}
}
``` 