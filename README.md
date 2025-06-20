# CoFT - Co-training with Frequency and Temporal domains

A **Co-training framework** that combines **Frequency-domain** and **Temporal-domain** contrastive learning for enhanced time series representation learning. Built on top of the CA-TCC (Context-Aware Time-Contrastive Clustering) architecture.

## 🌟 Key Features

### 🔀 **Co-training Architecture**
- **Dual-Branch Design**: Temporal branch (CA-TCC) + Frequency branch (FFT-based)
- **Cross-Domain Knowledge Transfer**: Pseudo-labeling between time and frequency domains
- **Feature Flag Control**: Clean A/B testing with `--enable_coft`

### 📊 **Multi-Dataset Support**
- **HAR** (Human Activity Recognition)
- **Sleep** (Sleep Stage Classification)  
- **Epilepsy** (Seizure Detection)
- **pFD** (Fault Detection)

### ⚡ **Performance Optimized**
- **Training Speed**: 77% speedup achieved (359s → 83s)
- **Zero Overhead**: No performance impact when CoFT is disabled
- **Dynamic Sizing**: Automatic tensor dimension handling

## 🚀 Quick Start

### Basic Usage

```bash
# Original CA-TCC (Baseline)
python main.py --selected_dataset HAR

# CoFT Enhanced (Frequency + Temporal Co-training)
python main.py --selected_dataset HAR --enable_coft
```

### Dataset-Specific Examples

```bash
# Human Activity Recognition
python main.py --selected_dataset HAR --enable_coft

# Sleep Stage Classification
python main.py --selected_dataset sleep --enable_coft

# Epilepsy Detection
python main.py --selected_dataset Epilepsy --enable_coft

# Fault Detection
python main.py --selected_dataset pFD --enable_coft
```

### Training Modes

```bash
# Self-supervised learning (default)
python main.py --selected_dataset HAR --training_mode self_supervised --enable_coft

# Supervised contrastive learning
python main.py --selected_dataset HAR --training_mode SupCon --enable_coft

# Fine-tuning with 1% labeled data
python main.py --selected_dataset HAR --training_mode ft_1p --enable_coft
```

## 🏗️ Architecture Overview

### CoFT Components

| Component | Description | File Location |
|-----------|-------------|---------------|
| **Frequency Model** | FFT-based CNN for frequency domain | `models/frequency_model.py` |
| **Frequency Contrastive** | Frequency-domain contrastive learning | `models/frequency_contrastive.py` |
| **Co-training Bridge** | Cross-domain knowledge transfer | `models/coft_cotraining.py` |
| **Hybrid Loss** | Combined temporal + frequency + co-training losses | `models/coft_loss.py` |
| **Enhanced Trainer** | Dual-branch training orchestration | `trainer/trainer_coft.py` |

### Training Pipeline

```
Input Time Series
    ├── Temporal Branch (CA-TCC)
    │   ├── CNN Feature Extraction
    │   ├── Temporal Contrastive Learning
    │   └── Temporal Predictions
    │
    ├── Frequency Branch (CoFT)
    │   ├── FFT Transform
    │   ├── Magnitude/Phase Processing  
    │   ├── Frequency Contrastive Learning
    │   └── Frequency Predictions
    │
    └── Co-training Bridge
        ├── Pseudo-label Generation
        ├── Cross-domain Consistency
        └── Ensemble Predictions
```

## 📈 Performance Results

### Training Speed (HAR Dataset)
- **Baseline**: 359-364 seconds
- **Optimized**: 83-84 seconds (**77% speedup**)
- **Accuracy**: ~76.7% (maintained within 2% of original)

### Expected Loss Patterns
```bash
# Baseline CA-TCC
Epoch 1: Loss 12.14 → Epoch 7: Loss 8.93

# CoFT Enhanced (Higher loss due to additional components)
Epoch 1: Loss 25.39 → Epoch 8: Loss 17.42
```

## 🔧 Advanced Options

### Complete Command Structure

```bash
python main.py \
    --selected_dataset {HAR,sleep,Epilepsy,pFD} \
    --enable_coft \
    --training_mode {self_supervised,SupCon,supervised,ft_1p} \
    --experiment_description "experiment_name" \
    --run_description "run_name" \
    --seed 42 \
    --device cuda:0
```

### Feature Flag Benefits

| Flag State | Behavior | Use Case |
|------------|----------|----------|
| `--enable_coft` | CoFT architecture with dual-branch co-training | Research, enhanced performance |
| *(omitted)* | Original CA-TCC baseline | Baseline comparison, production |

## 📦 Installation

```bash
# Clone repository
git clone <repository_url>
cd CoFT

# Install dependencies
pip install -r requirements.txt

# Prepare datasets (place in data/ directory)
data/
├── HAR/
├── sleep/
├── epilepsy/
└── pFD/
```

## 🧪 Experimental Validation

### A/B Testing Setup
```bash
# Run baseline experiment
python main.py --selected_dataset HAR --experiment_description "baseline_test"

# Run CoFT experiment  
python main.py --selected_dataset HAR --enable_coft --experiment_description "coft_test"

# Results saved to:
experiments_logs/baseline_test/
experiments_logs/coft_test/
```

### Ablation Studies
```bash
# Test individual components
python main.py --selected_dataset HAR --training_mode self_supervised  # Temporal only
python main.py --selected_dataset HAR --training_mode SupCon          # + Supervised contrastive
python main.py --selected_dataset HAR --enable_coft                    # + Frequency co-training
```

## 📚 Technical Details

### Co-training Hypothesis
> "Co-training with pseudo-labels can effectively transfer knowledge between Time and Frequency domains without direct feature fusion."

### Key Innovations
1. **FFT-based Frequency Processing**: Real FFT with magnitude/phase decomposition
2. **Dynamic Component Initialization**: Automatic tensor dimension handling
3. **Cross-domain Pseudo-labeling**: Confidence-based knowledge transfer
4. **Ensemble Predictions**: Multi-strategy fusion for robust inference

### Implementation Highlights
- **Zero Overhead Design**: No performance impact when disabled
- **Backwards Compatibility**: Original CA-TCC behavior preserved
- **Memory Efficient**: Dynamic component loading
- **Error Resilient**: Comprehensive dimension handling

## 🐛 Troubleshooting

### Common Issues
```bash
# If you get dimension mismatch errors
python main.py --selected_dataset HAR  # Test baseline first

# If dataset not found
ls data/  # Check data directory structure

# If CUDA out of memory
python main.py --device cpu  # Use CPU fallback
```

### Debug Mode
```bash
# Add verbose logging
python main.py --selected_dataset HAR --enable_coft | tee debug.log
```

## 🤝 Contributing

1. All new features should follow the **toggleable feature pattern**
2. Maintain backwards compatibility with original CA-TCC
3. Include comprehensive testing for both baseline and enhanced modes
4. Document performance implications

## 📄 License

This project builds upon the CA-TCC framework and includes novel co-training enhancements for frequency-temporal domain knowledge transfer.

---

**Ready to enhance your time series models with frequency-temporal co-training!** 🚀
