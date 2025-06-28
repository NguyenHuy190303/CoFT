# 🔧 Infrastructure Setup Guide

## 🚀 **Quick Setup**

### **1. Environment Creation**
```bash
# Create conda environment
conda create -n CoFT python=3.8
conda activate CoFT

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

### **2. GPU Setup (Recommended)**
```bash
# Check GPU availability
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# For CUDA issues, reinstall PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **3. Quick Test**
```bash
# Test basic functionality
./search.sh help

# Quick validation
./search.sh diagnostic HAR
```

## 💻 **Hardware Requirements**

### **Minimum Requirements**
- **GPU**: GTX 1060 / RTX 2060 (6GB+ VRAM)
- **RAM**: 16GB system memory
- **Storage**: 50GB available space
- **CPU**: 4+ cores recommended

### **Recommended Setup**
- **GPU**: RTX 3070 / RTX 4060 (8GB+ VRAM)
- **RAM**: 32GB system memory  
- **Storage**: 100GB+ SSD
- **CPU**: 8+ cores

### **High-Performance Setup**
- **GPU**: RTX 4080 / RTX 4090 / A100 (16GB+ VRAM)
- **RAM**: 64GB+ system memory
- **Storage**: NVMe SSD
- **CPU**: 16+ cores

## 🔧 **Configuration by Hardware**

### **RTX 4060 (8GB) - Entry Level**
```bash
# Memory-efficient settings
python main.py --reduced_batch_size 32 --mixed_precision --gradient_accumulation 4
```

### **RTX 4080 (16GB) - Balanced**
```bash
# Standard settings with performance boost
python main.py --mixed_precision
```

### **RTX 4090 (24GB) - High Performance**
```bash
# Full performance settings
python main.py --batch_size 256
```

## 🐍 **Python Dependencies**

### **Core Requirements**
```txt
torch>=1.9.0
numpy>=1.21.0
scikit-learn>=1.0.0
pandas>=1.3.0
```

### **Optional Optimizations**
```txt
einops  # For efficient tensor operations
tqdm    # Progress bars
wandb   # Experiment tracking (optional)
```

### **Version Compatibility**
- **Python**: 3.8-3.10 (3.8 recommended)
- **PyTorch**: 1.9+ (2.0+ for optimal performance)
- **CUDA**: 11.8+ (for GPU support)

## ⚡ **Performance Optimizations**

### **CUDA Settings**
```bash
# Enable optimal CUDA settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### **Memory Management**
```bash
# For memory-constrained systems
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
```

## 🛠️ **Troubleshooting**

### **Common Issues**

| Issue | Solution |
|-------|----------|
| **CUDA out of memory** | Reduce batch size: `--reduced_batch_size 32` |
| **Import errors** | Reinstall: `pip install -r requirements.txt --force-reinstall` |
| **Slow training** | Enable mixed precision: `--mixed_precision` |
| **Environment conflicts** | Fresh environment: `conda create -n CoFT_new python=3.8` |

### **Emergency Reset**
```bash
# Complete reset
conda remove -n CoFT --all
conda create -n CoFT python=3.8
conda activate CoFT
pip install -r requirements.txt
```

---
*Infrastructure ready → Start optimizing!* 🚀 