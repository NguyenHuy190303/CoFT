# 🚀 Google Colab Usage Guide for CoFT Optimization

## 📋 **Setup trong Colab**

### 1. **Upload Files và Install Dependencies**
```python
# Upload toàn bộ project CoFT lên Colab
from google.colab import files
import zipfile
import os

# Cách 1: Upload zip file
uploaded = files.upload()
for filename in uploaded.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('.')

# Cách 2: Clone từ GitHub (nếu có)
!git clone https://github.com/your-repo/CoFT.git
%cd CoFT

# Install dependencies
!pip install -r requirements.txt
```

### 2. **Setup Script Permissions**
```bash
# Trong Colab cell
!chmod +x optimize_coft_colab.sh
!ls -la *.sh  # Verify permissions
```

## 🏃 **Chạy Optimization**

### **Quick Test (5-10 experiments)**
```bash
# Chạy với dataset HAR
!./optimize_coft_colab.sh HAR

# Hoặc với dataset khác
!./optimize_coft_colab.sh Sleep
```

### **Monitor Progress**
```python
# Theo dõi kết quả real-time
import time
import os

def monitor_optimization():
    while True:
        # Check latest results
        if os.path.exists('optimization_results_*/optimization_log.csv'):
            !tail -5 optimization_results_*/optimization_log.csv
        time.sleep(30)  # Check every 30 seconds

# Run in background
monitor_optimization()
```

## 📊 **Xem Kết Quả**

### **Best Results**
```bash
# Xem best parameters
!cat optimization_results_*/best_parameters.txt

# Xem detailed log
!head -10 optimization_results_*/optimization_log.csv

# Xem analysis report
!cat optimization_results_*/colab_analysis_report.txt
```

### **Download Results**
```python
# Download results về máy local
from google.colab import files
import glob

# Download best model
model_files = glob.glob('optimization_results_*/best_model_*.pt')
for model in model_files:
    files.download(model)

# Download logs
log_files = glob.glob('optimization_results_*/optimization_log.csv')
for log in log_files:
    files.download(log)
```

## ⚠️ **Khác Biệt so với Local**

| Aspect | Local Version | Colab Version |
|--------|---------------|---------------|
| **Execution** | `conda run -n CoFT` | Direct `python` |
| **Timeout** | 600s (10 min) | 900s (15 min) |
| **Parameters** | Full grid space | Reduced for speed |
| **Dependencies** | Manual conda | Auto apt-get install |
| **File handling** | Standard sed | Colab-safe sed |

## 🔧 **Troubleshooting**

### **Common Issues & Solutions**

#### 1. **"Permission denied" Error**
```bash
!chmod +x optimize_coft_colab.sh
!chmod 755 optimization_results_*
```

#### 2. **"bc: command not found"**
```bash
!apt-get update -qq && apt-get install -y -qq bc
```

#### 3. **Timeout Issues**
```python
# Reduce parameter space further
# Edit optimize_coft_colab.sh:
LAMBDA_COTRAINING=(0.01 0.02)  # Reduce from (0.01 0.02 0.05)
TEMPORAL_LR=(1e-4)              # Single value only
```

#### 4. **Memory Issues**
```python
# Free up memory
import gc
import torch
gc.collect()
torch.cuda.empty_cache()

# Check memory usage
!nvidia-smi
```

#### 5. **File Not Found Errors**
```bash
# Verify file structure
!find . -name "*.py" | head -10
!ls -la models/
!ls -la trainer/
```

## 📈 **Performance Expectations**

### **Colab Free Tier**
- **Experiments**: 5-7 per session
- **Time per experiment**: 10-15 minutes
- **Total runtime**: ~2 hours max
- **Memory**: 12GB RAM limit

### **Colab Pro**
- **Experiments**: 15-20 per session
- **Time per experiment**: 8-12 minutes
- **Total runtime**: ~4 hours max
- **Memory**: 25GB RAM available

## 🎯 **Optimization Strategy cho Colab**

### **Phase 1: Quick Validation (30 min)**
```bash
# Test 1-2 key parameters only
LAMBDA_COTRAINING=(0.01)
ENSEMBLE_METHODS=("temporal_only")
```

### **Phase 2: Best Parameter Search (1-2 hours)**
```bash
# Expand around best values from Phase 1
LAMBDA_COTRAINING=(0.005 0.01 0.015)  # Around best value
```

### **Phase 3: Fine-tuning (Local)**
```bash
# Use local machine for full grid search
# Transfer best Colab parameters to local optimize_coft.sh
```

## 💡 **Best Practices**

### **1. Resource Management**
- Sử dụng Colab Pro nếu có thể
- Monitor GPU/RAM usage: `!nvidia-smi`
- Cleanup sau mỗi run: `!rm -rf old_results/`

### **2. Data Persistence**
```python
# Mount Google Drive để save results
from google.colab import drive
drive.mount('/content/drive')

# Save results to Drive
!cp -r optimization_results_* /content/drive/My\ Drive/CoFT_Results/
```

### **3. Session Management**
```python
# Save checkpoint after each experiment
import pickle
checkpoint = {
    'best_params': best_params,
    'completed_experiments': experiment_id,
    'results': results_dict
}
with open('checkpoint.pkl', 'wb') as f:
    pickle.dump(checkpoint, f)
```

## 🔄 **Resume từ Checkpoint**
```python
# Load previous results
with open('checkpoint.pkl', 'rb') as f:
    checkpoint = pickle.load(f)
    
# Continue from where you left off
start_experiment_id = checkpoint['completed_experiments']
```

## 📞 **Support**

Nếu gặp vấn đề, check theo thứ tự:
1. ✅ File permissions (`chmod +x`)
2. ✅ Dependencies installed (`bc`, `timeout`)
3. ✅ File structure intact (`main.py`, `models/`, `trainer/`)
4. ✅ Memory available (`nvidia-smi`)
5. ✅ Disk space (`df -h`)

---
**Lưu ý**: Colab version được optimize cho môi trường cloud với reduced parameter space để fit trong thời gian và resource limits của Colab. 