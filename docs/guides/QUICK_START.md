# 🚀 CoFT Quick Start Guide

## 📋 **Essential Commands**

### **Parameter Search (Most Important)**
```bash
# Quick validation (recommended first step)
./search.sh diagnostic HAR

# Quick optimization 
./search.sh quick HAR

# Full optimization (2-8 hours)
./search.sh optimize HAR
```

### **Manual Training Modes**
```bash
# Self-supervised pre-training
python main.py --training_mode self_supervised --selected_dataset HAR --enable_coft

# Linear evaluation with 1% labels
python main.py --training_mode train_linear_1p --selected_dataset HAR --enable_coft

# Fine-tuning with 1% labels
python main.py --training_mode ft_1p --selected_dataset HAR --enable_coft
```

## 🗂️ **Available Datasets**
- **HAR** - Human Activity Recognition (recommended for testing)
- **sleep** - Sleep stage classification  
- **Epilepsy** - Epileptic seizure detection
- **pFD** - Fault detection dataset

## 🎯 **Recommended Workflow**

### **For New Users:**
1. **Test environment**: `./search.sh diagnostic HAR`
2. **If successful**: `./search.sh quick HAR` 
3. **For production**: `./search.sh optimize HAR`

### **For Development:**
1. **Debug single run**: `python main.py --training_mode ft_1p --selected_dataset HAR --enable_coft`
2. **Parameter testing**: Use `search.sh` with different modes
3. **Compare baselines**: Run without `--enable_coft` flag

## ⚡ **Key Features**

- **✅ CoFT Mode**: `--enable_coft` (frequency + temporal co-training)
- **✅ Auto-preparation**: Missing models trained automatically
- **✅ Graceful shutdown**: Ctrl+C anytime to stop safely
- **✅ Progress tracking**: Real-time results and best parameters
- **✅ Multiple datasets**: HAR, sleep, Epilepsy, pFD support

## 🛡️ **Troubleshooting Quick Fixes**

| Issue | Solution |
|-------|----------|
| **Environment not found** | Check conda env: `conda info --envs` |
| **File not found errors** | Run from project root directory |
| **CUDA out of memory** | Use smaller batch size or restart Python |
| **Script interrupted** | Files auto-restored, safe to restart |
| **Identical results** | Check parameter verification (should show 3/3) |

## 📊 **Expected Results**

| Dataset | Baseline | CoFT Optimized | Improvement |
|---------|----------|----------------|-------------|
| **HAR** | ~55% | **76-85%** | +20-30% |
| **Sleep** | ~65% | **75-80%** | +10-15% |
| **Epilepsy** | ~70% | **80-85%** | +10-15% |

## 🔧 **Next Steps**

- **Good results?** → See `optimization/PARAMETER_GUIDE.md` for advanced tuning
- **Issues?** → Check `bugfixes/CRITICAL_FIXES.md` for solutions  
- **Setup problems?** → See `infrastructure/SETUP_GUIDE.md`

---
*Quick Start Complete - Ready to optimize!* 🎉 