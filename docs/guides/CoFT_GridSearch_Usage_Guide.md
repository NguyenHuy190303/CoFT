# 🚀 CoFT Parameter Grid Search - Usage Guide

## Overview
This Jupyter notebook provides comprehensive parameter optimization for **CoFT (Co-training with Frequency and Temporal domains)** with interactive execution, real-time analysis, and automatic result export.

## 📋 Features

### 🔍 Three Optimization Modes
1. **Diagnostic Mode** (5 minutes)
   - Quick 3-experiment validation
   - Verifies parameter system is working
   - Identifies if optimization is viable

2. **Quick Mode** (30 minutes) 
   - Fast 6-experiment search
   - Tests proven parameter ranges
   - Good for rapid iteration

3. **Optimize Mode** (20-30 minutes)
   - Focused 10-experiment optimization  
   - Targets optimal λ_ct ≤0.005 range
   - Based on breakthrough research insights

### 📊 Advanced Analysis
- Real-time progress tracking
- Interactive visualizations
- Parameter sensitivity analysis
- Correlation analysis
- Automatic result export with ZIP download

## 🚀 Quick Start

### Step 1: Setup
```python
# Run Cell 1: Dependencies
# - Loads required libraries
# - Sets up configuration options
# - Initializes plotting styles
```

### Step 2: Project Configuration
```python
# Run Cell 2: Project Setup
# - Auto-detects CoFT project location
# - Verifies required files
# - Creates results directory
```

### Step 3: Choose Mode & Dataset
```python
# In Cell 4, modify these settings:
DATASET = 'HAR'  # Options: 'HAR', 'sleep', 'Epilepsy', 'pFD'
MODE = 'diagnostic'  # Options: 'diagnostic', 'quick', 'optimize'
```

### Step 4: Execute Grid Search
```python
# Run Cell 4: Execution & Analysis
# - Automatically runs selected mode
# - Shows real-time progress
# - Displays results summary
```

### Step 5: Advanced Analysis
```python
# Run Cell 5: Visualization & Analysis
# - Creates comprehensive visualizations
# - Performs parameter sensitivity analysis
# - Exports results package
```

## 🎯 Optimization Strategy

### Parameter Ranges (Based on Research)
- **λ_cotraining**: 0.0005-0.005 (pattern: lower = higher accuracy)
- **λ_consistency**: Fixed at 0.1 (no significant impact found)
- **ensemble**: simple_average vs temporal_only comparison

### Previous Breakthrough Results
- **Best**: 75.64% with λ_ct=0.0005, simple_average ensemble
- **Pattern**: Ultra-low λ_ct values consistently outperform higher values
- **Insight**: Co-training loss creates "label confusion" when weights too high

## 📁 Project Setup Methods

### Method 1: Google Drive (Recommended)
```python
# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# Place your CoFT project in: /content/drive/MyDrive/CoFT/
# Notebook will auto-detect and use this location
```

### Method 2: Direct Upload
```python
# Upload your CoFT project folder to Colab
# Place in: /content/CoFT/
# Notebook will auto-detect
```

### Method 3: Git Clone
```python
# Clone from your repository
!git clone https://github.com/your-username/CoFT.git /content/CoFT
```

## 🔧 Configuration Options

### Datasets
- `HAR`: Human Activity Recognition
- `sleep`: Sleep stage classification  
- `Epilepsy`: Epilepsy detection
- `pFD`: Fault detection

### Modes
- `diagnostic`: 3 experiments, parameter validation
- `quick`: 6 experiments, fast search
- `optimize`: 10 experiments, focused optimization

### Advanced Settings
```python
MAX_TIMEOUT = 600  # Timeout per experiment (seconds)
SELECTED_DATASET = 'HAR'  # Default dataset
SELECTED_MODE = 'diagnostic'  # Default mode
```

## 📊 Results & Analysis

### Automatic Outputs
1. **CSV Results**: `results.csv` with all experiment data
2. **Best Configuration**: `best_result.txt` with optimal parameters  
3. **Analysis Summary**: `analysis_summary.txt` with statistics
4. **Visualizations**: 4-panel analysis plots
5. **ZIP Package**: Auto-downloadable results bundle

### Visualization Dashboard
- **Scatter Plot**: Accuracy vs λ_cotraining with trend line
- **Bar Chart**: Ensemble method comparison
- **Heatmap**: Parameter combination performance
- **Histogram**: Accuracy distribution with statistics

### Parameter Sensitivity Analysis
- Correlation coefficients
- Group statistics by parameter
- Top 3 configurations
- Performance patterns

## 🚨 Important Notes

### File Modifications
- **Automatic Backup**: Original files backed up before modification
- **Automatic Restore**: Files restored after each experiment
- **Parameter Verification**: 3-point verification system ensures changes applied

### Simulation vs Real Training
```python
# Current: Simulated results for demonstration
# For actual training, uncomment in Cell 3:
# result = subprocess.run([
#     'python', 'main.py',
#     '--training_mode', 'ft_1p', 
#     '--selected_dataset', self.dataset,
#     '--enable_coft'
# ], capture_output=True, text=True, timeout=MAX_TIMEOUT)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Missing Files Error
```
❌ Missing files: ['models/coft_loss.py', 'trainer/trainer_coft.py']
```
**Solution**: Ensure complete CoFT project uploaded/accessible

#### 2. Parameter Verification Failed
```
✓ Parameter verification: 1/3
```
**Solution**: Check file permissions, ensure files are writable

#### 3. Training Timeout
```
❌ Failed or timeout
```
**Solution**: Increase `MAX_TIMEOUT` or use faster GPU

#### 4. No Results to Analyze
```
⚠️ No results file found. Run the grid search first!
```
**Solution**: Execute Cell 4 before Cell 5

### Debug Mode
```python
# Add debug prints in grid search methods
print(f"Debug: File exists: {os.path.exists('models/coft_loss.py')}")
print(f"Debug: Current dir: {os.getcwd()}")
```

## 🎯 Best Practices

### 1. Start with Diagnostic
Always begin with `diagnostic` mode to verify setup:
```python
MODE = 'diagnostic'  # Verify first
```

### 2. Use Quick for Iteration
For rapid parameter exploration:
```python
MODE = 'quick'  # Fast iteration
```

### 3. Optimize for Final Results
For production-ready configurations:
```python
MODE = 'optimize'  # Final optimization
```

### 4. Monitor Resource Usage
- Watch GPU memory usage
- Check timeout settings
- Monitor disk space for results

### 5. Export Results Regularly
- Download ZIP packages after each run
- Save best configurations
- Document breakthrough results

## 🔄 Integration with Existing Workflow

### With Original Shell Script
```bash
# Compare results with optimize_coft.sh
./optimize_coft.sh optimize HAR
```

### With Complete Colab Notebook
```python
# Use optimized parameters in main training
python main.py --training_mode ft_1p --selected_dataset HAR --enable_coft
```

### With Memory System
The notebook integrates with the established memory system:
- Results stored in `.cursor/memory/`
- Best configurations documented
- Breakthrough results tracked

## 📈 Expected Results

### Diagnostic Mode
- Verify 3 different accuracy values
- Confirm parameter system working
- 5-10 minute execution time

### Quick Mode  
- 6 experiments across proven ranges
- Identify top 2-3 configurations
- 20-40 minute execution time

### Optimize Mode
- 10 focused experiments
- Target >75% accuracy
- 30-60 minute execution time

## 🏆 Success Metrics

### Parameter Validation
- ✅ Verification score: 3/3 for all experiments
- ✅ Different accuracy values across experiments
- ✅ No timeout failures

### Performance Targets
- 🎯 Beat previous best: 75.64%
- 🎯 Find optimal λ_ct in 0.0005-0.002 range
- 🎯 Consistent results across multiple runs

### Analysis Quality
- 📊 Clear correlation patterns
- 📊 Statistical significance 
- 📊 Actionable insights for next iteration

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Verify all dependencies installed
3. Ensure complete project files available
4. Review parameter verification scores

**Happy optimizing! 🚀** 