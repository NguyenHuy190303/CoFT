# Changelog

All notable changes to the CoFT project will be documented in this file.

## [2.0.0] - 2024-12-21

### 🚀 Major Release - Unified Optimization Architecture

#### 📁 Clean Architecture Implementation
- **Unified Script System**: Consolidated 5 scripts into 2 main scripts with multiple modes
- **Script Cleanup**: Removed `optimize_coft_BROKEN.sh`, `optimize_coft_FIXED.sh`, `quick_diagnostic_test.sh`, `quick_test_coft.sh`
- **File Cleanup**: Removed old optimization results and temporary files
- **Clean Repository**: Organized structure with only essential files

#### 🎯 New Unified Optimization System
- **`optimize_coft.sh`**: Main script with 4 modes (diagnostic, quick, optimize, help)
- **`optimize_coft_colab.sh`**: Google Colab version with reduced parameter space
- **Mode Selection**: Users can choose execution mode based on time available
- **Color Output**: Enhanced user experience with color-coded progress

#### 🔧 Optimization Modes
- **diagnostic**: Quick 5-minute validation (3 experiments)
- **quick**: Fast parameter search (6 experiments, 30 minutes)  
- **optimize**: Full optimization (24 experiments, 2-4 hours)
- **help**: Interactive usage guide

### 🛠️ Critical Bug Fixes - Parameter Optimization System

#### Fixed Major Optimization Bug
- **CRITICAL**: Fixed `optimize_coft.sh` giving identical results (0.7443%) for all experiments
- **Root Cause**: Parameter update functions were non-functional
  - `update_learning_rates()`: Only logged parameters, never modified files
  - Regex patterns `[0-9.]\+`: Didn't match decimal numbers like `0.01`
  - Ensemble switching: Non-functional due to debugging mode patterns
  - No parameter verification system

#### New Optimization Scripts
- **`optimize_coft_FIXED.sh`**: Fully functional optimization with working parameter updates
- **`quick_diagnostic_test.sh`**: 5-minute validation tool to verify parameter changes work
- **`optimize_coft_colab.sh`**: Google Colab compatible optimization script
- **Parameter verification system**: 3-point validation scores (3/3 = success)

#### Optimization Results Validation
- **Before Fix**: All experiments identical → 0.7443%
- **After Fix**: Different parameters show different results:
  - `λ_cotraining=0.001, temporal_only` → 75.30%
  - `λ_cotraining=0.05, simple_average` → 73.01%
  - `λ_cotraining=0.1, temporal_only` → [varies]

### 🔧 Technical Improvements
- **Fixed Regex Patterns**: `[0-9]*\.[0-9]*` properly matches decimal numbers
- **Working Learning Rate Updates**: Now actually modifies config files
- **Functional Ensemble Switching**: temporal_only and simple_average modes work
- **Parameter Verification**: Real-time validation that changes were applied

### 🌐 Multi-Platform Support
- **Local Environment**: Full conda-based optimization
- **Google Colab**: Cloud-optimized version with reduced parameter space
- **Cross-Platform**: Compatible with both Windows WSL and Linux environments

### 📚 Documentation Updates
- **README.md**: Added comprehensive Parameter Optimization section
- **COLAB_USAGE_GUIDE.md**: Complete Google Colab setup and usage guide
- **Task Documentation**: Detailed problem analysis and solution documentation

### 🧪 Testing & Validation
- **Diagnostic Testing**: Quick validation tool prevents wasted optimization time
- **Parameter Space**: Reduced to essential values for faster validation
- **Results Tracking**: CSV logs, best model saving, progress monitoring

## [1.1.0] - 2024-06-21

### 🌟 Major Features Added

#### CoFT (Co-training with Frequency and Temporal domains) Architecture
- **Dual-Branch Design**: Integrated frequency-domain processing alongside existing temporal branch
- **Cross-Domain Co-training**: Implemented pseudo-labeling and knowledge transfer between time and frequency domains  
- **Feature Flag Control**: Added `--enable_coft` flag for clean A/B testing and backwards compatibility

#### New Components
- **Frequency Model** (`models/frequency_model.py`): FFT-based CNN architecture for frequency domain features
- **Frequency Contrastive** (`models/frequency_contrastive.py`): Frequency-domain contrastive learning module
- **Co-training Bridge** (`models/coft_cotraining.py`): Cross-domain knowledge transfer and ensemble predictions
- **Hybrid Loss Function** (`models/coft_loss.py`): Unified loss computation combining temporal, frequency, and co-training losses
- **Enhanced Trainer** (`trainer/trainer_coft.py`): Dual-branch training orchestration

### 📊 Multi-Dataset Support
- **HAR** (Human Activity Recognition): Enhanced support with CoFT
- **Sleep** (Sleep Stage Classification): Full CoFT integration
- **Epilepsy** (Seizure Detection): CoFT-enabled training
- **pFD** (Fault Detection): Co-training capabilities

### ⚡ Performance Improvements
- **Training Speed**: Achieved 77% speedup (359s → 83s) with CUDA optimizations
- **Memory Efficiency**: Dynamic component loading with zero overhead when disabled
- **Accuracy**: Maintained ~76.7% accuracy (within 2% of original baseline)

### 🔧 Technical Enhancements

#### Dynamic Architecture
- **Dynamic Linear Layer Initialization**: Automatic tensor dimension handling for frequency features
- **FFT Processing**: Real FFT with magnitude/phase decomposition for neural network compatibility
- **Adaptive Component Sizing**: Runtime dimension calculation for robust deployment

#### Code Quality Improvements
- **Fixed Deprecation Warnings**: Updated LogSoftmax usage with explicit dimension parameters
- **Error Resilience**: Comprehensive tensor dimension mismatch handling
- **CLI Argument Parsing**: Improved boolean flag handling with `action='store_true'`

### 🛠️ Bug Fixes
- **RuntimeError Resolution**: Fixed tensor dimension mismatches in frequency model linear layers
- **Feature Tensor Handling**: Proper 3D to 2D feature flattening in co-training module
- **Adapter Network Initialization**: Dynamic cross-domain adapter creation based on actual feature dimensions
- **FFT Input Processing**: Proper 4D to 3D tensor conversion for FFT operations

### 📚 Documentation Updates
- **Comprehensive README**: Complete rewrite with usage examples, architecture overview, and troubleshooting
- **Dataset Usage Guide**: Clear instructions for running with different datasets
- **Performance Benchmarks**: Documented training speed improvements and expected loss patterns
- **API Documentation**: Detailed component descriptions and file locations

### 🧪 Testing & Validation
- **A/B Testing Framework**: Clean comparison between baseline CA-TCC and CoFT enhanced modes
- **Multi-Dataset Validation**: Tested across HAR, Sleep, Epilepsy, and pFD datasets
- **Backwards Compatibility**: Verified no regression in original CA-TCC behavior
- **Error-Free Training**: Resolved all runtime errors and warnings

### 🔄 Backwards Compatibility
- **Zero Breaking Changes**: Original CA-TCC functionality preserved exactly
- **Optional Enhancement**: CoFT features only activate with explicit `--enable_coft` flag
- **Clean Fallback**: Graceful degradation to baseline behavior when flag omitted

## [1.0.0] - Previous Release

### Initial Implementation
- CA-TCC (Context-Aware Time-Contrastive Clustering) baseline
- Temporal contrastive learning
- Multi-dataset support foundation
- Basic training pipeline

---

## Legend
- 🌟 Major Features
- ⚡ Performance  
- 🔧 Technical Improvements
- 🛠️ Bug Fixes
- 📚 Documentation
- 🧪 Testing
- 🔄 Compatibility 