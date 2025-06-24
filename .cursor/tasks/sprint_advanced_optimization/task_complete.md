# Task: Advanced Multi-Mode CoFT Optimization for 95% Target

## Overview
Enhanced `optimize_coft.sh` with comprehensive multi-mode search to recover the user's previously achieved 95% accuracy by addressing training mode limitations and expanding parameter space.

## Problem Analysis

### Root Cause of 75% vs 95% Gap
1. **Training Mode Issue**: Current optimization stuck in `ft_1p` (supervised) mode with ["label confusion"][[memory:6747281751153640138]]
2. **Limited Parameter Range**: λ_cotraining restricted to 0.0005-0.005 due to supervised mode constraints  
3. **Missing Training Modes**: No testing of `self_supervised` and `SupCon` modes where higher λ_cotraining is viable
4. **Narrow Search Space**: Only 3 parameters optimized vs. full hyperparameter space

## Solution: Advanced Mode Implementation

### Added "advanced" Mode
- **Purpose**: Multi-mode comprehensive search targeting 95% accuracy recovery
- **Experiments**: 72 experiments across 3 training modes (6-8 hours runtime)
- **Strategy**: Systematic exploration of expanded parameter space

### Phase Structure
1. **Phase 1: Self-Supervised Mode (30 experiments)**
   - λ_cotraining: 0.01, 0.05, 0.1, 0.2, 0.5 (much higher than supervised)
   - λ_consistency: 0.1, 0.2, 0.3
   - ensemble: temporal_only, simple_average
   - **Rationale**: No label confusion allows higher co-training weights

2. **Phase 2: SupCon Mode (16 experiments)**  
   - λ_cotraining: 0.005, 0.01, 0.02, 0.05 (intermediate range)
   - λ_consistency: 0.1, 0.2
   - ensemble: temporal_only, simple_average
   - **Rationale**: Supervised contrastive may balance benefits

3. **Phase 3: Refined Supervised (24 experiments)**
   - λ_cotraining: 0.0001, 0.0005, 0.001, 0.002 (refined low range)
   - λ_consistency: 0.05, 0.1, 0.15  
   - ensemble: temporal_only, simple_average
   - **Rationale**: Better understanding of supervised mode limits

### Technical Features
1. **Multi-Mode Support**
   - `run_experiment_with_mode()` function handles different training modes
   - Dynamic training mode selection per experiment
   - Mode-specific parameter validation

2. **Enhanced Analysis**
   - Performance comparison across training modes
   - Success threshold detection (≥90%, ≥80%)
   - Comprehensive mode analysis export

3. **Intelligent Progress Tracking**
   - Phase-specific progress indicators
   - Real-time best result tracking across modes
   - Mode-specific best configuration logging

## Usage
```bash
./optimize_coft.sh advanced HAR
```

## Expected Outcomes
- **High Probability**: Recover 90%+ accuracy in self-supervised mode
- **Target**: Find the 95% configuration through expanded search
- **Insight**: Understand which training mode and parameter ranges achieve peak performance

## Results Structure
```
advanced_YYYYMMDD_HHMMSS/
├── results.csv           # All experiment results with training modes
├── best_result.txt       # Best configuration across all modes
└── mode_analysis.txt     # Top 3 results per training mode
```

## Benefits
- **Comprehensive Coverage**: Tests all major training modes
- **Higher λ_cotraining**: Explores previously impossible parameter ranges
- **Scientific Approach**: Systematic elimination of mode-parameter combinations
- **Target-Focused**: Specifically designed to recover 95% performance

## Status
✅ **Completed** - Ready for comprehensive multi-mode optimization

## Assignee
Leo

## Date
2025-01-21

## Next Steps
If advanced mode achieves ≥90%:
1. Further hyperparameter optimization (temperature, learning rate)
2. Architecture-specific parameter tuning
3. Ensemble method refinement

If results still below 90%:
1. Investigate other hyperparameters (temperature, lr, dropout)
2. Check data preprocessing differences
3. Verify implementation consistency with original 95% setup 