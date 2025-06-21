# CoFT Integration Sprint - Overview

**Assignee**: Leo  
**Status**: In Progress  
**Priority**: High  
**Sprint**: coft_integration  

## Objective
Integrate Frequency branch into CA-TCC logic within CoFT project, controlled by feature flag `--enable_coft`.

## Hypothesis
Co-training with pseudo-labels can effectively transfer knowledge between Time and Frequency domains without direct feature fusion.

## Architecture Plan

### H0: Feature Flag Implementation ✅
- [x] Add CLI argument `--enable_coft` (boolean, default: false)
- [x] Update main training script to handle conditional logic

### H1: Guard Existing CA-TCC Block  
- [ ] Preserve existing TemporalContrastiveBlock
- [ ] Add conditional Classifier_Temporal when `enable_coft=True`

### H2: Frequency Branch Implementation
- [ ] Create FrequencyBranch module (conditional)
- [ ] Implement FFT-based feature extraction
- [ ] Add Classifier_Frequency

### H3: Co-training Bridge
- [ ] Implement cross-domain pseudo-label generation
- [ ] Add knowledge transfer mechanism
- [ ] Create co-training loop logic

### H4: Hybrid Loss Function
- [ ] Define conditional final loss: `L_hybrid` vs `L_original`
- [ ] Implement domain-specific loss components
- [ ] Add cross-domain consistency loss

## Success Criteria
- [ ] Feature flag enables/disables all new functionality cleanly
- [ ] Original CA-TCC performance maintained when `enable_coft=False`
- [ ] Improved performance when `enable_coft=True`
- [ ] Clean A/B testing capability

## Notes
- All new architectural blocks must be wrapped in conditional statements
- Preserve backwards compatibility with existing CA-TCC
- Enable easy performance comparison via CLI flag 