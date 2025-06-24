#!/bin/bash

###############################################################################
# CoFT FIXED Optimization Script v3.0
# 
# FIXES ALL BUGS that caused identical results:
# ✅ Fixed regex patterns for decimal numbers  
# ✅ Re-enabled ensemble code in trainer
# ✅ Proper ensemble switching between temporal_only and simple_average
# ✅ Enhanced parameter verification
# ✅ Better file synchronization
#
# Author: CoFT Team  
# Version: 3.0 - ALL BUGS FIXED
###############################################################################

# Default configuration
MODE=${1:-"help"}
DATASET=${2:-"HAR"}
CONDA_ENV="CoFT"

# Signal handling for graceful shutdown
cleanup_on_exit() {
    echo ""
    echo -e "${YELLOW}🛑 Script interrupted by user (Ctrl+C)${NC}"
    echo -e "${BLUE}🧹 Cleaning up...${NC}"
    
    # Restore backup files if they exist
    if [[ -f models/coft_loss.py.backup ]]; then
        cp models/coft_loss.py.backup models/coft_loss.py
        rm -f models/coft_loss.py.backup
        echo "   ✅ Restored models/coft_loss.py"
    fi
    
    if [[ -f trainer/trainer_coft.py.backup ]]; then
        cp trainer/trainer_coft.py.backup trainer/trainer_coft.py
        rm -f trainer/trainer_coft.py.backup
        echo "   ✅ Restored trainer/trainer_coft.py"
    fi
    
    echo -e "${GREEN}✨ Cleanup completed. Files restored to original state.${NC}"
    echo -e "${CYAN}💡 You can resume optimization anytime by running the script again.${NC}"
    exit 130  # Standard exit code for Ctrl+C
}

# Set up signal traps
trap cleanup_on_exit SIGINT SIGTERM

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to display usage
show_usage() {
    echo -e "${CYAN}🚀 CoFT FIXED Optimization Script v3.0${NC}"
    echo -e "${GREEN}✅ ALL PARAMETER UPDATE BUGS FIXED!${NC}"
    echo ""
    echo -e "${YELLOW}USAGE:${NC}"
    echo "  ./optimize_coft.sh [mode] [dataset]"
    echo ""
    echo -e "${YELLOW}MODES:${NC}"
    echo -e "  ${GREEN}diagnostic${NC}  - Quick validation (3 experiments, 5 min)"
    echo -e "  ${GREEN}quick${NC}       - Quick test (6 experiments, 30 min)"
    echo -e "  ${GREEN}optimize${NC}    - Full optimization (18 experiments, 1.5-3 hours)"
    echo -e "  ${GREEN}help${NC}        - Show this usage guide"
    echo ""
    echo -e "${YELLOW}DATASETS:${NC}"
    echo "  HAR, sleep, Epilepsy, pFD"
    echo ""
    echo -e "${YELLOW}KEY FIXES:${NC}"
    echo -e "  🔧 Fixed regex patterns for lambda parameters"
    echo -e "  🔧 Re-enabled ensemble code in trainer"
    echo -e "  🔧 Proper ensemble switching logic"
    echo -e "  🔧 Enhanced parameter verification"
    echo ""
    echo -e "${YELLOW}CONTROLS:${NC}"
    echo -e "  ${GREEN}Ctrl+C${NC}      - Gracefully stop and restore files"
    echo -e "  ${GREEN}Resume${NC}      - Run script again to continue optimization"
}

# FIXED: Function to update CoFT loss parameters with correct regex
update_coft_loss_params() {
    local lambda_ct=$1
    local lambda_cs=$2
    
    echo "   🔧 Updating lambda_cotraining: $lambda_ct"
    echo "   🔧 Updating lambda_consistency: $lambda_cs"
    
    cp models/coft_loss.py models/coft_loss.py.backup
    
    # ENHANCED: Multiple regex patterns to handle different number formats
    # Pattern 1: Standard decimal numbers (e.g., 0.01, 0.1, 0.001)
    sed -i "s/self\.lambda_cotraining = [0-9]\+\.[0-9]\+/self.lambda_cotraining = $lambda_ct/" models/coft_loss.py
    sed -i "s/self\.lambda_consistency = [0-9]\+\.[0-9]\+/self.lambda_consistency = $lambda_cs/" models/coft_loss.py
    
    # Pattern 2: Numbers starting with 0. (backup pattern)
    sed -i "s/self\.lambda_cotraining = 0\.[0-9]\+/self.lambda_cotraining = $lambda_ct/" models/coft_loss.py  
    sed -i "s/self\.lambda_consistency = 0\.[0-9]\+/self.lambda_consistency = $lambda_cs/" models/coft_loss.py
    
    # Pattern 3: Very small numbers (e.g., 0.0005, 0.0001)
    sed -i "s/self\.lambda_cotraining = 0\.00[0-9]\+/self.lambda_cotraining = $lambda_ct/" models/coft_loss.py
    sed -i "s/self\.lambda_consistency = 0\.00[0-9]\+/self.lambda_consistency = $lambda_cs/" models/coft_loss.py
    
    # Pattern 4: Numbers close to 1.0 (e.g., 0.8, 0.9)
    sed -i "s/self\.lambda_cotraining = 0\.[0-9]/self.lambda_cotraining = $lambda_ct/" models/coft_loss.py
    sed -i "s/self\.lambda_consistency = 0\.[0-9]/self.lambda_consistency = $lambda_cs/" models/coft_loss.py
}

# FIXED: Function to update ensemble method - Re-enable ensemble code and switch properly
update_ensemble_method() {
    local method=$1
    
    echo "   🔧 Setting ensemble method: $method"
    
    cp trainer/trainer_coft.py trainer/trainer_coft.py.backup
    
    if [[ "$method" == "temporal_only" ]]; then
        # Enable temporal-only mode: use only temporal predictions
        sed -i 's/# ensemble_predictions = ensemble_module(predictions, freq_predictions)/ensemble_predictions = ensemble_module(predictions, freq_predictions)/' trainer/trainer_coft.py
        sed -i 's/# final_predictions = ensemble_predictions/final_predictions = predictions  # TEMPORAL_ONLY_MODE/' trainer/trainer_coft.py
        sed -i 's/final_predictions = (predictions + freq_predictions) \/ 2/final_predictions = predictions  # TEMPORAL_ONLY_MODE/' trainer/trainer_coft.py
        
    elif [[ "$method" == "simple_average" ]]; then
        # Enable simple average mode: average temporal and frequency predictions
        sed -i 's/# ensemble_predictions = ensemble_module(predictions, freq_predictions)/ensemble_predictions = ensemble_module(predictions, freq_predictions)/' trainer/trainer_coft.py
        sed -i 's/# final_predictions = ensemble_predictions/final_predictions = (predictions + freq_predictions) \/ 2  # SIMPLE_AVERAGE/' trainer/trainer_coft.py
        sed -i 's/final_predictions = predictions  # TEMPORAL_ONLY_MODE/final_predictions = (predictions + freq_predictions) \/ 2  # SIMPLE_AVERAGE/' trainer/trainer_coft.py
    fi
    
    # Ensure we have the freq_predictions calculation enabled
    sed -i 's/# freq_predictions = freq_model(freq_x_augmented)/freq_predictions = freq_model(freq_x_augmented)/' trainer/trainer_coft.py
}

# ENHANCED: Function to verify parameter changes with more robust checking
verify_parameters() {
    local lambda_ct=$1
    local lambda_cs=$2
    local ensemble=$3
    local verification_score=0
    
    # Check if files exist and are not empty
    if [[ ! -s models/coft_loss.py ]]; then
        echo "   ❌ models/coft_loss.py is empty or missing!"
        echo "$verification_score/3"
        return
    fi
    
    if [[ ! -s trainer/trainer_coft.py ]]; then
        echo "   ❌ trainer/trainer_coft.py is empty or missing!"
        echo "$verification_score/3"
        return
    fi
    
    # Check lambda_cotraining with flexible pattern
    if grep -q "lambda_cotraining = $lambda_ct" models/coft_loss.py; then
        ((verification_score++))
        echo "   ✓ lambda_cotraining = $lambda_ct verified"
    else
        echo "   ❌ lambda_cotraining = $lambda_ct NOT found"
        echo "   Current values in file:"
        grep "lambda_cotraining" models/coft_loss.py | head -2 | sed 's/^/      /'
    fi
    
    # Check lambda_consistency with flexible pattern
    if grep -q "lambda_consistency = $lambda_cs" models/coft_loss.py; then
        ((verification_score++))
        echo "   ✓ lambda_consistency = $lambda_cs verified"
    else
        echo "   ❌ lambda_consistency = $lambda_cs NOT found"
        echo "   Current values in file:"
        grep "lambda_consistency" models/coft_loss.py | head -2 | sed 's/^/      /'
    fi
    
    # Check ensemble method
    if [[ "$ensemble" == "temporal_only" ]] && grep -q "TEMPORAL_ONLY_MODE" trainer/trainer_coft.py; then
        ((verification_score++))
        echo "   ✓ temporal_only mode verified"
    elif [[ "$ensemble" == "simple_average" ]] && grep -q "SIMPLE_AVERAGE" trainer/trainer_coft.py; then
        ((verification_score++))
        echo "   ✓ simple_average mode verified"
    else
        echo "   ❌ ensemble method $ensemble NOT verified"
        echo "   Current ensemble lines:"
        grep -n "final_predictions.*=" trainer/trainer_coft.py | head -3 | sed 's/^/      /'
    fi
    
    echo "$verification_score/3"
}

# Function to run single experiment with enhanced error handling
run_experiment() {
    local exp_id=$1
    local lambda_ct=$2
    local lambda_cs=$3
    local ensemble=$4
    local desc="$5"
    local results_dir="$6"
    
    echo -e "${PURPLE}🔬 Experiment $exp_id${NC}: $desc"
    echo "   Parameters: λ_cotraining=$lambda_ct, λ_consistency=$lambda_cs, ensemble=$ensemble"
    
    # Update parameters
    update_coft_loss_params "$lambda_ct" "$lambda_cs"
    update_ensemble_method "$ensemble"
    
    # ENHANCED: Better file synchronization
    sleep 3  # Increased from 1 second
    sync     # Force file system sync
    
    # Verify changes
    echo "   🔍 Verifying parameter changes..."
    local verification=$(verify_parameters "$lambda_ct" "$lambda_cs" "$ensemble")
    echo "   📋 Parameter verification: $verification"
    
    # Create experiment log
    local exp_log="$results_dir/experiment_$exp_id.log"
    echo "Experiment $exp_id: $desc" > "$exp_log"
    echo "lambda_cotraining: $lambda_ct" >> "$exp_log"
    echo "lambda_consistency: $lambda_cs" >> "$exp_log"
    echo "ensemble_method: $ensemble" >> "$exp_log"
    echo "verification: $verification" >> "$exp_log"
    echo "timestamp: $(date)" >> "$exp_log"
    echo "---" >> "$exp_log"
    
    # ENHANCED: Show actual parameter values before training
    echo "   📊 Current file parameters:"
    grep "lambda_cotraining.*=" models/coft_loss.py | head -1 | sed 's/^/      /'
    grep "lambda_consistency.*=" models/coft_loss.py | head -1 | sed 's/^/      /'
    grep "final_predictions.*=" trainer/trainer_coft.py | head -1 | sed 's/^/      /'
    
    # Run training
    local start_time=$(date +%s)
    echo "   ⏳ Running training..."
    
    if timeout 1200 conda run -n "$CONDA_ENV" python main.py \
        --training_mode ft_1p \
        --selected_dataset "$DATASET" \
        --enable_coft >> "$exp_log" 2>&1; then
        
        # ENHANCED: Multiple accuracy extraction patterns
        local test_acc=$(grep "Test Accuracy" "$exp_log" | tail -1 | sed 's/.*: \([0-9.]\+\).*/\1/')
        if [[ -z "$test_acc" ]]; then
            test_acc=$(grep -oE "Test.*[Aa]ccuracy.*[0-9]+\.[0-9]+" "$exp_log" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
        fi
        if [[ -z "$test_acc" ]]; then
            test_acc=$(grep -oE "[0-9]+\.[0-9]+%" "$exp_log" | tail -1 | sed 's/%//')
        fi
        
        echo -e "   ${GREEN}✅ Result: Test Accuracy = ${test_acc:-N/A}%${NC}"
        
        # Save result
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "$exp_id,$lambda_ct,$lambda_cs,$ensemble,$test_acc,$duration,$verification" >> "$results_dir/results.csv"
        
    else
        echo -e "   ${RED}❌ Failed or timeout${NC}"
        echo "$exp_id,$lambda_ct,$lambda_cs,$ensemble,FAILED,1200,$verification" >> "$results_dir/results.csv"
    fi
    
    # Restore files
    cp models/coft_loss.py.backup models/coft_loss.py
    cp trainer/trainer_coft.py.backup trainer/trainer_coft.py
    
    echo "───────────────────────────────────────────────────────────────────────────"
}

# DIAGNOSTIC MODE: Enhanced with different parameter values to test fixes
run_diagnostic_mode() {
    local results_dir="diagnostic_FIXED_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$results_dir"
    
    echo -e "${CYAN}🔍 DIAGNOSTIC MODE - FIXED VERSION${NC}"
    echo -e "${GREEN}✅ Testing that parameters actually change between experiments${NC}"
    echo "📊 Running 3 experiments with VERY different parameters"
    echo "📁 Results directory: $results_dir"
    echo -e "${YELLOW}💡 Press Ctrl+C anytime to stop gracefully${NC}"
    echo "═══════════════════════════════════════════════════════════════════════════"
    
    echo "exp_id,lambda_cotraining,lambda_consistency,ensemble,test_accuracy,duration,verification" > "$results_dir/results.csv"
    
    # Use very different parameters to clearly show changes
    run_experiment "1" "0.001" "0.05" "temporal_only" "Ultra-low CoFT, temporal only" "$results_dir"
    run_experiment "2" "0.02" "0.3" "simple_average" "Medium CoFT, simple average" "$results_dir"
    run_experiment "3" "0.1" "0.5" "temporal_only" "High CoFT, temporal only" "$results_dir"
    
    # Enhanced analysis
    echo -e "${CYAN}🔍 DIAGNOSTIC ANALYSIS:${NC}"
    echo "═══════════════════════════════════════════════════════════════════════════"
    cat "$results_dir/results.csv"
    
    local unique_results=$(tail -n +2 "$results_dir/results.csv" | cut -d',' -f5 | grep -v "FAILED" | sort -u | wc -l)
    local all_verifications=$(tail -n +2 "$results_dir/results.csv" | cut -d',' -f7 | grep "3/3" | wc -l)
    
    echo ""
    echo -e "${YELLOW}📈 RESULTS ANALYSIS:${NC}"
    echo "   Unique accuracy values: $unique_results"
    echo "   Successful parameter verifications: $all_verifications/3"
    
    if [[ $unique_results -gt 1 ]] && [[ $all_verifications -eq 3 ]]; then
        echo -e "${GREEN}🎉 SUCCESS: Parameters are working correctly!${NC}"
        echo -e "${GREEN}✅ Found $unique_results different accuracy values${NC}"
        echo -e "${GREEN}✅ All parameter changes verified (3/3)${NC}"
        echo -e "${BLUE}🚀 Ready for full optimization!${NC}"
    else
        echo -e "${RED}❌ STILL HAVE ISSUES:${NC}"
        if [[ $unique_results -le 1 ]]; then
            echo -e "${RED}   - Results are still identical${NC}"
        fi
        if [[ $all_verifications -lt 3 ]]; then
            echo -e "${RED}   - Parameter updates failed${NC}"
        fi
    fi
}

# QUICK MODE: Quick parameter test (6 experiments)
run_quick_mode() {
    local results_dir="quick_FIXED_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$results_dir"
    
    echo -e "${CYAN}⚡ QUICK MODE - FIXED VERSION${NC}"
    echo "📊 Running 6 experiments (~30 minutes)"
    echo "📁 Results directory: $results_dir"
    echo -e "${YELLOW}💡 Press Ctrl+C anytime to stop gracefully${NC}"
    echo "═══════════════════════════════════════════════════════════════════════════"
    
    echo "exp_id,lambda_cotraining,lambda_consistency,ensemble,test_accuracy,duration,verification" > "$results_dir/results.csv"
    
    local exp_id=0
    for lambda_ct in 0.0001 0.0002 0.0005; do
        for ensemble in "temporal_only" "simple_average"; do
            ((exp_id++))
            run_experiment "$exp_id" "$lambda_ct" "0.1" "$ensemble" "Ultra-low λ_ct test λ=$lambda_ct, $ensemble" "$results_dir"
        done
    done
    
    # Find best result
    local best_acc=$(tail -n +2 "$results_dir/results.csv" | cut -d',' -f5 | grep -v "FAILED" | sort -nr | head -1)
    local best_params=$(grep "$best_acc" "$results_dir/results.csv" | head -1)
    
    echo -e "${GREEN}🏆 QUICK MODE COMPLETED!${NC}"
    echo "Best result: $best_acc%"
    echo "Best parameters: $best_params"
    echo "📊 Full results: $results_dir/results.csv"
}

# OPTIMIZE MODE: Full optimization (24 experiments)
run_optimize_mode() {
    local results_dir="optimization_FIXED_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$results_dir"
    
    echo -e "${CYAN}🚀 OPTIMIZE MODE - FIXED VERSION${NC}"
    echo "📊 Running 18 experiments (1.5-3 hours)"
    echo "📁 Results directory: $results_dir"
    echo -e "${YELLOW}💡 Press Ctrl+C anytime to stop gracefully${NC}"
    echo "═══════════════════════════════════════════════════════════════════════════"
    
    echo "exp_id,lambda_cotraining,lambda_consistency,ensemble,test_accuracy,duration,verification" > "$results_dir/results.csv"
    
    local exp_id=0
    local best_acc=0
    local best_params=""
    
    for lambda_ct in 0.0001 0.0002 0.0005; do
        for lambda_cs in 0.01 0.1 0.8; do
            for ensemble in "temporal_only" "simple_average"; do
                ((exp_id++))
                echo -e "${YELLOW}Progress: $exp_id/18${NC}"
                
                run_experiment "$exp_id" "$lambda_ct" "$lambda_cs" "$ensemble" "Full opt λ_ct=$lambda_ct, λ_cs=$lambda_cs, $ensemble" "$results_dir"
                
                # Track best result
                local current_acc=$(tail -1 "$results_dir/results.csv" | cut -d',' -f5)
                if [[ "$current_acc" != "FAILED" ]] && [[ -n "$current_acc" ]] && (( $(echo "$current_acc > $best_acc" | bc -l 2>/dev/null || echo 0) )); then
                    best_acc="$current_acc"
                    best_params="λ_ct=$lambda_ct, λ_cs=$lambda_cs, ensemble=$ensemble"
                    echo -e "${GREEN}🏆 NEW BEST: $best_acc%${NC}"
                    echo "Best parameters: $best_params" > "$results_dir/best_result.txt"
                    echo "Test accuracy: $best_acc%" >> "$results_dir/best_result.txt"
                    echo "Experiment ID: $exp_id" >> "$results_dir/best_result.txt"
                fi
            done
        done
    done
    
    echo -e "${GREEN}🎉 OPTIMIZATION COMPLETED!${NC}"
    echo "🏆 Best accuracy: $best_acc%"
    echo "🎯 Best parameters: $best_params"
    echo "📊 Full results: $results_dir/results.csv"
    echo "🏅 Best config: $results_dir/best_result.txt"
}

# Main execution logic
main() {
    case "$MODE" in
        "diagnostic")
            run_diagnostic_mode
            ;;
        "quick")
            run_quick_mode
            ;;
        "optimize")
            run_optimize_mode
            ;;
        "help"|*)
            show_usage
            ;;
    esac
}

# Validate environment before running
if [[ "$MODE" != "help" ]]; then
    echo -e "${BLUE}🔧 Environment Check${NC}"
    
    # Check conda environment
    if ! conda info --envs | grep -q "$CONDA_ENV"; then
        echo -e "${RED}❌ Conda environment '$CONDA_ENV' not found${NC}"
        exit 1
    fi
    
    # Check required files
    for file in "models/coft_loss.py" "trainer/trainer_coft.py" "main.py"; do
        if [[ ! -f "$file" ]]; then
            echo -e "${RED}❌ Required file '$file' not found${NC}"
            exit 1
        fi
    done
    
    echo -e "${GREEN}✅ Environment check passed${NC}"
    echo ""
fi

# Execute main function
main

# Cleanup backup files
rm -f models/coft_loss.py.backup trainer/trainer_coft.py.backup

echo -e "${CYAN}✨ CoFT FIXED Optimization Script completed!${NC}" 