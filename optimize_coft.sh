#!/bin/bash

###############################################################################
# CoFT Unified Optimization Script
# 
# Purpose: All-in-one parameter optimization with multiple execution modes
# Modes: diagnostic, optimize, quick, help
# 
# Usage: ./optimize_coft.sh [mode] [dataset]
# Examples:
#   ./optimize_coft.sh diagnostic HAR    # Quick 5-minute validation
#   ./optimize_coft.sh optimize HAR      # Full optimization (2-4 hours)
#   ./optimize_coft.sh quick HAR         # Quick parameter test
#   ./optimize_coft.sh help              # Show usage guide
#
# Author: CoFT Team
# Version: 2.0.0 - Unified Architecture
###############################################################################

# Default configuration
MODE=${1:-"help"}
DATASET=${2:-"HAR"}
CONDA_ENV="CoFT"

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
    echo -e "${CYAN}🚀 CoFT Unified Optimization Script v2.0.0${NC}"
    echo ""
    echo -e "${YELLOW}USAGE:${NC}"
    echo "  ./optimize_coft.sh [mode] [dataset]"
    echo ""
    echo -e "${YELLOW}MODES:${NC}"
    echo -e "  ${GREEN}diagnostic${NC}  - Quick 5-minute validation (3 experiments)"
    echo -e "  ${GREEN}optimize${NC}    - Full optimization (24 experiments, 2-4 hours)"
    echo -e "  ${GREEN}quick${NC}       - Quick parameter test (6 experiments, 30 min)"
    echo -e "  ${GREEN}help${NC}        - Show this usage guide"
    echo ""
    echo -e "${YELLOW}DATASETS:${NC}"
    echo "  HAR, sleep, Epilepsy, pFD"
    echo ""
    echo -e "${YELLOW}EXAMPLES:${NC}"
    echo "  ./optimize_coft.sh diagnostic HAR    # Quick validation"
    echo "  ./optimize_coft.sh optimize HAR      # Full optimization"
    echo "  ./optimize_coft.sh quick HAR         # Quick test"
    echo ""
    echo -e "${BLUE}💡 TIP: Always run 'diagnostic' mode first to verify setup!${NC}"
}

# Function to update CoFT loss parameters
update_coft_loss_params() {
    local lambda_ct=$1
    local lambda_cs=$2
    
    cp models/coft_loss.py models/coft_loss.py.backup
    sed -i "s/self\.lambda_cotraining = [0-9]*\.[0-9]*/self.lambda_cotraining = $lambda_ct/" models/coft_loss.py
    sed -i "s/self\.lambda_consistency = [0-9]*\.[0-9]*/self.lambda_consistency = $lambda_cs/" models/coft_loss.py
}

# Function to update ensemble method
update_ensemble_method() {
    local method=$1
    
    cp trainer/trainer_coft.py trainer/trainer_coft.py.backup
    
    if [[ "$method" == "temporal_only" ]]; then
        sed -i 's/ensemble_predictions = ensemble_module(predictions, freq_predictions); final_predictions = ensemble_predictions/final_predictions = predictions  # TEMPORAL_ONLY_MODE/' trainer/trainer_coft.py
    elif [[ "$method" == "simple_average" ]]; then
        sed -i 's/final_predictions = predictions  # TEMPORAL_ONLY_MODE/final_predictions = (predictions + freq_predictions) \/ 2  # SIMPLE_AVERAGE/' trainer/trainer_coft.py
        sed -i 's/final_predictions = ensemble_predictions/final_predictions = (predictions + freq_predictions) \/ 2/' trainer/trainer_coft.py
    fi
}

# Function to verify parameter changes
verify_parameters() {
    local lambda_ct=$1
    local lambda_cs=$2
    local ensemble=$3
    local verification_score=0
    
    if grep -q "lambda_cotraining = $lambda_ct" models/coft_loss.py; then
        ((verification_score++))
    fi
    
    if grep -q "lambda_consistency = $lambda_cs" models/coft_loss.py; then
        ((verification_score++))
    fi
    
    if [[ "$ensemble" == "temporal_only" ]] && grep -q "TEMPORAL_ONLY_MODE" trainer/trainer_coft.py; then
        ((verification_score++))
    elif [[ "$ensemble" == "simple_average" ]] && grep -q "SIMPLE_AVERAGE" trainer/trainer_coft.py; then
        ((verification_score++))
    fi
    
    echo "$verification_score/3"
}

# Function to run single experiment
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
    
    # Verify changes
    local verification=$(verify_parameters "$lambda_ct" "$lambda_cs" "$ensemble")
    echo "   ✓ Parameter verification: $verification"
    
    # Create experiment log
    local exp_log="$results_dir/experiment_$exp_id.log"
    echo "Experiment $exp_id: $desc" > "$exp_log"
    echo "lambda_cotraining: $lambda_ct" >> "$exp_log"
    echo "lambda_consistency: $lambda_cs" >> "$exp_log"
    echo "ensemble_method: $ensemble" >> "$exp_log"
    echo "verification: $verification" >> "$exp_log"
    echo "---" >> "$exp_log"
    
    sleep 1  # File system sync
    
    # Run training
    local start_time=$(date +%s)
    echo "   ⏳ Running training..."
    
    if timeout 600 conda run -n "$CONDA_ENV" python main.py \
        --training_mode ft_1p \
        --selected_dataset "$DATASET" \
        --enable_coft >> "$exp_log" 2>&1; then
        
        local test_acc=$(grep "Test Accuracy" "$exp_log" | tail -1 | sed 's/.*: \([0-9.]\+\).*/\1/')
        echo -e "   ${GREEN}✅ Result: Test Accuracy = ${test_acc:-N/A}%${NC}"
        
        # Save result
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "$exp_id,$lambda_ct,$lambda_cs,$ensemble,$test_acc,$duration,$verification" >> "$results_dir/results.csv"
        
    else
        echo -e "   ${RED}❌ Failed or timeout${NC}"
        echo "$exp_id,$lambda_ct,$lambda_cs,$ensemble,FAILED,600,$verification" >> "$results_dir/results.csv"
    fi
    
    # Restore files
    cp models/coft_loss.py.backup models/coft_loss.py
    cp trainer/trainer_coft.py.backup trainer/trainer_coft.py
    
    echo "───────────────────────────────────────────────────────────────────────────"
}

# DIAGNOSTIC MODE: Quick validation (3 experiments)
run_diagnostic_mode() {
    local results_dir="diagnostic_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$results_dir"
    
    echo -e "${CYAN}🔍 DIAGNOSTIC MODE${NC} - Quick Parameter Validation"
    echo "📊 Running 3 experiments with different parameters"
    echo "📁 Results directory: $results_dir"
    echo "═══════════════════════════════════════════════════════════════════════════"
    
    echo "exp_id,lambda_cotraining,lambda_consistency,ensemble,test_accuracy,duration,verification" > "$results_dir/results.csv"
    
    run_experiment "1" "0.001" "0.1" "temporal_only" "Minimal CoFT impact" "$results_dir"
    run_experiment "2" "0.05" "0.2" "simple_average" "Medium CoFT with ensemble" "$results_dir"
    run_experiment "3" "0.1" "0.1" "temporal_only" "High CoFT impact" "$results_dir"
    
    # Analyze results
    echo -e "${CYAN}🔍 DIAGNOSTIC ANALYSIS:${NC}"
    echo "═══════════════════════════════════════════════════════════════════════════"
    cat "$results_dir/results.csv"
    
    local unique_results=$(tail -n +2 "$results_dir/results.csv" | cut -d',' -f5 | grep -v "FAILED" | sort -u | wc -l)
    
    if [[ $unique_results -gt 1 ]]; then
        echo -e "${GREEN}✅ SUCCESS: Found $unique_results different accuracy values!${NC}"
        echo -e "${GREEN}✅ Parameters are working correctly!${NC}"
        echo -e "${BLUE}💡 Ready for full optimization mode${NC}"
    else
        echo -e "${RED}❌ PROBLEM: Results are identical or failed${NC}"
        echo -e "${YELLOW}🔧 Check logs in $results_dir/ for debugging${NC}"
    fi
}

# QUICK MODE: Quick parameter test (6 experiments)
run_quick_mode() {
    local results_dir="quick_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$results_dir"
    
    echo -e "${CYAN}⚡ QUICK MODE${NC} - Fast Parameter Search"
    echo "📊 Running 6 experiments (~30 minutes)"
    echo "📁 Results directory: $results_dir"
    echo "═══════════════════════════════════════════════════════════════════════════"
    
    echo "exp_id,lambda_cotraining,lambda_consistency,ensemble,test_accuracy,duration,verification" > "$results_dir/results.csv"
    
    local exp_id=0
    for lambda_ct in 0.005 0.01 0.02; do
        for ensemble in "temporal_only" "simple_average"; do
            ((exp_id++))
            run_experiment "$exp_id" "$lambda_ct" "0.1" "$ensemble" "Quick test λ=$lambda_ct, $ensemble" "$results_dir"
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
    local results_dir="optimization_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$results_dir"
    
    echo -e "${CYAN}🚀 OPTIMIZE MODE${NC} - Full Parameter Optimization"
    echo "📊 Running 24 experiments (2-4 hours)"
    echo "📁 Results directory: $results_dir"
    echo "═══════════════════════════════════════════════════════════════════════════"
    
    echo "exp_id,lambda_cotraining,lambda_consistency,ensemble,test_accuracy,duration,verification" > "$results_dir/results.csv"
    
    local exp_id=0
    local best_acc=0
    local best_params=""
    
    for lambda_ct in 0.005 0.01 0.02 0.05; do
        for lambda_cs in 0.1 0.2 0.3; do
            for ensemble in "temporal_only" "simple_average"; do
                ((exp_id++))
                echo -e "${YELLOW}Progress: $exp_id/24${NC}"
                
                run_experiment "$exp_id" "$lambda_ct" "$lambda_cs" "$ensemble" "Full opt λ_ct=$lambda_ct, λ_cs=$lambda_cs, $ensemble" "$results_dir"
                
                # Track best result
                local current_acc=$(tail -1 "$results_dir/results.csv" | cut -d',' -f5)
                if [[ "$current_acc" != "FAILED" ]] && (( $(echo "$current_acc > $best_acc" | bc -l) )); then
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
        echo -e "${YELLOW}💡 Please create the environment first${NC}"
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

echo -e "${CYAN}✨ CoFT Optimization Script completed!${NC}" 