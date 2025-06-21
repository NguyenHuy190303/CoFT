#!/bin/bash

###############################################################################
# Quick CoFT Test Script - Immediate Validation
# 
# Purpose: Run 4 key experiments to validate optimization direction
# Before running the full 120+ experiment grid search
# 
# Usage: ./quick_test_coft.sh [dataset]
# Example: ./quick_test_coft.sh HAR
###############################################################################

DATASET=${1:-"HAR"}
RESULTS_DIR="quick_test_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/quick_test_results.csv"

mkdir -p "$RESULTS_DIR"
echo "experiment,lambda_cotraining,lambda_consistency,test_acc,status" > "$LOG_FILE"

echo "🚀 Quick CoFT Validation Test"
echo "📊 Running 4 key experiments to validate optimization strategy"
echo "🗂️ Dataset: $DATASET"
echo "═══════════════════════════════════════════════════════════════════"

# Test configurations based on expert analysis
declare -a TESTS=(
    "current:0.1:0.3"           # Current configuration
    "lower_cotraining:0.02:0.3"  # Much lower co-training weight
    "lowest_cotraining:0.01:0.2"  # Minimal co-training
    "minimal_impact:0.005:0.1"   # Almost disabled co-training
)

for i in "${!TESTS[@]}"; do
    IFS=':' read -r name lambda_ct lambda_cs <<< "${TESTS[$i]}"
    
    echo "🔬 Test $((i+1))/4: $name"
    echo "   λ_cotraining: $lambda_ct, λ_consistency: $lambda_cs"
    
    # Backup and update parameters
    cp models/coft_loss.py models/coft_loss.py.backup
    sed -i "s/self\.lambda_cotraining = [0-9.]\+/self.lambda_cotraining = $lambda_ct/" models/coft_loss.py
    sed -i "s/self\.lambda_consistency = [0-9.]\+/self.lambda_consistency = $lambda_cs/" models/coft_loss.py
    
    # Run experiment
    if timeout 300 python main.py --training_mode train_linear_1p --selected_dataset "$DATASET" --enable_coft > "$RESULTS_DIR/test_$((i+1)).log" 2>&1; then
        test_acc=$(grep "Test Accuracy" "$RESULTS_DIR/test_$((i+1)).log" | tail -1 | sed 's/.*: \([0-9.]\+\).*/\1/')
        echo "   ✅ Test Accuracy: ${test_acc:-N/A}%"
        echo "$name,$lambda_ct,$lambda_cs,${test_acc:-N/A},success" >> "$LOG_FILE"
    else
        echo "   ❌ Failed"
        echo "$name,$lambda_ct,$lambda_cs,N/A,failed" >> "$LOG_FILE"
    fi
    
    # Restore original
    cp models/coft_loss.py.backup models/coft_loss.py
    echo "───────────────────────────────────────────────────────────────────"
done

echo "🎉 Quick test completed!"
echo "📋 Results saved in: $LOG_FILE"
echo "💡 View results: cat $LOG_FILE"

# Cleanup
rm -f models/coft_loss.py.backup 