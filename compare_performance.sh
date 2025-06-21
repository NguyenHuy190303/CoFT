#!/bin/bash

echo "🚀 Performance Comparison: AMP vs Baseline"
echo "=========================================="

# Select dataset to test (default: HAR, fastest to test)
DATASET=${1:-"HAR"}

echo "📊 Testing dataset: $DATASET"
echo ""

# Function to run dataset commands
run_dataset() {
    local dataset=$1
    local exp_name="${dataset}_experiment"
    local run_name="$dataset"
    local dataset_param="$dataset"
    
    # Special case for SleepEDF
    if [ "$dataset" = "SleepEDF" ]; then
        dataset_param="sleep"
        exp_name="sleepEDF_experiment"
        run_name="sleepEDF"
    fi
    
    echo "Running $dataset dataset..."
    local start_time=$(date +%s)
    
    python3 main.py --experiment_description $exp_name --run_description $run_name --seed 0 --selected_dataset $dataset_param --training_mode "self_supervised"
    python3 main.py --experiment_description $exp_name --run_description $run_name --seed 0 --selected_dataset $dataset_param --training_mode "train_linear_1p"
    python3 main.py --experiment_description $exp_name --run_description $run_name --seed 0 --selected_dataset $dataset_param --training_mode "ft_1p"
    python3 main.py --experiment_description $exp_name --run_description $run_name --seed 0 --selected_dataset $dataset_param --training_mode "gen_pseudo_labels"
    python3 main.py --experiment_description $exp_name --run_description $run_name --seed 0 --selected_dataset $dataset_param --training_mode "SupCon"
    python3 main.py --experiment_description $exp_name --run_description $run_name --seed 0 --selected_dataset $dataset_param --training_mode "train_linear_SupCon_1p"
    
    local end_time=$(date +%s)
    echo "$dataset Dataset - Total execution time: $((end_time - start_time)) seconds"
}

# Function to run all datasets
run_all_datasets() {
    echo "Running comprehensive benchmark for all datasets..."
    local overall_start=$(date +%s)
    
    run_dataset "HAR"
    echo ""
    run_dataset "Epilepsy"
    echo ""
    run_dataset "SleepEDF"
    
    local overall_end=$(date +%s)
    echo "============================================="
    echo "All datasets completed in: $((overall_end - overall_start)) seconds"
    echo "============================================="
}

# Validate dataset choice
case $DATASET in
    "HAR"|"Epilepsy"|"SleepEDF")
        ;;
    "ALL")
        ;;
    *)
        echo "❌ Unknown dataset: $DATASET"
        echo "Usage: ./compare_performance.sh [HAR|Epilepsy|SleepEDF|ALL]"
        exit 1
        ;;
esac

# Create backup of current trainer
echo "💾 Backing up current trainer..."
cp trainer/trainer.py trainer/trainer_backup.py

# Test 1: With AMP (current version)
echo ""
echo "🔥 Testing with AMP..."
echo "========================"
AMP_START=$(date +%s)

if [ "$DATASET" = "ALL" ]; then
    run_all_datasets 2>&1 | tee amp_output.log
else
    run_dataset $DATASET 2>&1 | tee amp_output.log
fi

AMP_END=$(date +%s)
AMP_TIME=$((AMP_END - AMP_START))

# Extract accuracy from AMP run
AMP_ACCURACY=""
if grep -q "Test Accuracy" amp_output.log; then
    AMP_ACCURACY=$(grep "Test Accuracy" amp_output.log | tail -1 | grep -o '[0-9]*\.[0-9]*' | tail -1)
fi

echo "✅ AMP completed in: ${AMP_TIME} seconds"
if [ ! -z "$AMP_ACCURACY" ]; then
    echo "   Test Accuracy: ${AMP_ACCURACY}"
fi

# Test 2: Without AMP (baseline version)
echo ""
echo "📊 Testing without AMP (Baseline)..."
echo "====================================="
cp trainer/trainer_baseline.py trainer/trainer.py

BASELINE_START=$(date +%s)

if [ "$DATASET" = "ALL" ]; then
    run_all_datasets 2>&1 | tee baseline_output.log
else
    run_dataset $DATASET 2>&1 | tee baseline_output.log
fi

BASELINE_END=$(date +%s)
BASELINE_TIME=$((BASELINE_END - BASELINE_START))

# Extract accuracy from baseline run
BASELINE_ACCURACY=""
if grep -q "Test Accuracy" baseline_output.log; then
    BASELINE_ACCURACY=$(grep "Test Accuracy" baseline_output.log | tail -1 | grep -o '[0-9]*\.[0-9]*' | tail -1)
fi

echo "✅ Baseline completed in: ${BASELINE_TIME} seconds"
if [ ! -z "$BASELINE_ACCURACY" ]; then
    echo "   Test Accuracy: ${BASELINE_ACCURACY}"
fi

# Restore original trainer
cp trainer/trainer_backup.py trainer/trainer.py
rm trainer/trainer_backup.py

# Calculate performance metrics
echo ""
echo "📈 PERFORMANCE COMPARISON"
echo "========================="
echo "AMP Time:      ${AMP_TIME} seconds"
echo "Baseline Time: ${BASELINE_TIME} seconds"

if [ $BASELINE_TIME -gt 0 ]; then
    IMPROVEMENT=$(echo "scale=1; ($BASELINE_TIME - $AMP_TIME) * 100 / $BASELINE_TIME" | bc -l)
    SPEEDUP=$(echo "scale=2; $BASELINE_TIME / $AMP_TIME" | bc -l)
    
    echo "Time Difference: ${IMPROVEMENT}%"
    echo "Speedup Factor: ${SPEEDUP}x"
    
    if [ "$(echo "$IMPROVEMENT > 5" | bc -l)" -eq 1 ]; then
        echo "🎉 AMP provides significant speed improvement!"
    elif [ "$(echo "$IMPROVEMENT > 0" | bc -l)" -eq 1 ]; then
        echo "👍 AMP provides modest speed improvement"
    else
        echo "⚠️ AMP is slower than baseline"
    fi
else
    echo "❌ Cannot calculate improvement (baseline time is 0)"
fi

# Accuracy comparison
if [ ! -z "$AMP_ACCURACY" ] && [ ! -z "$BASELINE_ACCURACY" ]; then
    ACC_DIFF=$(echo "scale=2; $AMP_ACCURACY - $BASELINE_ACCURACY" | bc -l)
    echo ""
    echo "📊 ACCURACY COMPARISON"
    echo "====================="
    echo "AMP Accuracy:      ${AMP_ACCURACY}%"
    echo "Baseline Accuracy: ${BASELINE_ACCURACY}%"
    echo "Accuracy Difference: ${ACC_DIFF}%"
fi

# Save detailed results
echo ""
echo "💾 DETAILED RESULTS"
echo "=================="
echo "Results saved to:"
echo "- amp_output.log (AMP run output)"
echo "- baseline_output.log (Baseline run output)"

# Create summary file
cat > performance_summary.txt << EOF
Performance Comparison Results
=============================
Date: $(date)
Dataset: $DATASET

Timing Results:
- AMP Time: ${AMP_TIME} seconds
- Baseline Time: ${BASELINE_TIME} seconds
- Time Improvement: ${IMPROVEMENT}%
- Speedup Factor: ${SPEEDUP}x

Accuracy Results:
- AMP Accuracy: ${AMP_ACCURACY}%
- Baseline Accuracy: ${BASELINE_ACCURACY}%
- Accuracy Difference: ${ACC_DIFF}%

Conclusion:
EOF

if [ "$(echo "$IMPROVEMENT > 5" | bc -l)" -eq 1 ]; then
    echo "AMP provides significant performance benefits and should be enabled." >> performance_summary.txt
elif [ "$(echo "$IMPROVEMENT > 0" | bc -l)" -eq 1 ]; then
    echo "AMP provides modest performance benefits." >> performance_summary.txt
else
    echo "AMP may not be beneficial for this workload." >> performance_summary.txt
fi

echo "- performance_summary.txt (Summary report)"
echo ""
echo "🎯 To test different datasets:"
echo "   ./compare_performance.sh HAR"
echo "   ./compare_performance.sh Epilepsy" 
echo "   ./compare_performance.sh SleepEDF"
echo "   ./compare_performance.sh ALL"