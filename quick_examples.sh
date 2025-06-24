#!/bin/bash
# Quick Examples for CoFT Training with Different Configurations

echo "🚀 CoFT Quick Examples - All Configuration Options"
echo "================================================="

echo ""
echo "📊 1. LABEL PERCENTAGE EXAMPLES:"
echo "================================="

echo "# 1% Labels (Default)"
echo "python3 main.py --training_mode full_run --selected_dataset HAR --label_percentage 1"

echo ""
echo "# 5% Labels"
echo "python3 main.py --training_mode full_run --selected_dataset HAR --label_percentage 5"

echo ""
echo "# 75% Labels (Full dataset)"  
echo "python3 main.py --training_mode full_run --selected_dataset HAR --label_percentage 75"

echo ""
echo "🎨 2. INFOTS AUGMENTATION EXAMPLES:"
echo "=================================="

echo "# Enable InfoTS for HAR (already enabled by default)"
echo "python3 main.py --training_mode full_run --selected_dataset HAR --enable_infots"

echo ""
echo "# Enable InfoTS for Sleep dataset"
echo "python3 main.py --training_mode full_run --selected_dataset sleep --enable_infots"

echo ""
echo "# Enable InfoTS for Epilepsy dataset"
echo "python3 main.py --training_mode full_run --selected_dataset Epilepsy --enable_infots"

echo ""
echo "# Enable InfoTS for pFD dataset"
echo "python3 main.py --training_mode full_run --selected_dataset pFD --enable_infots"

echo ""
echo "🔥 3. COMBINED EXAMPLES (5% + InfoTS):"
echo "====================================="

echo "# HAR with 5% labels + InfoTS"
echo "python3 main.py --training_mode full_run --selected_dataset HAR --label_percentage 5 --enable_infots"

echo ""
echo "# Sleep with 5% labels + InfoTS"
echo "python3 main.py --training_mode full_run --selected_dataset sleep --label_percentage 5 --enable_infots"

echo ""
echo "# Epilepsy with 5% labels + InfoTS"
echo "python3 main.py --training_mode full_run --selected_dataset Epilepsy --label_percentage 5 --enable_infots"

echo ""
echo "# pFD with 5% labels + InfoTS"
echo "python3 main.py --training_mode full_run --selected_dataset pFD --label_percentage 5 --enable_infots"

echo ""
echo "⚡ 4. COFT + INFOTS + 5% EXAMPLES:"
echo "================================"

echo "# HAR with CoFT + InfoTS + 5% labels"
echo "python3 main.py --training_mode full_run --selected_dataset HAR --enable_coft --enable_infots --label_percentage 5"

echo ""
echo "# Sleep with CoFT + InfoTS + 5% labels"
echo "python3 main.py --training_mode full_run --selected_dataset sleep --enable_coft --enable_infots --label_percentage 5"

echo ""
echo "💡 5. YOUR SPECIFIC COMMANDS:"
echo "============================"

echo "# Your current command with 5% labels:"
echo "python3 main.py --training_mode full_run --selected_dataset HAR --seed 42 --experiment_description \"infots_test\" --run_description \"infots_enabled\" --label_percentage 5"

echo ""
echo "# Enable InfoTS for sleep dataset:"
echo "python3 main.py --training_mode full_run --selected_dataset sleep --seed 42 --experiment_description \"infots_test\" --run_description \"infots_enabled\" --enable_infots"

echo ""
echo "📋 USAGE SUMMARY:"
echo "================"
echo "• --label_percentage [1|5|75]  : Choose label percentage (default: 1)"
echo "• --enable_infots               : Enable InfoTS for ANY dataset"
echo "• --enable_coft                 : Enable CoFT frequency co-training"
echo "• --selected_dataset [HAR|sleep|Epilepsy|pFD] : Choose dataset"
echo ""
echo "🎯 Note: InfoTS is already enabled by default for HAR dataset" 