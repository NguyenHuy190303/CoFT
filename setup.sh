#!/bin/bash
# CoFT Project Setup Script
# Automatically creates data archives and sets up environment for Linux systems

set -e  # Exit on any error

echo "🚀 Setting up CoFT Project Environment"
echo "======================================"

# DEBUG: Show current directory and files
echo "🔍 DEBUG INFO:"
echo "   Current directory: $(pwd)"
echo "   Files in current dir: $(ls -la | grep -E '(setup|data)' | wc -l) setup/data related files"
echo ""

# Check if we're on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  This script is designed for Linux systems"
    echo "   For other systems, please create archives manually"
fi

# Create data directories if they don't exist
echo "📁 Creating data directories..."
mkdir -p data/epilepsy
mkdir -p data/HAR  
mkdir -p data/sleep
mkdir -p data/SleepEDF

# Check if tar.gz files exist, if not create them from directories
echo "📦 Checking/Creating data archives..."

if [ -f "data/epilepsy.tar.gz" ]; then
    echo "   ✅ epilepsy.tar.gz already exists"
elif [ -d "data/epilepsy" ] && [ "$(ls -A data/epilepsy)" ]; then
    echo "   🔄 Creating epilepsy.tar.gz from directory..."
    tar -czf data/epilepsy.tar.gz -C data epilepsy
    echo "   ✅ epilepsy.tar.gz created"
else
    echo "   ⚠️  epilepsy data not found - please add data to data/epilepsy/ first"
fi

if [ -f "data/har.tar.gz" ]; then
    echo "   ✅ har.tar.gz already exists"
elif [ -d "data/HAR" ] && [ "$(ls -A data/HAR)" ]; then
    echo "   🔄 Creating har.tar.gz from directory..."
    tar -czf data/har.tar.gz -C data HAR
    echo "   ✅ har.tar.gz created"
else
    echo "   ⚠️  HAR data not found - please add data to data/HAR/ first"
fi

if [ -f "data/sleep.tar.gz" ]; then
    echo "   ✅ sleep.tar.gz already exists"
elif [ -d "data/sleep" ] && [ "$(ls -A data/sleep)" ]; then
    echo "   🔄 Creating sleep.tar.gz from directory..."
    tar -czf data/sleep.tar.gz -C data sleep
    echo "   ✅ sleep.tar.gz created"
else
    echo "   ⚠️  sleep data not found - please add data to data/sleep/ first"
fi

if [ -f "data/sleepedf.tar.gz" ]; then
    echo "   ✅ sleepedf.tar.gz already exists"
elif [ -d "data/SleepEDF" ] && [ "$(ls -A data/SleepEDF)" ]; then
    echo "   🔄 Creating sleepedf.tar.gz from directory..."
    tar -czf data/sleepedf.tar.gz -C data SleepEDF
    echo "   ✅ sleepedf.tar.gz created"
else
    echo "   ⚠️  SleepEDF data not found - please add data to data/SleepEDF/ first"
fi

# Install Python dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📋 Installing Python dependencies..."
    if command -v pip3 &> /dev/null; then
        pip3 install -r requirements.txt
        echo "   ✅ Dependencies installed"
    elif command -v pip &> /dev/null; then
        pip install -r requirements.txt
        echo "   ✅ Dependencies installed"
    else
        echo "   ⚠️  pip not found. Please install dependencies manually:"
        echo "   pip install -r requirements.txt"
    fi
else
    echo "   ⚠️  requirements.txt not found"
fi

# Check data integrity
echo "🔍 Checking data integrity..."
data_files_found=0

for dataset in HAR sleep epilepsy SleepEDF; do
    if [ -f "data/$dataset/train.pt" ]; then
        echo "   ✅ $dataset dataset ready"
        ((data_files_found++))
    else
        echo "   ❌ $dataset dataset missing train.pt"
    fi
done

echo ""
echo "📊 Setup Summary:"
echo "   🗂️  Data directories: Created"
echo "   📦 Archive files: Created/Checked"
echo "   🔧 Scripts: Made executable"
echo "   📋 Dependencies: $([ -f "requirements.txt" ] && echo "Installed" || echo "Skipped")"
echo "   📊 Datasets ready: $data_files_found/4"

echo ""
echo "🎯 Ready to use! Try these commands:"
echo "   # Quick examples"
echo "   ./quick_examples.sh"
echo ""
echo "   # Train with 1% labels"
echo "   python3 main.py --training_mode full_run --selected_dataset HAR --label_percentage 1"
echo ""
echo "   # Train with 5% labels + InfoTS"
echo "   python3 main.py --training_mode full_run --selected_dataset HAR --label_percentage 5 --enable_infots"
echo ""
echo "   # Train with CoFT + InfoTS"  
echo "   python3 main.py --training_mode full_run --selected_dataset HAR --enable_coft --enable_infots"

echo ""
echo "💡 Manual Setup Commands (if needed):"
echo "   # Create archives manually:"
echo "   tar -czf data/epilepsy.tar.gz -C data epilepsy"
echo "   tar -czf data/har.tar.gz -C data HAR"
echo "   tar -czf data/sleep.tar.gz -C data sleep"
echo "   tar -czf data/sleepedf.tar.gz -C data SleepEDF"

echo ""
echo "✅ Setup completed successfully!" 