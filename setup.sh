#!/bin/bash
# CoFT Project Setup Script
# Automatically extracts data files and sets up environment for Linux systems

set -e  # Exit on any error

echo "🚀 Setting up CoFT Project Environment"
echo "======================================"

# Check if we're on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  This script is designed for Linux systems"
    echo "   For other systems, please extract data files manually"
fi

# Create data directories if they don't exist
echo "📁 Creating data directories..."
mkdir -p data/epilepsy
mkdir -p data/HAR  
mkdir -p data/sleep
mkdir -p data/SleepEDF

# Extract data files if .tar.gz exist
echo "📦 Extracting data files..."

if [ -f "data/epilepsy.tar.gz" ]; then
    echo "   🔄 Extracting epilepsy data..."
    tar -xzf data/epilepsy.tar.gz -C data/epilepsy/ --strip-components=1
    echo "   ✅ Epilepsy data extracted"
else
    echo "   ⚠️  epilepsy.tar.gz not found"
fi

if [ -f "data/har.tar.gz" ]; then
    echo "   🔄 Extracting HAR data..."
    tar -xzf data/har.tar.gz -C data/HAR/ --strip-components=1
    echo "   ✅ HAR data extracted"
else
    echo "   ⚠️  har.tar.gz not found"
fi

if [ -f "data/sleep.tar.gz" ]; then
    echo "   🔄 Extracting sleep data..."
    tar -xzf data/sleep.tar.gz -C data/sleep/ --strip-components=1
    echo "   ✅ Sleep data extracted"
else
    echo "   ⚠️  sleep.tar.gz not found"
fi

if [ -f "data/sleepedf.tar.gz" ]; then
    echo "   🔄 Extracting SleepEDF data..."
    tar -xzf data/sleepedf.tar.gz -C data/SleepEDF/ --strip-components=1
    echo "   ✅ SleepEDF data extracted"
else
    echo "   ⚠️  sleepedf.tar.gz not found"
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
echo "   📦 Archive files: Extracted"
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
echo "✅ Setup completed successfully!" 