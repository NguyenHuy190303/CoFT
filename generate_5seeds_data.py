#!/usr/bin/env python3
"""
CA-TCC Few-Shot Data Generation với 5 Random Seeds

Tạo few-shot training data theo methodology trong FIVE_SEEDS_METHODOLOGY.md:
- Random sampling with class guarantee  
- 5 random seeds [0,1,2,3,4]
- Percentages: 1%, 5%, 10%, 50%, 75%
- Tương thích hoàn toàn với CA-TCC pipeline
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
import sys
from pathlib import Path


def random_sample_with_class_guarantee(X, y, percentage, random_seed=42, max_attempts=50):
    """
    Random sampling với class guarantee - theo CA-TCC methodology
    
    Args:
        X: Input samples
        y: Labels  
        percentage: Percentage of data to sample
        random_seed: Random seed for reproducibility
        max_attempts: Maximum attempts to ensure all classes
    
    Returns:
        X_sample, y_sample: Sampled data with all classes present
    """
    unique_classes = np.unique(y)
    n_samples = len(y)
    target_size = max(len(unique_classes), int(n_samples * percentage / 100.0))
    
    print(f"🎯 Target: {percentage}% of {n_samples} samples = ~{target_size} samples")
    print(f"📊 Classes required: {len(unique_classes)} classes")
    
    # Multiple attempts với varied seeds
    for attempt in range(max_attempts):
        # Random sampling (không stratify - pure random như paper)
        _, X_sample, _, y_sample = train_test_split(
            X, y, test_size=target_size, random_state=random_seed + attempt, shuffle=True
        )
        
        # Check if all classes are present
        sample_classes = np.unique(y_sample)
        if len(sample_classes) == len(unique_classes):
            print(f"✅ Success on attempt {attempt + 1}")
            print(f"📈 Final sample size: {len(y_sample)}")
            
            # Print class distribution
            for cls in unique_classes:
                count = np.sum(y_sample == cls)
                original_count = np.sum(y == cls)
                percentage_actual = (count / original_count) * 100
                print(f"   Class {cls}: {count}/{original_count} ({percentage_actual:.1f}%)")
            
            return X_sample, y_sample
        
        # Adaptive target size increase
        target_size = min(target_size + len(unique_classes), n_samples)
    
    # Fallback to stratified if random fails
    print(f"⚠️  Random sampling failed after {max_attempts} attempts")
    print("🔄 Using stratified fallback...")
    
    _, X_sample, _, y_sample = train_test_split(
        X, y, test_size=percentage/100.0, random_state=random_seed, shuffle=True, stratify=y
    )
    
    return X_sample, y_sample


def generate_5seeds_data():
    """Generate 5-seeds few-shot data for all datasets"""
    
    datasets = {
        'sleep': 'data/sleep',
        'HAR': 'data/HAR', 
        'epilepsy': 'data/epilepsy',
        'SleepEDF': 'data/SleepEDF'
    }
    
    percentages = [1, 5, 10, 50, 75]
    seeds = [0, 1, 2, 3, 4]  # Fixed seeds theo paper
    
    for dataset_name, data_path in datasets.items():
        print(f"\n🔍 Processing {dataset_name} dataset...")
        
        train_file = os.path.join(data_path, 'train.pt')
        
        if not os.path.exists(train_file):
            print(f"❌ {train_file} not found, skipping...")
            continue
            
        # Load original training data
        print(f"📂 Loading {train_file}...")
        train_data = torch.load(train_file, map_location='cpu', weights_only=False)
        
        if isinstance(train_data, dict):
            X = train_data['samples']
            y = train_data['labels']
        else:
            # Handle tuple format (samples, labels)
            X, y = train_data
            
        print(f"📊 Original data: {len(X)} samples, {len(np.unique(y))} classes")
        
        # Convert to numpy for sklearn
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = np.array(X)
            
        if isinstance(y, torch.Tensor):
            y_np = y.numpy()
        else:
            y_np = np.array(y)
        
        # Create 5seeds directory
        seeds_dir = os.path.join(data_path, '5seeds')
        os.makedirs(seeds_dir, exist_ok=True)
        
        # Statistics tracking
        stats = {
            "dataset": dataset_name,
            "full_samples": len(X_np),
            "full_classes": len(np.unique(y_np)),
            "percentages": {}
        }
        
        # Generate for each percentage
        for perc in percentages:
            print(f"\n🎲 Generating {perc}% subsets...")
            
            perc_stats = {
                "percentage": perc,
                "seeds_results": {}
            }
            
            # Generate for each seed
            for seed in seeds:
                print(f"\n🌱 Seed {seed}:")
                
                X_few, y_few = random_sample_with_class_guarantee(
                    X_np, y_np, percentage=perc, random_seed=seed
                )
                
                # Convert back to tensors
                X_few_tensor = torch.tensor(X_few, dtype=X.dtype)
                y_few_tensor = torch.tensor(y_few, dtype=y.dtype)
                
                # Create few-shot data structure
                few_shot_data = {
                    'samples': X_few_tensor,
                    'labels': y_few_tensor,
                    'metadata': {
                        'dataset': dataset_name,
                        'percentage': perc,
                        'seed': seed,
                        'sampling_method': 'random_with_class_guarantee',
                        'original_samples': len(X_np),
                        'selected_samples': len(X_few),
                        'classes_present': len(np.unique(y_few)),
                        'all_classes_present': len(np.unique(y_few)) == len(np.unique(y_np))
                    }
                }
                
                # Save with 5seeds naming
                seeds_file = os.path.join(seeds_dir, f'train_{perc}perc_seed{seed}.pt')
                torch.save(few_shot_data, seeds_file)
                print(f"💾 Saved: {seeds_file}")
                
                # Also save compatibility files for main directory (seed 0 only)
                if seed == 0:
                    compat_file = os.path.join(data_path, f'train_{perc}perc.pt')
                    torch.save(few_shot_data, compat_file)
                    print(f"🔗 Compatibility: {compat_file}")
                    
                    # Special handling for train_1p.pt (both naming conventions)
                    if perc == 1:
                        train_1p_file = os.path.join(data_path, 'train_1p.pt')
                        torch.save(few_shot_data, train_1p_file)
                        print(f"🔗 Compatibility: {train_1p_file}")
                    
                    if perc == 5:
                        train_5p_file = os.path.join(data_path, 'train_5p.pt')
                        torch.save(few_shot_data, train_5p_file)
                        print(f"🔗 Compatibility: {train_5p_file}")
                
                # Record stats
                seed_stats = {
                    "samples": len(X_few),
                    "actual_percentage": (len(X_few) / len(X_np)) * 100,
                    "classes_present": len(np.unique(y_few)),
                    "all_classes_present": len(np.unique(y_few)) == len(np.unique(y_np)),
                    "class_distribution": {str(cls): int(np.sum(y_few == cls)) for cls in np.unique(y_np)}
                }
                
                perc_stats["seeds_results"][str(seed)] = seed_stats
            
            stats["percentages"][str(perc)] = perc_stats
        
        # Save statistics
        stats_file = os.path.join(seeds_dir, 'stats_5seeds.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n📊 Statistics saved: {stats_file}")
        
        print(f"\n✅ {dataset_name} 5-seeds data generation completed!")
    
    print("\n🎉 All 5-seeds datasets generated successfully!")
    print("\n📋 Generated files:")
    print("   📁 data/*/5seeds/train_{1,5,10,50,75}perc_seed{0,1,2,3,4}.pt")
    print("   🔗 data/*/train_{1,5,10,50,75}perc.pt (compatibility)")
    print("   📊 data/*/5seeds/stats_5seeds.json")


def verify_generated_data():
    """Verify the generated data integrity"""
    
    print("\n🔍 Verifying generated 5-seeds data...")
    
    datasets = ['sleep', 'HAR', 'epilepsy', 'SleepEDF']
    percentages = [1, 5, 10, 50, 75]
    seeds = [0, 1, 2, 3, 4]
    
    for dataset in datasets:
        print(f"\n📊 {dataset}:")
        
        # Check 5seeds directory
        seeds_dir = f'data/{dataset}/5seeds'
        if not os.path.exists(seeds_dir):
            print(f"   ❌ {seeds_dir} not found")
            continue
        
        # Count files
        total_expected = len(percentages) * len(seeds)
        files_found = 0
        
        for perc in percentages:
            for seed in seeds:
                file_path = f'{seeds_dir}/train_{perc}perc_seed{seed}.pt'
                if os.path.exists(file_path):
                    files_found += 1
                    
                    # Quick load test
                    try:
                        data = torch.load(file_path, map_location='cpu', weights_only=False)
                        samples = len(data['samples'])
                        classes = len(torch.unique(data['labels']))
                        print(f"   ✅ {perc}% seed{seed}: {samples} samples, {classes} classes")
                    except Exception as e:
                        print(f"   ❌ {perc}% seed{seed}: Load error - {e}")
                else:
                    print(f"   ❌ {perc}% seed{seed}: File missing")
        
        print(f"   📈 Summary: {files_found}/{total_expected} files found")
        
        # Check compatibility files
        for perc in [1, 5]:
            compat_file = f'data/{dataset}/train_{perc}perc.pt'
            if os.path.exists(compat_file):
                print(f"   🔗 Compatibility: train_{perc}perc.pt ✅")
            else:
                print(f"   🔗 Compatibility: train_{perc}perc.pt ❌")


if __name__ == "__main__":
    print("🚀 CA-TCC 5-Seeds Few-Shot Data Generator")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_generated_data()
    else:
        generate_5seeds_data()
        print("\n" + "=" * 50)
        verify_generated_data()
    
    print("\n✅ Process completed!")
    print("\nUsage:")
    print("  python main.py --training_mode full_run --selected_dataset sleep --enable_coft --label_percentage 1") 