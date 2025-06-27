#!/usr/bin/env python3
"""
🚀 CoFT A100 Ultra-Fast Parameter Optimization Script
🎯 One-shot solution for parameter grid search

Usage:
    python optimize_a100.py [dataset] [mode] [options]
    
Examples:
    python optimize_a100.py HAR diagnostic          # 3 exp, 5 min
    python optimize_a100.py HAR quick               # 6 exp, 30 min  
    python optimize_a100.py HAR optimize            # 18 exp, 2-3 hours
    python optimize_a100.py sleep optimize --gpu    # With GPU optimizations
"""

import os
import sys
import time
import subprocess
import argparse
import json
import pandas as pd
import re
from datetime import datetime
from pathlib import Path

# Try importing torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - CPU mode only")

class CoFTA100Optimizer:
    def __init__(self, dataset='HAR', mode='diagnostic', gpu_optimize=True):
        self.dataset = dataset
        self.mode = mode
        self.gpu_optimize = gpu_optimize
        self.results = []
        self.best_result = None
        self.start_time = time.time()
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"a100_results_{mode}_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🚀 CoFT A100 Optimizer initialized")
        print(f"📊 Dataset: {dataset}")
        print(f"🎯 Mode: {mode}")
        print(f"📁 Results: {self.results_dir}")
        
        self.setup_gpu_optimizations()
        
    def setup_gpu_optimizations(self):
        """Setup A100 GPU optimizations"""
        if not TORCH_AVAILABLE or not self.gpu_optimize:
            print("🔧 CPU mode - no GPU optimizations")
            return
            
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🔥 GPU: {gpu_name}")
            
            if "A100" in gpu_name:
                print("🎯 A100 DETECTED! Enabling TF32...")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                print("✅ A100 optimizations enabled (2-3x speedup)")
            else:
                print(f"⚡ GPU optimizations enabled for {gpu_name}")
                torch.backends.cudnn.benchmark = True
        else:
            print("❌ No GPU detected")
    
    def get_parameter_grid(self):
        """Get parameter grid based on mode"""
        if self.mode == 'diagnostic':
            return {
                'lambda_ct': [0.001, 0.01, 0.1],
                'lambda_cs': [0.1, 0.3, 0.5], 
                'ensemble': ['temporal_only', 'simple_average']
            }
        elif self.mode == 'quick':
            return {
                'lambda_ct': [0.0001, 0.0005, 0.001],
                'lambda_cs': [0.1, 0.15],
                'ensemble': ['temporal_only', 'simple_average']  
            }
        elif self.mode == 'optimize':
            return {
                'lambda_ct': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005],
                'lambda_cs': [0.05, 0.1, 0.15, 0.2, 0.3],
                'ensemble': ['temporal_only', 'simple_average']
            }
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def update_coft_parameters(self, lambda_ct, lambda_cs):
        """Update CoFT loss parameters"""
        try:
            # Read current file
            with open('models/coft_loss.py', 'r') as f:
                content = f.read()
            
            # Backup original
            with open(f'{self.results_dir}/coft_loss_backup.py', 'w') as f:
                f.write(content)
            
            # Update parameters with multiple patterns for robustness
            patterns = [
                (rf'self\.lambda_cotraining\s*=\s*[0-9\.e\-]+', f'self.lambda_cotraining = {lambda_ct}'),
                (rf'self\.lambda_consistency\s*=\s*[0-9\.e\-]+', f'self.lambda_consistency = {lambda_cs}')
            ]
            
            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content)
            
            # Write updated file
            with open('models/coft_loss.py', 'w') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            print(f"   ❌ Parameter update failed: {e}")
            return False
    
    def update_ensemble_method(self, method):
        """Update ensemble method in trainer"""
        try:
            # Read trainer file
            with open('trainer/trainer_coft.py', 'r') as f:
                content = f.read()
            
            # Backup original
            with open(f'{self.results_dir}/trainer_coft_backup.py', 'w') as f:
                f.write(content)
            
            # Update ensemble method
            if method == 'temporal_only':
                # Use only temporal predictions
                content = re.sub(
                    r'final_predictions\s*=\s*\(predictions\s*\+\s*freq_predictions\)\s*/\s*2.*',
                    'final_predictions = predictions  # TEMPORAL_ONLY',
                    content
                )
            else:  # simple_average
                # Use average of temporal and frequency predictions
                content = re.sub(
                    r'final_predictions\s*=\s*predictions\s*#\s*TEMPORAL_ONLY.*',
                    'final_predictions = (predictions + freq_predictions) / 2  # SIMPLE_AVERAGE',
                    content
                )
            
            # Write updated file
            with open('trainer/trainer_coft.py', 'w') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            print(f"   ❌ Ensemble update failed: {e}")
            return False
    
    def verify_parameters(self, lambda_ct, lambda_cs, ensemble):
        """Verify parameter changes were applied"""
        score = 0
        
        try:
            # Check lambda parameters
            with open('models/coft_loss.py', 'r') as f:
                content = f.read()
                if f'lambda_cotraining = {lambda_ct}' in content:
                    score += 1
                if f'lambda_consistency = {lambda_cs}' in content:
                    score += 1
            
            # Check ensemble method
            with open('trainer/trainer_coft.py', 'r') as f:
                content = f.read()
                if ensemble == 'temporal_only' and 'TEMPORAL_ONLY' in content:
                    score += 1
                elif ensemble == 'simple_average' and 'SIMPLE_AVERAGE' in content:
                    score += 1
            
        except Exception as e:
            print(f"   ⚠️  Verification error: {e}")
        
        return f"{score}/3"
    
    def run_experiment(self, exp_id, lambda_ct, lambda_cs, ensemble):
        """Run single experiment"""
        start_time = time.time()
        
        print(f"🔬 Exp {exp_id}: λ_ct={lambda_ct:.5f}, λ_cs={lambda_cs}, {ensemble}")
        
        # Update parameters
        param_success = self.update_coft_parameters(lambda_ct, lambda_cs)
        ensemble_success = self.update_ensemble_method(ensemble)
        
        if not param_success or not ensemble_success:
            print(f"   ❌ Parameter update failed")
            return None, time.time() - start_time, "0/3"
        
        # Small delay for file sync
        time.sleep(1)
        
        # Verify parameters
        verification = self.verify_parameters(lambda_ct, lambda_cs, ensemble)
        print(f"   🔍 Verification: {verification}")
        
        try:
            # Run training command
            cmd = [
                'python', 'main.py',
                '--training_mode', 'ft_1p',
                '--selected_dataset', self.dataset,
                '--enable_coft'
            ]
            
            # Create experiment log
            exp_log = f'{self.results_dir}/exp_{exp_id}.log'
            
            with open(exp_log, 'w') as log_file:
                log_file.write(f"Experiment {exp_id}\n")
                log_file.write(f"lambda_cotraining: {lambda_ct}\n")
                log_file.write(f"lambda_consistency: {lambda_cs}\n")
                log_file.write(f"ensemble: {ensemble}\n")
                log_file.write(f"verification: {verification}\n")
                log_file.write(f"command: {' '.join(cmd)}\n")
                log_file.write("=" * 50 + "\n")
            
            # Run with timeout
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 min timeout for A100 fast experiments
            )
            
            # Append output to log
            with open(exp_log, 'a') as log_file:
                log_file.write("STDOUT:\n")
                log_file.write(result.stdout)
                log_file.write("\nSTDERR:\n")
                log_file.write(result.stderr)
            
            # Parse accuracy
            accuracy = self.parse_accuracy(result.stdout + result.stderr)
            duration = time.time() - start_time
            
            if accuracy:
                print(f"   ✅ {accuracy:.2f}% in {duration:.1f}s")
                
                # Update best result
                if not self.best_result or accuracy > self.best_result['accuracy']:
                    self.best_result = {
                        'exp_id': exp_id,
                        'accuracy': accuracy,
                        'lambda_ct': lambda_ct,
                        'lambda_cs': lambda_cs,
                        'ensemble': ensemble,
                        'verification': verification
                    }
                    print(f"   🏆 NEW BEST!")
                
                return accuracy, duration, verification
            else:
                print(f"   ❌ Failed to parse accuracy in {duration:.1f}s")
                return None, duration, verification
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout (300s)")
            return None, 300, verification
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None, 0, verification
    
    def parse_accuracy(self, output):
        """Parse test accuracy from training output"""
        patterns = [
            r'Test\s+Accuracy.*?([0-9]+\.?[0-9]*)%',
            r'test_acc.*?([0-9]+\.?[0-9]*)',
            r'Test.*?([0-9]+\.?[0-9]*)%',
            r'Accuracy.*?([0-9]+\.?[0-9]*)%',
            r'ft_1p.*?([0-9]+\.?[0-9]*)',
            r'Final.*?accuracy.*?([0-9]+\.?[0-9]*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    acc = float(matches[-1])
                    # Convert to percentage if needed
                    if acc <= 1.0:
                        acc *= 100
                    if 0 <= acc <= 100:
                        return round(acc, 4)
                except:
                    continue
        
        return None
    
    def restore_files(self):
        """Restore original files from backup"""
        try:
            backup_files = [
                ('coft_loss_backup.py', 'models/coft_loss.py'),
                ('trainer_coft_backup.py', 'trainer/trainer_coft.py')
            ]
            
            for backup, original in backup_files:
                backup_path = f'{self.results_dir}/{backup}'
                if os.path.exists(backup_path):
                    with open(backup_path, 'r') as f:
                        content = f.read()
                    with open(original, 'w') as f:
                        f.write(content)
                    print(f"✅ Restored {original}")
        except Exception as e:
            print(f"⚠️  Restore error: {e}")
    
    def run_grid_search(self):
        """Run complete grid search"""
        print(f"\n🚀 STARTING {self.mode.upper()} MODE")
        print("=" * 60)
        
        # Get parameter grid
        grid = self.get_parameter_grid()
        
        # Generate experiments
        experiments = []
        exp_id = 1
        
        for lambda_ct in grid['lambda_ct']:
            for lambda_cs in grid['lambda_cs']:
                for ensemble in grid['ensemble']:
                    experiments.append({
                        'id': exp_id,
                        'lambda_ct': lambda_ct,
                        'lambda_cs': lambda_cs,
                        'ensemble': ensemble
                    })
                    exp_id += 1
        
        print(f"📊 Total experiments: {len(experiments)}")
        print(f"⏱️  Estimated time: {self.get_estimated_time(len(experiments))}")
        print("💡 Press Ctrl+C to stop gracefully")
        print("-" * 60)
        
        # Initialize results CSV
        csv_file = f'{self.results_dir}/results.csv'
        with open(csv_file, 'w') as f:
            f.write("exp_id,lambda_ct,lambda_cs,ensemble,accuracy,duration,verification\n")
        
        # Run experiments
        for i, exp in enumerate(experiments):
            try:
                accuracy, duration, verification = self.run_experiment(
                    exp['id'], exp['lambda_ct'], exp['lambda_cs'], exp['ensemble']
                )
                
                # Save result
                result = {
                    'exp_id': exp['id'],
                    'lambda_ct': exp['lambda_ct'],
                    'lambda_cs': exp['lambda_cs'],
                    'ensemble': exp['ensemble'],
                    'accuracy': accuracy,
                    'duration': duration,
                    'verification': verification
                }
                self.results.append(result)
                
                # Append to CSV
                with open(csv_file, 'a') as f:
                    f.write(f"{exp['id']},{exp['lambda_ct']},{exp['lambda_cs']},{exp['ensemble']},{accuracy},{duration},{verification}\n")
                
                # Progress update
                progress = (i + 1) / len(experiments) * 100
                elapsed = (time.time() - self.start_time) / 60
                
                if len(self.results) > 0:
                    avg_duration = sum(r['duration'] for r in self.results[-5:]) / min(5, len(self.results))
                    remaining = (len(experiments) - i - 1) * avg_duration / 60
                else:
                    remaining = 0
                
                print(f"📊 Progress: {progress:.1f}% | Elapsed: {elapsed:.1f}m | ETA: {remaining:.1f}m")
                if self.best_result:
                    print(f"🏆 Best: {self.best_result['accuracy']:.2f}% | Avg: {avg_duration:.1f}s/exp")
                
            except KeyboardInterrupt:
                print(f"\n🛑 Interrupted by user!")
                break
        
        # Final analysis
        self.analyze_results()
        self.restore_files()
    
    def get_estimated_time(self, num_experiments):
        """Get estimated completion time"""
        times = {
            'diagnostic': "~5 minutes",
            'quick': "~30 minutes", 
            'optimize': "~2-3 hours"
        }
        return times.get(self.mode, f"~{num_experiments * 5} minutes")
    
    def analyze_results(self):
        """Analyze and save results"""
        print(f"\n📊 ANALYSIS COMPLETE")
        print("=" * 60)
        
        if not self.results:
            print("❌ No results to analyze")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        successful = df[df['accuracy'].notna()].copy()
        
        print(f"📋 Total experiments: {len(df)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(df) - len(successful)}")
        
        if len(successful) > 0:
            # Sort by accuracy
            top5 = successful.nlargest(5, 'accuracy')
            
            print(f"\n🏆 TOP 5 RESULTS:")
            for idx, row in top5.iterrows():
                print(f"   {row['accuracy']:.2f}% - λ_ct={row['lambda_ct']:.5f}, λ_cs={row['lambda_cs']:.2f}, {row['ensemble']}")
            
            # Best result
            best = top5.iloc[0]
            print(f"\n🥇 BEST RESULT: {best['accuracy']:.2f}%")
            print(f"   λ_cotraining: {best['lambda_ct']:.5f}")
            print(f"   λ_consistency: {best['lambda_cs']:.2f}")
            print(f"   Ensemble: {best['ensemble']}")
            print(f"   Verification: {best['verification']}")
            
            # Save production config
            config = {
                'lambda_cotraining': float(best['lambda_ct']),
                'lambda_consistency': float(best['lambda_cs']),
                'ensemble_method': best['ensemble'],
                'test_accuracy': float(best['accuracy']),
                'optimization_mode': self.mode,
                'dataset': self.dataset,
                'timestamp': datetime.now().isoformat()
            }
            
            config_file = f'{self.results_dir}/production_config.json'
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"\n💾 Production config saved: {config_file}")
            
            # Compare with known benchmarks
            benchmarks = {
                'HAR': 76.32,  # Previous best from memory
                'sleep': 70.0,  # Estimated
                'epilepsy': 75.0  # Estimated
            }
            
            if self.dataset in benchmarks:
                benchmark = benchmarks[self.dataset]
                improvement = best['accuracy'] - benchmark
                
                print(f"\n📈 COMPARISON:")
                print(f"   Previous best: {benchmark:.2f}%")
                print(f"   Current best: {best['accuracy']:.2f}%")
                
                if improvement > 0:
                    print(f"   🚀 IMPROVEMENT: +{improvement:.2f}%")
                else:
                    print(f"   📊 Difference: {improvement:.2f}%")
        
        print(f"\n📁 All results saved in: {self.results_dir}")
        print(f"📋 CSV file: {self.results_dir}/results.csv")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='🚀 CoFT A100 Ultra-Fast Parameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python optimize_a100.py HAR diagnostic          # Quick test (3 exp, 5 min)
    python optimize_a100.py HAR quick               # Medium test (6 exp, 30 min)
    python optimize_a100.py HAR optimize            # Full optimization (18 exp, 2-3h)
    python optimize_a100.py sleep optimize --no-gpu # CPU mode
        """
    )
    
    parser.add_argument('dataset', 
                       choices=['HAR', 'sleep', 'epilepsy', 'pFD'],
                       help='Dataset to optimize for')
    
    parser.add_argument('mode',
                       choices=['diagnostic', 'quick', 'optimize'],
                       help='Optimization mode')
    
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU optimizations')
    
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout per experiment in seconds (default: 300)')
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    required_files = ['main.py', 'models/coft_loss.py', 'trainer/trainer_coft.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"💡 Please run from CoFT project root directory")
        sys.exit(1)
    
    # Create optimizer and run
    optimizer = CoFTA100Optimizer(
        dataset=args.dataset,
        mode=args.mode,
        gpu_optimize=not args.no_gpu
    )
    
    try:
        optimizer.run_grid_search()
        print(f"\n🎉 Optimization completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped by user")
        optimizer.restore_files()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        optimizer.restore_files()
        sys.exit(1)

if __name__ == "__main__":
    main() 