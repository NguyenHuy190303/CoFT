import argparse
import os
import sys
from datetime import datetime, timedelta
import warnings

import numpy as np
import torch

from dataloader.dataloader import data_generator
from models.TC import TC
from models.model import base_Model
from trainer.trainer import Trainer, model_evaluate, gen_pseudo_labels
from utils import _calc_metrics, copy_Files
from utils import _logger, set_requires_grad

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def safe_torch_load(filepath, device=None, **kwargs):
    """Safe torch.load wrapper that avoids weights_only for compatibility"""
    if device:
        return torch.load(filepath, map_location=device, **kwargs)
    else:
        return torch.load(filepath, **kwargs)

def execute_training_mode(args, mode_name, overall_start_time):
    """
    Execute a single training mode with proper setup and cleanup.
    
    Args:
        args: Command line arguments (modified with current mode)
        mode_name: Current training mode to execute
        overall_start_time: Start time of the full orchestrator run
        
    Returns:
        bool: True if execution successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting Training Mode: {mode_name}")
    print(f"{'='*60}")
    
    # Update args for current mode
    args.training_mode = mode_name
    training_mode = mode_name
    
    device = torch.device(args.device)
    experiment_description = args.experiment_description
    data_type = args.selected_dataset.replace("-", "_")
    run_description = args.run_description

    logs_save_dir = args.logs_save_dir
    os.makedirs(logs_save_dir, exist_ok=True)

    # Dynamic import of dataset config
    if data_type == "HAR":
        from config_files.HAR_Configs import Config as Configs
    elif data_type == "sleep":
        from config_files.sleep_Configs import Config as Configs  
    elif data_type == "Epilepsy":
        from config_files.Epilepsy_Configs import Config as Configs
    elif data_type == "pFD":
        from config_files.pFD_Configs import Config as Configs
    else:
        raise ValueError(f"Unknown dataset: {data_type}")
    
    configs = Configs()

    # ##### fix random seeds for reproducibility ########
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(SEED)
    #####################################################
    
    # Override InfoTS setting if specified via command line
    # --- BEGIN LEGACY CODE ---
    # if args.enable_infots:
    #     configs.augmentation.use_infots_augmentation = True
    #     print(f"🎨 InfoTS augmentation ENABLED via command line for {data_type} dataset")
    # --- END LEGACY CODE ---

    # Log augmentation status
    # --- BEGIN LEGACY CODE ---
    # infots_status = getattr(configs.augmentation, 'use_infots_augmentation', False)
    # if infots_status:
    #     print(f"🎨 InfoTS augmentation ENABLED via config for {data_type} dataset")
    # else:
    #     print(f"📊 InfoTS augmentation DISABLED for {data_type} dataset (using CoFT baseline)")
        # --- END LEGACY CODE ---

    # Memory Optimization Configuration (Reduced Output)
    if args.memory_efficient or args.reduced_batch_size or args.enable_coft:
        # Auto-detect memory constraints and apply optimizations
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            device_name = torch.cuda.get_device_name(0)
            
            # Simplified GPU info
            print(f"🔍 GPU: {device_name} ({total_memory_gb:.1f} GB)")
            
            # High-end GPU configuration (24GB+)
            if total_memory_gb >= 24.0:
                # print(f"🚀 High-end GPU detected ({total_memory_gb:.1f}GB) - Premium optimizations available")
                
                # For high-end GPUs, focus on speed rather than memory savings
                if args.reduced_batch_size:
                    configs.batch_size = min(configs.batch_size * 2, 512)  # Conservative boost
                    # print(f"📊 Batch size boosted to {configs.batch_size} for high-end GPU")
                
                # Enable mixed precision by default for high-end GPUs
                if not args.mixed_precision:
                    args.mixed_precision = True
                    # print("🚀 Auto-enabled mixed precision for high-end GPU")
                    
            # Mid-range GPU configuration (8-24GB)  
            elif total_memory_gb >= 8.0:
                # print(f"⚡ Mid-range GPU detected ({total_memory_gb:.1f}GB) - Balanced optimizations")
                
                if args.reduced_batch_size:
                    configs.batch_size = int(configs.batch_size * 1.5)  # Moderate boost
                    # print(f"📊 Batch size adjusted to {configs.batch_size}")
                    
            # Entry-level GPU configuration (<8GB)
            else:
                # print(f"💾 Entry-level GPU detected ({total_memory_gb:.1f}GB) - Memory-focused optimizations")
                
                # Aggressive memory optimizations for entry-level GPUs
                if not args.memory_efficient:
                    args.memory_efficient = True
                    # print("💾 Auto-enabled memory optimizations")
                
                if not args.mixed_precision:
                    args.mixed_precision = True
                    # print("🚀 Auto-enabled mixed precision for memory savings")
                
                if args.gradient_accumulation < 2:
                    args.gradient_accumulation = 2
                    # print("📈 Auto-enabled gradient accumulation")
                
                if args.reduced_batch_size:
                    configs.batch_size = max(configs.batch_size // 2, 16)  # Reduce batch size
                    # print(f"📊 Reduced batch size to {configs.batch_size} for memory efficiency")
                    
            # Batch size configuration summary
            if args.reduced_batch_size or args.memory_efficient or args.enable_coft:
                original_batch_size = 128  # Default from configs
                effective_batch_size = configs.batch_size * args.gradient_accumulation
                
                # print(f"\n📊 BATCH SIZE CONFIGURATION:")
                # print(f"   Original Batch Size: {original_batch_size}")
                # print(f"   New Batch Size: {configs.batch_size}")
                # print(f"   Gradient Accumulation: {args.gradient_accumulation}")
                # print(f"   Effective Batch Size: {effective_batch_size}")
                
                if effective_batch_size > original_batch_size:
                    boost_percentage = ((effective_batch_size - original_batch_size) / original_batch_size) * 100
                    # print(f"🎯 ACCURACY BOOST: Effective batch size increased by {boost_percentage:.1f}%")
                    # print(f"   💡 Expected: Better gradient estimates and potentially higher accuracy")
                elif effective_batch_size < original_batch_size:
                    reduction_percentage = ((original_batch_size - effective_batch_size) / original_batch_size) * 100
                    # print(f"💾 MEMORY SAVING: Effective batch size reduced by {reduction_percentage:.1f}%")
                    # print(f"   💡 Expected: Lower memory usage, maintain performance via grad accumulation")
                    
            # Memory optimization summary
            optimizations = []
            estimated_savings = 0
            
            if args.mixed_precision:
                optimizations.append("Mixed Precision (FP16) - Minimal impact (<0.1%)")
                estimated_savings += 30
            if args.gradient_accumulation > 1:
                optimizations.append(f"Gradient Accumulation (x{args.gradient_accumulation}) - No impact")
            if args.gradient_checkpointing:
                optimizations.append("Gradient Checkpointing - Small impact (~1-2%)")
                estimated_savings += 40
            if args.clear_cache_freq > 0:
                optimizations.append("Memory Management (every 50 batches) - No impact")
            
            if optimizations:
                # print(f"\n🚀 MEMORY OPTIMIZATION SUMMARY:")
                for opt in optimizations:
                    # print(f"   ✅ {opt}")
                    pass
                if estimated_savings > 0:
                    # print(f"   💾 Estimated Memory Savings: ~{estimated_savings}%")
                    pass
            
            # Learning rate scaling recommendation
            if args.reduced_batch_size and effective_batch_size != original_batch_size:
                scale_factor = (effective_batch_size / original_batch_size) ** 0.5
                recommended_lr = configs.lr * scale_factor
                
                # print(f"\n🎯 LEARNING RATE RECOMMENDATION:")
                # print(f"   Current LR: {configs.lr}")
                # print(f"   Recommended LR: {recommended_lr:.2e} (scale factor: {scale_factor:.3f})")
                # print(f"   💡 Add --lr_auto_scale to apply automatically")
                
                if args.lr_auto_scale:
                    configs.lr = recommended_lr
                    # print(f"   ✅ Auto-applied: LR scaled to {configs.lr:.2e}")
        else:
            # print("⚠️  GPU not available - CPU mode with conservative settings")
            pass

    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                      training_mode + f"_seed_{SEED}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    # Logging
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Mode:    {training_mode}')
    logger.debug(f'CoFT:    {"Enabled" if args.enable_coft else "Disabled"}')
    logger.debug("=" * 45)

    try:
        # Load datasets
        data_path = os.path.join(args.data_path, data_type)
        train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode, args.enable_coft)
        logger.debug("Data loaded ...")

        # Load Model
        model = base_Model(configs).to(device)
        temporal_contr_model = TC(configs, device).to(device)

        # CoFT: Initialize frequency branch and co-training components conditionally
        frequency_model = None
        frequency_contr_model = None
        frequency_optimizer = None
        frequency_contr_optimizer = None

        if args.enable_coft:
            from models.frequency_model import FrequencyModel
            from models.frequency_contrastive import FrequencyContrastive
            
            frequency_model = FrequencyModel(configs).to(device)
            frequency_contr_model = FrequencyContrastive(configs, device).to(device)
            
            # Initialize optimizers for frequency components
            frequency_optimizer = torch.optim.Adam(frequency_model.parameters(), lr=configs.lr,
                                                  betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
            frequency_contr_optimizer = torch.optim.Adam(frequency_contr_model.parameters(), lr=configs.lr,
                                                       betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
            
            logger.debug("CoFT: Frequency branch and optimizers initialized")

        # Model loading logic based on training mode
        if "fine_tune" in training_mode or "ft_" in training_mode:
            # load saved model of this experiment
            if 'SupCon' not in training_mode:
                load_from = os.path.join(
                    os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                                 "saved_models"))
            else:
                load_from = os.path.join(
                    os.path.join(logs_save_dir, experiment_description, run_description, f"SupCon_seed_{SEED}", "saved_models"))
            chkpoint = safe_torch_load(os.path.join(load_from, "ckp_last.pt"), device)
            pretrained_dict = chkpoint["model_state_dict"]
            model_dict = model.state_dict()
            del_list = ['logits']
            pretrained_dict_copy = pretrained_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                    if j in i:
                        del pretrained_dict[i]
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        if training_mode == "gen_pseudo_labels":
            ft_perc = f"{args.label_percentage}p"
            load_from = os.path.join(
                os.path.join(logs_save_dir, experiment_description, run_description, f"ft_{ft_perc}_seed_{SEED}", "saved_models"))
            chkpoint = safe_torch_load(os.path.join(load_from, "ckp_last.pt"), device)
            pretrained_dict = chkpoint["model_state_dict"]
            model.load_state_dict(pretrained_dict)
            gen_pseudo_labels(model, train_dl, device, data_path)
            # Don't exit here - return success for orchestrator to continue
            logger.debug(f"SUCCESS: {mode_name} completed successfully!")
            return True

        if "train_linear" in training_mode or "tl" in training_mode:
            if 'SupCon' not in training_mode:
                load_from = os.path.join(
                    os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                                 "saved_models"))
            else:
                load_from = os.path.join(
                    os.path.join(logs_save_dir, experiment_description, run_description, f"SupCon_seed_{SEED}", "saved_models"))
            chkpoint = safe_torch_load(os.path.join(load_from, "ckp_last.pt"), device)
            pretrained_dict = chkpoint["model_state_dict"]
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # delete these parameters (Ex: the linear layer at the end)
            del_list = ['logits']
            pretrained_dict_copy = pretrained_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                    if j in i:
                        del pretrained_dict[i]

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

        if training_mode == "random_init":
            model_dict = model.state_dict()

            # delete all the parameters except for logits
            del_list = ['logits']
            pretrained_dict_copy = model_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                    if j in i:
                        del model_dict[i]
            set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.

        if training_mode == "SupCon":
            data_perc = str(args.label_percentage)  # Use configurable percentage
            load_from = os.path.join(       
                os.path.join(logs_save_dir, experiment_description, run_description, f"ft_{data_perc}p_seed_{SEED}", "saved_models"))      
            chkpoint = safe_torch_load(os.path.join(load_from, "ckp_last.pt"), device)      
            pretrained_dict = chkpoint["model_state_dict"]      
            model.load_state_dict(pretrained_dict)     

        model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                           weight_decay=3e-4)

        temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
                                                    betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

        if training_mode == "self_supervised" or training_mode == "SupCon":  # to do it only once
            copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

        # Conservative CUDA Optimizations - Only proven beneficial settings
        if torch.cuda.is_available():
            # Enable benchmark optimization for consistent input sizes
            torch.backends.cudnn.benchmark = True
            
            # Keep deterministic = False for reasonable speed vs reproducibility balance  
            torch.backends.cudnn.deterministic = False
            
            # Disable TF32 to maintain accuracy (small speed cost for better precision)
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            
            # Basic memory management
            torch.cuda.empty_cache()
            
            #print(f"CUDA optimizations enabled for {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print("Conservative settings: TF32 disabled for accuracy preservation")
        else:
            print("CUDA not available, running on CPU")

        # Trainer - Choose between original and CoFT trainer based on feature flag
        if args.enable_coft:
            from trainer.trainer_coft import CoFTTrainer
            CoFTTrainer(model, temporal_contr_model, frequency_model, frequency_contr_model,
                       model_optimizer, temporal_contr_optimizer, frequency_optimizer, frequency_contr_optimizer,
                       train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode, args.enable_coft,
                       # Memory optimization arguments
                       memory_efficient=args.memory_efficient,
                       gradient_accumulation=args.gradient_accumulation,
                       mixed_precision=args.mixed_precision,
                       gradient_checkpointing=args.gradient_checkpointing,
                       clear_cache_freq=args.clear_cache_freq)
        else:
            Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device,
                    logger, configs, experiment_log_dir, training_mode,
                    # Memory optimization arguments
                    memory_efficient=args.memory_efficient,
                    gradient_accumulation=args.gradient_accumulation,
                    mixed_precision=args.mixed_precision,
                    gradient_checkpointing=args.gradient_checkpointing,
                    clear_cache_freq=args.clear_cache_freq)

        if training_mode != "self_supervised" and training_mode != "SupCon" and training_mode != "SupCon_pseudo":
            # Testing
            outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
            total_loss, total_acc, pred_labels, true_labels = outs
            metrics = _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)
            
            # Log comprehensive metrics
            if metrics:
                logger.debug(f"Final Metrics - Accuracy: {metrics['accuracy']:.4f}, F1 Macro: {metrics['f1_macro']:.4f}")

        # Calculate mode execution time
        mode_end_time = datetime.now()
        mode_duration = mode_end_time - overall_start_time
        
        logger.debug(f"SUCCESS: {mode_name} completed successfully! Mode duration: {mode_duration}")
        print(f"\n✅ {mode_name} completed successfully!")
        print(f"   ⏱️  Mode Execution Time: {mode_duration}")
        return True
        
    except Exception as e:
        print(f"❌ Error in {mode_name}: {str(e)}")
        logger.error(f"❌ Error in {mode_name}: {str(e)}")
        return False


def training_orchestrator(args):
    """
    Orchestrator function to run multiple training modes sequentially.
    
    Args:
        args: Command line arguments
    """
    # Define the complete training pipeline sequence with configurable label percentage
    label_perc = f"{args.label_percentage}p"
    TRAINING_PIPELINE = [
        "self_supervised",
        f"train_linear_{label_perc}", 
        f"ft_{label_perc}",
        "gen_pseudo_labels",
        "SupCon",
        f"train_linear_SupCon_{label_perc}"
    ]
    
    overall_start_time = datetime.now()
    
    print(f"\n🎯 Starting Full Training Pipeline")
    print(f"📋 Pipeline: {' → '.join(TRAINING_PIPELINE)}")
    print(f"🗂️ Dataset: {args.selected_dataset}")
    print(f"📊 Label Percentage: {args.label_percentage}%")
    print(f"🔄 CoFT: {'Enabled' if args.enable_coft else 'Disabled'}")
    # --- BEGIN LEGACY CODE ---
    # print(f"🎨 InfoTS: {'Enabled' if args.enable_infots else 'Disabled'}")
    # --- END LEGACY CODE ---
    print(f"⏰ Start Time: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful_modes = []
    failed_modes = []
    mode_times = {}
    
    for i, mode in enumerate(TRAINING_PIPELINE, 1):
        mode_start_time = datetime.now()
        print(f"\n📍 Step {i}/{len(TRAINING_PIPELINE)}: {mode}")
        print(f"🕐 Started at: {mode_start_time.strftime('%H:%M:%S')}")
        
        # Execute the training mode
        success = execute_training_mode(args, mode, overall_start_time)
        
        mode_end_time = datetime.now()
        mode_duration = mode_end_time - mode_start_time
        mode_times[mode] = mode_duration
        
        if success:
            successful_modes.append(mode)
            print(f"✅ Step {i} completed: {mode}")
            print(f"   ⏱️  Step Duration: {mode_duration}")
        else:
            failed_modes.append(mode)
            print(f"❌ Step {i} failed: {mode}")
            print(f"   ⏱️  Step Duration: {mode_duration}")
            print(f"🛑 Stopping pipeline due to failure in {mode}")
            break
    
    # Final summary
    total_time = datetime.now() - overall_start_time
    print(f"\n{'='*80}")
    print(f"🏁 TRAINING PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"⏱️ Total Pipeline Time: {total_time}")
    print(f"🕐 Started: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏁 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Successful: {len(successful_modes)}/{len(TRAINING_PIPELINE)} modes")
    
    if successful_modes:
        print(f"\n📊 DETAILED TIME BREAKDOWN:")
        for mode in successful_modes:
            duration = mode_times.get(mode, timedelta(0))
            percentage = (duration.total_seconds() / total_time.total_seconds()) * 100
            print(f"   ⏱️  {mode:25}: {duration} ({percentage:.1f}%)")
    
    if failed_modes:
        print(f"\n❌ Failed Modes: {failed_modes}")
        for mode in failed_modes:
            duration = mode_times.get(mode, timedelta(0))
            print(f"   ❌ {mode:25}: {duration} (failed)")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    if successful_modes:
        avg_time_per_mode = total_time / len(successful_modes)
        print(f"   ⏱️  Average per Mode: {avg_time_per_mode}")
        print(f"   🚀 CoFT Status: {'Enabled' if args.enable_coft else 'Disabled'}")
        print(f"   🗂️ Dataset: {args.selected_dataset}")
    
    print(f"{'='*80}")
    
    if len(successful_modes) == len(TRAINING_PIPELINE):
        print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
        return True
    else:
        print("⚠️ PIPELINE INCOMPLETE - CHECK FAILED MODES")
        return False


if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser()

    ######################## Model parameters ########################
    home_dir = os.getcwd()
    parser.add_argument('--experiment_description',     default='HAR_experiments',  type=str,   help='Experiment Description')
    parser.add_argument('--run_description',            default='test1',            type=str,   help='Experiment Description')
    parser.add_argument('--seed',                       default=0,                  type=int,   help='seed value')
    parser.add_argument('--training_mode',              default='self_supervised',  type=str,
                        help='Modes of choice: random_init, supervised, self_supervised, SupCon, ft_1p, gen_pseudo_labels, OR full_run for complete pipeline')

    parser.add_argument('--selected_dataset',           default='HAR',              type=str,   help='Dataset of choice: EEG, HAR, Epilepsy, pFD')
    parser.add_argument('--data_path',                  default=r'data/',           type=str,   help='Path containing dataset')

    parser.add_argument('--logs_save_dir',              default='experiments_logs', type=str,   help='saving directory')
    parser.add_argument('--device',                     default='cuda:0',           type=str,   help='cpu or cuda')
    parser.add_argument('--home_path',                  default=home_dir,           type=str,   help='Project home directory')

    ######################## CoFT Feature Flag ########################
    parser.add_argument('--enable_coft',                action='store_true',        default=False,
                        help='Enable Co-training with Frequency and Temporal domains (CoFT). When specified, activates frequency branch and co-training logic.')

    ######################## Memory Optimization ########################
    parser.add_argument('--memory_efficient',          action='store_true',        default=False,
                        help='Enable memory-efficient training with reduced batch sizes and optimizations')
    parser.add_argument('--reduced_batch_size',        type=int,                   default=None,
                        help='Override config batch size for memory-constrained systems (e.g., 32 or 64)')
    parser.add_argument('--gradient_accumulation',     type=int,                   default=1,
                        help='Number of gradient accumulation steps to maintain effective batch size')
    parser.add_argument('--mixed_precision',           action='store_true',        default=False,
                        help='Enable mixed precision training (FP16) to reduce memory usage')
    parser.add_argument('--gradient_checkpointing',    action='store_true',        default=False,
                        help='Enable gradient checkpointing to trade compute for memory')
    parser.add_argument('--clear_cache_freq',          type=int,                   default=10,
                        help='Clear CUDA cache every N batches (0 to disable)')

    ######################## Accuracy Preservation & Validation ########################
    parser.add_argument('--preserve_accuracy',         action='store_true',        default=False,
                        help='Enable aggressive accuracy preservation (may use more memory)')
    parser.add_argument('--accuracy_validation',       action='store_true',        default=False,
                        help='Run baseline comparison to validate accuracy preservation')
    parser.add_argument('--lr_auto_scale',             action='store_true',        default=False,
                        help='Automatically scale learning rate based on effective batch size')
    parser.add_argument('--high_precision_mode',       action='store_true',        default=False,
                        help='Use higher precision settings for critical accuracy requirements')

    ######################## Label Percentage Configuration ########################
    parser.add_argument('--label_percentage',          type=int,                   default=1,
                        help='Label percentage for training (1, 5, 75). Controls which dataset split to use.')

    ######################## InfoTS Augmentation Configuration ########################  
    # --- BEGIN LEGACY CODE ---
    # parser.add_argument('--enable_infots',             action='store_true',        default=False,
    #                     help='Enable InfoTS augmentation for ALL datasets (overrides config file settings)')
    # --- END LEGACY CODE ---

    args = parser.parse_args()

    # Validate label percentage
    valid_percentages = [1, 5, 75]
    if args.label_percentage not in valid_percentages:
        print(f"❌ Error: Invalid label percentage {args.label_percentage}%")
        print(f"   Valid options: {valid_percentages}")
        print(f"   Example: --label_percentage 5 for 5% labels")
        sys.exit(1)

    # Check if orchestrator mode is requested
    if args.training_mode == "full_run":
        print("🚀 FULL TRAINING PIPELINE MODE ACTIVATED")
        success = training_orchestrator(args)
        sys.exit(0 if success else 1)
    else:
        # Single mode execution (original behavior)
        print(f"🎯 SINGLE MODE EXECUTION: {args.training_mode}")
        success = execute_training_mode(args, args.training_mode, start_time)
        sys.exit(0 if success else 1)