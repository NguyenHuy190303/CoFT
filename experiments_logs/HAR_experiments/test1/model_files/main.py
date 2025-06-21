import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

from dataloader.dataloader import data_generator
from models.TC import TC
from models.model import base_Model
from trainer.trainer import Trainer, model_evaluate, gen_pseudo_labels
from utils import _calc_metrics, copy_Files
from utils import _logger, set_requires_grad

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
        train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
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
            chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
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
            ft_perc = "1p"
            load_from = os.path.join(
                os.path.join(logs_save_dir, experiment_description, run_description, f"ft_{ft_perc}_seed_{SEED}", "saved_models"))
            chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
            pretrained_dict = chkpoint["model_state_dict"]
            model.load_state_dict(pretrained_dict)
            gen_pseudo_labels(model, train_dl, device, data_path)
            # Don't exit here - return success for orchestrator to continue
            logger.debug(f"✅ {mode_name} completed successfully!")
            return True

        if "train_linear" in training_mode or "tl" in training_mode:
            if 'SupCon' not in training_mode:
                load_from = os.path.join(
                    os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                                 "saved_models"))
            else:
                load_from = os.path.join(
                    os.path.join(logs_save_dir, experiment_description, run_description, f"SupCon_seed_{SEED}", "saved_models"))
            chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
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
            data_perc = "1"  # Define data_perc with an appropriate value
            load_from = os.path.join(       
                os.path.join(logs_save_dir, experiment_description, run_description, f"ft_{data_perc}p_seed_{SEED}", "saved_models"))      
            chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)      
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
            
            print(f"CUDA optimizations enabled for {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print("Conservative settings: TF32 disabled for accuracy preservation")
        else:
            print("CUDA not available, running on CPU")

        # Trainer - Choose between original and CoFT trainer based on feature flag
        if args.enable_coft:
            from trainer.trainer_coft import CoFTTrainer
            CoFTTrainer(model, temporal_contr_model, frequency_model, frequency_contr_model,
                       model_optimizer, temporal_contr_optimizer, frequency_optimizer, frequency_contr_optimizer,
                       train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode, args.enable_coft)
        else:
            Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device,
                    logger, configs, experiment_log_dir, training_mode)

        if training_mode != "self_supervised" and training_mode != "SupCon" and training_mode != "SupCon_pseudo":
            # Testing
            outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
            total_loss, total_acc, pred_labels, true_labels = outs
            _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

        elapsed = datetime.now() - overall_start_time
        logger.debug(f"✅ {mode_name} completed successfully! Total elapsed: {elapsed}")
        print(f"✅ {mode_name} completed successfully!")
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
    # Define the complete training pipeline sequence
    TRAINING_PIPELINE = [
        "self_supervised",
        "train_linear_1p", 
        "ft_1p",
        "gen_pseudo_labels",
        "SupCon",
        "train_linear_SupCon_1p"
    ]
    
    overall_start_time = datetime.now()
    
    print(f"\n🎯 Starting Full Training Pipeline")
    print(f"📋 Pipeline: {' → '.join(TRAINING_PIPELINE)}")
    print(f"🗂️ Dataset: {args.selected_dataset}")
    print(f"🔄 CoFT: {'Enabled' if args.enable_coft else 'Disabled'}")
    print(f"⏰ Start Time: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful_modes = []
    failed_modes = []
    
    for i, mode in enumerate(TRAINING_PIPELINE, 1):
        print(f"\n📍 Step {i}/{len(TRAINING_PIPELINE)}: {mode}")
        
        # Execute the training mode
        success = execute_training_mode(args, mode, overall_start_time)
        
        if success:
            successful_modes.append(mode)
            print(f"✅ Step {i} completed: {mode}")
        else:
            failed_modes.append(mode)
            print(f"❌ Step {i} failed: {mode}")
            print(f"🛑 Stopping pipeline due to failure in {mode}")
            break
    
    # Final summary
    total_time = datetime.now() - overall_start_time
    print(f"\n{'='*80}")
    print(f"🏁 TRAINING PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"⏱️ Total Time: {total_time}")
    print(f"✅ Successful: {len(successful_modes)}/{len(TRAINING_PIPELINE)} modes")
    if successful_modes:
        print(f"   {' → '.join(successful_modes)}")
    if failed_modes:
        print(f"❌ Failed: {failed_modes}")
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

    args = parser.parse_args()

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