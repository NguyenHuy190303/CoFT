import os
import sys
import time
from datetime import datetime, timedelta

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from models.coft_loss import CoFTHybridLoss
from models.coft_cotraining import CoFTCoTraining, CoFTEnsemble
from models.frequency_model import FrequencyAugmentation


def CoFTTrainer(model, temporal_contr_model, frequency_model, frequency_contr_model,
                model_optimizer, temp_cont_optimizer, freq_optimizer, freq_cont_optimizer,
                train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, 
                training_mode, enable_coft=True,
                # Memory optimization arguments
                memory_efficient=False, gradient_accumulation=1, mixed_precision=False,
                gradient_checkpointing=False, clear_cache_freq=10):
    """
    CoFT Trainer that handles both temporal and frequency branches with co-training.
    Includes memory optimization features for resource-constrained systems.
    """
    # Start timing
    training_start_time = time.time()
    logger.debug("CoFT Training started ....")
    print(f"🕐 CoFT Training started at: {datetime.now().strftime('%H:%M:%S')}")
    
    # Memory optimization setup
    if mixed_precision and hasattr(torch, 'amp'):
        scaler = torch.cuda.amp.GradScaler()
        logger.debug("🚀 Mixed precision training enabled")
    elif mixed_precision:
        # Fallback for older PyTorch versions
        mixed_precision = False  # Disable mixed precision if not available
        scaler = None
        logger.debug("⚠️ Mixed precision not available on this PyTorch version - disabled")
    else:
        scaler = None
    
    if gradient_checkpointing:
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        if enable_coft and hasattr(frequency_model, 'enable_gradient_checkpointing'):
            frequency_model.enable_gradient_checkpointing()
        logger.debug("🚀 Gradient checkpointing enabled")
    
    # Log memory optimization settings
    if memory_efficient:
        memory_features = []
        if mixed_precision:
            memory_features.append("FP16")
        if gradient_accumulation > 1:
            memory_features.append(f"GradAccum x{gradient_accumulation}")
        if gradient_checkpointing:
            memory_features.append("Checkpointing")
        if clear_cache_freq > 0:
            memory_features.append(f"CacheFlush /{clear_cache_freq}")
        
        logger.debug(f"🎯 Memory Optimizations Active: {', '.join(memory_features)}")
        logger.debug(f"📊 Batch Size: {config.batch_size}, Effective: {config.batch_size * gradient_accumulation}")
    
    # Initialize CoFT-specific components
    hybrid_loss = CoFTHybridLoss(device, config, training_mode)
    cotraining_module = CoFTCoTraining(config) if enable_coft else None
    ensemble_module = CoFTEnsemble() if enable_coft else None
    freq_augmentation = FrequencyAugmentation() if enable_coft else None
    
    # Move CoFT components to device
    if enable_coft:
        cotraining_module = cotraining_module.to(device)
        ensemble_module = ensemble_module.to(device)
        freq_augmentation = freq_augmentation.to(device)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Update loss weights dynamically
        hybrid_loss.update_weights(epoch, config.num_epoch)
        
        # Train and validate
        train_loss, train_acc = coft_model_train(
            model, temporal_contr_model, frequency_model, frequency_contr_model,
            model_optimizer, temp_cont_optimizer, freq_optimizer, freq_cont_optimizer,
            criterion, train_dl, config, device, training_mode, enable_coft,
            hybrid_loss, cotraining_module, freq_augmentation, ensemble_module,
            # Memory optimization arguments
            scaler=scaler, gradient_accumulation=gradient_accumulation,
            mixed_precision=mixed_precision, clear_cache_freq=clear_cache_freq
        )
        
        valid_loss, valid_acc, _, _ = coft_model_evaluate(
            model, temporal_contr_model, frequency_model, frequency_contr_model,
            valid_dl, device, training_mode, enable_coft, ensemble_module, config
        )
        
        if (training_mode != "self_supervised") and (training_mode != "SupCon"):
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')
        
        # Memory management
        if memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            current_memory = torch.cuda.memory_allocated() / 1e6  # MB
            if epoch % 5 == 0:  # Log every 5 epochs
                logger.debug(f"💾 GPU Memory: {current_memory:.1f} MB")

    # Save the model after training
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {
        'model_state_dict': model.state_dict(),
        'temporal_contr_model_state_dict': temporal_contr_model.state_dict()
    }
    
    # Save frequency models if CoFT is enabled
    if enable_coft and frequency_model is not None:
        chkpoint['frequency_model_state_dict'] = frequency_model.state_dict()
        chkpoint['frequency_contr_model_state_dict'] = frequency_contr_model.state_dict()
        chkpoint['cotraining_module_state_dict'] = cotraining_module.state_dict()
        chkpoint['ensemble_module_state_dict'] = ensemble_module.state_dict()
    
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if (training_mode != "self_supervised") and (training_mode != "SupCon"):
        # Evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, pred_labels, true_labels = coft_model_evaluate(
            model, temporal_contr_model, frequency_model, frequency_contr_model,
            test_dl, device, training_mode, enable_coft, ensemble_module, config
        )
        
        # Calculate F1 score if we have predictions
        if len(pred_labels) > 0 and len(true_labels) > 0:
            f1_macro = f1_score(true_labels, pred_labels, average='macro')
            f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
            
            logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')
            logger.debug(f'F1 Score (Macro): {f1_macro:2.4f}\t | F1 Score (Weighted): {f1_weighted:2.4f}')
            
            print(f"\n📊 CoFT TRAINING COMPLETED - TEST METRICS:")
            print(f"   🎯 Test Accuracy: {test_acc*100:.2f}%")
            print(f"   📈 F1 Score (Macro): {f1_macro*100:.2f}%")
            print(f"   📊 F1 Score (Weighted): {f1_weighted*100:.2f}%")
        else:
            logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')

    # Calculate and display total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    training_duration = str(timedelta(seconds=int(total_training_time)))
    
    logger.debug(f"\n################## CoFT Training is Done! #########################")
    logger.debug(f"CoFT Training time is : {training_duration}")
    
    print(f"\n⏰ CoFT TRAINING COMPLETED:")
    print(f"   🕐 Total Training Time: {training_duration}")
    print(f"   ⏱️  Average per Epoch: {total_training_time/config.num_epoch:.2f} seconds")
    print(f"   🏁 Finished at: {datetime.now().strftime('%H:%M:%S')}")


def coft_model_train(model, temporal_contr_model, frequency_model, frequency_contr_model,
                     model_optimizer, temp_cont_optimizer, freq_optimizer, freq_cont_optimizer,
                     criterion, train_loader, config, device, training_mode, enable_coft,
                     hybrid_loss, cotraining_module, freq_augmentation, ensemble_module,
                     # Memory optimization arguments
                     scaler=None, gradient_accumulation=1, mixed_precision=False, clear_cache_freq=10):
    """Training loop for CoFT model with memory optimizations."""
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()
    
    if enable_coft:
        frequency_model.train()
        frequency_contr_model.train()
        cotraining_module.train()

    # Initialize accumulated loss tracking
    accumulated_loss = 0.0
    step_count = 0

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # Send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # Forward pass with mixed precision if enabled and available
        # Check if torch.amp is available (PyTorch >= 1.6)
        has_amp = hasattr(torch, 'amp') and mixed_precision
        if has_amp:
            autocast_context = torch.amp.autocast('cuda', enabled=True)
        else:
            from contextlib import nullcontext
            autocast_context = nullcontext()
            
        with autocast_context:
            temporal_outputs = {}
            frequency_outputs = {}

            if training_mode == "self_supervised" or training_mode == "SupCon":
                # Temporal branch processing
                predictions1, features1 = model(aug1)
                predictions2, features2 = model(aug2)

                # Normalize projection feature vectors
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

                temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(features1, features2)
                temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(features2, features1)

                temporal_outputs = {
                    'contrastive_loss': (temp_cont_loss1, temp_cont_loss2),
                    'contrastive_features': (temp_cont_feat1, temp_cont_feat2),
                    'logits': predictions1,  # For co-training
                    'features': features1
                }

                # Frequency branch processing (if enabled)
                if enable_coft:
                    # Apply frequency-domain augmentations
                    freq_aug1, freq_aug2 = freq_augmentation(data)
                    
                    freq_predictions1, freq_features1 = frequency_model(freq_aug1)
                    freq_predictions2, freq_features2 = frequency_model(freq_aug2)

                    # Normalize frequency features
                    freq_features1 = F.normalize(freq_features1, dim=1)
                    freq_features2 = F.normalize(freq_features2, dim=1)

                    freq_cont_loss1, freq_cont_feat1 = frequency_contr_model(freq_features1, freq_features2)
                    freq_cont_loss2, freq_cont_feat2 = frequency_contr_model(freq_features2, freq_features1)

                    frequency_outputs = {
                        'contrastive_loss': (freq_cont_loss1, freq_cont_loss2),
                        'contrastive_features': (freq_cont_feat1, freq_cont_feat2),
                        'logits': freq_predictions1,  # For co-training
                        'features': freq_features1
                    }

            else:
                # Supervised training mode
                predictions, features = model(data)
                temporal_outputs = {
                    'logits': predictions,
                    'features': features
                }
                
                if enable_coft:
                    freq_predictions, freq_features = frequency_model(data)
                    frequency_outputs = {
                        'logits': freq_predictions,
                        'features': freq_features
                    }

            # Compute hybrid loss
            if enable_coft:
                loss, loss_dict = hybrid_loss(
                    temporal_outputs, frequency_outputs, labels, cotraining_module
                )
                
                # Add accuracy calculation for CoFT supervised modes
                if training_mode not in ["self_supervised", "SupCon"] and 'logits' in temporal_outputs:
                    total_acc.append(labels.eq(temporal_outputs['logits'].detach().argmax(dim=1)).float().mean())
            else:
                # Fall back to original loss computation
                if training_mode == "self_supervised":
                    lambda1 = 1
                    lambda2 = 0.7
                    from models.loss import NTXentLoss
                    nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                                   config.Context_Cont.use_cosine_similarity)
                    loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                           nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2
                elif training_mode == "SupCon":
                    # Paper specs for CA-TCC:
                    # λ3 (Temporal Contrasting Loss): 0.01
                    # λ4 (Supervised Contextual Contrasting Loss): 0.7
                    lambda1 = 0.01
                    lambda2 = 0.7  # Paper: 0.7 instead of 0.1 - CRITICAL FIX!
                    from models.loss import SupConLoss
                    Sup_contrastive_criterion = SupConLoss(device)

                    supCon_features = torch.cat([temp_cont_feat1.unsqueeze(1), temp_cont_feat2.unsqueeze(1)], dim=1)
                    loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + Sup_contrastive_criterion(supCon_features,
                                                                                                     labels) * lambda2
                else:
                    loss = criterion(predictions, labels)
                    total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation
            accumulated_loss += loss.item()

        # Backward pass with mixed precision support
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        step_count += 1

        # Update parameters when we've accumulated enough gradients
        if step_count % gradient_accumulation == 0:
            # Apply gradient clipping for stability
            if scaler is not None:
                # Mixed precision gradient clipping
                scaler.unscale_(model_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if training_mode == "self_supervised" or training_mode == "SupCon":
                    scaler.unscale_(temp_cont_optimizer)
                    torch.nn.utils.clip_grad_norm_(temporal_contr_model.parameters(), max_norm=1.0)
                    
                    if enable_coft:
                        scaler.unscale_(freq_optimizer)
                        scaler.unscale_(freq_cont_optimizer)
                        torch.nn.utils.clip_grad_norm_(frequency_model.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(frequency_contr_model.parameters(), max_norm=1.0)

                # Update with scaler
                scaler.step(model_optimizer)
                if training_mode == "self_supervised" or training_mode == "SupCon":
                    scaler.step(temp_cont_optimizer)
                    if enable_coft:
                        scaler.step(freq_optimizer)
                        scaler.step(freq_cont_optimizer)
                scaler.update()
            else:
                # Standard gradient clipping and update
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                model_optimizer.step()
                
                if training_mode == "self_supervised" or training_mode == "SupCon":
                    torch.nn.utils.clip_grad_norm_(temporal_contr_model.parameters(), max_norm=1.0)
                    temp_cont_optimizer.step()
                    
                    if enable_coft:
                        torch.nn.utils.clip_grad_norm_(frequency_model.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(frequency_contr_model.parameters(), max_norm=1.0)
                        freq_optimizer.step()
                        freq_cont_optimizer.step()

            # Zero gradients after update
            model_optimizer.zero_grad()
            temp_cont_optimizer.zero_grad()
            if enable_coft:
                freq_optimizer.zero_grad()
                freq_cont_optimizer.zero_grad()

            # Store the accumulated loss
            total_loss.append(accumulated_loss)
            accumulated_loss = 0.0

        # Memory management - clear cache periodically
        if clear_cache_freq > 0 and batch_idx % clear_cache_freq == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Handle any remaining accumulated loss
    if step_count % gradient_accumulation != 0:
        total_loss.append(accumulated_loss)

    total_loss = torch.tensor(total_loss).mean()

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    
    return total_loss, total_acc


def coft_model_evaluate(model, temporal_contr_model, frequency_model, frequency_contr_model,
                        test_dl, device, training_mode, enable_coft, ensemble_module, config=None):
    """Evaluation loop for CoFT model."""
    model.eval()
    temporal_contr_model.eval()
    
    if enable_coft:
        frequency_model.eval()
        frequency_contr_model.eval()
        ensemble_module.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if (training_mode == "self_supervised") or (training_mode == "SupCon"):
                pass
            else:
                # Get temporal predictions
                temporal_output = model(data)
                predictions, features = temporal_output

                # Get frequency predictions and ensemble if CoFT enabled
                if enable_coft:
                    freq_predictions, freq_features = frequency_model(data)
                    
                    # Use optimal ensemble method from HAR config if available
                    if config is not None and hasattr(config, 'CoFT') and hasattr(config.CoFT, 'ensemble_method'):
                        if config.CoFT.ensemble_method == "temporal_only":
                            final_predictions = predictions  # TEMPORAL_ONLY (HAR optimal: 85.54%)
                        elif config.CoFT.ensemble_method == "frequency_only":
                            final_predictions = freq_predictions  # FREQUENCY_ONLY
                        elif config.CoFT.ensemble_method == "simple_average":
                            final_predictions = (predictions + freq_predictions) / 2  # SIMPLE_AVERAGE
                        else:
                            # Fallback to temporal_only for unknown methods
                            final_predictions = predictions  # TEMPORAL_ONLY (default)
                    else:
                        # Legacy fallback: simple average
                        final_predictions = (predictions + freq_predictions) / 2  # SIMPLE_AVERAGE
                    
                    # ensemble_predictions = ensemble_module(predictions, freq_predictions)  # Advanced ensemble
                else:
                    # CA-TCC baseline: only temporal predictions
                    final_predictions = predictions

                # Compute loss and accuracy
                loss = criterion(final_predictions, labels)
                total_acc.append(labels.eq(final_predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = final_predictions.max(1, keepdim=True)[1]
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_loss = 0
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_loss = torch.tensor(total_loss).mean()
        total_acc = torch.tensor(total_acc).mean()
        return total_loss, total_acc, outs, trgs 