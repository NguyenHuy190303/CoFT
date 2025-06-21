import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.coft_loss import CoFTHybridLoss
from models.coft_cotraining import CoFTCoTraining, CoFTEnsemble
from models.frequency_model import FrequencyAugmentation


def CoFTTrainer(model, temporal_contr_model, frequency_model, frequency_contr_model,
                model_optimizer, temp_cont_optimizer, freq_optimizer, freq_cont_optimizer,
                train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, 
                training_mode, enable_coft=True):
    """
    CoFT Trainer that handles both temporal and frequency branches with co-training.
    """
    # Start training
    logger.debug("CoFT Training started ....")
    
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
            hybrid_loss, cotraining_module, freq_augmentation, ensemble_module
        )
        
        valid_loss, valid_acc, _, _ = coft_model_evaluate(
            model, temporal_contr_model, frequency_model, frequency_contr_model,
            valid_dl, device, training_mode, enable_coft, ensemble_module
        )
        
        if (training_mode != "self_supervised") and (training_mode != "SupCon"):
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

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
        test_loss, test_acc, _, _ = coft_model_evaluate(
            model, temporal_contr_model, frequency_model, frequency_contr_model,
            test_dl, device, training_mode, enable_coft, ensemble_module
        )
        logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')

    logger.debug("\n################## CoFT Training is Done! #########################")


def coft_model_train(model, temporal_contr_model, frequency_model, frequency_contr_model,
                     model_optimizer, temp_cont_optimizer, freq_optimizer, freq_cont_optimizer,
                     criterion, train_loader, config, device, training_mode, enable_coft,
                     hybrid_loss, cotraining_module, freq_augmentation, ensemble_module):
    """Training loop for CoFT model."""
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()
    
    if enable_coft:
        frequency_model.train()
        frequency_contr_model.train()
        cotraining_module.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # Send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # Zero gradients
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()
        if enable_coft:
            freq_optimizer.zero_grad()
            freq_cont_optimizer.zero_grad()

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
            
            # FIXED: Add accuracy calculation for CoFT supervised modes
            if training_mode not in ["self_supervised", "SupCon"] and 'logits' in temporal_outputs:
                # DEBUGGING: Use only temporal predictions for now to isolate issue
                total_acc.append(labels.eq(temporal_outputs['logits'].detach().argmax(dim=1)).float().mean())
                
                # DEBUG: Log prediction comparison if needed
                # if 'logits' in frequency_outputs:
                #     temp_acc = labels.eq(temporal_outputs['logits'].detach().argmax(dim=1)).float().mean()
                #     freq_acc = labels.eq(frequency_outputs['logits'].detach().argmax(dim=1)).float().mean()
                #     print(f"DEBUG - Temporal Acc: {temp_acc:.4f}, Freq Acc: {freq_acc:.4f}")
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
                lambda1 = 0.01
                lambda2 = 0.1
                from models.loss import SupConLoss
                Sup_contrastive_criterion = SupConLoss(device)
                supCon_features = torch.cat([temp_cont_feat1.unsqueeze(1), temp_cont_feat2.unsqueeze(1)], dim=1)
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + Sup_contrastive_criterion(supCon_features, labels) * lambda2
            else:
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())

        # Backward pass
        loss.backward()
        
        # Update parameters
        model_optimizer.step()
        if training_mode == "self_supervised" or training_mode == "SupCon":
            temp_cont_optimizer.step()
            if enable_coft:
                freq_optimizer.step()
                freq_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    
    return total_loss, total_acc


def coft_model_evaluate(model, temporal_contr_model, frequency_model, frequency_contr_model,
                        test_dl, device, training_mode, enable_coft, ensemble_module):
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
                    # DEBUGGING: Use only temporal predictions for now
                    ensemble_predictions = ensemble_module(predictions, freq_predictions); final_predictions = ensemble_predictions
                    # TODO: Re-enable ensemble after debugging
                    # ensemble_predictions = ensemble_module(predictions, freq_predictions)
                    # final_predictions = ensemble_predictions
                else:
                    ensemble_predictions = ensemble_module(predictions, freq_predictions); final_predictions = ensemble_predictions

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