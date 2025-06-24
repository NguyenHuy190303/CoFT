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

from models.loss import NTXentLoss, SupConLoss


def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode,
            # Memory optimization arguments (for consistency)
            memory_efficient=False, gradient_accumulation=1, mixed_precision=False,
            gradient_checkpointing=False, clear_cache_freq=10):

    # Start timing
    training_start_time = time.time()
    logger.debug("Training started ....")
    print(f"🕐 Training started at: {datetime.now().strftime('%H:%M:%S')}")

    # Memory optimization setup (basic support)
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        logger.debug("🚀 Mixed precision training enabled (baseline)")
    else:
        scaler = None

    if memory_efficient:
        logger.debug(f"🎯 Memory optimizations active - Batch: {config.batch_size}, GradAccum: {gradient_accumulation}")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode,
                                            scaler=scaler, gradient_accumulation=gradient_accumulation,
                                            mixed_precision=mixed_precision, clear_cache_freq=clear_cache_freq)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if (training_mode != "self_supervised") and (training_mode != "SupCon"):
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')
        
        # Memory management
        if memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if epoch % 5 == 0:  # Log every 5 epochs
                current_memory = torch.cuda.memory_allocated() / 1e6  # MB
                logger.debug(f"💾 GPU Memory: {current_memory:.1f} MB")

    # save the model after training ...
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if (training_mode != "self_supervised") and (training_mode != "SupCon"):
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, pred_labels, true_labels = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        
        # Calculate F1 score if we have predictions
        if len(pred_labels) > 0 and len(true_labels) > 0:
            f1_macro = f1_score(true_labels, pred_labels, average='macro')
            f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
            
            logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')
            logger.debug(f'F1 Score (Macro): {f1_macro:2.4f}\t | F1 Score (Weighted): {f1_weighted:2.4f}')
            
            print(f"\n📊 TRAINING COMPLETED - TEST METRICS:")
            print(f"   🎯 Test Accuracy: {test_acc*100:.2f}%")
            print(f"   📈 F1 Score (Macro): {f1_macro*100:.2f}%")
            print(f"   📊 F1 Score (Weighted): {f1_weighted*100:.2f}%")
        else:
            logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')

    # Calculate and display total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    training_duration = str(timedelta(seconds=int(total_training_time)))
    
    logger.debug(f"\n################## Training is Done! #########################")
    logger.debug(f"Training time is : {training_duration}")
    
    print(f"\n⏰ TRAINING COMPLETED:")
    print(f"   🕐 Total Training Time: {training_duration}")
    print(f"   ⏱️  Average per Epoch: {total_training_time/config.num_epoch:.2f} seconds")
    print(f"   🏁 Finished at: {datetime.now().strftime('%H:%M:%S')}")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config,
                device, training_mode,
                scaler=None, gradient_accumulation=1, mixed_precision=False, clear_cache_freq=10):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    # Initialize accumulated loss tracking
    accumulated_loss = 0.0
    step_count = 0

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # Forward pass with mixed precision if enabled
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            if training_mode == "self_supervised":
                predictions1, features1 = model(aug1)
                predictions2, features2 = model(aug2)

                # normalize projection feature vectors
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

                temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(features1, features2)
                temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(features2, features1)

                # Paper specs for TS-TCC (unsupervised):
                # λ1 (Temporal Contrasting Loss): 1
                # λ2 (Contextual Contrasting Loss): 0.7
                lambda1 = 1
                lambda2 = 0.7
                nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                               config.Context_Cont.use_cosine_similarity)
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                       nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2

            elif training_mode == "SupCon":
                predictions1, features1 = model(aug1)
                predictions2, features2 = model(aug2)

                # normalize projection feature vectors
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

                temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(features1, features2)
                temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(features2, features1)

                lambda1 = 0.01
                lambda2 = 0.7
                Sup_contrastive_criterion = SupConLoss(device)
                supCon_features = torch.cat([temp_cont_feat1.unsqueeze(1), temp_cont_feat2.unsqueeze(1)], dim=1)
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + Sup_contrastive_criterion(supCon_features, labels) * lambda2

            else:
                predictions, features = model(data)
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
            if scaler is not None:
                # Mixed precision gradient clipping and update
                scaler.unscale_(model_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if training_mode == "self_supervised" or training_mode == "SupCon":
                    scaler.unscale_(temp_cont_optimizer)
                    torch.nn.utils.clip_grad_norm_(temporal_contr_model.parameters(), max_norm=1.0)

                scaler.step(model_optimizer)
                if training_mode == "self_supervised" or training_mode == "SupCon":
                    scaler.step(temp_cont_optimizer)
                scaler.update()
            else:
                # Standard gradient clipping and update
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                model_optimizer.step()
                if training_mode == "self_supervised" or training_mode == "SupCon":
                    torch.nn.utils.clip_grad_norm_(temporal_contr_model.parameters(), max_norm=1.0)
                    temp_cont_optimizer.step()

            # Zero gradients after update
            model_optimizer.zero_grad()
            temp_cont_optimizer.zero_grad()

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


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

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
                output = model(data)

            # compute loss
            if (training_mode != "self_supervised") and (training_mode != "SupCon"):
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_loss = 0
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        return total_loss, total_acc, outs, trgs


def gen_pseudo_labels(model, dataloader, device, experiment_log_dir):
    from sklearn.metrics import accuracy_score
    model.eval()
    softmax = nn.Softmax(dim=1)

    # saving output data
    all_pseudo_labels = np.array([])
    all_labels = np.array([])
    all_data = []

    with torch.no_grad():
        for data, labels, _, _ in dataloader:
            data = data.float().to(device)
            labels = labels.view((-1)).long().to(device)

            # forward pass
            predictions, features = model(data)

            normalized_preds = softmax(predictions)
            pseudo_labels = normalized_preds.max(1, keepdim=True)[1].squeeze()
            all_pseudo_labels = np.append(all_pseudo_labels, pseudo_labels.cpu().numpy())

            all_labels = np.append(all_labels, labels.cpu().numpy())
            all_data.append(data)

    all_data = torch.cat(all_data, dim=0)

    data_save = dict()
    data_save["samples"] = all_data.cpu()
    data_save["labels"] = torch.LongTensor(torch.from_numpy(all_pseudo_labels).long())
    file_name = f"pseudo_train_data.pt"
    torch.save(data_save, os.path.join(experiment_log_dir, file_name))
    print("Pseudo labels generated ...") 