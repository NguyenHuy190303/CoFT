import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import NTXentLoss, SupConLoss


class CoFTHybridLoss(nn.Module):
    """
    Hybrid Loss Function for CoFT Architecture.
    Combines temporal contrastive, frequency contrastive, and co-training losses.
    """
    
    def __init__(self, device, configs, mode='self_supervised'):
        super(CoFTHybridLoss, self).__init__()
        
        self.device = device
        self.mode = mode
        self.batch_size = configs.batch_size
        
        # Loss weight configuration - Use dataset-specific optimal parameters if available
        self.lambda_temporal = 1.0      # Temporal contrastive weight
        self.lambda_frequency = 1.0     # Frequency contrastive weight  
        self.lambda_temporal_ntxent = 0.7   # Temporal NT-Xent weight
        self.lambda_freq_ntxent = 0.7       # Frequency NT-Xent weight
        
        # Use dataset-specific optimal parameters if available
        if hasattr(configs, 'CoFT'):
            # HAR dataset optimal parameters (85.54% accuracy)
            self.lambda_cotraining = configs.CoFT.lambda_cotraining     # 0.0001 (HAR optimal)
            self.lambda_consistency = configs.CoFT.lambda_consistency   # 0.01 (HAR optimal)
            self.ensemble_method = configs.CoFT.ensemble_method         # "temporal_only"
            print(f"🎯 Using HAR optimal CoFT parameters: λ_ct={self.lambda_cotraining}, λ_cs={self.lambda_consistency}, ensemble={self.ensemble_method}")
        else:
            # Fallback to previous optimal values
            self.lambda_cotraining = 0.0001        # Co-training weight (OPTIMAL: 0.7632%)
            self.lambda_consistency = 0.15         # Cross-domain consistency weight (OPTIMAL: 0.7632%)
            self.ensemble_method = "temporal_only"  # Default to best performing ensemble
            print(f"⚠️  Using fallback CoFT parameters: λ_ct={self.lambda_cotraining}, λ_cs={self.lambda_consistency}")
        
        # Initialize loss functions
        self.nt_xent_criterion = NTXentLoss(
            device, configs.batch_size, 
            configs.Context_Cont.temperature,
            configs.Context_Cont.use_cosine_similarity
        )
        
        if mode == 'SupCon':
            self.sup_contrastive_criterion = SupConLoss(device)
            self.lambda_supcon_temporal = 0.1
            self.lambda_supcon_freq = 0.1
        
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def compute_temporal_losses(self, temp_cont_loss1, temp_cont_loss2, 
                               temp_cont_feat1, temp_cont_feat2, labels=None):
        """Compute temporal domain losses."""
        temporal_contrastive_loss = (temp_cont_loss1 + temp_cont_loss2) * self.lambda_temporal
        
        if self.mode == 'self_supervised':
            temporal_ntxent_loss = self.nt_xent_criterion(
                temp_cont_feat1, temp_cont_feat2
            ) * self.lambda_temporal_ntxent
            
            return temporal_contrastive_loss + temporal_ntxent_loss
            
        elif self.mode == 'SupCon' and labels is not None:
            supcon_features = torch.cat([
                temp_cont_feat1.unsqueeze(1), 
                temp_cont_feat2.unsqueeze(1)
            ], dim=1)
            
            temporal_supcon_loss = self.sup_contrastive_criterion(
                supcon_features, labels
            ) * self.lambda_supcon_temporal
            
            return temporal_contrastive_loss + temporal_supcon_loss
            
        else:
            return temporal_contrastive_loss
    
    def compute_frequency_losses(self, freq_cont_loss1, freq_cont_loss2,
                                freq_cont_feat1, freq_cont_feat2, labels=None):
        """Compute frequency domain losses."""
        frequency_contrastive_loss = (freq_cont_loss1 + freq_cont_loss2) * self.lambda_frequency
        
        if self.mode == 'self_supervised':
            freq_ntxent_loss = self.nt_xent_criterion(
                freq_cont_feat1, freq_cont_feat2
            ) * self.lambda_freq_ntxent
            
            return frequency_contrastive_loss + freq_ntxent_loss
            
        elif self.mode == 'SupCon' and labels is not None:
            supcon_features = torch.cat([
                freq_cont_feat1.unsqueeze(1),
                freq_cont_feat2.unsqueeze(1)
            ], dim=1)
            
            freq_supcon_loss = self.sup_contrastive_criterion(
                supcon_features, labels
            ) * self.lambda_supcon_freq
            
            return frequency_contrastive_loss + freq_supcon_loss
            
        else:
            return frequency_contrastive_loss
    
    def compute_cotraining_loss(self, temporal_logits, freq_logits, labels,
                               temporal_features, freq_features, cotraining_module):
        """Compute co-training and consistency losses."""
        if cotraining_module is None:
            return torch.tensor(0.0, device=self.device), {}
            
        cotraining_loss, stats = cotraining_module.co_training_loss(
            temporal_logits, freq_logits, labels,
            temporal_features, freq_features
        )
        
        return cotraining_loss * self.lambda_cotraining, stats
    
    def forward(self, temporal_outputs, frequency_outputs, labels=None, 
                cotraining_module=None):
        """
        Compute hybrid loss for CoFT.
        
        Args:
            temporal_outputs: Dict with temporal domain outputs
            frequency_outputs: Dict with frequency domain outputs  
            labels: Ground truth labels (optional)
            cotraining_module: Co-training module for cross-domain losses
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        
        # Temporal domain losses
        if 'contrastive_loss' in temporal_outputs:
            temporal_loss = self.compute_temporal_losses(
                temporal_outputs['contrastive_loss'][0],
                temporal_outputs['contrastive_loss'][1],
                temporal_outputs['contrastive_features'][0],
                temporal_outputs['contrastive_features'][1],
                labels
            )
            loss_dict['temporal_loss'] = temporal_loss
        else:
            temporal_loss = torch.tensor(0.0, device=self.device)
            loss_dict['temporal_loss'] = temporal_loss
        
        # Frequency domain losses  
        if 'contrastive_loss' in frequency_outputs:
            frequency_loss = self.compute_frequency_losses(
                frequency_outputs['contrastive_loss'][0],
                frequency_outputs['contrastive_loss'][1], 
                frequency_outputs['contrastive_features'][0],
                frequency_outputs['contrastive_features'][1],
                labels
            )
            loss_dict['frequency_loss'] = frequency_loss
        else:
            frequency_loss = torch.tensor(0.0, device=self.device)
            loss_dict['frequency_loss'] = frequency_loss
        
        # Co-training losses
        cotraining_loss = torch.tensor(0.0, device=self.device)
        if cotraining_module is not None and 'logits' in temporal_outputs and 'logits' in frequency_outputs:
            cotraining_loss, cotraining_stats = self.compute_cotraining_loss(
                temporal_outputs['logits'],
                frequency_outputs['logits'],
                labels,
                temporal_outputs.get('features'),
                frequency_outputs.get('features'),
                cotraining_module
            )
            loss_dict['cotraining_loss'] = cotraining_loss
            loss_dict.update(cotraining_stats)
        else:
            loss_dict['cotraining_loss'] = cotraining_loss
        
        # Supervised classification losses (for fine-tuning modes)
        supervised_loss = torch.tensor(0.0, device=self.device)
        if labels is not None and self.mode not in ['self_supervised', 'SupCon']:
            if 'logits' in temporal_outputs:
                supervised_loss += self.cross_entropy(temporal_outputs['logits'], labels)
            if 'logits' in frequency_outputs:
                supervised_loss += self.cross_entropy(frequency_outputs['logits'], labels)
            loss_dict['supervised_loss'] = supervised_loss
        
        # Total loss
        total_loss = temporal_loss + frequency_loss + cotraining_loss + supervised_loss
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
    
    def update_weights(self, epoch, total_epochs):
        """
        Dynamically adjust loss weights during training.
        Gradually increase co-training weight as training progresses.
        """
        # Warm up co-training loss (reduced weights for debugging)
        warmup_epochs = total_epochs // 4
        if epoch < warmup_epochs:
            self.lambda_cotraining = 0.0001 * (epoch / warmup_epochs)
        else:
            self.lambda_cotraining = 0.0001
            
        # Adjust consistency weight based on training progress
        progress = epoch / total_epochs
        self.lambda_consistency = 0.15 + 0.05 * progress  # Increase from 0.15 to 0.20 (OPTIMAL RANGE) 