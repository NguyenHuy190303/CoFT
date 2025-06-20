import torch
import torch.nn as nn
import torch.nn.functional as F


class CoFTCoTraining(nn.Module):
    """
    Co-training Bridge for Time-Frequency Domain Knowledge Transfer.
    Implements pseudo-labeling and cross-domain consistency for CoFT.
    """
    
    def __init__(self, configs, temperature=0.07, consistency_weight=0.5):
        super(CoFTCoTraining, self).__init__()
        
        self.temperature = temperature
        self.consistency_weight = consistency_weight
        self.num_classes = configs.num_classes
        
        # Cross-domain alignment modules
        self.temporal_to_freq_adapter = nn.Sequential(
            nn.Linear(configs.final_out_channels, configs.final_out_channels),
            nn.ReLU(),
            nn.Linear(configs.final_out_channels, configs.final_out_channels)
        )
        
        self.freq_to_temporal_adapter = nn.Sequential(
            nn.Linear(configs.final_out_channels, configs.final_out_channels),
            nn.ReLU(),
            nn.Linear(configs.final_out_channels, configs.final_out_channels)
        )
        
        # Confidence thresholds for pseudo-labeling
        self.confidence_threshold = 0.95
        
    def generate_pseudo_labels(self, logits, threshold=None):
        """
        Generate pseudo-labels from model predictions.
        
        Args:
            logits: Model output logits [batch_size, num_classes]
            threshold: Confidence threshold for pseudo-labeling
            
        Returns:
            pseudo_labels: Generated pseudo-labels
            mask: Boolean mask indicating confident predictions
        """
        if threshold is None:
            threshold = self.confidence_threshold
            
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        
        # Create confidence mask
        confidence_mask = max_probs > threshold
        
        return pseudo_labels, confidence_mask
    
    def cross_domain_consistency_loss(self, temporal_features, freq_features, 
                                     temporal_logits, freq_logits):
        """
        Compute cross-domain consistency loss.
        
        Args:
            temporal_features: Features from temporal branch
            freq_features: Features from frequency branch
            temporal_logits: Logits from temporal branch  
            freq_logits: Logits from frequency branch
            
        Returns:
            consistency_loss: Cross-domain consistency loss
        """
        # Flatten features if they are 3D (from conv layers)
        if len(temporal_features.shape) == 3:
            temporal_features = temporal_features.flatten(1)  # [batch, channels*spatial]
        if len(freq_features.shape) == 3:
            freq_features = freq_features.flatten(1)  # [batch, channels*spatial]
        
        # Initialize adapters dynamically based on actual feature dimensions
        if not hasattr(self, '_adapters_initialized'):
            temporal_dim = temporal_features.shape[1]
            freq_dim = freq_features.shape[1]
            
            # Recreate adapters with correct dimensions
            self.temporal_to_freq_adapter = nn.Sequential(
                nn.Linear(temporal_dim, freq_dim),
                nn.ReLU(),
                nn.Linear(freq_dim, freq_dim)
            ).to(temporal_features.device)
            
            self.freq_to_temporal_adapter = nn.Sequential(
                nn.Linear(freq_dim, temporal_dim),
                nn.ReLU(),
                nn.Linear(temporal_dim, temporal_dim)
            ).to(freq_features.device)
            
            self._adapters_initialized = True
        
        # Feature alignment loss
        temporal_aligned = self.temporal_to_freq_adapter(temporal_features)
        freq_aligned = self.freq_to_temporal_adapter(freq_features)
        
        feature_consistency_loss = F.mse_loss(temporal_aligned, freq_features) + \
                                  F.mse_loss(freq_aligned, temporal_features)
        
        # Prediction consistency loss
        temporal_probs = F.softmax(temporal_logits / self.temperature, dim=1)
        freq_probs = F.softmax(freq_logits / self.temperature, dim=1)
        
        prediction_consistency_loss = F.kl_div(
            F.log_softmax(temporal_logits / self.temperature, dim=1),
            freq_probs,
            reduction='batchmean'
        ) + F.kl_div(
            F.log_softmax(freq_logits / self.temperature, dim=1),
            temporal_probs,
            reduction='batchmean'
        )
        
        return (feature_consistency_loss + prediction_consistency_loss) * self.consistency_weight
    
    def co_training_loss(self, temporal_logits, freq_logits, labels, 
                        temporal_features, freq_features):
        """
        Compute co-training loss with pseudo-labeling.
        
        Args:
            temporal_logits: Logits from temporal branch
            freq_logits: Logits from frequency branch  
            labels: True labels (if available)
            temporal_features: Features from temporal branch
            freq_features: Features from frequency branch
            
        Returns:
            co_training_loss: Combined co-training loss
            stats: Dictionary with loss statistics
        """
        
        # Generate pseudo-labels from each domain
        temporal_pseudo, temporal_mask = self.generate_pseudo_labels(temporal_logits)
        freq_pseudo, freq_mask = self.generate_pseudo_labels(freq_logits)
        
        # Cross-domain pseudo-labeling losses
        temporal_pseudo_loss = F.cross_entropy(
            freq_logits[temporal_mask], 
            temporal_pseudo[temporal_mask],
            reduction='mean'
        ) if temporal_mask.sum() > 0 else torch.tensor(0.0, device=temporal_logits.device)
        
        freq_pseudo_loss = F.cross_entropy(
            temporal_logits[freq_mask],
            freq_pseudo[freq_mask], 
            reduction='mean'
        ) if freq_mask.sum() > 0 else torch.tensor(0.0, device=freq_logits.device)
        
        # Cross-domain consistency loss
        consistency_loss = self.cross_domain_consistency_loss(
            temporal_features, freq_features, temporal_logits, freq_logits
        )
        
        # Supervised loss (if labels available)
        supervised_loss = torch.tensor(0.0, device=temporal_logits.device)
        if labels is not None:
            supervised_loss = F.cross_entropy(temporal_logits, labels) + \
                            F.cross_entropy(freq_logits, labels)
        
        total_loss = temporal_pseudo_loss + freq_pseudo_loss + consistency_loss + supervised_loss
        
        stats = {
            'temporal_pseudo_loss': temporal_pseudo_loss.item(),
            'freq_pseudo_loss': freq_pseudo_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'supervised_loss': supervised_loss.item(),
            'temporal_confident_ratio': temporal_mask.float().mean().item(),
            'freq_confident_ratio': freq_mask.float().mean().item()
        }
        
        return total_loss, stats


class CoFTEnsemble(nn.Module):
    """
    Ensemble prediction module for CoFT.
    Combines temporal and frequency domain predictions.
    """
    
    def __init__(self, ensemble_method='weighted_average', temporal_weight=0.6):
        super(CoFTEnsemble, self).__init__()
        
        self.ensemble_method = ensemble_method
        self.temporal_weight = temporal_weight
        self.freq_weight = 1.0 - temporal_weight
        
        # Learnable ensemble weights (if using learnable method)
        if ensemble_method == 'learnable':
            self.ensemble_weights = nn.Parameter(torch.tensor([temporal_weight, 1-temporal_weight]))
            
    def forward(self, temporal_logits, freq_logits):
        """
        Ensemble predictions from both domains.
        
        Args:
            temporal_logits: Predictions from temporal branch
            freq_logits: Predictions from frequency branch
            
        Returns:
            ensemble_logits: Combined predictions
        """
        if self.ensemble_method == 'average':
            return (temporal_logits + freq_logits) / 2.0
            
        elif self.ensemble_method == 'weighted_average':
            return self.temporal_weight * temporal_logits + self.freq_weight * freq_logits
            
        elif self.ensemble_method == 'learnable':
            weights = F.softmax(self.ensemble_weights, dim=0)
            return weights[0] * temporal_logits + weights[1] * freq_logits
            
        elif self.ensemble_method == 'max':
            # Take the prediction with higher confidence
            temporal_conf = F.softmax(temporal_logits, dim=1).max(dim=1)[0]
            freq_conf = F.softmax(freq_logits, dim=1).max(dim=1)[0]
            mask = temporal_conf > freq_conf
            ensemble_logits = temporal_logits.clone()
            ensemble_logits[~mask] = freq_logits[~mask]
            return ensemble_logits
            
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}") 