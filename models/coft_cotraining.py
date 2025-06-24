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
        
        # Numerical stability constant
        self.eps = 1e-8
        
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
            
        # Check for NaN in input logits
        if torch.isnan(logits).any():
            print(f"⚠️  WARNING: NaN detected in logits during pseudo-label generation")
            # Return fallback values
            batch_size = logits.shape[0]
            device = logits.device
            pseudo_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            confidence_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return pseudo_labels, confidence_mask
            
        # Apply softmax to get probabilities with numerical stability
        probs = F.softmax(logits, dim=1) + self.eps
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        
        # Create confidence mask
        confidence_mask = max_probs > threshold
        
        return pseudo_labels, confidence_mask
    
    def cross_domain_consistency_loss(self, temporal_features, freq_features, 
                                     temporal_logits, freq_logits):
        """
        Compute cross-domain consistency loss with numerical stability.
        
        Args:
            temporal_features: Features from temporal branch
            freq_features: Features from frequency branch
            temporal_logits: Logits from temporal branch  
            freq_logits: Logits from frequency branch
            
        Returns:
            consistency_loss: Cross-domain consistency loss
        """
        # Check for NaN inputs
        if torch.isnan(temporal_features).any() or torch.isnan(freq_features).any():
            print(f"⚠️  WARNING: NaN detected in features during consistency loss")
            return torch.tensor(0.0, device=temporal_features.device, requires_grad=True)
            
        if torch.isnan(temporal_logits).any() or torch.isnan(freq_logits).any():
            print(f"⚠️  WARNING: NaN detected in logits during consistency loss")
            return torch.tensor(0.0, device=temporal_logits.device, requires_grad=True)
        
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
        
        try:
            # Feature alignment loss with gradient clipping
            temporal_aligned = self.temporal_to_freq_adapter(temporal_features)
            freq_aligned = self.freq_to_temporal_adapter(freq_features)
            
            # Check for NaN in aligned features
            if torch.isnan(temporal_aligned).any() or torch.isnan(freq_aligned).any():
                print(f"⚠️  WARNING: NaN detected in aligned features")
                return torch.tensor(0.0, device=temporal_features.device, requires_grad=True)
            
            feature_consistency_loss = F.mse_loss(temporal_aligned, freq_features) + \
                                      F.mse_loss(freq_aligned, temporal_features)
            
            # Prediction consistency loss with numerical stability
            temporal_probs = F.softmax(temporal_logits / self.temperature, dim=1) + self.eps
            freq_probs = F.softmax(freq_logits / self.temperature, dim=1) + self.eps
            
            # Normalize probabilities to sum to 1
            temporal_probs = temporal_probs / temporal_probs.sum(dim=1, keepdim=True)
            freq_probs = freq_probs / freq_probs.sum(dim=1, keepdim=True)
            
            prediction_consistency_loss = F.kl_div(
                torch.log(temporal_probs + self.eps),
                freq_probs,
                reduction='batchmean'
            ) + F.kl_div(
                torch.log(freq_probs + self.eps),
                temporal_probs,
                reduction='batchmean'
            )
            
            total_consistency_loss = (feature_consistency_loss + prediction_consistency_loss) * self.consistency_weight
            
            # Final NaN check
            if torch.isnan(total_consistency_loss):
                print(f"⚠️  WARNING: NaN detected in final consistency loss")
                return torch.tensor(0.0, device=temporal_features.device, requires_grad=True)
                
            return total_consistency_loss
            
        except Exception as e:
            print(f"⚠️  ERROR in consistency loss computation: {e}")
            return torch.tensor(0.0, device=temporal_features.device, requires_grad=True)
    
    def co_training_loss(self, temporal_logits, freq_logits, labels, 
                        temporal_features, freq_features):
        """
        Compute co-training loss with pseudo-labeling and improved numerical stability.
        
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
        device = temporal_logits.device
        
        # Initialize default values
        temporal_pseudo_loss = torch.tensor(0.0, device=device, requires_grad=True)
        freq_pseudo_loss = torch.tensor(0.0, device=device, requires_grad=True)
        consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)
        supervised_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        try:
            # Generate pseudo-labels from each domain
            temporal_pseudo, temporal_mask = self.generate_pseudo_labels(temporal_logits)
            freq_pseudo, freq_mask = self.generate_pseudo_labels(freq_logits)
            
            # Cross-domain pseudo-labeling losses with improved handling
            if temporal_mask.sum() > 0 and not torch.isnan(freq_logits).any():
                temporal_pseudo_loss = F.cross_entropy(
                    freq_logits[temporal_mask], 
                    temporal_pseudo[temporal_mask],
                    reduction='mean'
                )
                
            if freq_mask.sum() > 0 and not torch.isnan(temporal_logits).any():
                freq_pseudo_loss = F.cross_entropy(
                    temporal_logits[freq_mask],
                    freq_pseudo[freq_mask], 
                    reduction='mean'
                )
            
            # Cross-domain consistency loss
            if temporal_features is not None and freq_features is not None:
                consistency_loss = self.cross_domain_consistency_loss(
                    temporal_features, freq_features, temporal_logits, freq_logits
                )
            
            # Supervised loss (if labels available)
            if labels is not None and not torch.isnan(temporal_logits).any() and not torch.isnan(freq_logits).any():
                temporal_sup_loss = F.cross_entropy(temporal_logits, labels)
                freq_sup_loss = F.cross_entropy(freq_logits, labels)
                
                # Check for NaN in supervised losses
                if not torch.isnan(temporal_sup_loss) and not torch.isnan(freq_sup_loss):
                    supervised_loss = temporal_sup_loss + freq_sup_loss
            
            # Check all loss components for NaN
            losses = [temporal_pseudo_loss, freq_pseudo_loss, consistency_loss, supervised_loss]
            for i, loss in enumerate(losses):
                if torch.isnan(loss):
                    print(f"⚠️  WARNING: NaN detected in loss component {i}")
                    losses[i] = torch.tensor(0.0, device=device, requires_grad=True)
            
            total_loss = sum(losses)
            
            # Final safety check
            if torch.isnan(total_loss):
                print(f"⚠️  WARNING: NaN detected in total co-training loss, returning 0")
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            stats = {
                'temporal_pseudo_loss': losses[0].item(),
                'freq_pseudo_loss': losses[1].item(),
                'consistency_loss': losses[2].item(),
                'supervised_loss': losses[3].item(),
                'temporal_confident_ratio': temporal_mask.float().mean().item() if temporal_mask.numel() > 0 else 0.0,
                'freq_confident_ratio': freq_mask.float().mean().item() if freq_mask.numel() > 0 else 0.0
            }
            
            return total_loss, stats
            
        except Exception as e:
            print(f"⚠️  ERROR in co-training loss computation: {e}")
            stats = {
                'temporal_pseudo_loss': 0.0,
                'freq_pseudo_loss': 0.0,
                'consistency_loss': 0.0,
                'supervised_loss': 0.0,
                'temporal_confident_ratio': 0.0,
                'freq_confident_ratio': 0.0
            }
            return torch.tensor(0.0, device=device, requires_grad=True), stats


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
        Ensemble predictions from both domains with NaN handling.
        
        Args:
            temporal_logits: Predictions from temporal branch
            freq_logits: Predictions from frequency branch
            
        Returns:
            ensemble_logits: Combined predictions
        """
        # Check for NaN inputs
        if torch.isnan(temporal_logits).any() or torch.isnan(freq_logits).any():
            print(f"⚠️  WARNING: NaN detected in ensemble inputs")
            # Return the non-NaN input or zeros if both are NaN
            if not torch.isnan(temporal_logits).any():
                return temporal_logits
            elif not torch.isnan(freq_logits).any():
                return freq_logits
            else:
                return torch.zeros_like(temporal_logits)
        
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