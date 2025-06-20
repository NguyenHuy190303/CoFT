import numpy as np
import torch
import torch.nn as nn

from .attention import Seq_Transformer


class FrequencyContrastive(nn.Module):
    """
    Frequency-domain Contrastive Learning module for CoFT.
    Similar to TC but operates on frequency-domain features.
    """
    
    def __init__(self, configs, device):
        super(FrequencyContrastive, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps  # Reuse temporal contrastive timesteps
        self.device = device
        
        # Frequency-specific prediction heads
        self.Wk_freq = nn.ModuleList([
            nn.Linear(configs.TC.hidden_dim, self.num_channels) 
            for i in range(self.timestep)
        ])
        
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        
        # Frequency-domain projection head
        self.freq_projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )
        
        # Frequency-aware sequence transformer
        self.freq_seq_transformer = Seq_Transformer(
            patch_size=self.num_channels, 
            dim=configs.TC.hidden_dim, 
            depth=4,
            heads=4, 
            mlp_dim=64
        )

    def forward(self, z_freq_aug1, z_freq_aug2):
        """
        Forward pass for frequency contrastive learning.
        
        Args:
            z_freq_aug1: Frequency features from first augmentation
            z_freq_aug2: Frequency features from second augmentation
            
        Returns:
            freq_nce_loss: Frequency-domain contrastive loss
            freq_features: Projected frequency features
        """
        seq_len = z_freq_aug1.shape[2]
        
        # Transpose to match transformer input format
        z_freq_aug1 = z_freq_aug1.transpose(1, 2)
        z_freq_aug2 = z_freq_aug2.transpose(1, 2)
        
        batch = z_freq_aug1.shape[0]
        
        # Randomly sample time stamps (treating frequency bins as temporal sequence)
        t_samples = torch.randint(
            seq_len - self.timestep, size=(1,)
        ).long().to(self.device)
        
        freq_nce = 0  # Average over timestep and batch
        encode_samples = torch.empty(
            (self.timestep, batch, self.num_channels)
        ).float().to(self.device)
        
        # Encode future frequency samples
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_freq_aug2[:, t_samples + i, :].view(
                batch, self.num_channels
            )
        
        # Context from past frequency sequence
        forward_seq = z_freq_aug1[:, :t_samples + 1, :]
        
        # Transform frequency sequence
        c_t_freq = self.freq_seq_transformer(forward_seq)
        
        # Predict future frequency representations
        pred = torch.empty(
            (self.timestep, batch, self.num_channels)
        ).float().to(self.device)
        
        for i in np.arange(0, self.timestep):
            linear = self.Wk_freq[i]
            pred[i] = linear(c_t_freq)
        
        # Compute frequency contrastive loss
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            freq_nce += torch.sum(torch.diag(self.lsoftmax(total)))
        
        freq_nce /= -1. * batch * self.timestep
        
        return freq_nce, self.freq_projection_head(c_t_freq) 