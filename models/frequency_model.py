import torch
import torch.nn as nn
import torch.fft


class FrequencyModel(nn.Module):
    """
    Frequency-domain branch for CoFT architecture.
    Processes FFT-transformed input to learn frequency-domain representations.
    """
    
    def __init__(self, configs):
        super(FrequencyModel, self).__init__()
        
        self.input_channels = configs.input_channels
        
        # FFT processing parameters
        self.fft_norm = 'ortho'  # Orthogonal normalization for stable gradients
        
        # Frequency-domain convolution blocks
        # Note: After FFT, we get complex values, so we'll work with magnitude/phase
        self.freq_conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels * 2, 32, kernel_size=configs.kernel_size,  # *2 for real/imag
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )
        
        self.freq_conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        
        self.freq_conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        
        # Frequency-specific classifier
        # For frequency domain, we need to calculate the actual output dimensions
        # This will be dynamically determined in the first forward pass
        self.num_classes = configs.num_classes
        self.freq_logits = None  # Will be initialized in first forward pass
        
    def forward(self, x_in):
        """
        Forward pass through frequency branch.
        
        Args:
            x_in: Input time series [batch_size, channels, time_steps]
            
        Returns:
            logits: Classification logits from frequency domain
            freq_features: Extracted frequency features for contrastive learning
        """
        # Ensure input is 3D for FFT processing
        if len(x_in.shape) == 4:
            # If 4D [batch, channels, 1, time_steps], squeeze the extra dimension
            x_in = x_in.squeeze(2)
        
        # Apply FFT to convert to frequency domain
        x_fft = torch.fft.rfft(x_in, norm=self.fft_norm)  # Real FFT for real-valued input
        
        # Convert complex to real representation (magnitude and phase)
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # Concatenate magnitude and phase as separate channels
        x_freq = torch.cat([magnitude, phase], dim=1)  # [batch, channels*2, freq_bins]
        
        # Process through frequency-specific conv blocks
        x = self.freq_conv_block1(x_freq)
        x = self.freq_conv_block2(x)
        x = self.freq_conv_block3(x)
        
        # Flatten for classification - calculate the actual dimensions dynamically
        x_flat = x.reshape(x.shape[0], -1)
        
        # Initialize linear layer if this is the first forward pass
        if self.freq_logits is None:
            actual_features = x_flat.shape[1]
            self.freq_logits = nn.Linear(actual_features, self.num_classes).to(x_flat.device)
        
        freq_logits = self.freq_logits(x_flat)
        
        return freq_logits, x


class FrequencyAugmentation(nn.Module):
    """
    Frequency-domain augmentation for contrastive learning.
    """
    
    def __init__(self, noise_ratio=0.01, freq_mask_ratio=0.1):
        super(FrequencyAugmentation, self).__init__()
        self.noise_ratio = noise_ratio
        self.freq_mask_ratio = freq_mask_ratio
        
    def forward(self, x):
        """
        Apply frequency-domain augmentations.
        
        Args:
            x: Input time series [batch_size, channels, time_steps]
            
        Returns:
            aug1, aug2: Two augmented versions of the input
        """
        aug1 = self._apply_freq_noise(x, self.noise_ratio)
        aug2 = self._apply_freq_mask(x, self.freq_mask_ratio)
        
        return aug1, aug2
    
    def _apply_freq_noise(self, x, noise_ratio):
        """Add noise in frequency domain."""
        x_fft = torch.fft.rfft(x, norm='ortho')
        noise = torch.randn_like(x_fft) * noise_ratio
        x_noisy_fft = x_fft + noise
        x_noisy = torch.fft.irfft(x_noisy_fft, n=x.shape[-1], norm='ortho')
        return x_noisy
    
    def _apply_freq_mask(self, x, mask_ratio):
        """Mask random frequency components."""
        x_fft = torch.fft.rfft(x, norm='ortho')
        
        # Create frequency mask
        freq_bins = x_fft.shape[-1]
        mask_size = int(freq_bins * mask_ratio)
        mask = torch.ones_like(x_fft)
        
        # Randomly mask frequency bins
        start_idx = torch.randint(0, freq_bins - mask_size + 1, (1,)).item()
        mask[..., start_idx:start_idx + mask_size] = 0
        
        x_masked_fft = x_fft * mask
        x_masked = torch.fft.irfft(x_masked_fft, n=x.shape[-1], norm='ortho')
        return x_masked 