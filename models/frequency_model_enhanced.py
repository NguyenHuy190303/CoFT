import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math


class SpectralCNN(nn.Module):
    """
    Multi-scale Spectral CNN for frequency pattern extraction.
    Uses different kernel sizes to capture frequency patterns at multiple scales.
    """
    
    def __init__(self, input_channels, hidden_dim=128):
        super(SpectralCNN, self).__init__()
        
        # Multi-scale frequency kernels for different frequency patterns
        self.high_freq_conv = nn.Sequential(
            nn.Conv1d(input_channels * 2, hidden_dim//4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.mid_freq_conv = nn.Sequential(
            nn.Conv1d(input_channels * 2, hidden_dim//4, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.low_freq_conv = nn.Sequential(
            nn.Conv1d(input_channels * 2, hidden_dim//4, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.ultra_low_freq_conv = nn.Sequential(
            nn.Conv1d(input_channels * 2, hidden_dim//4, kernel_size=31, padding=15, bias=False),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Frequency fusion layer
        self.freq_fusion = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)  # Fixed output length
        )
        
    def forward(self, freq_data):
        """
        Extract multi-scale frequency patterns.
        
        Args:
            freq_data: [batch, channels*2, freq_bins] (magnitude + phase)
        
        Returns:
            spectral_features: [batch, hidden_dim, 64]
        """
        # Extract patterns at different frequency scales
        high_patterns = self.high_freq_conv(freq_data)      # High frequency details
        mid_patterns = self.mid_freq_conv(freq_data)        # Mid frequency patterns  
        low_patterns = self.low_freq_conv(freq_data)        # Low frequency trends
        ultra_low_patterns = self.ultra_low_freq_conv(freq_data)  # Ultra-low frequency baseline
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([
            high_patterns, mid_patterns, low_patterns, ultra_low_patterns
        ], dim=1)
        
        # Fuse and normalize
        spectral_features = self.freq_fusion(multi_scale_features)
        
        return spectral_features


class FrequencyAttention(nn.Module):
    """
    Frequency-specific attention mechanism.
    Learns to focus on important frequency bands for each activity.
    """
    
    def __init__(self, freq_dim, num_heads=8, dropout=0.1):
        super(FrequencyAttention, self).__init__()
        
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.head_dim = freq_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Frequency-specific query, key, value projections
        self.freq_qkv = nn.Linear(freq_dim, freq_dim * 3, bias=False)
        self.freq_out = nn.Sequential(
            nn.Linear(freq_dim, freq_dim),
            nn.Dropout(dropout)
        )
        
        # Frequency band importance learning
        self.freq_importance = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(freq_dim, freq_dim//4),
            nn.ReLU(),
            nn.Linear(freq_dim//4, freq_dim),
            nn.Sigmoid()
        )
        
    def forward(self, spectral_features):
        """
        Apply frequency attention.
        
        Args:
            spectral_features: [batch, freq_dim, seq_len]
            
        Returns:
            attended_features: [batch, freq_dim, seq_len]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, freq_dim, seq_len = spectral_features.shape
        
        # Transpose for attention: [batch, seq_len, freq_dim]
        x = spectral_features.transpose(1, 2)
        
        # Compute QKV
        qkv = self.freq_qkv(x)  # [batch, seq_len, freq_dim*3]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, freq_dim)
        
        # Output projection
        attended = self.freq_out(attended)
        
        # Transpose back: [batch, freq_dim, seq_len]
        attended_features = attended.transpose(1, 2)
        
        # Apply frequency importance weighting
        freq_weights = self.freq_importance(spectral_features)  # [batch, freq_dim]
        freq_weights = freq_weights.unsqueeze(-1)  # [batch, freq_dim, 1]
        attended_features = attended_features * freq_weights
        
        return attended_features, attention_weights


class EnhancedFrequencyModel(nn.Module):
    """
    Enhanced Frequency Model with Spectral CNN + Frequency Attention.
    
    Architecture:
    1. Advanced FFT processing (magnitude, phase, power spectral density)
    2. Multi-scale Spectral CNN for pattern extraction
    3. Frequency Attention for smart band selection
    4. Frequency-optimized classifier
    """
    
    def __init__(self, configs):
        super(EnhancedFrequencyModel, self).__init__()
        
        self.input_channels = configs.input_channels
        self.num_classes = configs.num_classes
        
        # Enhanced FFT processing
        self.fft_norm = 'ortho'
        
        # Stage 1: Multi-scale Spectral CNN
        self.spectral_cnn = SpectralCNN(
            input_channels=configs.input_channels,
            hidden_dim=256
        )
        
        # Stage 2: Frequency Attention
        self.freq_attention = FrequencyAttention(
            freq_dim=256,
            num_heads=8,
            dropout=0.1
        )
        
        # Stage 3: Frequency-optimized processing
        self.freq_processor = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Dropout(0.2)
        )
        
        # Stage 4: Advanced classifier
        self.freq_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, configs.num_classes)
        )
        
    def forward(self, x_in):
        """
        Enhanced frequency forward pass.
        
        Args:
            x_in: Input time series [batch_size, channels, time_steps]
            
        Returns:
            freq_logits: Enhanced frequency predictions
            freq_features: Rich frequency features for contrastive learning
            attention_weights: Frequency attention weights for interpretability
        """
        # Handle input dimensions
        if len(x_in.shape) == 4:
            x_in = x_in.squeeze(2)
        
        # Advanced FFT processing
        x_fft = torch.fft.rfft(x_in, norm=self.fft_norm)
        
        # Extract comprehensive frequency representations
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # Power spectral density for additional frequency info
        power = magnitude ** 2
        
        # Enhanced frequency representation: magnitude + phase (power could be added later)
        freq_data = torch.cat([magnitude, phase], dim=1)  # [batch, channels*2, freq_bins]
        
        # Stage 1: Multi-scale spectral pattern extraction
        spectral_features = self.spectral_cnn(freq_data)  # [batch, 256, 64]
        
        # Stage 2: Frequency attention for smart band selection
        attended_features, attention_weights = self.freq_attention(spectral_features)
        
        # Stage 3: Further frequency processing
        processed_features = self.freq_processor(attended_features)  # [batch, 128, 32]
        
        # Stage 4: Classification
        freq_logits = self.freq_classifier(processed_features)
        
        # Prepare contrastive features: keep 3D shape for frequency contrastive model compatibility
        # attended_features: [batch, 256, 64] - keep this shape for FrequencyContrastive model
        contrastive_features = attended_features  # [batch, 256, 64] - 3D shape required
        
        return freq_logits, contrastive_features, attention_weights


class EnhancedFrequencyAugmentation(nn.Module):
    """
    Enhanced frequency augmentations with more sophisticated techniques.
    """
    
    def __init__(self, noise_ratio=0.01, freq_mask_ratio=0.1, freq_shift_ratio=0.05):
        super(EnhancedFrequencyAugmentation, self).__init__()
        self.noise_ratio = noise_ratio
        self.freq_mask_ratio = freq_mask_ratio
        self.freq_shift_ratio = freq_shift_ratio
        
    def forward(self, x):
        """
        Apply enhanced frequency augmentations.
        """
        aug1 = self._apply_spectral_noise(x)
        aug2 = self._apply_freq_mixup(x)
        
        return aug1, aug2
    
    def _apply_spectral_noise(self, x):
        """Add frequency-selective noise."""
        x_fft = torch.fft.rfft(x, norm='ortho')
        
        # Add noise with frequency-dependent scaling
        freq_bins = x_fft.shape[-1]
        freq_scale = torch.linspace(1.0, 0.1, freq_bins, device=x.device)
        noise = torch.randn_like(x_fft) * self.noise_ratio * freq_scale
        
        x_noisy_fft = x_fft + noise
        x_noisy = torch.fft.irfft(x_noisy_fft, n=x.shape[-1], norm='ortho')
        return x_noisy
    
    def _apply_freq_mixup(self, x):
        """Mix frequency components from different samples."""
        x_fft = torch.fft.rfft(x, norm='ortho')
        
        # Random permutation for frequency mixing
        batch_size = x.shape[0]
        indices = torch.randperm(batch_size, device=x.device)
        
        # Mix frequencies with random ratio
        mix_ratio = torch.rand(1, device=x.device) * 0.3 + 0.1  # 0.1-0.4
        x_mixed_fft = x_fft * (1 - mix_ratio) + x_fft[indices] * mix_ratio
        
        x_mixed = torch.fft.irfft(x_mixed_fft, n=x.shape[-1], norm='ortho')
        return x_mixed


# Backward compatibility alias
FrequencyModel = EnhancedFrequencyModel
FrequencyAugmentation = EnhancedFrequencyAugmentation 