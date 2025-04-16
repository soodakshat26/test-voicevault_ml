import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Union


class SincConv(nn.Module):
    """
    Sinc-based convolution for RawNet3
    
    This implements parameterized sinc filters which are more efficient
    and interpretable than standard CNN filters for raw audio.
    """
    def __init__(
        self,
        out_channels: int = 80,
        kernel_size: int = 1024,
        sample_rate: int = 16000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        min_low_hz: int = 0,
        min_band_hz: int = 0
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        # Initialize filterbanks with mel scale center frequencies
        low_hz = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        
        # mel scale
        mel = np.linspace(
            self._to_mel(low_hz),
            self._to_mel(high_hz),
            out_channels + 1
        )
        hz = self._to_hz(mel)
        
        # Filter lower and upper frequency bounds
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        
        # Hamming window
        n_lin = torch.linspace(0, kernel_size - 1, steps=kernel_size)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)
        self.register_buffer('window', window.view(1, 1, -1))
        
        # (kernel_size, 1) for time-wise convolution
        n = (kernel_size - 1) / 2
        self.register_buffer(
            'n',
            (2 * math.pi * torch.arange(-n, 0).view(1, 1, -1)) / sample_rate
        )
        self.register_buffer(
            'n_',
            (2 * math.pi * torch.arange(1, n + 1).view(1, 1, -1)) / sample_rate
        )
        
        # Stride, padding etc.
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def _to_mel(self, hz):
        """Convert Hz to Mel scale"""
        return 2595 * np.log10(1 + hz / 700)
    
    def _to_hz(self, mel):
        """Convert Mel scale to Hz"""
        return 700 * (10 ** (mel / 2595) - 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input waveform (batch, channels, time)
            
        Returns:
            Filtered output
        """
        low = self.min_low_hz + torch.abs(self.low_hz_)  # Adding min_low_hz
        high = low + self.min_band_hz + torch.abs(self.band_hz_)  # Adding min_band_hz
        
        # Compute filter coefficients
        band = (high - low)[:, 0]
        f_times_t = torch.matmul(low, self.n)
        f_times_t_ = torch.matmul(high, self.n_)
        
        # Left and right filter parts
        band_pass_left = ((torch.sin(f_times_t) / self.n) * self.window)
        band_pass_right = ((torch.sin(f_times_t_) / self.n_) * self.window)
        
        # Concatenate
        band_pass = torch.cat([band_pass_left, band_pass_right], dim=2)
        
        # Normalize
        band_pass = band_pass / (2 * band.view(-1, 1, 1))
        
        # Expand to match in_channels
        filters = band_pass.expand(self.out_channels, -1, -1, -1).contiguous()
        
        return F.conv1d(
            x,
            filters.view(self.out_channels, -1, self.kernel_size),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1
        )


class FMS(nn.Module):
    """
    Feature Map Selection module for RawNet3
    
    This module enhances certain channels based on their importance.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(1, dim, 1))
        
    def forward(self, x):
        """Forward pass with feature selection"""
        # x: (B, C, T)
        weight = torch.sigmoid(self.weight)
        return x * weight


class ResidualBlock(nn.Module):
    """
    Residual block for RawNet3
    
    This implements a residual block with bottleneck structure for efficiency.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Bottleneck architecture
        bottleneck_dim = out_channels // 4
        
        self.conv1 = nn.Conv1d(
            in_channels, bottleneck_dim, kernel_size=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm1d(bottleneck_dim)
        
        self.conv2 = nn.Conv1d(
            bottleneck_dim, bottleneck_dim, kernel_size=kernel_size,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(bottleneck_dim)
        
        self.conv3 = nn.Conv1d(
            bottleneck_dim, out_channels, kernel_size=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.fms = FMS(out_channels)
        
        # Shortcut connection if dimensions don't match
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        """Forward pass with residual connection"""
        # x: (B, C, T)
        residual = self.shortcut(x)
        
        # Bottleneck path
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Apply feature map selection
        x = self.fms(x)
        
        # Add residual connection
        x = x + residual
        x = self.relu(x)
        
        return x


class Rawnet3(nn.Module):
    """
    RawNet3 architecture for anti-spoofing
    
    This model processes raw waveforms directly using SincConv followed
    by residual blocks with increasing dilation rates.
    """
    def __init__(
        self,
        filters: List[int] = [20, 20, 128, 128, 256, 256],
        first_kernel_size: int = 1024,
        in_channels: int = 1,
        num_classes: int = 2,
        sample_rate: int = 16000,
        max_seq_length: int = 64000
    ):
        super().__init__()
        self.filters = filters
        self.first_kernel_size = first_kernel_size
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        
        # First layer: SincConv
        self.sinc_conv = SincConv(
            out_channels=filters[0],
            kernel_size=first_kernel_size,
            sample_rate=sample_rate,
            in_channels=in_channels,
            stride=1,
            padding=first_kernel_size // 2
        )
        
        # Max pooling after first layer
        self.max_pool = nn.MaxPool1d(3)
        
        # Residual blocks with increasing dilation rates
        self.residual_blocks = nn.ModuleList()
        
        dilations = [1, 2, 4, 8, 16, 1]
        kernel_sizes = [3, 3, 3, 3, 3, 3]
        
        in_channels = filters[0]
        for i in range(len(filters) - 1):
            out_channels = filters[i+1]
            self.residual_blocks.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i]
                )
            )
            in_channels = out_channels
        
        # Attention-based statistical pooling
        attention_dim = filters[-1] // 8
        self.attention = nn.Sequential(
            nn.Conv1d(filters[-1], attention_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_dim),
            nn.Conv1d(attention_dim, filters[-1], kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # Output classifier
        self.fc1 = nn.Linear(filters[-1] * 2, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass for RawNet3
        
        Args:
            x: Input waveform (batch_size, 1, time)
            
        Returns:
            Classification output and embedding
        """
        # Ensure the input has the correct shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Apply pre-emphasis
        # x = self._pre_emphasis(x)
        
        # Apply first SincConv layer
        x = self.sinc_conv(x)
        x = F.leaky_relu(x, 0.2)
        x = self.max_pool(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Apply attention-based statistical pooling
        attention_weights = self.attention(x)
        
        # Weighted statistics pooling
        means = torch.sum(x * attention_weights, dim=2, keepdim=True)
        var = torch.sum(attention_weights * ((x - means) ** 2), dim=2, keepdim=True)
        std = torch.sqrt(var.clamp(min=1e-9))
        
        # Concatenate mean and std for utterance-level embedding
        embedding = torch.cat([means, std], dim=1).squeeze(2)
        
        # Final fully connected layers
        x = F.leaky_relu(self.bn_fc1(self.fc1(embedding)), 0.2)
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output, embedding
    
    def _pre_emphasis(self, x, coef=0.97):
        """Apply pre-emphasis to the input signal"""
        # x: (batch, channels, time)
        # Apply pre-emphasis filter: y[n] = x[n] - coef*x[n-1]
        y = torch.zeros_like(x)
        y[:, :, 0] = x[:, :, 0]
        y[:, :, 1:] = x[:, :, 1:] - coef * x[:, :, :-1]
        return y
