import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from models.antispoofing.rawnet import SincConv


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer for channel attention"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class GatedResidualBlock(nn.Module):
    """Gated Residual Block for RawGAT-ST"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            dilation=dilation, padding=dilation * (kernel_size - 1) // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size,
            dilation=dilation, padding=dilation * (kernel_size - 1) // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Gate mechanism
        self.gate_conv = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size,
            dilation=dilation, padding=dilation * (kernel_size - 1) // 2
        )
        self.gate_bn = nn.BatchNorm1d(out_channels)
        
        # SE layer
        self.se = SELayer(out_channels)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = self.residual(x)
        
        # First conv
        out = self.relu(self.bn1(self.conv1(x)))
        
        # Second conv
        out = self.bn2(self.conv2(out))
        
        # Gate
        gate = torch.sigmoid(self.gate_bn(self.gate_conv(out)))
        out = out * gate
        
        # SE attention
        out = self.se(out)
        
        # Residual connection
        out = out + residual
        out = self.relu(out)
        
        return out


class RawGAT_ST(nn.Module):
    """
    RawGAT-ST: Raw Waveform based Graph Attention Network with Self-supervised pre-training
    
    This model combines raw waveform processing with graph attention mechanisms
    for anti-spoofing, enhanced with self-supervised pre-training.
    """
    def __init__(
        self,
        sinc_channels: int = 20,
        res_channels: List[int] = [20, 32, 64, 128],
        graph_channels: int = 128,
        num_classes: int = 2,
        sample_rate: int = 16000,
        max_seq_length: int = 64000,
        n_fft: int = 512
    ):
        super().__init__()
        self.sinc_channels = sinc_channels
        self.max_seq_length = max_seq_length
        self.n_fft = n_fft
        
        # First layer: SincConv
        self.sinc_conv = SincConv(
            out_channels=sinc_channels,
            kernel_size=1024,
            sample_rate=sample_rate,
            in_channels=1,
            stride=1,
            padding=512
        )
        
        # Gated residual blocks with increasing dilation
        self.res_blocks = nn.ModuleList()
        dilations = [1, 2, 4, 8]
        
        for i in range(len(res_channels) - 1):
            self.res_blocks.append(
                GatedResidualBlock(
                    in_channels=res_channels[i],
                    out_channels=res_channels[i+1],
                    kernel_size=3,
                    dilation=dilations[i]
                )
            )
        
        # Graph representation
        self.graph_transform = nn.Sequential(
            nn.Conv1d(res_channels[-1], graph_channels, kernel_size=1),
            nn.BatchNorm1d(graph_channels),
            nn.ReLU()
        )
        
        # Graph attention pooling
        self.graph_attention = nn.Sequential(
            nn.Conv1d(graph_channels, graph_channels // 4, kernel_size=1),
            nn.BatchNorm1d(graph_channels // 4),
            nn.ReLU(),
            nn.Conv1d(graph_channels // 4, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # Output classifier
        self.fc1 = nn.Linear(graph_channels * 2, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
        # Self-supervised pre-training related components
        self.pretraining = False
        self.spec_reconstruct = nn.Sequential(
            nn.Conv1d(graph_channels, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(256, n_fft // 2 + 1, kernel_size=1)
        )
        
    def forward(self, x, return_spec_recon=False):
        """
        Forward pass for RawGAT-ST
        
        Args:
            x: Input waveform (batch_size, 1, time)
            return_spec_recon: Whether to return spectrogram reconstruction
                (for self-supervised pre-training)
            
        Returns:
            Classification output and embedding
        """
        # Ensure the input has the correct shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Apply SincConv
        x = self.sinc_conv(x)
        
        # Apply gated residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Transform for graph representation
        graph_rep = self.graph_transform(x)  # (batch, graph_channels, time)
        
        # Graph attention pooling
        attention_weights = self.graph_attention(graph_rep)  # (batch, 1, time)
        
        # Weighted statistics pooling
        means = torch.sum(graph_rep * attention_weights, dim=2, keepdim=True)
        var = torch.sum(attention_weights * ((graph_rep - means) ** 2), dim=2, keepdim=True)
        std = torch.sqrt(var.clamp(min=1e-9))
        
        # Concatenate mean and std for utterance-level embedding
        embedding = torch.cat([means, std], dim=1).squeeze(2)  # (batch, graph_channels*2)
        
        # Output classifier
        x = F.relu(self.bn_fc1(self.fc1(embedding)))
        x = self.dropout(x)
        output = self.fc2(x)
        
        # For self-supervised pre-training
        if return_spec_recon or self.pretraining:
            # Compute spectrogram of input for self-supervised loss
            with torch.no_grad():
                spec = torch.stft(
                    x.squeeze(1),
                    n_fft=self.n_fft,
                    hop_length=self.n_fft // 4,
                    return_complex=False
                )
                spec_magnitude = torch.sqrt(spec[..., 0]**2 + spec[..., 1]**2)
                log_spec = torch.log(spec_magnitude + 1e-9)
            
            # Predict spectrogram from graph representation
            spec_recon = self.spec_reconstruct(graph_rep)
            
            return output, embedding, spec_recon, log_spec
        
        return output, embedding
    
    def set_pretraining_mode(self, pretraining=True):
        """Set model to pretraining mode"""
        self.pretraining = pretraining
