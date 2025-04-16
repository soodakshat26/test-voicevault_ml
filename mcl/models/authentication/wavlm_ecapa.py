import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from typing import Optional, Tuple
from transformers import WavLMConfig


class AttentiveStatPooling(nn.Module):
    """Attentive statistical pooling with learnable attention"""
    def __init__(self, in_dim, attention_channels=128, global_context=True):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(attention_channels, in_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )
        self.global_context = global_context
        
    def forward(self, x):
        # x: (B, C, T)
        attention_weights = self.attention(x)
        
        # Weighted mean
        mean = torch.sum(x * attention_weights, dim=2, keepdim=True)
        
        if self.global_context:
            # Weighted standard deviation
            variance = torch.sum(attention_weights * (x - mean) ** 2, dim=2, keepdim=True)
            std = torch.sqrt(variance + 1e-8)
            
            # Concatenate mean and std along channel dimension
            pooled = torch.cat([mean, std], dim=1)
            pooled = pooled.view(x.size(0), -1)
        else:
            pooled = mean.view(x.size(0), -1)
            
        return pooled


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        batch, channels, _ = x.size()
        y = torch.mean(x, dim=2)  # Global average pooling
        y = self.fc(y)
        y = y.view(batch, channels, 1)
        return x * y


class ConvBlock(nn.Module):
    """Convolutional block with SE module"""
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        dilation=1, 
        stride=1, 
        groups=1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation,
            stride=stride,
            padding=(kernel_size-1)//2 * dilation,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.se = SEBlock(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x


class Bottle2neck(nn.Module):
    """Res2Net bottleneck block for ECAPA-TDNN"""
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3, 
        dilation=1, 
        scale=8
    ):
        super().__init__()
        width = out_channels // scale
        self.scale = scale
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.nums = scale - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv1d(
                width, width, kernel_size=kernel_size, 
                dilation=dilation, padding=dilation*(kernel_size-1)//2
            ))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        
        # first pointwise conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # split and multi-scale processing
        spx = torch.split(out, out.shape[1] // self.scale, dim=1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](sp)
            sp = self.relu(sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        
        # second pointwise conv
        out = self.conv3(out)
        out = self.bn3(out)
        
        # SE and residual
        out = self.se(out)
        out = out + residual
        out = self.relu(out)
        
        return out


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head self-attention pooling for variable-length utterances"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # Projection layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        # Context vector for content-based attention
        self.context_vector = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
    def reshape_for_attention(self, x, batch_size):
        """Reshape to [batch_size, num_heads, seq_len, head_dim]"""
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)  # [B, H, T, D]
        return x
        
    def forward(self, x):
        # x: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)  # [B, T, D]
        k = self.k_proj(x)  # [B, T, D]
        v = self.v_proj(x)  # [B, T, D]
        
        # Reshape for multi-head attention
        q = self.reshape_for_attention(q, batch_size)  # [B, H, T, D/H]
        k = self.reshape_for_attention(k, batch_size)  # [B, H, T, D/H]
        v = self.reshape_for_attention(v, batch_size)  # [B, H, T, D/H]
        
        # Use context vector for query (content-based attention)
        context = self.context_vector.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, D/H]
        context = context.unsqueeze(2)  # [B, H, 1, D/H]
        
        # Compute attention scores
        attn_scores = torch.matmul(context, k.transpose(-2, -1))  # [B, H, 1, T]
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, 1, T]
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # [B, H, 1, D/H]
        context = context.permute(0, 2, 1, 3)  # [B, 1, H, D/H]
        context = context.reshape(batch_size, 1, self.dim)  # [B, 1, D]
        
        # Project back to original dimension
        output = self.o_proj(context).squeeze(1)  # [B, D]
        
        # Calculate mean and standard deviation
        mean = output
        
        # Weighted standard deviation
        residuals = v - mean.view(batch_size, self.num_heads, 1, self.head_dim)
        variance = torch.matmul(attn_weights, residuals.pow(2)).squeeze(2)  # [B, H, D/H]
        std = torch.sqrt(variance.sum(dim=1) / self.num_heads + 1e-8)  # [B, D/H]
        
        # Concatenate mean and std
        stats = torch.cat([mean, std], dim=1)  # [B, 2*D]
        return stats


class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN speaker embedding model"""
    def __init__(
        self, 
        input_dim=80,
        channels=[512, 512, 512, 512, 1536],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        embedding_dim=192
    ):
        super().__init__()
        self.conv1 = ConvBlock(input_dim, channels[0], kernel_sizes[0], dilations[0])
        self.res2net_blocks = nn.ModuleList([
            Bottle2neck(channels[i-1], channels[i], kernel_sizes[i], dilations[i])
            for i in range(1, len(channels))
        ])
        
        # Attention pooling
        self.attention = AttentiveStatPooling(channels[-1], attention_channels)
        
        # Final embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(channels[-1] * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
    
    def forward(self, x):
        # x: (batch, channels, time)
        x = self.conv1(x)
        
        # Apply Res2Net blocks and collect outputs
        res_outputs = []
        for res2net_block in self.res2net_blocks:
            x = res2net_block(x)
            res_outputs.append(x)
        
        # Concatenate at channel dimension
        x = torch.cat(res_outputs, dim=1)
        
        # Apply attention pooling
        x = self.attention(x)
        
        # Final embedding
        x = self.embedding_layer(x)
        
        return x


class WavLM_ECAPA_TDNN(nn.Module):
    """
    Speaker authentication model using WavLM + ECAPA-TDNN architecture
    
    This model uses WavLM as the feature extractor and ECAPA-TDNN for speaker embedding.
    """
    def __init__(
        self,
        wavlm_model_name: str = "microsoft/wavlm-base-plus",
        ecapa_channels: list = [512, 512, 512, 512, 1536],
        ecapa_kernel_sizes: list = [5, 3, 3, 3, 1],
        ecapa_dilations: list = [1, 2, 3, 4, 1],
        embedding_dim: int = 192,
        freeze_wavlm: bool = True,
        pretrained: bool = True
    ):
        super().__init__()
        self.freeze_wavlm = freeze_wavlm
        
        # Load WavLM model and feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_model_name)
        
        if pretrained:
            self.wavlm = WavLMModel.from_pretrained(wavlm_model_name)
        else:
            config = WavLMConfig.from_pretrained(wavlm_model_name)
            self.wavlm = WavLMModel(config)
        
        if freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
        # Get feature dimension from WavLM
        wavlm_dim = self.wavlm.config.hidden_size
        
        # Create ECAPA-TDNN model
        self.ecapa_tdnn = ECAPA_TDNN(
            input_dim=wavlm_dim,
            channels=ecapa_channels,
            kernel_sizes=ecapa_kernel_sizes,
            dilations=ecapa_dilations,
            embedding_dim=embedding_dim
        )
        
        # Multi-head self-attention pooling
        self.attention_pooling = MultiHeadAttentionPooling(
            dim=wavlm_dim, 
            num_heads=8
        )
    
    def extract_wavlm_features(
        self, 
        audio_input, 
        attention_mask=None, 
        return_dict=False
    ):
        """Extract contextualized embeddings from WavLM"""
        if self.freeze_wavlm:
            with torch.no_grad():
                outputs = self.wavlm(
                    audio_input,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=return_dict
                )
        else:
            outputs = self.wavlm(
                audio_input,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=return_dict
            )
        
        if return_dict:
            # Use last hidden state
            hidden_states = outputs.last_hidden_state
        else:
            # For older transformers versions
            hidden_states = outputs[0]
        
        return hidden_states
    
    def forward(self, audio_input, attention_mask=None):
        """
        Forward pass for WavLM-ECAPA-TDNN model
        
        Args:
            audio_input: Raw waveform input (batch, time)
            attention_mask: Optional mask for padded inputs
            
        Returns:
            Speaker embedding (batch, embedding_dim)
        """
        # Extract features from WavLM
        hidden_states = self.extract_wavlm_features(
            audio_input, 
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Convert to ECAPA-TDNN input format
        x = hidden_states.transpose(1, 2)  # (batch, hidden_size, time)
        
        # Forward through ECAPA-TDNN
        x = self.ecapa_tdnn(x)
        
        return x
    
    def prepare_input(self, waveform, sample_rate=16000):
        """
        Prepare waveform input for the model
        
        Args:
            waveform: Raw audio waveform
            sample_rate: Sample rate of the audio
            
        Returns:
            Processed input tensor
        """
        # Resample if needed
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        
        # Process with feature extractor
        inputs = self.feature_extractor(
            waveform, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        return inputs.input_values
