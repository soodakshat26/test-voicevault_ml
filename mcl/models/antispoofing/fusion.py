import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

class AttentionFusion(nn.Module):
    """
    Attention-weighted fusion of multiple anti-spoofing models
    """
    def __init__(
        self, 
        model_dims: List[int], 
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        self.model_dims = model_dims
        self.num_models = len(model_dims)
        
        # Self-attention for each model embedding
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for dim in model_dims
        ])
        
        # Projection layers to same dimension
        self.proj_layers = nn.ModuleList([
            nn.Linear(dim, hidden_dim) 
            for dim in model_dims
        ])
        
        # Fusion layer
        fusion_dim = hidden_dim
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, embeddings: List[torch.Tensor]):
        """
        Forward pass for attention-weighted fusion
        
        Args:
            embeddings: List of embeddings from different models
                [(batch, dim1), (batch, dim2), ...]
            
        Returns:
            Classification output
        """
        assert len(embeddings) == self.num_models, f"Expected {self.num_models} embeddings, got {len(embeddings)}"
        
        # Project all embeddings to same dimension
        proj_embeddings = [
            self.proj_layers[i](embeddings[i])
            for i in range(self.num_models)
        ]
        
        # Compute attention weights
        attention_logits = [
            self.attention_layers[i](embeddings[i])
            for i in range(self.num_models)
        ]
        
        # Concatenate attention logits
        attention_logits = torch.cat(attention_logits, dim=1)  # (batch, num_models)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_logits, dim=1)  # (batch, num_models)
        
        # Apply attention weights to embeddings
        fused_embedding = torch.zeros_like(proj_embeddings[0])
        for i in range(self.num_models):
            fused_embedding += attention_weights[:, i:i+1] * proj_embeddings[i]
        
        # Final classification
        output = self.classifier(fused_embedding)
        
        return output, attention_weights


class MetaFusion(nn.Module):
    """
    Meta-learning wrapper to adapt fusion weights to attack types
    """
    def __init__(
        self, 
        model_dims: List[int],
        meta_input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        self.model_dims = model_dims
        self.num_models = len(model_dims)
        
        # Meta-learning network to predict fusion weights
        self.meta_net = nn.Sequential(
            nn.Linear(meta_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_models),
            nn.Softmax(dim=1)
        )
        
        # Projection layers to same dimension
        self.proj_layers = nn.ModuleList([
            nn.Linear(dim, hidden_dim) 
            for dim in model_dims
        ])
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, embeddings: List[torch.Tensor], meta_features: torch.Tensor):
        """
        Forward pass for meta-learning based fusion
        
        Args:
            embeddings: List of embeddings from different models
                [(batch, dim1), (batch, dim2), ...]
            meta_features: Meta-features for adaptive weighting (batch, meta_dim)
            
        Returns:
            Classification output
        """
        # Project all embeddings to same dimension
        proj_embeddings = [
            self.proj_layers[i](embeddings[i])
            for i in range(self.num_models)
        ]
        
        # Predict fusion weights from meta-features
        fusion_weights = self.meta_net(meta_features)  # (batch, num_models)
        
        # Apply fusion weights to embeddings
        fused_embedding = torch.zeros_like(proj_embeddings[0])
        for i in range(self.num_models):
            fused_embedding += fusion_weights[:, i:i+1] * proj_embeddings[i]
        
        # Final classification
        output = self.classifier(fused_embedding)
        
        return output, fusion_weights


class AntiSpoofingSystem(nn.Module):
    """
    Complete anti-spoofing system combining multiple models
    """
    def __init__(
        self,
        rawnet_config: Dict = None,
        aasist_config: Dict = None, 
        rawgat_config: Dict = None,
        fusion_type: str = 'attention',
        num_classes: int = 2,
        sample_rate: int = 16000
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fusion_type = fusion_type
        
        # Default configs if not provided
        if rawnet_config is None:
            rawnet_config = {'num_classes': num_classes}
        if aasist_config is None:
            aasist_config = {'num_classes': num_classes}
        if rawgat_config is None:
            rawgat_config = {'num_classes': num_classes}
        
        # Initialize individual models
        self.rawnet = Rawnet3(**rawnet_config)
        self.aasist = AASIST(**aasist_config)
        self.rawgat = RawGAT_ST(**rawgat_config)
        
        # Model embedding dimensions
        rawnet_dim = 256 * 2  # Assuming the last filter size is 256 and we use mean + std
        aasist_dim = 32  # From graph_out_size
        rawgat_dim = 128 * 2  # Assuming graph_channels = 128 and we use mean + std
        
        model_dims = [rawnet_dim, aasist_dim, rawgat_dim]
        
        # Initialize fusion model
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(
                model_dims=model_dims,
                hidden_dim=128,
                num_classes=num_classes
            )
        elif fusion_type == 'meta':
            # Use concatenated embeddings as meta-features
            meta_input_dim = sum(model_dims)
            self.fusion = MetaFusion(
                model_dims=model_dims,
                meta_input_dim=meta_input_dim,
                hidden_dim=128,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
    def forward(self, x):
        """
        Forward pass for the complete anti-spoofing system
        
        Args:
            x: Input waveform (batch, 1, time) or (batch, time)
            
        Returns:
            Classification output and fusion weights
        """
        # Ensure the input has the correct shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Process with each model
        rawnet_output, rawnet_embedding = self.rawnet(x)
        aasist_output, aasist_embedding = self.aasist(x)
        rawgat_output, rawgat_embedding = self.rawgat(x)
        
        # Combine with fusion model
        embeddings = [rawnet_embedding, aasist_embedding, rawgat_embedding]
        
        if self.fusion_type == 'meta':
            # Concatenate embeddings as meta-features
            meta_features = torch.cat(embeddings, dim=1)
            final_output, fusion_weights = self.fusion(embeddings, meta_features)
        else:
            final_output, fusion_weights = self.fusion(embeddings)
        
        # For interpretability, also return individual outputs
        individual_outputs = {
            'rawnet': rawnet_output,
            'aasist': aasist_output,
            'rawgat': rawgat_output
        }
        
        embeddings_dict = {
            'rawnet': rawnet_embedding,
            'aasist': aasist_embedding,
            'rawgat': rawgat_embedding
        }
        
        return final_output, fusion_weights, individual_outputs, embeddings_dict
