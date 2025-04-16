import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Union


class AngularPrototypicalLoss(nn.Module):
    """
    Angular Prototypical Loss for speaker embedding learning
    
    This loss improves upon standard Prototypical Loss by using angular distance
    for better generalization.
    """
    def __init__(
        self,
        scale: float = 30.0,
        margin: float = 0.2,
        easy_margin: bool = False,
        temperature: float = 0.1
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.temperature = temperature
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold to control easy margin
        self.th = math.cos(math.pi - margin)
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Angular Prototypical Loss
        
        Args:
            embeddings: Speaker embeddings (batch_size, embedding_dim)
            labels: Speaker labels (batch_size)
            
        Returns:
            Loss value
        """
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Get unique speaker labels
        unique_labels = torch.unique(labels)
        num_classes = len(unique_labels)
        
        # Create a mapping for original labels to integers starting from 0
        label_map = {label.item(): i for i, label in enumerate(unique_labels)}
        
        # Map original labels to mapped labels
        mapped_labels = torch.tensor([label_map[label.item()] for label in labels], 
                                    device=labels.device)
        
        # Compute prototypes for each speaker
        prototypes = torch.zeros(num_classes, embeddings.size(1), 
                                device=embeddings.device)
        
        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            class_embeddings = embeddings[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes[i] = F.normalize(prototype, p=2, dim=0)
        
        # Compute cosine similarity
        cosine = torch.mm(embeddings, prototypes.t())
        
        # Get one-hot encoding for labels
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, mapped_labels.view(-1, 1), 1)
        
        # Compute angular margin
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.margin)
        
        # Apply angular margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale by temperature
        output = output / self.temperature
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(output, mapped_labels)
        
        return loss


class AAMSoftmaxLoss(nn.Module):
    """
    Additive Angular Margin Softmax Loss for speaker verification
    
    This is a powerful loss function for angular face recognition,
    adapted for speaker verification tasks.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
        easy_margin: bool = False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        
        # Parameters for each class
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        
        # Pre-compute cos(margin) and sin(margin)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        
        # Threshold for adjusting decision boundary
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute AAM-Softmax Loss
        
        Args:
            embeddings: Speaker embeddings (batch_size, embedding_dim)
            labels: Speaker labels (batch_size)
            
        Returns:
            Loss value
        """
        # L2 normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)
        
        # Add angular margin
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding for target labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply angular margin to target classes only
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale outputs
        output = output * self.scale
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(output, labels)
        
        return loss


class ContrastiveLearningLoss(nn.Module):
    """
    Contrastive Learning for Audio (COLA) approach for speaker verification
    
    This loss function implements contrastive learning for audio embeddings,
    creating positive pairs from augmented versions of the same speaker and
    negative pairs from different speakers.
    """
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        contrast_mode: str = 'all'
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode
        
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            features: Feature embeddings (batch_size, embedding_dim)
            labels: Speaker labels (batch_size)
            mask: Optional binary mask for supervised contrastive learning
            
        Returns:
            Loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize embeddings
        features = F.normalize(features, p=2, dim=1)
        
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask for positive pairs based on speaker labels
        if mask is None:
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
            # Remove self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss
