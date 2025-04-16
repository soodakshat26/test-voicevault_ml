import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union

class LocalDifferentialPrivacy:
    """
    Local Differential Privacy for federated learning
    
    This class implements mechanisms for ensuring differential privacy
    in federated learning by adding calibrated noise to model updates.
    """
    def __init__(
        self,
        epsilon: float = 0.5,  # Privacy budget
        delta: float = 1e-5,   # Failure probability
        clip_norm: float = 1.0,  # Gradient clipping norm
        mechanism: str = 'gaussian'  # 'gaussian' or 'laplace'
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.mechanism = mechanism
    
    def add_noise_to_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Add calibrated noise to gradients to ensure differential privacy
        
        Args:
            gradients: List of gradient tensors
            
        Returns:
            Noisy gradients
        """
        # Clip gradients
        gradients = self._clip_gradients(gradients)
        
        # Add noise based on mechanism
        if self.mechanism == 'gaussian':
            return self._add_gaussian_noise(gradients)
        elif self.mechanism == 'laplace':
            return self._add_laplace_noise(gradients)
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
    
    def _clip_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Clip gradients to ensure bounded sensitivity"""
        # Compute global norm
        global_norm = torch.norm(
            torch.stack([torch.norm(g.detach()) for g in gradients])
        )
        
        # Apply clipping
        clipping_factor = min(1.0, self.clip_norm / (global_norm + 1e-10))
        clipped_gradients = [g.detach() * clipping_factor for g in gradients]
        
        return clipped_gradients
    
    def _add_gaussian_noise(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Add calibrated Gaussian noise to ensure (ε, δ)-DP"""
        # Calculate noise scale using Gaussian mechanism
        # Based on Abadi et al. "Deep Learning with Differential Privacy"
        c = self.clip_norm
        sigma = c * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        # Add noise to each gradient
        noisy_gradients = []
        for g in gradients:
            noise = torch.randn_like(g) * sigma
            noisy_gradients.append(g + noise)
        
        return noisy_gradients
    
    def _add_laplace_noise(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Add calibrated Laplace noise to ensure ε-DP"""
        # Calculate noise scale using Laplace mechanism
        c = self.clip_norm
        scale = c / self.epsilon
        
        # Add noise to each gradient
        noisy_gradients = []
        for g in gradients:
            # Generate Laplace noise
            uniform = torch.rand_like(g) - 0.5
            noise = -scale * torch.sign(uniform) * torch.log(1 - 2 * torch.abs(uniform))
            noisy_gradients.append(g + noise)
        
        return noisy_gradients


class KnowledgeDistillation:
    """
    Knowledge Distillation for federated learning
    
    This class implements knowledge distillation to maintain lightweight
    client models that preserve privacy.
    """
    def __init__(
        self,
        global_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.5  # Weight for distillation loss vs. standard loss
    ):
        self.global_model = global_model
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor, 
        targets: torch.Tensor,
        teacher_model: nn.Module = None,
        inputs: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss
        
        Args:
            student_logits: Output from student model
            targets: Ground truth labels
            teacher_model: Teacher model (default: global model)
            inputs: Input data for teacher model
            
        Returns:
            Combined loss with distillation
        """
        if teacher_model is None:
            teacher_model = self.global_model
        
        # Standard cross-entropy loss
        ce_loss = nn.CrossEntropyLoss()(student_logits, targets)
        
        # If no inputs provided, can't compute distillation loss
        if inputs is None:
            return ce_loss
        
        # Get soft targets from teacher
        with torch.no_grad():
            teacher_model.eval()
            teacher_logits = teacher_model(inputs)
        
        # Compute soft targets with temperature
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        log_probs = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss
        kd_loss = -torch.sum(soft_targets * log_probs, dim=1).mean() * (self.temperature ** 2)
        
        # Combine losses
        combined_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
        
        return combined_loss


class GradientCompression:
    """
    Gradient compression techniques for reducing communication overhead
    
    This class implements Top-k and Quantization methods for compressing
    gradients in federated learning.
    """
    def __init__(
        self,
        compression_method: str = 'topk',  # 'topk' or 'quantize'
        k_ratio: float = 0.1,  # For Top-k: fraction of gradients to keep
        num_bits: int = 8,     # For quantization: number of bits
        error_feedback: bool = True  # Whether to use error feedback
    ):
        self.compression_method = compression_method
        self.k_ratio = k_ratio
        self.num_bits = num_bits
        self.error_feedback = error_feedback
        self.error_residuals = {}  # Store residuals for error feedback
    
    def compress(self, gradients: List[torch.Tensor], client_id: int = None) -> Dict:
        """
        Compress gradients based on selected method
        
        Args:
            gradients: List of gradient tensors
            client_id: Client ID for error feedback
            
        Returns:
            Compressed gradients information
        """
        # Initialize error residuals for this client if needed
        if self.error_feedback and client_id is not None:
            if client_id not in self.error_residuals:
                self.error_residuals[client_id] = [torch.zeros_like(g) for g in gradients]
            
            # Add error residuals to gradients
            gradients = [g + r for g, r in zip(gradients, self.error_residuals[client_id])]
        
        # Apply compression
        if self.compression_method == 'topk':
            compressed_grads, indices, shapes = self._topk_compression(gradients)
            compression_info = {'compressed_grads': compressed_grads, 'indices': indices, 'shapes': shapes}
        elif self.compression_method == 'quantize':
            compressed_grads, scale_factors = self._quantize_compression(gradients)
            compression_info = {'compressed_grads': compressed_grads, 'scale_factors': scale_factors}
        else:
            raise ValueError(f"Unknown compression method: {self.compression_method}")
        
        # Update error residuals if using error feedback
        if self.error_feedback and client_id is not None:
            decompressed_grads = self.decompress(compression_info)
            self.error_residuals[client_id] = [g - d for g, d in zip(gradients, decompressed_grads)]
        
        return compression_info
    
    def _topk_compression(self, gradients: List[torch.Tensor]) -> Tuple[List, List, List]:
        """Apply Top-k compression to gradients"""
        compressed_grads = []
        indices = []
        shapes = []
        
        for grad in gradients:
            # Flatten gradient
            shape = grad.shape
            flat_grad = grad.flatten()
            
            # Calculate number of elements to keep
            k = max(1, int(self.k_ratio * flat_grad.numel()))
            
            # Get Top-k values and indices
            abs_grad = torch.abs(flat_grad)
            _, top_indices = torch.topk(abs_grad, k)
            top_values = flat_grad[top_indices]
            
            compressed_grads.append(top_values)
            indices.append(top_indices)
            shapes.append(shape)
        
        return compressed_grads, indices, shapes
    
    def _quantize_compression(self, gradients: List[torch.Tensor]) -> Tuple[List, List]:
        """Apply quantization to gradients"""
        compressed_grads = []
        scale_factors = []
        
        for grad in gradients:
            # Calculate scale factor
            max_val = torch.max(torch.abs(grad))
            scale_factor = max_val / (2 ** (self.num_bits - 1) - 1) if max_val > 0 else 1.0
            
            # Quantize
            quantized = torch.round(grad / scale_factor).to(torch.int8)
            
            compressed_grads.append(quantized)
            scale_factors.append(scale_factor)
        
        return compressed_grads, scale_factors
    
    def decompress(self, compression_info: Dict) -> List[torch.Tensor]:
        """
        Decompress gradients based on compression method
        
        Args:
            compression_info: Dictionary with compression information
            
        Returns:
            Decompressed gradients
        """
        if self.compression_method == 'topk':
            return self._topk_decompression(
                compression_info['compressed_grads'],
                compression_info['indices'],
                compression_info['shapes']
            )
        elif self.compression_method == 'quantize':
            return self._quantize_decompression(
                compression_info['compressed_grads'],
                compression_info['scale_factors']
            )
        else:
            raise ValueError(f"Unknown compression method: {self.compression_method}")
    
    def _topk_decompression(
        self, 
        compressed_grads: List[torch.Tensor],
        indices: List[torch.Tensor],
        shapes: List[torch.Size]
    ) -> List[torch.Tensor]:
        """Decompress Top-k compressed gradients"""
        decompressed_grads = []
        
        for values, idx, shape in zip(compressed_grads, indices, shapes):
            # Create empty tensor with original shape
            decompressed = torch.zeros(shape.numel(), device=values.device)
            
            # Fill with values at saved indices
            decompressed[idx] = values
            
            # Reshape back to original shape
            decompressed = decompressed.reshape(shape)
            decompressed_grads.append(decompressed)
        
        return decompressed_grads
    
    def _quantize_decompression(
        self, 
        compressed_grads: List[torch.Tensor],
        scale_factors: List[float]
    ) -> List[torch.Tensor]:
        """Decompress quantized gradients"""
        decompressed_grads = []
        
        for quantized, scale_factor in zip(compressed_grads, scale_factors):
            # Dequantize
            decompressed = quantized.float() * scale_factor
            decompressed_grads.append(decompressed)
        
        return decompressed_grads
