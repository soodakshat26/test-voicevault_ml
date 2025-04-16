import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

class Generator(nn.Module):
    """
    Generator for CycleGAN-based defense
    
    Transforms synthetic speech to reveal artifacts.
    """
    def __init__(
        self, 
        in_channels: int = 1,
        out_channels: int = 1,
        n_residual_blocks: int = 6,
        n_filters: int = 64
    ):
        super().__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad1d(3),
            nn.Conv1d(in_channels, n_filters, kernel_size=7),
            nn.InstanceNorm1d(n_filters),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        for _ in range(2):
            model += [
                nn.Conv1d(n_filters, n_filters*2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm1d(n_filters*2),
                nn.ReLU(inplace=True)
            ]
            n_filters *= 2
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(n_filters)]
        
        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose1d(n_filters, n_filters//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm1d(n_filters//2),
                nn.ReLU(inplace=True)
            ]
            n_filters //= 2
        
        # Output layer
        model += [
            nn.ReflectionPad1d(3),
            nn.Conv1d(n_filters, out_channels, kernel_size=7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    """Residual Block for Generator"""
    def __init__(self, n_filters):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(n_filters, n_filters, kernel_size=3),
            nn.InstanceNorm1d(n_filters),
            nn.ReLU(inplace=True),
            nn.ReflectionPad1d(1),
            nn.Conv1d(n_filters, n_filters, kernel_size=3),
            nn.InstanceNorm1d(n_filters)
        )
    
    def forward(self, x):
        return x + self.block(x)


class Discriminator(nn.Module):
    """
    Discriminator for Wasserstein GAN with gradient penalty
    
    Classifies between real and synthetic speech.
    """
    def __init__(
        self, 
        in_channels: int = 1,
        n_filters: int = 64,
        n_layers: int = 4
    ):
        super().__init__()
        
        # Initial convolution
        layers = [
            nn.Conv1d(in_channels, n_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Additional layers with increased filters
        for i in range(1, n_layers):
            layers += [
                nn.Conv1d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm1d(n_filters*2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            n_filters *= 2
        
        # Final layers
        layers += [
            nn.Conv1d(n_filters, n_filters*2, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm1d(n_filters*2),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Output layer
        layers += [nn.Conv1d(n_filters*2, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class GAN_Defense(nn.Module):
    """
    GAN-based Defense for anti-spoofing
    
    Uses CycleGAN to transform synthetic speech and reveal artifacts.
    """
    def __init__(
        self, 
        generator_config: Dict = None,
        discriminator_config: Dict = None
    ):
        super().__init__()
        
        # Default configs if not provided
        if generator_config is None:
            generator_config = {}
        if discriminator_config is None:
            discriminator_config = {}
        
        # Initialize generator and discriminator
        self.generator = Generator(**generator_config)
        self.discriminator = Discriminator(**discriminator_config)
        
        # Initialize cycle generator (synthetic -> real -> synthetic)
        self.cycle_generator = Generator(**generator_config)
        
    def forward(self, x, mode='inference'):
        """
        Forward pass for GAN-based defense
        
        Args:
            x: Input waveform (batch, 1, time)
            mode: 'inference', 'generator_train', or 'discriminator_train'
            
        Returns:
            Processed outputs based on mode
        """
        if mode == 'inference':
            # Transform speech to reveal artifacts
            transformed = self.generator(x)
            return transformed
        
        elif mode == 'generator_train':
            # Generator training: transform, discriminate, cycle consistency
            transformed = self.generator(x)
            fake_validity = self.discriminator(transformed)
            
            # Cycle consistency
            reconstructed = self.cycle_generator(transformed)
            
            return {
                'transformed': transformed,
                'fake_validity': fake_validity,
                'reconstructed': reconstructed
            }
            
        elif mode == 'discriminator_train':
            # Discriminator training: real vs. fake
            with torch.no_grad():
                fake = self.generator(x)
            real_validity = self.discriminator(x)
            fake_validity = self.discriminator(fake.detach())
            
            # For WGAN-GP, also return interpolated samples for gradient penalty
            batch_size = x.size(0)
            
            # Random interpolation factor
            alpha = torch.rand(batch_size, 1, 1, device=x.device)
            
            # Interpolate between real and fake
            interpolated = alpha * x + (1 - alpha) * fake.detach()
            interpolated.requires_grad_(True)
            
            # Discriminator output on interpolated
            interpolated_validity = self.discriminator(interpolated)
            
            return {
                'real_validity': real_validity,
                'fake_validity': fake_validity,
                'interpolated': interpolated,
                'interpolated_validity': interpolated_validity
            }
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
    def compute_gradient_penalty(self, real_samples, fake_samples, lambda_gp=10):
        """
        Compute gradient penalty for WGAN-GP
        
        Args:
            real_samples: Real audio samples
            fake_samples: Generated audio samples
            lambda_gp: Gradient penalty weight
            
        Returns:
            Gradient penalty loss term
        """
        # Random interpolation factor
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, device=real_samples.device)
        
        # Interpolate between real and fake
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Discriminator output on interpolated
        disc_interpolated = self.discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
