import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Tuple
import wandb
import time
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from pathlib import Path

from models.authentication.wavlm_ecapa import WavLM_ECAPA_TDNN
from models.authentication.loss import AngularPrototypicalLoss, AAMSoftmaxLoss, ContrastiveLearningLoss
from preprocessing.augmentation import AudioAugmenter

class SpeakerVerificationTrainer:
    """
    Trainer for speaker verification models
    
    This trainer implements the complete training pipeline for the WavLM-ECAPA-TDNN
    speaker verification model, including contrastive learning, data augmentation,
    and evaluation metrics.
    """
    def __init__(
        self,
        model_config: Dict,
        training_config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'checkpoints'
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize loss functions
        self.criterion = self._initialize_loss()
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self._initialize_optimizer()
        
        # Initialize augmenter
        self.augmenter = AudioAugmenter()
        
        # Metrics tracking
        self.best_eer = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.eers = []
        
    def _initialize_model(self) -> nn.Module:
        """Initialize and return the WavLM-ECAPA-TDNN model"""
        model = WavLM_ECAPA_TDNN(
            wavlm_model_name=self.model_config.get('wavlm_model_name', 'microsoft/wavlm-base-plus'),
            ecapa_channels=self.model_config.get('ecapa_channels', [512, 512, 512, 512, 1536]),
            ecapa_kernel_sizes=self.model_config.get('ecapa_kernel_sizes', [5, 3, 3, 3, 1]),
            ecapa_dilations=self.model_config.get('ecapa_dilations', [1, 2, 3, 4, 1]),
            embedding_dim=self.model_config.get('embedding_dim', 192),
            freeze_wavlm=self.model_config.get('freeze_wavlm', True),
            pretrained=self.model_config.get('pretrained', True)
        )
        return model.to(self.device)
    
    def _initialize_loss(self) -> nn.Module:
        """Initialize the loss function based on configuration"""
        loss_type = self.training_config.get('loss_type', 'angular_proto')
        
        if loss_type == 'angular_proto':
            return AngularPrototypicalLoss(
                scale=self.training_config.get('scale', 30.0),
                margin=self.training_config.get('margin', 0.2),
                temperature=self.training_config.get('temperature', 0.1)
            )
        elif loss_type == 'aam_softmax':
            return AAMSoftmaxLoss(
                embedding_dim=self.model_config.get('embedding_dim', 192),
                num_classes=self.training_config.get('num_speakers', 1000),
                margin=self.training_config.get('margin', 0.2),
                scale=self.training_config.get('scale', 30.0)
            )
        elif loss_type == 'contrastive':
            return ContrastiveLearningLoss(
                temperature=self.training_config.get('temperature', 0.07)
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def _initialize_optimizer(self) -> Tuple[optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        """Initialize optimizer and learning rate scheduler"""
        optimizer_type = self.training_config.get('optimizer', 'adam')
        lr = self.training_config.get('learning_rate', 1e-4)
        weight_decay = self.training_config.get('weight_decay', 1e-5)
        
        # Get trainable parameters
        if self.model_config.get('freeze_wavlm', True):
            parameters = [p for p in self.model.parameters() if p.requires_grad]
        else:
            # Different LR for WavLM and ECAPA-TDNN if WavLM is not frozen
            wavlm_params = list(self.model.wavlm.parameters())
            ecapa_params = list(self.model.ecapa_tdnn.parameters())
            
            parameters = [
                {'params': wavlm_params, 'lr': lr * 0.1},  # Lower LR for WavLM
                {'params': ecapa_params}
            ]
        
        # Initialize optimizer
        if optimizer_type == 'adam':
            optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Initialize learning rate scheduler
        scheduler_type = self.training_config.get('scheduler', None)
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.training_config.get('epochs', 100),
                eta_min=self.training_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_type == 'warmup_cosine':
            from transformers import get_cosine_schedule_with_warmup
            
            num_training_steps = self.training_config.get('epochs', 100) * \
                                 self.training_config.get('steps_per_epoch', 1000)
            num_warmup_steps = int(num_training_steps * 0.1)  # 10% warmup
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            scheduler = None
        
        return optimizer, scheduler
    
    def _apply_augmentation(self, batch):
        """Apply random augmentations to the batch"""
        aug_types = self.training_config.get('augmentations', [])
        
        if not aug_types:
            return batch
        
        waveforms = batch['waveform']
        sample_rate = batch.get('sample_rate', 16000)
        augmented_waveforms = []
        
        for waveform in waveforms:
            # Apply random augmentations
            for aug_type in aug_types:
                if random.random() < 0.5:  # 50% chance to apply each augmentation
                    if aug_type == 'noise':
                        waveform = self.augmenter.add_noise(waveform, snr_db=random.uniform(5, 20))
                    elif aug_type == 'reverb':
                        waveform = self.augmenter.add_reverberation(waveform, sample_rate)
                    elif aug_type == 'pitch':
                        waveform = self.augmenter.change_pitch(waveform, sample_rate, semitones=random.uniform(-2, 2))
                    elif aug_type == 'speed':
                        waveform = self.augmenter.change_speed(waveform, speed_factor=random.uniform(0.9, 1.1))
                    elif aug_type == 'vtlp':
                        waveform = self.augmenter.vocal_tract_length_perturbation(waveform, sample_rate)
                    # Add more augmentations as needed
            
            augmented_waveforms.append(waveform)
        
        batch['waveform'] = torch.stack(augmented_waveforms)
        return batch
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            # Apply augmentations
            batch = self._apply_augmentation(batch)
            
            # Move batch to device
            waveforms = batch['waveform'].to(self.device)
            labels = batch['speaker_id'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.model(waveforms)
            
            # Compute loss
            loss = self.criterion(embeddings, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.training_config.get('clip_grad_norm', False):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.get('max_grad_norm', 3.0)
                )
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Step learning rate scheduler if per-step scheduler
            if self.scheduler is not None and self.training_config.get('scheduler', None) == 'warmup_cosine':
                self.scheduler.step()
        
        # Compute average loss
        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate(self, dataloader, compute_metrics=True):
        """Evaluate the model on the validation set"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        # For computing EER
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validation")
            for batch in progress_bar:
                # Move batch to device
                waveforms = batch['waveform'].to(self.device)
                labels = batch['speaker_id'].to(self.device)
                
                # Forward pass
                embeddings = self.model(waveforms)
                
                # Compute loss
                loss = self.criterion(embeddings, labels)
                
                # Update metrics
                val_loss += loss.item()
                num_batches += 1
                
                # Store embeddings and labels for EER computation
                if compute_metrics:
                    all_embeddings.append(embeddings.cpu())
                    all_labels.append(labels.cpu())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute average loss
        avg_loss = val_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # Compute EER if requested
        eer = None
        if compute_metrics and all_embeddings:
            eer = self._compute_eer(
                torch.cat(all_embeddings),
                torch.cat(all_labels)
            )
            self.eers.append(eer)
        
        return avg_loss, eer
    
    def _compute_eer(self, embeddings, labels):
        """
        Compute Equal Error Rate (EER) using cosine similarity
        
        Args:
            embeddings: Speaker embeddings
            labels: Speaker labels
            
        Returns:
            EER value
        """
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Create trial pairs
        num_samples = embeddings.size(0)
        
        # Find unique speakers
        unique_speakers = torch.unique(labels)
        
        # Create positive and negative pairs
        scores = []
        ground_truth = []
        
        # Sample positive and negative pairs
        num_pairs = min(10000, num_samples * 5)  # Limit number of pairs for efficiency
        
        for _ in range(num_pairs):
            # Select two random indices
            idx1 = random.randint(0, num_samples - 1)
            
            # Decide if this should be a positive or negative pair
            if random.random() < 0.5:  # Positive pair
                # Find another sample from the same speaker
                same_speaker_indices = (labels == labels[idx1]).nonzero(as_tuple=True)[0]
                
                if len(same_speaker_indices) > 1:  # Ensure at least 2 samples from this speaker
                    idx2 = same_speaker_indices[random.randint(0, len(same_speaker_indices) - 1)]
                    while idx2 == idx1:  # Avoid same sample
                        idx2 = same_speaker_indices[random.randint(0, len(same_speaker_indices) - 1)]
                    
                    is_same_speaker = 1
                else:
                    # If not enough samples, create a negative pair instead
                    diff_speaker_indices = (labels != labels[idx1]).nonzero(as_tuple=True)[0]
                    idx2 = diff_speaker_indices[random.randint(0, len(diff_speaker_indices) - 1)]
                    is_same_speaker = 0
            else:  # Negative pair
                # Find sample from different speaker
                diff_speaker_indices = (labels != labels[idx1]).nonzero(as_tuple=True)[0]
                idx2 = diff_speaker_indices[random.randint(0, len(diff_speaker_indices) - 1)]
                is_same_speaker = 0
            
            # Compute cosine similarity
            similarity = torch.dot(embeddings[idx1], embeddings[idx2]).item()
            
            scores.append(similarity)
            ground_truth.append(is_same_speaker)
        
        # Convert to numpy
        scores = np.array(scores)
        ground_truth = np.array(ground_truth)
        
        # Compute EER
        fpr, tpr, thresholds = roc_curve(ground_truth, scores)
        fnr = 1 - tpr
        
        # Find threshold where FPR == FNR (EER)
        eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
        eer = np.mean([fpr[np.nanargmin(np.absolute(fnr - fpr))], 
                      fnr[np.nanargmin(np.absolute(fnr - fpr))]])
        
        return eer
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = None,
        log_wandb: bool = False
    ):
        """
        Train the model
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            num_epochs: Number of epochs to train (overrides config)
            log_wandb: Whether to log metrics to wandb
        """
        if num_epochs is None:
            num_epochs = self.training_config.get('epochs', 100)
        
        # Initialize wandb
        if log_wandb:
            wandb.init(
                project=self.training_config.get('wandb_project', 'speaker-verification'),
                name=self.training_config.get('wandb_run_name', None),
                config={**self.model_config, **self.training_config}
            )
            # Watch model
            wandb.watch(self.model)
        
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_dataloader)
            
            # Validate
            val_loss, eer = self.evaluate(val_dataloader)
            
            # Step learning rate scheduler if per-epoch scheduler
            if self.scheduler is not None and self.training_config.get('scheduler', None) != 'warmup_cosine':
                if self.training_config.get('scheduler', None) == 'reduce_on_plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, EER: {eer:.4f}")
            
            # Log to wandb
            if log_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'eer': eer,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if eer < self.best_eer:
                self.best_eer = eer
                self.save_checkpoint(f'best_model.pt')
                print(f"New best EER: {eer:.4f}")
            
            # Save checkpoint every N epochs
            if (epoch + 1) % self.training_config.get('save_every', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Final save
        self.save_checkpoint('final_model.pt')
        
        # Finish wandb
        if log_wandb:
            wandb.finish()
        
        # Print training summary
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best EER: {self.best_eer:.4f}")
        
        return self.best_eer
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_eer': self.best_eer,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load best EER
        if 'best_eer' in checkpoint:
            self.best_eer = checkpoint['best_eer']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Best EER: {self.best_eer:.4f}")
