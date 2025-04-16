# training/retraining.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import time
from typing import List, Dict, Tuple, Any, Optional, Union
import copy
from datetime import datetime
import torch.nn.functional as F


class AutomaticRetrainingPipeline:
    """
    Pipeline for automatic retraining of voice authentication models
    """
    def __init__(
        self,
        model: nn.Module,
        retraining_config: Dict,
        data_manager: Any,
        model_save_dir: str = 'retraining_models',
        log_dir: str = 'retraining_logs'
    ):
        self.model = model
        self.retraining_config = retraining_config
        self.data_manager = data_manager
        self.model_save_dir = model_save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize retraining log
        self.retraining_log = []
        self._load_retraining_log()
        
        # Active learning selector
        self.active_learning = ActiveLearningSelector(
            selection_method=retraining_config.get('active_learning_method', 'uncertainty')
        )
    
    def _load_retraining_log(self):
        """Load retraining log from disk"""
        log_path = os.path.join(self.log_dir, 'retraining_log.json')
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                self.retraining_log = json.load(f)
    
    def _save_retraining_log(self):
        """Save retraining log to disk"""
        log_path = os.path.join(self.log_dir, 'retraining_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.retraining_log, f, indent=2)
    
    def should_retrain(
        self,
        performance_metrics: Dict[str, float],
        threshold: float = 0.05,
        min_interval_days: float = 7.0
    ) -> bool:
        """
        Determine if retraining is needed based on performance metrics
        
        Args:
            performance_metrics: Current performance metrics
            threshold: Performance degradation threshold
            min_interval_days: Minimum days between retraining
            
        Returns:
            True if retraining is needed, False otherwise
        """
        # Check if this is the first training
        if not self.retraining_log:
            return True
        
        # Get last retraining timestamp
        last_retraining = self.retraining_log[-1]
        last_timestamp = last_retraining['timestamp']
        
        # Check if minimum interval has passed
        current_time = time.time()
        days_since_last = (current_time - last_timestamp) / (24 * 3600)
        if days_since_last < min_interval_days:
            return False
        
        # Compare metrics to determine if performance has degraded
        last_metrics = last_retraining['metrics']
        for metric, current_value in performance_metrics.items():
            if metric in last_metrics:
                # Check if metric has degraded by more than threshold
                if metric in ['accuracy', 'auc', 'f1']:
                    # Higher is better, so check for decrease
                    if last_metrics[metric] - current_value > threshold:
                        return True
                elif metric in ['error', 'eer', 'loss']:
                    # Lower is better, so check for increase
                    if current_value - last_metrics[metric] > threshold:
                        return True
        
        return False
    
    def select_retraining_data(
        self,
        candidate_samples: List[Dict],
        sample_features: List[np.ndarray],
        sample_labels: List[int],
        sample_metadata: List[Dict],
        max_samples: int = 1000
    ) -> List[int]:
        """
        Select most informative samples for retraining
        
        Args:
            candidate_samples: List of sample data
            sample_features: Features for candidate samples
            sample_labels: Labels for candidate samples
            sample_metadata: Metadata for candidate samples
            max_samples: Maximum number of samples to select
            
        Returns:
            Indices of selected samples
        """
        # Use active learning to select most informative samples
        selected_indices = self.active_learning.select_samples(
            self.model,
            sample_features,
            sample_labels,
            max_samples=max_samples
        )
        
        # Apply diversity enforcement if enabled
        if self.retraining_config.get('enforce_diversity', True):
            selected_indices = self._enforce_diversity(
                selected_indices,
                sample_features,
                sample_metadata,
                max_samples=max_samples
            )
        
        return selected_indices
    
    def _enforce_diversity(
        self,
        selected_indices: List[int],
        sample_features: List[np.ndarray],
        sample_metadata: List[Dict],
        max_samples: int = 1000
    ) -> List[int]:
        """
        Enforce diversity in selected samples
        
        Args:
            selected_indices: Initially selected sample indices
            sample_features: Features for all candidate samples
            sample_metadata: Metadata for all candidate samples
            max_samples: Maximum number of samples to select
            
        Returns:
            Indices of diverse samples
        """
        # If no diversity attributes specified, return original selection
        diversity_attributes = self.retraining_config.get('diversity_attributes', [])
        if not diversity_attributes:
            return selected_indices[:max_samples]
        
        # Convert features to numpy array
        features = np.array([sample_features[i] for i in selected_indices])
        
        # Extract metadata for selected samples
        metadata = [sample_metadata[i] for i in selected_indices]
        
        # Group samples by diversity attributes
        groups = {}
        for i, meta in enumerate(metadata):
            # Create a key from all diversity attributes
            key_parts = []
            for attr in diversity_attributes:
                if attr in meta:
                    key_parts.append(f"{attr}:{meta[attr]}")
                else:
                    key_parts.append(f"{attr}:unknown")
            
            group_key = '+'.join(key_parts)
            
            if group_key not in groups:
                groups[group_key] = []
            
            groups[group_key].append((i, selected_indices[i]))
        
        # Determine how many samples to select from each group
        # Use proportional allocation with minimum guarantees
        num_groups = len(groups)
        min_per_group = min(2, max_samples // num_groups)
        remaining = max_samples - min_per_group * num_groups
        
        # Allocate remaining samples proportionally to group size
        group_sizes = {k: len(v) for k, v in groups.items()}
        total_size = sum(group_sizes.values())
        
        group_allocations = {}
        for group, size in group_sizes.items():
            # Minimum plus proportional share of remaining
            allocation = min_per_group
            if total_size > 0:
                allocation += int(remaining * size / total_size)
            
            # Ensure we don't allocate more than we have
            allocation = min(allocation, size)
            
            group_allocations[group] = allocation
        
        # Select samples from each group
        final_indices = []
        
        for group, samples in groups.items():
            # How many to select from this group
            to_select = group_allocations[group]
            
            # If we need all samples in this group, take them all
            if to_select >= len(samples):
                final_indices.extend([s[1] for s in samples])
            else:
                # Otherwise, select diverse samples from this group using clustering
                # For simplicity, we'll just take a random subset here
                selected = np.random.choice(
                    range(len(samples)), 
                    size=to_select, 
                    replace=False
                )
                
                final_indices.extend([samples[i][1] for i in selected])
        
        return final_indices
    
    def retrain_model(
        self,
        selected_data: List[Dict],
        validation_data: List[Dict]
    ) -> nn.Module:
        """
        Retrain the model on selected data
        
        Args:
            selected_data: Selected training data
            validation_data: Validation data
            
        Returns:
            Retrained model
        """
        # Create a copy of the current model as starting point
        retrained_model = copy.deepcopy(self.model)
        
        # Extract hyperparameters for retraining
        learning_rate = self.retraining_config.get('learning_rate', 0.0001)
        weight_decay = self.retraining_config.get('weight_decay', 1e-5)
        num_epochs = self.retraining_config.get('num_epochs', 20)
        batch_size = self.retraining_config.get('batch_size', 32)
        
        # Prepare data loaders
        train_dataloader = self.data_manager.create_dataloader(
            selected_data, batch_size=batch_size, shuffle=True
        )
        
        val_dataloader = self.data_manager.create_dataloader(
            validation_data, batch_size=batch_size, shuffle=False
        )
        
        # Extract device
        device = next(retrained_model.parameters()).device
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(
            retrained_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Use criterion based on task type
        if self.retraining_config.get('task_type', 'binary') == 'binary':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_model = None
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            retrained_model.train()
            train_loss = 0.0
            
            for batch in train_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs, _ = retrained_model(inputs)
                
                # Compute loss
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            retrained_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs, _ = retrained_model(inputs)
                    
                    # Compute loss
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
            
            # Log progress
            print(f"Epoch {epoch+1}/{num_epochs}, "
                 f"Train Loss: {train_loss/len(train_dataloader):.4f}, "
                 f"Val Loss: {val_loss/len(val_dataloader):.4f}")
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(retrained_model)
        
        # Return best model
        return best_model if best_model is not None else retrained_model
    
    def evaluate_retrained_model(
        self,
        model: nn.Module,
        test_data: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate retrained model on test data
        
        Args:
            model: Retrained model
            test_data: Test data
            
        Returns:
            Evaluation metrics
        """
        # Prepare data loader
        batch_size = self.retraining_config.get('batch_size', 32)
        test_dataloader = self.data_manager.create_dataloader(
            test_data, batch_size=batch_size, shuffle=False
        )
        
        # Extract device
        device = next(model.parameters()).device
        
        # Evaluation phase
        model.eval()
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs, _ = model(inputs)
                
                # Store outputs and labels
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all outputs and labels
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics based on task type
        if self.retraining_config.get('task_type', 'binary') == 'binary':
            # Convert to probabilities
            probs = torch.sigmoid(all_outputs)
            
            # Compute metrics
            from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
            
            # Convert to numpy for sklearn
            probs_np = probs.numpy()
            labels_np = all_labels.numpy()
            
            # Binary predictions
            preds_np = (probs_np > 0.5).astype(int)
            
            # Compute metrics
            accuracy = accuracy_score(labels_np, preds_np)
            auc = roc_auc_score(labels_np, probs_np)
            f1 = f1_score(labels_np, preds_np)
            
            metrics = {
                'accuracy': accuracy,
                'auc': auc,
                'f1': f1
            }
        else:
            # Convert to probabilities
            probs = F.softmax(all_outputs, dim=1)
            
            # Compute metrics
            from sklearn.metrics import accuracy_score, f1_score
            
            # Convert to numpy for sklearn
            preds_np = torch.argmax(probs, dim=1).numpy()
            labels_np = all_labels.numpy()
            
            # Compute metrics
            accuracy = accuracy_score(labels_np, preds_np)
            f1 = f1_score(labels_np, preds_np, average='macro')
            
            metrics = {
                'accuracy': accuracy,
                'f1': f1
            }
        
        return metrics
    
    def deploy_model(
        self,
        model: nn.Module,
        metrics: Dict[str, float]
    ):
        """
        Deploy retrained model if it meets criteria
        
        Args:
            model: Retrained model
            metrics: Evaluation metrics
        """
        # Check if new model is better than current
        improvement_threshold = self.retraining_config.get('improvement_threshold', 0.01)
        
        # Compare to last retraining metrics
        if self.retraining_log:
            last_metrics = self.retraining_log[-1]['metrics']
            
            # Check if any key metric has improved
            improvement = False
            for metric, value in metrics.items():
                if metric in last_metrics:
                    if metric in ['accuracy', 'auc', 'f1']:
                        # Higher is better
                        if value - last_metrics[metric] > improvement_threshold:
                            improvement = True
                            break
                    elif metric in ['error', 'eer', 'loss']:
                        # Lower is better
                        if last_metrics[metric] - value > improvement_threshold:
                            improvement = True
                            break
            
            if not improvement:
                print("Retrained model does not show significant improvement. Not deploying.")
                return
        
        # Model is better or this is the first training, deploy it
        # 1. Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(
            self.model_save_dir, 
            f"model_retrained_{timestamp}.pt"
        )
        
        torch.save(model.state_dict(), model_path)
        
        # 2. Update the current model
        self.model.load_state_dict(model.state_dict())
        
        # 3. Log the retraining
        retraining_entry = {
            'timestamp': time.time(),
            'human_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': model_path,
            'metrics': metrics
        }
        
        self.retraining_log.append(retraining_entry)
        self._save_retraining_log()
        
        print(f"Model deployed successfully. Saved to {model_path}")
    
    def run_retraining_pipeline(
        self,
        current_metrics: Dict[str, float],
        candidate_pool: List[Dict] = None
    ) -> bool:
        """
        Run the complete retraining pipeline
        
        Args:
            current_metrics: Current performance metrics
            candidate_pool: Pool of new data for potential retraining
            
        Returns:
            True if retraining was performed, False otherwise
        """
        # 1. Check if retraining is needed
        if not self.should_retrain(current_metrics):
            print("Retraining not required at this time.")
            return False
        
        print("Starting retraining pipeline...")
        
        # 2. Get candidate data if not provided
        if candidate_pool is None:
            candidate_pool = self.data_manager.get_candidate_pool()
        
        # 3. Extract features, labels, and metadata
        sample_features, sample_labels, sample_metadata = self.data_manager.extract_data(
            candidate_pool
        )
        
        # 4. Select most informative samples
        selected_indices = self.select_retraining_data(
            candidate_pool,
            sample_features,
            sample_labels,
            sample_metadata,
            max_samples=self.retraining_config.get('max_samples', 1000)
        )
        
        # 5. Get selected data
        selected_data = [candidate_pool[i] for i in selected_indices]
        
        # 6. Get validation and test data
        validation_data = self.data_manager.get_validation_data()
        test_data = self.data_manager.get_test_data()
        
        # 7. Retrain model
        retrained_model = self.retrain_model(selected_data, validation_data)
        
        # 8. Evaluate retrained model
        metrics = self.evaluate_retrained_model(retrained_model, test_data)
        
        # 9. Deploy if better
        self.deploy_model(retrained_model, metrics)
        
        return True


class ActiveLearningSelector:
    """
    Active learning for selecting the most informative samples
    """
    def __init__(
        self,
        selection_method: str = 'uncertainty',
        uncertainty_threshold: float = 0.1
    ):
        self.selection_method = selection_method
        self.uncertainty_threshold = uncertainty_threshold
    
    def select_samples(
        self,
        model: nn.Module,
        sample_features: List[np.ndarray],
        sample_labels: List[int],
        max_samples: int = 1000
    ) -> List[int]:
        """
        Select most informative samples using active learning
        
        Args:
            model: Current model
            sample_features: Features for candidate samples
            sample_labels: Labels for candidate samples
            max_samples: Maximum number of samples to select
            
        Returns:
            Indices of selected samples
        """
        if self.selection_method == 'uncertainty':
            return self._uncertainty_sampling(model, sample_features, max_samples)
        elif self.selection_method == 'diversity':
            return self._diversity_sampling(sample_features, max_samples)
        elif self.selection_method == 'hybrid':
            uncertain_indices = self._uncertainty_sampling(
                model, sample_features, max_samples * 2
            )
            # Get features for uncertain samples
            uncertain_features = [sample_features[i] for i in uncertain_indices]
            # Apply diversity sampling on uncertain samples
            diverse_indices = self._diversity_sampling(uncertain_features, max_samples)
            # Map back to original indices
            return [uncertain_indices[i] for i in diverse_indices]
        else:
            # Random sampling
            return np.random.choice(
                range(len(sample_features)), 
                size=min(max_samples, len(sample_features)), 
                replace=False
            ).tolist()
    
    def _uncertainty_sampling(
        self,
        model: nn.Module,
        sample_features: List[np.ndarray],
        max_samples: int
    ) -> List[int]:
        """Select samples with highest model uncertainty"""
        # Extract device
        device = next(model.parameters()).device
        
        # Put model in eval mode
        model.eval()
        
        # Convert features to tensors
        features_tensor = torch.tensor(
            np.array(sample_features), 
            dtype=torch.float32,
            device=device
        )
        
        # Get predictions in batches
        batch_size = 32
        uncertainties = []
        
        with torch.no_grad():
            for i in range(0, len(features_tensor), batch_size):
                batch = features_tensor[i:i+batch_size]
                # Get model outputs
                outputs, _ = model(batch)
                
                # Calculate uncertainty based on model task
                if outputs.shape[1] == 1:  # Binary classification
                    # Uncertainty = |0.5 - p|
                    probs = torch.sigmoid(outputs)
                    uncertainty = 0.5 - torch.abs(probs - 0.5)
                else:  # Multi-class classification
                    # Uncertainty = 1 - max(p)
                    probs = F.softmax(outputs, dim=1)
                    max_probs, _ = torch.max(probs, dim=1)
                    uncertainty = 1.0 - max_probs
                
                uncertainties.append(uncertainty.cpu().numpy())
        
        # Concatenate all uncertainties
        uncertainties = np.concatenate(uncertainties)
        
        # Get indices of most uncertain samples
        uncertain_indices = np.argsort(uncertainties)[::-1][:max_samples]
        
        return uncertain_indices.tolist()
    
    def _diversity_sampling(
        self,
        sample_features: List[np.ndarray],
        max_samples: int
    ) -> List[int]:
        """Select diverse set of samples using clustering"""
        from sklearn.cluster import KMeans
        
        # Convert to numpy array
        features = np.array(sample_features)
        
        # Determine number of clusters (limited by both max_samples and data size)
        n_clusters = min(max_samples, len(features))
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # For each cluster, select the sample closest to the centroid
        selected_indices = []
        
        for i in range(n_clusters):
            # Get samples in this cluster
            cluster_samples = np.where(clusters == i)[0]
            
            if len(cluster_samples) == 0:
                continue
            
            # Get distance to centroid for each sample
            centroid = kmeans.cluster_centers_[i]
            distances = np.array([
                np.linalg.norm(features[j] - centroid) 
                for j in cluster_samples
            ])
            
            # Select sample closest to centroid
            closest_idx = cluster_samples[np.argmin(distances)]
            selected_indices.append(closest_idx)
        
        return selected_indices
