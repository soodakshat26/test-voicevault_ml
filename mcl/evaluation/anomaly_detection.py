# evaluation/anomaly_detection.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import os   

class AnomalogyDetector:
    """
    Framework for detecting anomalous authentication attempts
    """
    def __init__(
        self,
        detection_method: str = 'isolation_forest',
        contamination: float = 0.01,
        feature_type: str = 'embedding'  # 'embedding' or 'scores'
    ):
        self.detection_method = detection_method
        self.contamination = contamination
        self.feature_type = feature_type
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Training state
        self.is_fitted = False
        self.normal_embedding_mean = None
        self.normal_embedding_std = None
    
    def _initialize_model(self):
        """Initialize anomaly detection model"""
        if self.detection_method == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.detection_method == 'one_class_svm':
            return OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
        elif self.detection_method == 'vae':
            # For simplicity, we'll use a numpy-based approach
            # A full implementation would use a proper VAE model
            return None
        else:
            raise ValueError(f"Unsupported detection method: {self.detection_method}")
    
    def fit(
        self,
        normal_samples: Union[List[np.ndarray], np.ndarray],
        sample_metadata: Optional[List[Dict]] = None
    ):
        """
        Train the anomaly detector on normal authentication samples
        
        Args:
            normal_samples: Normal authentication embeddings or scores
            sample_metadata: Optional metadata for each sample
        """
        if isinstance(normal_samples, list):
            normal_samples = np.array(normal_samples)
        
        # For embedding-based detection, compute mean and std
        if self.feature_type == 'embedding':
            self.normal_embedding_mean = np.mean(normal_samples, axis=0)
            self.normal_embedding_std = np.std(normal_samples, axis=0)
        
        # Train the model
        if self.detection_method in ['isolation_forest', 'one_class_svm']:
            self.model.fit(normal_samples)
            self.is_fitted = True
        elif self.detection_method == 'vae':
            # Skip actual VAE training for this simplified version
            self.is_fitted = True
    
    def predict(
        self,
        samples: Union[List[np.ndarray], np.ndarray],
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in authentication attempts
        
        Args:
            samples: Authentication embeddings or scores to evaluate
            threshold: Optional anomaly score threshold
            
        Returns:
            (is_anomaly, anomaly_scores)
        """
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector must be fitted before prediction")
        
        if isinstance(samples, list):
            samples = np.array(samples)
        
        # Compute anomaly scores
        if self.detection_method == 'isolation_forest':
            # Anomaly score (-1 for anomalies, 1 for normal)
            predictions = self.model.predict(samples)
            scores = self.model.score_samples(samples)
            # Convert predictions to boolean (True for anomalies)
            is_anomaly = (predictions == -1)
            
        elif self.detection_method == 'one_class_svm':
            # Anomaly score (negative for anomalies)
            predictions = self.model.predict(samples)
            scores = self.model.decision_function(samples)
            # Convert predictions to boolean (True for anomalies)
            is_anomaly = (predictions == -1)
            
        elif self.detection_method == 'vae':
            # Simplified approach using Mahalanobis distance
            # Calculate squared Mahalanobis distance
            diff = samples - self.normal_embedding_mean
            cov_inv = np.linalg.inv(np.diag(self.normal_embedding_std**2))
            scores = np.array([d.T.dot(cov_inv).dot(d) for d in diff])
            
            # Apply threshold if provided
            if threshold is None:
                threshold = np.percentile(scores, 95)
            
            is_anomaly = scores > threshold
        
        return is_anomaly, scores
    
    def save_model(self, path: str):
        """Save anomaly detection model"""
        import joblib
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Store model and parameters
        model_data = {
            'model': self.model,
            'detection_method': self.detection_method,
            'contamination': self.contamination,
            'feature_type': self.feature_type,
            'is_fitted': self.is_fitted,
            'normal_embedding_mean': self.normal_embedding_mean,
            'normal_embedding_std': self.normal_embedding_std
        }
        
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load anomaly detection model"""
        import joblib
        
        # Load model and parameters
        model_data = joblib.load(path)
        
        # Set attributes
        self.model = model_data['model']
        self.detection_method = model_data['detection_method']
        self.contamination = model_data['contamination']
        self.feature_type = model_data['feature_type']
        self.is_fitted = model_data['is_fitted']
        self.normal_embedding_mean = model_data['normal_embedding_mean']
        self.normal_embedding_std = model_data['normal_embedding_std']
        
        return self


class SequenceAnomalyDetector:
    """
    LSTM-based sequence modeling for detecting temporal attack patterns
    """
    def __init__(
        self,
        input_dim: int = 2,  # Default: [score, duration]
        hidden_dim: int = 64,
        num_layers: int = 2,
        sequence_length: int = 10,
        threshold: float = 3.0  # Standard deviations for anomaly
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.threshold = threshold
        
        # Initialize LSTM model
        self.model = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer to predict next values
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        # Training state
        self.is_fitted = False
        self.prediction_errors = []
    
    def fit(
        self,
        sequences: List[torch.Tensor],
        num_epochs: int = 100,
        learning_rate: float = 0.001
    ):
        """
        Train the sequence model on normal authentication patterns
        
        Args:
            sequences: List of authentication sequence tensors
                Each tensor should be (sequence_length, input_dim)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        # Prepare model for training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.output_layer.to(device)
        
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.output_layer.parameters()),
            lr=learning_rate
        )
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for seq in sequences:
                # Move to device
                seq = seq.to(device)
                
                # Create input and target sequences
                x = seq[:-1]  # All but last
                y = seq[1:]   # All but first
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output, _ = self.model(x.unsqueeze(0))
                prediction = self.output_layer(output.squeeze(0))
                
                # Compute loss
                loss = criterion(prediction, y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(sequences):.6f}")
        
        # Compute prediction errors on training data
        self.prediction_errors = []
        
        with torch.no_grad():
            for seq in sequences:
                seq = seq.to(device)
                x = seq[:-1]
                y = seq[1:]
                
                output, _ = self.model(x.unsqueeze(0))
                prediction = self.output_layer(output.squeeze(0))
                
                # Compute error
                error = torch.mean((prediction - y)**2, dim=1)
                self.prediction_errors.extend(error.cpu().numpy())
        
        # Compute error statistics
        self.error_mean = np.mean(self.prediction_errors)
        self.error_std = np.std(self.prediction_errors)
        
        self.is_fitted = True
    
    def predict(
        self,
        sequence: torch.Tensor
    ) -> Tuple[bool, float]:
        """
        Detect anomalies in authentication sequence
        
        Args:
            sequence: Authentication sequence tensor (sequence_length, input_dim)
            
        Returns:
            (is_anomaly, anomaly_score)
        """
        if not self.is_fitted:
            raise RuntimeError("Sequence anomaly detector must be fitted before prediction")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with torch.no_grad():
            # Move to device
            sequence = sequence.to(device)
            
            # Create input and expected output
            x = sequence[:-1]  # All but last
            y = sequence[1:]   # All but first
            
            # Get prediction
            output, _ = self.model(x.unsqueeze(0))
            prediction = self.output_layer(output.squeeze(0))
            
            # Compute error
            error = torch.mean((prediction - y)**2, dim=1)
            
            # Convert to numpy for analysis
            error_np = error.cpu().numpy()
            
            # Compute anomaly score (z-score)
            anomaly_score = (np.mean(error_np) - self.error_mean) / self.error_std
            
            # Detect anomaly if error is above threshold
            is_anomaly = anomaly_score > self.threshold
        
        return is_anomaly, anomaly_score
