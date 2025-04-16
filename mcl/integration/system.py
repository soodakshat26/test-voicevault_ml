# integration/system.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import time
import os
import json
from datetime import datetime
import torch.nn.functional as F


from models.authentication.wavlm_ecapa import WavLM_ECAPA_TDNN
from models.antispoofing.fusion import AntiSpoofingSystem
from evaluation.anomaly_detection import AnomalogyDetector, SequenceAnomalyDetector

class VoiceAuthenticationSystem:
    """
    Complete end-to-end voice authentication system
    
    This class integrates speaker verification and anti-spoofing with
    decision logic, security policies, and explanation features.
    """
    def __init__(
        self,
        auth_model: Optional[nn.Module] = None,
        antispoofing_model: Optional[nn.Module] = None,
        config_path: str = 'config/system.json',
        models_dir: str = 'models',
        device: str = None
    ):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize models
        self.auth_model = auth_model or self._initialize_auth_model()
        self.antispoofing_model = antispoofing_model or self._initialize_antispoofing_model()
        
        # Initialize anomaly detector
        self.anomaly_detector = AnomalogyDetector(
            detection_method=self.config.get('anomaly_detection_method', 'isolation_forest'),
            contamination=self.config.get('anomaly_contamination', 0.01)
        )
        
        # Initialize sequence anomaly detector
        self.sequence_detector = None
        if self.config.get('use_sequence_detection', False):
            self.sequence_detector = SequenceAnomalyDetector(
                input_dim=self.config.get('sequence_input_dim', 2),
                hidden_dim=self.config.get('sequence_hidden_dim', 64),
                sequence_length=self.config.get('sequence_length', 10)
            )
        
        # Initialize security policy
        self.security_policy = SecurityPolicy(
            threshold=self.config.get('auth_threshold', 0.5),
            antispoofing_threshold=self.config.get('antispoofing_threshold', 0.5),
            liveness_required=self.config.get('liveness_required', True)
        )
        
        # Initialize explanation module
        self.explainer = AuthenticationExplainer(
            auth_model=self.auth_model,
            antispoofing_model=self.antispoofing_model
        )
        
        # Authentication history
        self.auth_history = []
    
    def _load_config(self, config_path: str) -> Dict:
        """Load system configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                'auth_threshold': 0.5,
                'antispoofing_threshold': 0.5,
                'liveness_required': True,
                'anomaly_detection_method': 'isolation_forest',
                'anomaly_contamination': 0.01,
                'use_sequence_detection': False,
                'auth_model_path': 'models/authentication/wavlm_ecapa.pt',
                'antispoofing_model_path': 'models/antispoofing/fusion_model.pt'
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    
    def _initialize_auth_model(self) -> nn.Module:
        """Initialize speaker authentication model"""
        # Create model
        model = WavLM_ECAPA_TDNN(
            wavlm_model_name=self.config.get('wavlm_model_name', 'microsoft/wavlm-base-plus'),
            embedding_dim=self.config.get('embedding_dim', 192)
        )
        
        # Load weights if available
        model_path = self.config.get('auth_model_path')
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        return model.to(self.device)
    
    def _initialize_antispoofing_model(self) -> nn.Module:
        """Initialize anti-spoofing model"""
        # Create model
        model = AntiSpoofingSystem(
            fusion_type=self.config.get('fusion_type', 'attention'),
            num_classes=2
        )
        
        # Load weights if available
        model_path = self.config.get('antispoofing_model_path')
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        return model.to(self.device)
    
    def authenticate(
        self,
        audio: torch.Tensor,
        enrolled_embedding: torch.Tensor,
        metadata: Dict = None
    ) -> Dict:
        """
        Authenticate a user based on voice
        
        Args:
            audio: Input audio waveform
            enrolled_embedding: Enrolled user embedding
            metadata: Additional metadata for authentication
            
        Returns:
            Authentication results
        """
        # Ensure audio is on the correct device
        audio = audio.to(self.device)
        enrolled_embedding = enrolled_embedding.to(self.device)
        
        # Extract embeddings from auth model
        self.auth_model.eval()
        with torch.no_grad():
            _, auth_embedding = self.auth_model(audio)
        
        # Calculate similarity to enrolled embedding
        similarity = self._calculate_similarity(auth_embedding, enrolled_embedding)
        
        # Run anti-spoofing check
        self.antispoofing_model.eval()
        with torch.no_grad():
            antispoofing_output, _, _, _ = self.antispoofing_model(audio)
        
        # Get probability of real (not spoofed) voice
        real_prob = torch.sigmoid(antispoofing_output[:, 0]).item()
        
        # Check for anomalies if anomaly detector is fitted
        anomaly_detected = False
        anomaly_score = 0.0
        
        if hasattr(self.anomaly_detector, 'is_fitted') and self.anomaly_detector.is_fitted:
            # Use embedding for anomaly detection
            is_anomaly, anomaly_score = self.anomaly_detector.predict(
                auth_embedding.cpu().numpy()
            )
            anomaly_detected = bool(is_anomaly)
        
        # Check sequence anomaly if applicable
        sequence_anomaly = False
        sequence_score = 0.0
        
        if (self.sequence_detector is not None and 
            hasattr(self.sequence_detector, 'is_fitted') and 
            self.sequence_detector.is_fitted and
            metadata is not None and 'sequence' in metadata):
            
            sequence = metadata['sequence']
            is_sequence_anomaly, sequence_score = self.sequence_detector.predict(sequence)
            sequence_anomaly = bool(is_sequence_anomaly)
        
        # Apply security policy
        auth_decision = self.security_policy.evaluate(
            similarity=similarity,
            antispoofing_score=real_prob,
            anomaly_detected=anomaly_detected,
            sequence_anomaly=sequence_anomaly
        )
        
        # Generate explanation
        explanation = self.explainer.explain(
            auth_embedding=auth_embedding,
            enrolled_embedding=enrolled_embedding,
            similarity=similarity,
            antispoofing_score=real_prob,
            anomaly_score=anomaly_score,
            sequence_score=sequence_score,
            auth_decision=auth_decision
        )
        
        # Record authentication attempt
        self._record_authentication(
            similarity=similarity,
            antispoofing_score=real_prob,
            auth_decision=auth_decision,
            anomaly_detected=anomaly_detected,
            metadata=metadata
        )
        
        # Return results
        result = {
            'authenticated': auth_decision,
            'similarity': float(similarity),
            'antispoofing_score': float(real_prob),
            'anomaly_detected': anomaly_detected,
            'anomaly_score': float(anomaly_score),
            'sequence_anomaly': sequence_anomaly,
            'sequence_score': float(sequence_score),
            'explanation': explanation,
            'auth_embedding': auth_embedding.cpu().numpy(),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _calculate_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:
        """Calculate similarity between embeddings"""
        # Normalize embeddings
        embedding1 = F.normalize(embedding1, p=2, dim=1)
        embedding2 = F.normalize(embedding2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.sum(embedding1 * embedding2, dim=1).item()
        
        return similarity
    
    def _record_authentication(
        self,
        similarity: float,
        antispoofing_score: float,
        auth_decision: bool,
        anomaly_detected: bool,
        metadata: Dict = None
    ):
        """Record authentication attempt for later analysis"""
        timestamp = time.time()
        
        record = {
            'timestamp': timestamp,
            'human_time': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'similarity': similarity,
            'antispoofing_score': antispoofing_score,
            'auth_decision': auth_decision,
            'anomaly_detected': anomaly_detected
        }
        
        # Add metadata if available
        if metadata:
            record['metadata'] = metadata
        
        # Add to history
        self.auth_history.append(record)
        
        # Limit history size
        max_history = self.config.get('max_history_size', 1000)
        if len(self.auth_history) > max_history:
            self.auth_history = self.auth_history[-max_history:]
    
    def enroll_user(
        self,
        audio_samples: List[torch.Tensor],
        user_id: str
    ) -> Dict:
        """
        Enroll a new user in the system
        
        Args:
            audio_samples: List of audio samples from the user
            user_id: Unique identifier for the user
            
        Returns:
            Enrollment results
        """
        # Process each audio sample to get embeddings
        embeddings = []
        
        self.auth_model.eval()
        with torch.no_grad():
            for audio in audio_samples:
                audio = audio.to(self.device)
                _, embedding = self.auth_model(audio)
                embeddings.append(embedding)
        
        # Average embeddings
        embeddings = torch.cat(embeddings, dim=0)
        mean_embedding = torch.mean(embeddings, dim=0, keepdim=True)
        
        # Normalize embedding
        enrolled_embedding = F.normalize(mean_embedding, p=2, dim=1)
        
        # Return enrollment data
        return {
            'user_id': user_id,
            'enrolled_embedding': enrolled_embedding.cpu().numpy(),
            'num_samples': len(audio_samples),
            'timestamp': datetime.now().isoformat()
        }
    
    def update_anomaly_detector(self, normal_embeddings: List[np.ndarray]):
        """
        Update anomaly detector with new normal embeddings
        
        Args:
            normal_embeddings: List of normal authentication embeddings
        """
        self.anomaly_detector.fit(normal_embeddings)
        
        print(f"Anomaly detector updated with {len(normal_embeddings)} normal samples.")
    
    def update_sequence_detector(self, normal_sequences: List[torch.Tensor]):
        """
        Update sequence anomaly detector with new normal sequences
        
        Args:
            normal_sequences: List of normal authentication sequences
        """
        if self.sequence_detector is None:
            print("Sequence detector not enabled in configuration.")
            return
        
        self.sequence_detector.fit(normal_sequences)
        
        print(f"Sequence detector updated with {len(normal_sequences)} normal sequences.")
    
    def save_models(self, save_dir: str = 'models'):
        """Save all models to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save authentication model
        auth_path = os.path.join(save_dir, 'authentication', 'wavlm_ecapa.pt')
        os.makedirs(os.path.dirname(auth_path), exist_ok=True)
        torch.save(self.auth_model.state_dict(), auth_path)
        
        # Save anti-spoofing model
        antispoofing_path = os.path.join(save_dir, 'antispoofing', 'fusion_model.pt')
        os.makedirs(os.path.dirname(antispoofing_path), exist_ok=True)
        torch.save(self.antispoofing_model.state_dict(), antispoofing_path)
        
        # Save anomaly detector
        if hasattr(self.anomaly_detector, 'is_fitted') and self.anomaly_detector.is_fitted:
            anomaly_path = os.path.join(save_dir, 'anomaly', 'detector.joblib')
            os.makedirs(os.path.dirname(anomaly_path), exist_ok=True)
            self.anomaly_detector.save_model(anomaly_path)
        
        # Save sequence detector
        if (self.sequence_detector is not None and 
            hasattr(self.sequence_detector, 'is_fitted') and 
            self.sequence_detector.is_fitted):
            
            sequence_path = os.path.join(save_dir, 'anomaly', 'sequence_detector.pt')
            os.makedirs(os.path.dirname(sequence_path), exist_ok=True)
            torch.save(self.sequence_detector.state_dict(), sequence_path)
        
        # Update config with new paths
        self.config['auth_model_path'] = auth_path
        self.config['antispoofing_model_path'] = antispoofing_path
        
        # Save updated config
        config_path = 'config/system.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Models saved to {save_dir}")
        print(f"Configuration updated and saved to {config_path}")


class SecurityPolicy:
    """
    Security policy for voice authentication system
    
    This class defines policies for accepting or rejecting authentication
    attempts based on various security factors.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        antispoofing_threshold: float = 0.5,
        liveness_required: bool = True,
        adaptive_thresholds: bool = False
    ):
        self.threshold = threshold
        self.antispoofing_threshold = antispoofing_threshold
        self.liveness_required = liveness_required
        self.adaptive_thresholds = adaptive_thresholds
        
        # History for adaptive thresholds
        if adaptive_thresholds:
            self.auth_history = []
            self.max_history = 100
    
    def evaluate(
        self,
        similarity: float,
        antispoofing_score: float,
        anomaly_detected: bool = False,
        sequence_anomaly: bool = False
    ) -> bool:
        """
        Evaluate authentication request based on security policy
        
        Args:
            similarity: Similarity score to enrolled voice
            antispoofing_score: Anti-spoofing confidence score
            anomaly_detected: Whether an anomaly was detected
            sequence_anomaly: Whether a sequence anomaly was detected
            
        Returns:
            True if authentication is approved, False otherwise
        """
        # Get current thresholds
        current_threshold = self._get_threshold() if self.adaptive_thresholds else self.threshold
        
        # Basic authentication check
        authenticated = similarity >= current_threshold
        
        # Antispoofing check if required
        if self.liveness_required:
            authenticated = authenticated and (antispoofing_score >= self.antispoofing_threshold)
        
        # Anomaly checks
        if anomaly_detected or sequence_anomaly:
            authenticated = False
        
        # Update history for adaptive thresholds
        if self.adaptive_thresholds:
            self._update_history(similarity, authenticated)
        
        return authenticated
    
    def _get_threshold(self) -> float:
        """Get adaptive threshold based on recent authentication history"""
        if not self.auth_history:
            return self.threshold
        
        # Calculate threshold based on recent successful authentications
        successful_scores = [entry['similarity'] for entry in self.auth_history 
                           if entry['authenticated']]
        
        if not successful_scores:
            return self.threshold
        
        # Use a percentile-based approach
        import numpy as np
        adaptive_threshold = max(
            self.threshold,
            np.percentile(successful_scores, 10)  # 10th percentile of successful scores
        )
        
        return adaptive_threshold
    
    def _update_history(self, similarity: float, authenticated: bool):
        """Update authentication history"""
        self.auth_history.append({
            'similarity': similarity,
            'authenticated': authenticated,
            'timestamp': time.time()
        })
        
        # Limit history size
        if len(self.auth_history) > self.max_history:
            self.auth_history = self.auth_history[-self.max_history:]
    
    def update_thresholds(
        self,
        new_threshold: Optional[float] = None,
        new_antispoofing_threshold: Optional[float] = None,
        new_liveness_required: Optional[bool] = None
    ):
        """Update security policy thresholds"""
        if new_threshold is not None:
            self.threshold = new_threshold
        
        if new_antispoofing_threshold is not None:
            self.antispoofing_threshold = new_antispoofing_threshold
        
        if new_liveness_required is not None:
            self.liveness_required = new_liveness_required


class AuthenticationExplainer:
    """
    Explanation module for voice authentication system
    
    This class provides explanations for authentication decisions,
    helping users and administrators understand the system's behavior.
    """
    def __init__(
        self,
        auth_model: nn.Module,
        antispoofing_model: nn.Module
    ):
        self.auth_model = auth_model
        self.antispoofing_model = antispoofing_model
    
    def explain(
        self,
        auth_embedding: torch.Tensor,
        enrolled_embedding: torch.Tensor,
        similarity: float,
        antispoofing_score: float,
        anomaly_score: float,
        sequence_score: float,
        auth_decision: bool
    ) -> Dict:
        """
        Generate explanation for authentication decision
        
        Args:
            auth_embedding: Authentication embedding
            enrolled_embedding: Enrolled user embedding
            similarity: Similarity score
            antispoofing_score: Anti-spoofing score
            anomaly_score: Anomaly detection score
            sequence_score: Sequence anomaly score
            auth_decision: Final authentication decision
            
        Returns:
            Dictionary with explanation details
        """
        # Simple explanation with the main factors
        factors = []
        
        # Voice similarity factor
        similarity_factor = {
            'name': 'Voice Similarity',
            'score': similarity,
            'influence': 'high',
            'description': f"How closely the voice matches the enrolled user (score: {similarity:.3f})"
        }
        factors.append(similarity_factor)
        
        # Anti-spoofing factor
        if antispoofing_score is not None:
            spoofing_factor = {
                'name': 'Voice Authenticity',
                'score': antispoofing_score,
                'influence': 'high',
                'description': f"Confidence that the voice is real, not synthetic (score: {antispoofing_score:.3f})"
            }
            factors.append(spoofing_factor)
        
        # Anomaly factor
        if anomaly_score != 0:
            anomaly_factor = {
                'name': 'Anomaly Detection',
                'score': 1.0 - min(1.0, max(0.0, anomaly_score)),
                'influence': 'medium',
                'description': f"Unusual voice patterns compared to normal authentication (score: {anomaly_score:.3f})"
            }
            factors.append(anomaly_factor)
        
        # Sequence factor
        if sequence_score != 0:
            sequence_factor = {
                'name': 'Authentication Pattern',
                'score': 1.0 - min(1.0, max(0.0, sequence_score)),
                'influence': 'low',
                'description': f"Unusual patterns in authentication behavior (score: {sequence_score:.3f})"
            }
            factors.append(sequence_factor)
        
        # Overall outcome
        decision_explanation = (
            "Authentication successful. Voice verified." if auth_decision else
            "Authentication failed. Voice could not be verified."
        )
        
        # Generate detailed text explanation
        text_explanation = self._generate_text_explanation(
            similarity=similarity,
            antispoofing_score=antispoofing_score,
            anomaly_score=anomaly_score,
            sequence_score=sequence_score,
            auth_decision=auth_decision
        )
        
        # Complete explanation
        explanation = {
            'decision': auth_decision,
            'decision_explanation': decision_explanation,
            'factors': factors,
            'text_explanation': text_explanation
        }
        
        return explanation
    
    def _generate_text_explanation(
        self,
        similarity: float,
        antispoofing_score: float,
        anomaly_score: float,
        sequence_score: float,
        auth_decision: bool
    ) -> str:
        """Generate detailed text explanation for authentication decision"""
        # Format for readability
        sim_str = f"{similarity:.3f}"
        spoof_str = f"{antispoofing_score:.3f}" if antispoofing_score is not None else "N/A"
        anomaly_str = f"{anomaly_score:.3f}" if anomaly_score != 0 else "N/A"
        seq_str = f"{sequence_score:.3f}" if sequence_score != 0 else "N/A"
        
        parts = []
        
        # Introduction based on decision
        if auth_decision:
            parts.append("Authentication successful.")
        else:
            parts.append("Authentication failed.")
        
        # Voice similarity explanation
        if similarity >= 0.7:
            parts.append(f"Voice similarity is very high ({sim_str}).")
        elif similarity >= 0.5:
            parts.append(f"Voice similarity is good ({sim_str}).")
        else:
            parts.append(f"Voice similarity is low ({sim_str}).")
        
        # Anti-spoofing explanation
        if antispoofing_score is not None:
            if antispoofing_score >= 0.8:
                parts.append(f"Voice authenticity is very high ({spoof_str}).")
            elif antispoofing_score >= 0.5:
                parts.append(f"Voice authenticity is acceptable ({spoof_str}).")
            else:
                parts.append(f"Voice authenticity is questionable ({spoof_str}).")
        
        # Anomaly explanation
        if anomaly_score != 0:
            if anomaly_score >= 3.0:
                parts.append(f"Significant anomalies detected ({anomaly_str}).")
            elif anomaly_score >= 1.0:
                parts.append(f"Minor anomalies detected ({anomaly_str}).")
        
        # Sequence explanation
        if sequence_score != 0:
            if sequence_score >= 3.0:
                parts.append(f"Unusual authentication pattern detected ({seq_str}).")
            elif sequence_score >= 1.0:
                parts.append(f"Slightly unusual authentication pattern ({seq_str}).")
        
        # Conclusion
        if auth_decision:
            parts.append("All security checks passed.")
        else:
            # Explain failure reason
            if similarity < 0.5:
                parts.append("Voice does not match enrolled user.")
            elif antispoofing_score is not None and antispoofing_score < 0.5:
                parts.append("Voice may be synthetic or manipulated.")
            elif anomaly_score >= 3.0:
                parts.append("Authentication blocked due to security anomalies.")
            elif sequence_score >= 3.0:
                parts.append("Authentication blocked due to unusual access pattern.")
            else:
                parts.append("Multiple security factors did not meet required thresholds.")
        
        return " ".join(parts)
