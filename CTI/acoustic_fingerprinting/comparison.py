import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import time
from scipy.spatial.distance import euclidean, mahalanobis
from fastdtw import fastdtw
from scipy.stats import multivariate_normal
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F

from .speaker_patterns import BiometricFeatureExtractor, SpeakerProfileGenerator

class DTWComparator:
    """
    Dynamic Time Warping for comparing speech patterns with rate variations.
    """
    
    def __init__(self, radius: int = 10):
        """
        Initialize the DTW comparator.
        
        Args:
            radius: Sakoe-Chiba band radius for FastDTW
        """
        self.radius = radius
    
    def compare(
        self, 
        features1: np.ndarray, 
        features2: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compare two feature sequences using FastDTW.
        
        Args:
            features1: First feature sequence
            features2: Second feature sequence
            
        Returns:
            Tuple of (distance, path)
        """
        # Ensure features are 2D arrays
        if features1.ndim == 1:
            features1 = features1.reshape(-1, 1)
        if features2.ndim == 1:
            features2 = features2.reshape(-1, 1)
        
        # Calculate DTW distance and path
        distance, path = fastdtw(features1, features2, dist=euclidean, radius=self.radius)
        
        return distance, path
    
    def normalized_distance(
        self, 
        features1: np.ndarray, 
        features2: np.ndarray
    ) -> float:
        """
        Calculate normalized DTW distance between two feature sequences.
        
        Args:
            features1: First feature sequence
            features2: Second feature sequence
            
        Returns:
            Normalized distance
        """
        # Calculate DTW distance
        distance, _ = self.compare(features1, features2)
        
        # Normalize by sequence lengths
        norm_factor = (len(features1) + len(features2)) / 2
        normalized_distance = distance / norm_factor
        
        return normalized_distance


class VectorComparator:
    """
    Comparator for high-dimensional feature vectors.
    """
    
    def __init__(self, method: str = 'cosine'):
        """
        Initialize the vector comparator.
        
        Args:
            method: Comparison method ('cosine', 'euclidean', 'mahalanobis')
        """
        self.method = method
    
    def compare(
        self, 
        vector1: np.ndarray, 
        vector2: np.ndarray,
        cov_matrix: np.ndarray = None
    ) -> float:
        """
        Compare two feature vectors.
        
        Args:
            vector1: First feature vector
            vector2: Second feature vector
            cov_matrix: Covariance matrix for Mahalanobis distance
            
        Returns:
            Similarity score (higher is more similar)
        """
        if self.method == 'cosine':
            # Cosine similarity (higher is more similar)
            similarity = cosine_similarity(
                vector1.reshape(1, -1), 
                vector2.reshape(1, -1)
            )[0, 0]
            
            # Scale to [0, 1]
            similarity = (similarity + 1) / 2
            
        elif self.method == 'euclidean':
            # Euclidean distance (lower is more similar)
            distance = np.linalg.norm(vector1 - vector2)
            
            # Convert to similarity score (higher is more similar)
            similarity = 1 / (1 + distance)
            
        elif self.method == 'mahalanobis':
            if cov_matrix is None:
                raise ValueError("Covariance matrix required for Mahalanobis distance")
            
            # Ensure covariance matrix is invertible
            try:
                inv_cov = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                inv_cov = np.linalg.pinv(cov_matrix)
            
            # Mahalanobis distance (lower is more similar)
            distance = mahalanobis(vector1, vector2, inv_cov)
            
            # Convert to similarity score (higher is more similar)
            similarity = 1 / (1 + distance)
            
        else:
            raise ValueError(f"Unknown comparison method: {self.method}")
        
        return similarity


class ConfidenceScorer:
    """
    System for scoring confidence in speaker verification.
    """
    
    def __init__(
        self, 
        threshold: float = 0.7,
        secondary_threshold: float = 0.5,
        adaptive: bool = True
    ):
        """
        Initialize the confidence scorer.
        
        Args:
            threshold: Primary similarity threshold for acceptance
            secondary_threshold: Secondary threshold for borderline cases
            adaptive: Whether to use adaptive thresholding
        """
        self.threshold = threshold
        self.secondary_threshold = secondary_threshold
        self.adaptive = adaptive
        
        # History of similarity scores for adapting threshold
        self.genuine_scores = []
        self.impostor_scores = []
    
    def score_verification(
        self, 
        similarity: float, 
        feature_vector: np.ndarray, 
        profile: Dict
    ) -> Dict:
        """
        Score a verification attempt with confidence measure.
        
        Args:
            similarity: Similarity score
            feature_vector: Feature vector of verification attempt
            profile: Speaker profile to compare against
            
        Returns:
            Dictionary with verification result and confidence
        """
        # Get profile statistics
        feature_mean = profile['feature_mean']
        feature_cov = profile['feature_cov']
        
        # Calculate additional probabilistic score using multivariate Gaussian
        try:
            likelihood = multivariate_normal.pdf(
                feature_vector, 
                mean=feature_mean, 
                cov=feature_cov
            )
        except (np.linalg.LinAlgError, ValueError):
            # Fallback if covariance matrix is singular
            likelihood = multivariate_normal.pdf(
                feature_vector, 
                mean=feature_mean, 
                cov=np.diag(np.diag(feature_cov))
            )
        
        # Normalize likelihood to [0, 1] range
        # This is a heuristic since PDF values can be very small
        normalized_likelihood = 1 - np.exp(-likelihood * 1e6)
        
        # Combine similarity with likelihood for final score
        combined_score = 0.7 * similarity + 0.3 * normalized_likelihood
        
        # Get threshold (adaptive or fixed)
        current_threshold = self._get_adaptive_threshold() if self.adaptive else self.threshold
        
        # Determine verification result
        verified = combined_score >= current_threshold
        
        # Calculate confidence level
        if verified:
            # Higher confidence as score exceeds threshold
            confidence = (combined_score - current_threshold) / (1 - current_threshold)
            confidence = min(max(confidence, 0), 1)  # Clamp to [0, 1]
        else:
            # Lower confidence as score approaches threshold
            confidence = 1 - (current_threshold - combined_score) / current_threshold
            confidence = min(max(confidence, 0), 1)  # Clamp to [0, 1]
        
        # Handle borderline cases with secondary threshold
        borderline = abs(combined_score - current_threshold) < 0.1
        
        # Result dictionary
        result = {
            'verified': verified,
            'similarity_score': similarity,
            'likelihood_score': normalized_likelihood,
            'combined_score': combined_score,
            'threshold': current_threshold,
            'confidence': confidence,
            'borderline': borderline
        }
        
        return result
    
    def update_score_history(
        self, 
        similarity: float, 
        genuine: bool
    ):
        """
        Update history of similarity scores for adaptive thresholding.
        
        Args:
            similarity: Similarity score
            genuine: Whether this was a genuine match or impostor
        """
        if genuine:
            self.genuine_scores.append(similarity)
            # Keep only recent scores
            if len(self.genuine_scores) > 100:
                self.genuine_scores = self.genuine_scores[-100:]
        else:
            self.impostor_scores.append(similarity)
            # Keep only recent scores
            if len(self.impostor_scores) > 100:
                self.impostor_scores = self.impostor_scores[-100:]
    
    def _get_adaptive_threshold(self) -> float:
        """
        Calculate adaptive threshold based on score history.
        
        Returns:
            Adaptive threshold value
        """
        # Use default threshold if not enough data
        if len(self.genuine_scores) < 10 or len(self.impostor_scores) < 10:
            return self.threshold
        
        # Calculate statistics
        genuine_mean = np.mean(self.genuine_scores)
        genuine_std = np.std(self.genuine_scores)
        impostor_mean = np.mean(self.impostor_scores)
        impostor_std = np.std(self.impostor_scores)
        
        # Aim for equal error rate
        # Find threshold that maximizes separation
        if impostor_mean + impostor_std < genuine_mean - genuine_std:
            # Good separation
            threshold = (impostor_mean + impostor_std + genuine_mean - genuine_std) / 2
        else:
            # Poor separation, be more conservative
            threshold = (impostor_mean + genuine_mean) / 2 + genuine_std / 2
        
        # Ensure threshold is reasonable
        threshold = max(min(threshold, 0.95), 0.5)
        
        return threshold


class AcousticFingerprinter:
    """
    Complete acoustic fingerprinting system for speaker verification.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        feature_dim: int = 128,
        dtw_radius: int = 10,
        vector_comparison: str = 'cosine',
        threshold: float = 0.7,
        adaptive_threshold: bool = True
    ):
        """
        Initialize the acoustic fingerprinting system.
        
        Args:
            sample_rate: Audio sample rate in Hz
            feature_dim: Dimension of feature vectors
            dtw_radius: Radius for DTW comparison
            vector_comparison: Method for vector comparison
            threshold: Verification threshold
            adaptive_threshold: Whether to use adaptive thresholding
        """
        self.sample_rate = sample_rate
        
        # Initialize components
        self.feature_extractor = BiometricFeatureExtractor(
            sample_rate=sample_rate,
            feature_dim=feature_dim
        )
        
        self.profile_generator = SpeakerProfileGenerator(
            feature_extractor=self.feature_extractor
        )
        
        self.dtw_comparator = DTWComparator(radius=dtw_radius)
        
        self.vector_comparator = VectorComparator(method=vector_comparison)
        
        self.confidence_scorer = ConfidenceScorer(
            threshold=threshold,
            adaptive=adaptive_threshold
        )
    
    def enroll_speaker(
        self, 
        audio_samples: List[np.ndarray], 
        speaker_id: str
    ) -> Dict:
        """
        Enroll a new speaker in the system.
        
        Args:
            audio_samples: List of audio samples from the speaker
            speaker_id: Unique identifier for the speaker
            
        Returns:
            Speaker profile
        """
        # Generate profile
        profile = self.profile_generator.generate_profile(
            audio_samples=audio_samples,
            speaker_id=speaker_id
        )
        
        # Save profile
        profile_path = self.profile_generator.save_profile(profile)
        
        # Add path to profile
        profile['profile_path'] = profile_path
        
        return profile
    
    def verify_speaker(
        self, 
        audio: np.ndarray, 
        profile: Dict
    ) -> Dict:
        """
        Verify if audio matches the speaker profile.
        
        Args:
            audio: Audio to verify
            profile: Speaker profile to compare against
            
        Returns:
            Verification result
        """
        # Extract features
        feature_vector = self.feature_extractor.extract_biometric_features(audio)
        
        # Compare with profile
        similarity = self.vector_comparator.compare(
            feature_vector,
            profile['feature_mean'],
            profile.get('feature_cov')
        )
        
        # Score verification
        result = self.confidence_scorer.score_verification(
            similarity=similarity,
            feature_vector=feature_vector,
            profile=profile
        )
        
        # Add speaker ID to result
        result['speaker_id'] = profile['speaker_id']
        
        return result
    
    def update_speaker_profile(
        self, 
        profile: Dict, 
        new_audio_samples: List[np.ndarray]
    ) -> Dict:
        """
        Update a speaker profile with new audio samples.
        
        Args:
            profile: Existing speaker profile
            new_audio_samples: New audio samples
            
        Returns:
            Updated speaker profile
        """
        # Update profile
        updated_profile = self.profile_generator.update_profile(
            profile=profile,
            new_audio_samples=new_audio_samples
        )
        
        # Save updated profile
        profile_path = self.profile_generator.save_profile(updated_profile)
        
        # Add path to profile
        updated_profile['profile_path'] = profile_path
        
        return updated_profile
    
    def compare_audio_samples(
        self, 
        audio1: np.ndarray, 
        audio2: np.ndarray
    ) -> Dict:
        """
        Compare two audio samples directly.
        
        Args:
            audio1: First audio sample
            audio2: Second audio sample
            
        Returns:
            Comparison result
        """
        # Extract features
        features1 = self.feature_extractor.extract_biometric_features(audio1)
        features2 = self.feature_extractor.extract_biometric_features(audio2)
        
        # Compare feature vectors
        vector_similarity = self.vector_comparator.compare(features1, features2)
        
        # Calculate raw features for DTW (MFCCs)
        import librosa
        mfcc1 = librosa.feature.mfcc(
            y=audio1, 
            sr=self.sample_rate,
            n_mfcc=20
        ).T
        
        mfcc2 = librosa.feature.mfcc(
            y=audio2, 
            sr=self.sample_rate,
            n_mfcc=20
        ).T
        
        # DTW comparison
        dtw_distance, _ = self.dtw_comparator.compare(mfcc1, mfcc2)
        dtw_similarity = 1 / (1 + dtw_distance)
        
        # Combine similarities
        combined_similarity = 0.7 * vector_similarity + 0.3 * dtw_similarity
        
        # Result
        result = {
            'vector_similarity': vector_similarity,
            'dtw_similarity': dtw_similarity,
            'combined_similarity': combined_similarity,
            'match_likelihood': 'high' if combined_similarity > 0.8 else 
                               ('medium' if combined_similarity > 0.6 else 'low')
        }
        
        return result
