import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import json
import time
from datetime import datetime
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from scipy.stats import norm

from .signal_decomposition import SignalDecomposer, SubAudioAnalyzer, extract_formants


class FormantTracker:
    """
    Track and analyze formants in speech for speaker identification.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the formant tracker.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
    
    def extract_formant_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract formant-based features from speech.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of formant features
        """
        # Extract formants
        formants = extract_formants(audio, self.sample_rate)
        
        # Calculate formant statistics (mean, std, median)
        formant_means = np.nanmean(formants, axis=1)
        formant_stds = np.nanstd(formants, axis=1)
        formant_medians = np.nanmedian(formants, axis=1)
        
        # Calculate formant ratios (F2/F1, F3/F2, F4/F3)
        # These are particularly useful for speaker identification
        f2_f1_ratio = formants[1] / (formants[0] + 1e-10)
        f3_f2_ratio = formants[2] / (formants[1] + 1e-10)
        f4_f3_ratio = formants[3] / (formants[2] + 1e-10)
        
        # Calculate formant bandwidths (approximation)
        formant_bandwidths = formant_stds * 2
        
        # Calculate formant trajectories (delta)
        formant_deltas = np.diff(formants, axis=1)
        formant_delta_means = np.nanmean(formant_deltas, axis=1)
        formant_delta_stds = np.nanstd(formant_deltas, axis=1)
        
        # Package features
        features = {
            'formants': formants,
            'formant_means': formant_means,
            'formant_stds': formant_stds,
            'formant_medians': formant_medians,
            'formant_bandwidths': formant_bandwidths,
            'f2_f1_ratio': f2_f1_ratio,
            'f3_f2_ratio': f3_f2_ratio,
            'f4_f3_ratio': f4_f3_ratio,
            'formant_delta_means': formant_delta_means,
            'formant_delta_stds': formant_delta_stds
        }
        
        return features


class BiometricFeatureExtractor:
    """
    Extract biometric features from speech for speaker identification.
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        use_formants: bool = True,
        use_mfcc: bool = True,
        use_pitch: bool = True,
        feature_dim: int = 128
    ):
        """
        Initialize the biometric feature extractor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            use_formants: Whether to use formant features
            use_mfcc: Whether to use MFCC features
            use_pitch: Whether to use pitch features
            feature_dim: Dimension of the final feature vector
        """
        self.sample_rate = sample_rate
        self.use_formants = use_formants
        self.use_mfcc = use_mfcc
        self.use_pitch = use_pitch
        self.feature_dim = feature_dim
        
        # Initialize components
        self.sub_audio_analyzer = SubAudioAnalyzer(sample_rate=sample_rate)
        self.formant_tracker = FormantTracker(sample_rate=sample_rate)
        
        # Initialize dimensionality reduction
        self.pca = None
        self.scaler = StandardScaler()
    
    def extract_biometric_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract speaker-specific biometric features.
        
        Args:
            audio: Audio signal
            
        Returns:
            Biometric feature vector
        """
        # Get core acoustic features
        acoustic_features = self.sub_audio_analyzer.extract_features(audio)
        
        # Get formant features if enabled
        formant_features = None
        if self.use_formants:
            formant_features = self.formant_tracker.extract_formant_features(audio)
        
        # Collect features to use
        feature_list = []
        
        # Add MFCC features if enabled
        if self.use_mfcc:
            # Compute statistics of MFCCs
            mfcc_means = np.mean(acoustic_features['mfcc'], axis=1)
            mfcc_stds = np.std(acoustic_features['mfcc'], axis=1)
            
            mfcc_delta_means = np.mean(acoustic_features['mfcc_delta'], axis=1)
            mfcc_delta_stds = np.std(acoustic_features['mfcc_delta'], axis=1)
            
            mfcc_delta2_means = np.mean(acoustic_features['mfcc_delta2'], axis=1)
            mfcc_delta2_stds = np.std(acoustic_features['mfcc_delta2'], axis=1)
            
            # Append to feature list
            feature_list.extend([
                mfcc_means, mfcc_stds, 
                mfcc_delta_means, mfcc_delta_stds,
                mfcc_delta2_means, mfcc_delta2_stds
            ])
        
        # Add pitch features if enabled
        if self.use_pitch:
            # Get pitch statistics (ignoring unvoiced regions)
            f0 = acoustic_features['f0']
            voiced_mask = acoustic_features['voiced_flag'] > 0
            
            if np.any(voiced_mask):
                f0_voiced = f0[voiced_mask]
                f0_mean = np.mean(f0_voiced)
                f0_std = np.std(f0_voiced)
                f0_min = np.min(f0_voiced)
                f0_max = np.max(f0_voiced)
                f0_range = f0_max - f0_min
            else:
                f0_mean = 0
                f0_std = 0
                f0_min = 0
                f0_max = 0
                f0_range = 0
            
            # Append to feature list
            feature_list.extend([
                np.array([f0_mean]),
                np.array([f0_std]),
                np.array([f0_min]),
                np.array([f0_max]),
                np.array([f0_range])
            ])
        
        # Add formant features if enabled
        if self.use_formants and formant_features is not None:
            # Append formant statistics to feature list
            feature_list.extend([
                formant_features['formant_means'],
                formant_features['formant_stds'],
                formant_features['formant_delta_means'],
                np.mean(formant_features['f2_f1_ratio']),
                np.mean(formant_features['f3_f2_ratio']),
                np.mean(formant_features['f4_f3_ratio'])
            ])
        
        # Add spectral features
        spec_centroid_mean = np.mean(acoustic_features['spectral_centroid'])
        spec_bandwidth_mean = np.mean(acoustic_features['spectral_bandwidth'])
        spec_contrast_mean = np.mean(acoustic_features['spectral_contrast'], axis=1)
        spec_flatness_mean = np.mean(acoustic_features['spectral_flatness'])
        
        # Append to feature list
        feature_list.extend([
            np.array([spec_centroid_mean]),
            np.array([spec_bandwidth_mean]),
            spec_contrast_mean,
            np.array([spec_flatness_mean])
        ])
        
        # Concatenate all features
        concatenated_features = np.concatenate([f.flatten() for f in feature_list])
        
        # Standardize features
        scaled_features = self.scaler.fit_transform(concatenated_features.reshape(1, -1))
        
        # Apply dimensionality reduction if feature dimension is specified
        if self.pca is None and self.feature_dim < len(concatenated_features):
            self.pca = PCA(n_components=self.feature_dim)
            reduced_features = self.pca.fit_transform(scaled_features).flatten()
        elif self.pca is not None:
            reduced_features = self.pca.transform(scaled_features).flatten()
        else:
            reduced_features = scaled_features.flatten()
        
        return reduced_features


class SpeakerProfileGenerator:
    """
    Generate and store speaker profiles from biometric features.
    """
    
    def __init__(
        self, 
        feature_extractor: BiometricFeatureExtractor,
        storage_dir: str = 'data/speaker_profiles'
    ):
        """
        Initialize the speaker profile generator.
        
        Args:
            feature_extractor: Biometric feature extractor
            storage_dir: Directory to store speaker profiles
        """
        self.feature_extractor = feature_extractor
        self.storage_dir = storage_dir
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
    
    def generate_profile(
        self, 
        audio_samples: List[np.ndarray], 
        speaker_id: str
    ) -> Dict:
        """
        Generate a speaker profile from multiple audio samples.
        
        Args:
            audio_samples: List of audio samples from the speaker
            speaker_id: Unique identifier for the speaker
            
        Returns:
            Speaker profile dictionary
        """
        # Extract features from each audio sample
        features = []
        for audio in audio_samples:
            feature_vector = self.feature_extractor.extract_biometric_features(audio)
            features.append(feature_vector)
        
        # Convert to numpy array
        features = np.array(features)
        
        # Calculate profile statistics
        feature_mean = np.mean(features, axis=0)
        feature_cov = np.cov(features, rowvar=False)
        
        # Create a statistical model of the feature distribution
        # We'll use a Gaussian model (mean and covariance)
        profile = {
            'speaker_id': speaker_id,
            'feature_mean': feature_mean,
            'feature_cov': feature_cov,
            'num_samples': len(audio_samples),
            'created_at': datetime.now().isoformat(),
            'feature_dim': feature_mean.shape[0]
        }
        
        return profile
    
    def save_profile(self, profile: Dict, speaker_id: str = None) -> str:
        """
        Save speaker profile to storage.
        
        Args:
            profile: Speaker profile
            speaker_id: Override speaker ID (optional)
            
        Returns:
            Path to saved profile
        """
        speaker_id = speaker_id or profile['speaker_id']
        
        # Create filename with timestamp
        timestamp = int(time.time())
        filename = f"{speaker_id}_{timestamp}.pkl"
        
        # Full path
        profile_path = os.path.join(self.storage_dir, filename)
        
        # Save using pickle
        with open(profile_path, 'wb') as f:
            pickle.dump(profile, f)
        
        return profile_path
    
    def load_profile(self, profile_path: str) -> Dict:
        """
        Load speaker profile from storage.
        
        Args:
            profile_path: Path to profile file
            
        Returns:
            Speaker profile dictionary
        """
        with open(profile_path, 'rb') as f:
            profile = pickle.load(f)
        
        return profile
    
    def update_profile(
        self, 
        profile: Dict, 
        new_audio_samples: List[np.ndarray]
    ) -> Dict:
        """
        Update an existing speaker profile with new audio samples.
        
        Args:
            profile: Existing speaker profile
            new_audio_samples: New audio samples
            
        Returns:
            Updated speaker profile
        """
        # Extract features from new samples
        new_features = []
        for audio in new_audio_samples:
            feature_vector = self.feature_extractor.extract_biometric_features(audio)
            new_features.append(feature_vector)
        
        # Convert to numpy array
        new_features = np.array(new_features)
        
        # Get existing statistics
        feature_mean = profile['feature_mean']
        feature_cov = profile['feature_cov']
        num_samples = profile['num_samples']
        
        # Calculate weighted mean and covariance
        total_samples = num_samples + len(new_audio_samples)
        weight_old = num_samples / total_samples
        weight_new = len(new_audio_samples) / total_samples
        
        # Update mean
        new_mean = np.mean(new_features, axis=0)
        updated_mean = weight_old * feature_mean + weight_new * new_mean
        
        # Update covariance (simplified approach)
        new_cov = np.cov(new_features, rowvar=False)
        updated_cov = weight_old * feature_cov + weight_new * new_cov
        
        # Update profile
        updated_profile = profile.copy()
        updated_profile['feature_mean'] = updated_mean
        updated_profile['feature_cov'] = updated_cov
        updated_profile['num_samples'] = total_samples
        updated_profile['last_updated'] = datetime.now().isoformat()
        
        return updated_profile
