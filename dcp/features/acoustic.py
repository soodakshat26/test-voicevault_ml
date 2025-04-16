import numpy as np
import librosa
import tensorflow as tf
import torch
import yaml
import os
from scipy import signal
import warnings

class AcousticFeatureExtractor:
    """Base class for acoustic feature extraction"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['features']['acoustic']
        
        self.sample_rate = 16000  # Default sample rate
        
    def extract(self, y, sr):
        """
        Extract acoustic features from audio signal
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
            
        Returns:
        --------
        features : dict
            Dictionary containing feature arrays
        """
        # Base implementation just returns basic features
        features = {
            'rms': self._extract_rms(y)
        }
        
        return features
    
    def _extract_rms(self, y):
        """Extract RMS energy"""
        # Frame the signal
        frames = librosa.util.frame(y, frame_length=512, hop_length=256)
        # Calculate RMS for each frame
        rms = np.sqrt(np.mean(frames**2, axis=0))
        return rms


class MFCCExtractor(AcousticFeatureExtractor):
    """MFCC feature extraction with advanced options"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.n_mfcc = self.config.get('mfcc', {}).get('coefficients', 20)
        self.lifter = self.config.get('mfcc', {}).get('liftering', 22)
        self.n_fft = 2048
        self.hop_length = 512
        self.delta_order = 2
        
    def extract(self, y, sr):
        """Extract MFCC features with deltas and energy"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            lifter=self.lifter
        )
        
        # Compute delta features
        delta_mfccs = librosa.feature.delta(mfccs, order=1)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Compute energy
        energy = np.sum(np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))**2, axis=0)
        energy = energy.reshape(1, -1)
        log_energy = np.log(energy + 1e-10)
        
        # Combine features
        features = {
            'mfcc': mfccs,
            'delta_mfcc': delta_mfccs,
            'delta2_mfcc': delta2_mfccs,
            'log_energy': log_energy
        }
        
        return features


class XVectorExtractor(AcousticFeatureExtractor):
    """X-vector embedding extraction using neural network"""
    
    def __init__(self, model_path=None, config_path="config/processing.yaml"):
        super().__init__(config_path)
        
        # Default model path
        if model_path is None:
            model_path = "models/xvector/xvector_model"
        
        self.embedding_size = self.config.get('x_vectors', {}).get('embedding_size', 512)
        self.pooling = self.config.get('x_vectors', {}).get('pooling', 'temporal_statistics')
        
        # Load or create model
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Load or create x-vector model"""
        if os.path.exists(model_path):
            try:
                # Try TensorFlow first
                return tf.keras.models.load_model(model_path)
            except:
                try:
                    # Try PyTorch
                    return torch.load(model_path)
                except:
                    warnings.warn(f"Could not load model from {model_path}. Creating new model.")
                    return self._create_model()
        else:
            warnings.warn(f"Model path {model_path} does not exist. Creating new model.")
            return self._create_model()
    
    def _create_model(self):
        """Create a simple x-vector model"""
        # This is a simplified model for demonstration purposes
        # A production system would use a more sophisticated architecture
        
        # Input shape: (time, features)
        inputs = tf.keras.layers.Input(shape=(None, 40))
        
        # Frame-level feature extraction
        x = tf.keras.layers.Conv1D(512, 5, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Temporal pooling
        if self.pooling == 'temporal_statistics':
            # Statistics pooling (mean and std)
            # Create a custom layer for statistics pooling
            class StatsPooling(tf.keras.layers.Layer):
                def call(self, inputs):
                    mean = tf.reduce_mean(inputs, axis=1)
                    std = tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=1) - tf.square(mean) + 1e-10)
                    return tf.concat([mean, std], axis=1)
            
            x = StatsPooling()(x)
        else:
            # Simple average pooling
            x = tf.keras.layers.GlobalAveragePooling1D()(x)

        
        # Embedding layers
        x = tf.keras.layers.Dense(self.embedding_size, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Final embedding
        embedding = tf.keras.layers.Dense(self.embedding_size, name='embedding')(x)
        
        # Create model
        model = tf.keras.models.Model(inputs=inputs, outputs=embedding)
        
        return model
    
    def extract(self, y, sr):
        """Extract x-vector embedding from audio signal"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Extract MFCCs as input features
        mfcc_extractor = MFCCExtractor()
        mfcc_features = mfcc_extractor.extract(y, sr)
        
        # Prepare input features (MFCCs + deltas)
        features = np.vstack([
            mfcc_features['mfcc'],
            mfcc_features['delta_mfcc']
        ])
        
        # Transpose to time x features
        features = features.T
        
        # Add batch dimension
        features = np.expand_dims(features, 0)
        
        # Extract embedding
        embedding = self.model.predict(features)
        
        return {'x_vector': embedding[0]}


class PNCCExtractor(AcousticFeatureExtractor):
    """Power Normalized Cepstral Coefficients for noise robustness"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.n_coeffs = 20
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 40
        
    def extract(self, y, sr):
        """Extract PNCC features"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # STFT
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # Convert to power
        S_power = S**2
        
        # Mel filterbank
        mel_basis = librosa.filters.mel(sr=sr, n_fft=self.n_fft, n_mels=self.n_mels)
        mel_spec = np.dot(mel_basis, S_power)
        
        # Power normalization
        # 1. Medium-time power normalization
        window_size = 5
        power_padded = np.pad(mel_spec, ((0, 0), (window_size//2, window_size//2)), mode='edge')
        power_smoothed = np.zeros_like(mel_spec)
        
        for i in range(mel_spec.shape[1]):
            power_smoothed[:, i] = np.mean(power_padded[:, i:i+window_size], axis=1)
        
        # 2. Asymmetric noise suppression
        lambda_factor = 0.999
        power_floor = np.zeros_like(mel_spec)
        
        for i in range(1, mel_spec.shape[1]):
            power_floor[:, i] = np.maximum(
                power_smoothed[:, i],
                lambda_factor * power_floor[:, i-1]
            )
        
        # 3. Subtract noise floor and apply half-wave rectification
        power_subtracted = np.maximum(power_smoothed - power_floor, 0)
        
        # 4. Power function nonlinearity
        power_exp = 1/15
        power_nonlin = power_subtracted**power_exp
        
        # 5. DCT
        pncc = librosa.feature.inverse.mel_to_audio(power_nonlin, sr=sr, n_fft=self.n_fft)
        pncc = np.log(pncc + 1e-10)
        
        # Extract first n coefficients
        pncc = pncc[:self.n_coeffs, :]
        
        return {'pncc': pncc}


class AcousticFeatureFactory:
    """Factory for creating acoustic feature extractors"""
    
    @staticmethod
    def create_extractor(method, config_path="config/processing.yaml"):
        """Create feature extractor based on method name"""
        if method == "mfcc":
            return MFCCExtractor(config_path)
        elif method == "xvector":
            return XVectorExtractor(config_path=config_path)
        elif method == "pncc":
            return PNCCExtractor(config_path)
        else:
            print(f"Unknown feature extractor: {method}. Using default MFCC extractor.")
            return MFCCExtractor(config_path)
