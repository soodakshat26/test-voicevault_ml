import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
import os
import yaml
from scipy import signal
import matplotlib.pyplot as plt

class EnergyVAD:
    """Simple energy-based Voice Activity Detection"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['preprocessing']['vad']
        
        self.frame_size_ms = self.config.get('frame_size', 20)
        self.overlap_ms = self.config.get('overlap', 10)
        self.threshold = self.config.get('decision_threshold', 0.5)
        self.context_frames = self.config.get('context_frames', 3)
    
    def detect_speech(self, y, sr):
        """Detect speech segments in audio signal using energy thresholding"""
        # Calculate frame length in samples
        frame_length = int(sr * self.frame_size_ms / 1000)
        hop_length = int(sr * (self.frame_size_ms - self.overlap_ms) / 1000)
        
        # Compute short-time energy
        energy = np.array([
            np.sum(y[i:i+frame_length]**2) 
            for i in range(0, len(y)-frame_length, hop_length)
        ])
        
        # Normalize energy
        energy_norm = energy / np.max(energy)
        
        # Calculate adaptive threshold
        # Use Otsu's method for adaptive thresholding
        hist, bin_edges = np.histogram(energy_norm, bins=100, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find threshold that maximizes between-class variance
        best_threshold = 0
        max_variance = 0
        
        weight1 = hist.cumsum() / hist.sum()
        weight2 = 1 - weight1
        
        mean1 = np.cumsum(hist * bin_centers) / hist.cumsum()
        mean2 = (np.sum(hist * bin_centers) - np.cumsum(hist * bin_centers)) / (hist.sum() - hist.cumsum())
        
        # Replace NaNs with 0
        mean1 = np.nan_to_num(mean1)
        mean2 = np.nan_to_num(mean2)
        
        # Calculate between-class variance
        between_var = weight1 * weight2 * (mean1 - mean2)**2
        
        # Find the threshold that maximizes between-class variance
        idx = np.argmax(between_var)
        best_threshold = bin_centers[idx]
        
        # Adjust threshold using config parameter (lower = more sensitive)
        threshold = best_threshold * self.threshold
        
        # Apply threshold
        speech_frames = energy_norm > threshold
        
        # Apply context-based smoothing
        speech_frames = self._smooth_decisions(speech_frames)
        
        # Convert frame-level decisions to sample-level
        speech_mask = np.zeros_like(y, dtype=bool)
        
        for i, is_speech in enumerate(speech_frames):
            start = i * hop_length
            end = min(start + frame_length, len(y))
            if is_speech:
                speech_mask[start:end] = True
        
        return speech_mask
    
    def _smooth_decisions(self, speech_frames):
        """Apply smoothing to speech frame decisions to reduce false transitions"""
        smoothed = np.copy(speech_frames)
        context = self.context_frames
        
        # Forward pass
        for i in range(len(smoothed) - context):
            if np.count_nonzero(smoothed[i:i+context]) >= context // 2:
                smoothed[i] = True
        
        # Backward pass
        for i in range(len(smoothed) - 1, context, -1):
            if np.count_nonzero(smoothed[i-context:i]) >= context // 2:
                smoothed[i] = True
        
        return smoothed
    
    def extract_speech_segments(self, y, sr, min_duration_ms=100):
        """Extract continuous speech segments"""
        speech_mask = self.detect_speech(y, sr)
        
        # Find contiguous speech segments
        changes = np.diff(np.concatenate(([0], speech_mask.astype(int), [0])))
        rising_edges = np.where(changes == 1)[0]
        falling_edges = np.where(changes == -1)[0]
        
        # Minimum segment duration in samples
        min_samples = int(sr * min_duration_ms / 1000)
        
        segments = []
        for start, end in zip(rising_edges, falling_edges):
            if end - start >= min_samples:
                segments.append((start, end))
        
        # Extract audio segments
        speech_segments = []
        for start, end in segments:
            speech_segments.append(y[start:end])
        
        return speech_segments, segments


class NeuralVAD:
    """RNN-based Voice Activity Detection"""
    
    def __init__(self, model_path=None, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['preprocessing']['vad']
        
        # Default model path
        if model_path is None:
            model_path = "models/vad/neural_vad_model"
        
        # Parameters
        self.frame_size_ms = self.config.get('frame_size', 20)
        self.hop_length_ms = self.frame_size_ms - self.config.get('overlap', 10)
        self.context_frames = self.config.get('context_frames', 40)
        self.threshold = self.config.get('decision_threshold', 0.5)
        self.sr = 16000  # Fixed sample rate for model
        
        # Load or create model
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self._create_model()
            print(f"Warning: No model found at {model_path}. Created new model.")
    
    def _create_model(self):
        """Create a simple RNN-based VAD model"""
        # This is a simplified model for demonstration purposes
        # A production system would use a more sophisticated architecture
        
        # Input features shape
        feature_dim = 60  # MFCC features
        context_frames = self.context_frames
        
        # Model architecture
        inputs = tf.keras.layers.Input(shape=(context_frames, feature_dim))
        
        # Bidirectional RNN layers
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        )(inputs)
        
        # Self-attention mechanism
        attention = tf.keras.layers.Dense(1, activation='tanh')(x)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(128)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # Apply attention to the RNN outputs
        x = tf.keras.layers.Multiply()([x, attention])
        x = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def extract_features(self, y, sr):
        """Extract MFCC features for VAD"""
        # Resample audio if needed
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
            sr = self.sr
        
        # Convert ms to samples
        frame_length = int(self.frame_size_ms * sr / 1000)
        hop_length = int(self.hop_length_ms * sr / 1000)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=20,
            n_fft=frame_length,
            hop_length=hop_length
        )
        
        # Add delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine features
        features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        
        # Transpose to time x features
        features = features.T
        
        return features
    
    def detect_speech(self, y, sr):
        """Detect speech segments using neural network"""
        # Extract features
        features = self.extract_features(y, sr)
        
        # Create context windows
        context_frames = self.context_frames
        windowed_features = []

        # Ensure all windows will have the same dimensions
        feature_dim = features.shape[1] if features.shape[0] > 0 else 0

        for i in range(len(features)):
            # Create context window, ensuring total size doesn't exceed context_frames
            half_ctx = context_frames // 2
            
            # Calculate initial window bounds
            start = max(0, i - half_ctx)
            end = min(len(features), i + half_ctx + (context_frames % 2))  # Add 1 for odd context_frames
            
            # If window is too large, reduce it
            if end - start > context_frames:
                # Prioritize future frames if needed
                if i < half_ctx:  # Near the beginning
                    end = start + context_frames
                else:  # Otherwise, adjust start
                    start = end - context_frames
            
            # Extract window
            window = features[start:end]
            
            # Create a fixed-size array filled with zeros
            fixed_window = np.zeros((context_frames, feature_dim))
            
            # Copy actual data into the fixed window
            fixed_window[:len(window)] = window
            
            windowed_features.append(fixed_window)


        # Convert to numpy array - now all elements have the same shape
        windowed_features = np.array(windowed_features)

        
        # Make predictions
        predictions = self.model.predict(windowed_features)
        
        # Apply threshold
        speech_frames = predictions[:, 0] > self.threshold
        
        # Convert frame decisions to sample-level mask
        hop_length = int(self.hop_length_ms * sr / 1000)
        speech_mask = np.zeros_like(y, dtype=bool)
        
        for i, is_speech in enumerate(speech_frames):
            start = i * hop_length
            end = min(start + hop_length, len(y))
            if is_speech:
                speech_mask[start:end] = True
        
        return speech_mask
    
    def extract_speech_segments(self, y, sr, min_duration_ms=100):
        """Extract continuous speech segments"""
        speech_mask = self.detect_speech(y, sr)
        
        # Find contiguous speech segments
        changes = np.diff(np.concatenate(([0], speech_mask.astype(int), [0])))
        rising_edges = np.where(changes == 1)[0]
        falling_edges = np.where(changes == -1)[0]
        
        # Minimum segment duration in samples
        min_samples = int(sr * min_duration_ms / 1000)
        
        segments = []
        for start, end in zip(rising_edges, falling_edges):
            if end - start >= min_samples:
                segments.append((start, end))
        
        # Extract audio segments
        speech_segments = []
        for start, end in segments:
            speech_segments.append(y[start:end])
        
        return speech_segments, segments


class VADFactory:
    """Factory class to create appropriate VAD objects"""
    
    @staticmethod
    def create_vad(method, config_path="config/processing.yaml"):
        """Create a VAD processor based on method name"""
        if method == "energy":
            return EnergyVAD(config_path)
        elif method == "rnn_attention":
            return NeuralVAD(config_path=config_path)
        else:
            print(f"Unknown VAD method: {method}. Using energy-based VAD.")
            return EnergyVAD(config_path)
