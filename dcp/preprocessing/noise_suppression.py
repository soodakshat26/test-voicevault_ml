import numpy as np
import soundfile as sf
import librosa
import scipy.signal as signal
import tensorflow as tf
import os
import yaml
from scipy import linalg

class SpectralSubtractor:
    """Advanced spectral subtraction noise reduction"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['preprocessing']['noise_suppression']
        
        self.n_fft = 2048
        self.hop_length = self.n_fft // 4
        self.bands = self.config.get('bands', 64)
        self.use_psychoacoustic = self.config.get('psychoacoustic_masking', True)
        self.phase_preserve = self.config.get('phase_preservation', True)
        self.alpha = 1.5  # Over-subtraction factor
        self.beta = 0.01  # Spectral floor
        
    def estimate_noise(self, y, sr, n_noise_frames=10):
        """Estimate noise spectrum from first n frames"""
        # Calculate spectrogram
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Estimate noise from first n frames
        if n_noise_frames >= S.shape[1]:
            n_noise_frames = max(1, S.shape[1] // 4)
            
        noise_spec = np.mean(np.abs(S[:, :n_noise_frames])**2, axis=1)
        
        return noise_spec
    
    def suppress_noise(self, y, sr, noise_spec=None):
        """Apply spectral subtraction"""
        # Calculate spectrogram
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # If noise spectrum not provided, estimate from signal
        if noise_spec is None:
            noise_spec = self.estimate_noise(y, sr)
            
        # Reshape noise spectrum for broadcasting
        noise_spec = noise_spec.reshape(-1, 1)
        
        # Calculate power spectrogram
        S_power = np.abs(S)**2
        
        # Apply spectral subtraction with oversubtraction
        S_clean_power = S_power - self.alpha * noise_spec
        
        # Apply spectral floor
        S_clean_power = np.maximum(S_clean_power, self.beta * S_power)
        
        # Convert back to magnitude
        S_clean_mag = np.sqrt(S_clean_power)
        
        # Apply psychoacoustic masking if enabled
        if self.use_psychoacoustic:
            S_clean_mag = self._apply_psychoacoustic_masking(S_clean_mag, S, sr)
        
        # Reconstruct signal
        if self.phase_preserve:
            # Preserve original phase
            S_clean = S_clean_mag * np.exp(1j * np.angle(S))
        else:
            # Use magnitude-only reconstruction
            S_clean = S_clean_mag
        
        # Inverse STFT
        y_clean = librosa.istft(S_clean, hop_length=self.hop_length)
        
        # Ensure same length as input
        if len(y_clean) > len(y):
            y_clean = y_clean[:len(y)]
        elif len(y_clean) < len(y):
            y_clean = np.pad(y_clean, (0, len(y) - len(y_clean)))
            
        return y_clean
    
    def _apply_psychoacoustic_masking(self, S_mag, S_original, sr):
        """Apply psychoacoustic masking to reduce musical noise"""
        # Convert to bark scale for psychoacoustic model
        bark_scale = 13 * np.arctan(0.00076 * sr/2) + 3.5 * np.arctan((sr/2 / 7500)**2)
        
        # Simple spreading function in Bark domain (more sophisticated models exist)
        def bark_spreading(bark_width):
            spread = np.zeros(self.n_fft//2 + 1)
            
            # Create a triangular spreading function
            for i in range(len(spread)):
                freq = i * sr / self.n_fft
                bark = 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500)**2)
                
                # Apply spreading to neighboring frequencies
                for j in range(len(spread)):
                    freq_j = j * sr / self.n_fft
                    bark_j = 13 * np.arctan(0.00076 * freq_j) + 3.5 * np.arctan((freq_j / 7500)**2)
                    
                    # Triangular spreading function
                    spread_val = max(0, 1 - abs(bark - bark_j) / bark_width)
                    spread[i] = max(spread[i], spread_val)
            
            return spread
        
        # Create spreading function (simplified)
        spread = bark_spreading(0.5)  # 0.5 Bark spreading
        
        # Apply masking threshold
        masking_threshold = np.zeros_like(S_mag)
        
        for i in range(S_mag.shape[1]):
            # Convolve with spreading function
            col_spread = np.convolve(S_mag[:, i], spread, mode='same')
            
            # Apply masking
            masking_threshold[:, i] = 0.1 * col_spread  # 10% masking threshold
            
        # Apply mask: if below masking threshold, keep original value to reduce musical noise
        mask = S_mag < masking_threshold
        S_mag[mask] = np.abs(S_original)[mask]
        
        return S_mag


class DeepLearningDenoiser:
    """Neural network-based speech enhancement"""
    
    def __init__(self, model_path=None, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['preprocessing']['noise_suppression']
        
        # Default model path
        if model_path is None:
            model_path = "models/denoising/denoiser_model"
        
        # Load or create model
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self._create_model()
            print(f"Warning: No model found at {model_path}. Created new model.")
        
        self.n_fft = 512
        self.hop_length = 128
        self.sr = 16000  # Target sample rate for processing
        
    def _create_model(self):
        """Create a simple denoising model"""
        # This is a simplified model for demonstration purposes
        # A production system would use a more sophisticated architecture
        
        # Simple UNet-style model for spectral denoising
        inputs = tf.keras.layers.Input(shape=(257, None, 1))  # Magnitude spectrogram chunks
        
        # Encoder
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        
        # Decoder
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        
        # Output layer - mask
        mask = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        # Apply mask to input
        outputs = tf.keras.layers.Multiply()([inputs, mask])
        
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def process_file(self, input_path, output_path):
        """Process an audio file with the denoising model"""
        # Load and resample audio
        y, sr = librosa.load(input_path, sr=self.sr)
        
        # Process the audio
        y_clean = self.suppress_noise(y)
        
        # Save the denoised audio
        sf.write(output_path, y_clean, self.sr)
        
        print(f"Processed {input_path} -> {output_path}")
        return output_path
    
    def suppress_noise(self, y):
        """Apply neural network denoising to the audio signal"""
        # Compute spectrogram
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Get magnitude and phase
        S_mag = np.abs(S)
        S_phase = np.angle(S)
        
        # Reshape for model input
        S_mag = S_mag.reshape((1, S_mag.shape[0], S_mag.shape[1], 1))
        
        # Process through model
        S_mag_clean = self.model.predict(S_mag)[0, :, :, 0]
        
        # Reconstruct complex spectrogram
        S_clean = S_mag_clean * np.exp(1j * S_phase)
        
        # Invert to time domain
        y_clean = librosa.istft(S_clean, hop_length=self.hop_length)
        
        # Ensure same length as input
        if len(y_clean) > len(y):
            y_clean = y_clean[:len(y)]
        elif len(y_clean) < len(y):
            y_clean = np.pad(y_clean, (0, len(y) - len(y_clean)))
            
        return y_clean


class WaveletDenoiser:
    """Wavelet-based denoising"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['preprocessing']['noise_suppression']
        
        self.wavelet = 'db4'  # Daubechies wavelet
        self.level = 5  # Decomposition level
        self.threshold_type = 'soft'  # 'soft' or 'hard'
    
    def suppress_noise(self, y, sr):
        """Apply wavelet denoising"""
        # Import PyWavelets (pywt) lazily as it might not be installed
        try:
            import pywt
        except ImportError:
            print("PyWavelets not installed. Install with: pip install PyWavelets")
            return y
        
        # Decompose signal using wavelet transform
        coeffs = pywt.wavedec(y, self.wavelet, level=self.level)
        
        # Estimate noise level from first detail coefficient
        sigma = np.median(np.abs(coeffs[1])) / 0.6745
        
        # Calculate threshold
        threshold = sigma * np.sqrt(2 * np.log(len(y)))
        
        # Apply thresholding
        for i in range(1, len(coeffs)):
            if self.threshold_type == 'hard':
                coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='hard')
            else:
                coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
        
        # Reconstruct signal
        y_clean = pywt.waverec(coeffs, self.wavelet)
        
        # Ensure same length as input
        if len(y_clean) > len(y):
            y_clean = y_clean[:len(y)]
        elif len(y_clean) < len(y):
            y_clean = np.pad(y_clean, (0, len(y) - len(y_clean)))
            
        return y_clean


class NoiseSuppressionFactory:
    """Factory class to create appropriate noise suppression objects"""
    
    @staticmethod
    def create_suppressor(method, config_path="config/processing.yaml"):
        """Create a noise suppressor based on method name"""
        if method == "spectral_subtraction":
            return SpectralSubtractor(config_path)
        elif method == "deep_learning":
            return DeepLearningDenoiser(config_path=config_path)
        elif method == "wavelet":
            return WaveletDenoiser(config_path)
        else:
            print(f"Unknown method: {method}. Using spectral subtraction.")
            return SpectralSubtractor(config_path)
