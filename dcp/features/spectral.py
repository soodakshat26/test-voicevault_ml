import numpy as np
import librosa
import yaml
from scipy import signal
from scipy import linalg
import warnings

class SpectralFeatureExtractor:
    """Base class for spectral feature extraction"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['features']['spectral']
        
        self.sample_rate = 16000  # Default sample rate
        self.n_fft = 2048
        self.hop_length = 512
        
    def extract(self, y, sr):
        """Extract basic spectral features"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Calculate spectrogram
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # Basic spectral features
        spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        
        features = {
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'spectral_contrast': spectral_contrast
        }
        
        return features


class ReassignedSpectrogramExtractor(SpectralFeatureExtractor):
    """Reassigned spectrogram for improved energy localization"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.use_reassigned = self.config.get('reassigned_spectrogram', True)
        
    def extract(self, y, sr):
        """Extract reassigned spectrogram"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        if not self.use_reassigned:
            # Fall back to standard spectrogram
            S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
            return {'spectrogram': S}
        
        # Time-frequency reassignment
        # Calculate STFT
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Calculate time and frequency derivatives
        time_derivative = np.zeros_like(S)
        freq_derivative = np.zeros_like(S)
        
        # Time derivative using finite differences
        time_derivative[:, 1:-1] = (S[:, 2:] - S[:, :-2]) / 2
        
        # Frequency derivative
        freq_derivative[1:-1, :] = (S[2:, :] - S[:-2, :]) / 2
        
        # Calculate reassignment operators
        time_shift = np.zeros_like(S, dtype=float)
        freq_shift = np.zeros_like(S, dtype=float)
        
        # Avoid division by zero
        mask = np.abs(S) > 1e-10
        
        time_shift[mask] = -np.real(time_derivative[mask] * np.conj(S[mask]) / (np.abs(S[mask])**2))
        freq_shift[mask] = np.real(freq_derivative[mask] * np.conj(S[mask]) / (np.abs(S[mask])**2))
        
        # Limit reassignment to prevent extreme shifts
        time_shift = np.clip(time_shift, -2, 2)
        freq_shift = np.clip(freq_shift, -2, 2)
        
        # Create reassigned spectrogram
        S_reassigned = np.zeros_like(S, dtype=float)
        
        # Apply reassignment
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if mask[i, j]:
                    i_new = int(i + freq_shift[i, j])
                    j_new = int(j + time_shift[i, j])
                    
                    if 0 <= i_new < S.shape[0] and 0 <= j_new < S.shape[1]:
                        S_reassigned[i_new, j_new] += np.abs(S[i, j])
        
        return {'reassigned_spectrogram': S_reassigned}


class MultiTaperSpectralEstimator(SpectralFeatureExtractor):
    """Multi-taper spectrum estimation for reduced variance"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.use_multi_taper = self.config.get('multi_taper', True)
        self.taper_count = self.config.get('taper_count', 8)
        
    def extract(self, y, sr):
        """Extract multi-taper spectrum"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        if not self.use_multi_taper:
            # Fall back to periodogram
            f, Pxx = signal.periodogram(y, fs=sr, nfft=self.n_fft)
            return {'periodogram': Pxx}
        
        # Multi-taper spectrum estimation
        # Generate DPSS tapers
        tapers, eigenvalues = signal.windows.dpss(len(y), 4, Kmax=self.taper_count, return_ratios=True)
        
        # Initialize spectrum
        Pxx_mt = np.zeros(self.n_fft // 2 + 1)
        
        # Apply each taper and accumulate spectra
        for k in range(self.taper_count):
            # Apply taper
            y_tapered = y * tapers[k]
            
            # Calculate periodogram
            f, Pxx = signal.periodogram(y_tapered, fs=sr, nfft=self.n_fft)
            
            # Add to multi-taper estimate, weighted by eigenvalue
            Pxx_mt += eigenvalues[k] * Pxx
        
        # Normalize
        Pxx_mt /= np.sum(eigenvalues)
        
        return {
            'multi_taper_spectrum': Pxx_mt,
            'frequencies': f
        }


class GroupDelayProcessor(SpectralFeatureExtractor):
    """Modified group delay functions for phase-enhanced representation"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.use_group_delay = self.config.get('group_delay', True)
        self.gamma = 0.4  # Scaling parameter
        self.alpha = 0.2  # Compression parameter
        
    def extract(self, y, sr):
        """Extract modified group delay features"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        if not self.use_group_delay:
            return {}
        
        # Calculate signal and differential FFTs
        X = np.fft.fft(y * np.hamming(len(y)), n=self.n_fft)
        
        # Calculate differential signal
        y_diff = np.diff(y, prepend=0)
        Y = np.fft.fft(y_diff * np.hamming(len(y_diff)), n=self.n_fft)
        
        # Calculate cepstrally smoothed spectrum
        X_smooth = self._cepstral_smooth(X)
        
        # Calculate group delay
        tau = np.real(Y * np.conj(X)) / (np.abs(X_smooth)**2)
        
        # Modified group delay
        tau_mod = np.sign(tau) * (np.abs(tau)**self.alpha)
        
        # Scale and threshold
        tau_mod = tau_mod / np.max(np.abs(tau_mod)) * self.gamma
        tau_mod = np.clip(tau_mod, -1, 1)
        
        # Keep only the first half (positive frequencies)
        tau_mod = tau_mod[:self.n_fft//2 + 1]
        
        return {'group_delay': tau_mod}
    
    def _cepstral_smooth(self, X):
        """Apply cepstral smoothing to spectrum"""
        log_X = np.log(np.abs(X) + 1e-10)
        ceps = np.fft.ifft(log_X)
        
        # Lifter cepstrum (keep low quefrency)
        lifter = np.ones_like(ceps)
        lifter[20:len(lifter)-20] = 0
        
        # Transform back
        smooth_log_X = np.fft.fft(ceps * lifter)
        X_smooth = np.exp(smooth_log_X)
        
        return X_smooth


class SpectralFeatureFactory:
    """Factory for creating spectral feature extractors"""
    
    @staticmethod
    def create_extractor(method, config_path="config/processing.yaml"):
        """Create feature extractor based on method name"""
        if method == "reassigned_spectrogram":
            return ReassignedSpectrogramExtractor(config_path)
        elif method == "multi_taper":
            return MultiTaperSpectralEstimator(config_path)
        elif method == "group_delay":
            return GroupDelayProcessor(config_path)
        else:
            print(f"Unknown feature extractor: {method}. Using basic spectral extractor.")
            return SpectralFeatureExtractor(config_path)
