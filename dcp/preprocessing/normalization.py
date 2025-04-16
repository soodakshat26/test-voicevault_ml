import numpy as np
import librosa
import soundfile as sf
import yaml
from scipy import signal
from scipy import linalg

class ChannelNormalizer:
    """Class for audio channel equalization and normalization"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['preprocessing']['normalization']
        
        self.apply_cmvn = self.config.get('cepstral_mean_variance', True)
        self.apply_channel_eq = self.config.get('channel_equalization', True)
        self.apply_phase_norm = self.config.get('phase_normalization', True)
        self.apply_mod_spectrum = self.config.get('modulation_spectrum', True)
        
    def normalize(self, y, sr):
        """Apply comprehensive normalization to audio signal"""
        if self.apply_channel_eq:
            y = self.equalize_channel(y, sr)
        
        if self.apply_phase_norm:
            y = self.normalize_phase(y, sr)
        
        if self.apply_mod_spectrum:
            y = self.normalize_modulation_spectrum(y, sr)
        
        return y
    
    def equalize_channel(self, y, sr, target_response=None):
        """
        Apply channel equalization through inverse filtering
        
        Parameters:
        -----------
        y : ndarray
            Input audio signal
        sr : int
            Sample rate
        target_response : ndarray, optional
            Target frequency response (flat if None)
            
        Returns:
        --------
        y_eq : ndarray
            Equalized audio signal
        """
        # Calculate current spectrum
        n_fft = 2048
        S = librosa.stft(y, n_fft=n_fft)
        S_mag = np.abs(S)
        S_phase = np.angle(S)
        
        # Calculate average spectrum
        avg_spectrum = np.mean(S_mag, axis=1)
        
        # If no target response provided, use a flat spectrum
        if target_response is None:
            # Smooth the spectrum to avoid extreme correction
            window_length = 11
            avg_spectrum_smooth = np.convolve(
                avg_spectrum, 
                np.ones(window_length)/window_length, 
                mode='same'
            )
            target_response = np.ones_like(avg_spectrum_smooth) * np.mean(avg_spectrum_smooth)
        
        # Create equalization filter
        eq_filter = target_response / (avg_spectrum + 1e-10)
        
        # Smooth the filter
        window_length = 11
        eq_filter = np.convolve(eq_filter, np.ones(window_length)/window_length, mode='same')
        
        # Limit extreme values
        eq_filter = np.clip(eq_filter, 0.1, 10.0)
        
        # Apply filter to spectrum
        eq_filter = eq_filter.reshape(-1, 1)  # Reshape for broadcasting
        S_eq_mag = S_mag * eq_filter
        
        # Reconstruct complex spectrum
        S_eq = S_eq_mag * np.exp(1j * S_phase)
        
        # Convert back to time domain
        y_eq = librosa.istft(S_eq)
        
        # Ensure same length as input
        if len(y_eq) > len(y):
            y_eq = y_eq[:len(y)]
        elif len(y_eq) < len(y):
            y_eq = np.pad(y_eq, (0, len(y) - len(y_eq)))
        
        return y_eq
    
    def normalize_phase(self, y, sr):
        """
        Normalize phase using all-pass filtering
        
        Parameters:
        -----------
        y : ndarray
            Input audio signal
        sr : int
            Sample rate
            
        Returns:
        --------
        y_norm : ndarray
            Phase-normalized audio signal
        """
        # Calculate group delay
        n_fft = 2048
        S = librosa.stft(y, n_fft=n_fft)
        S_mag = np.abs(S)
        S_phase = np.angle(S)
        
        # Calculate unwrapped phase
        unwrapped_phase = np.unwrap(S_phase, axis=0)
        
        # Estimate group delay (negative derivative of phase)
        group_delay = -np.diff(unwrapped_phase, axis=0, prepend=unwrapped_phase[0:1])
        
        # Calculate mean group delay
        mean_group_delay = np.mean(group_delay, axis=1, keepdims=True)
        
        # Calculate correction phase
        correction_phase = -np.cumsum(mean_group_delay, axis=0)
        
        # Apply phase correction
        S_corrected = S_mag * np.exp(1j * (S_phase - correction_phase))
        
        # Convert back to time domain
        y_norm = librosa.istft(S_corrected)
        
        # Ensure same length as input
        if len(y_norm) > len(y):
            y_norm = y_norm[:len(y)]
        elif len(y_norm) < len(y):
            y_norm = np.pad(y_norm, (0, len(y) - len(y_norm)))
        
        return y_norm
    
    def normalize_modulation_spectrum(self, y, sr):
        """
        Normalize modulation spectrum for speaking rate invariance
        
        Parameters:
        -----------
        y : ndarray
            Input audio signal
        sr : int
            Sample rate
            
        Returns:
        --------
        y_norm : ndarray
            Modulation-normalized audio signal
        """
        # Check if the input is too short for processing
        if len(y) < 512:
            return y  # Return original signal if too short
        
        try:
            # Calculate envelope
            analytic_signal = signal.hilbert(y)
            envelope = np.abs(analytic_signal)
            
            # Downsample envelope
            target_sr = 200  # Hz
            downsampling_factor = max(1, int(sr / target_sr))
            envelope_downsampled = envelope[::downsampling_factor]
            
            # Ensure we have enough samples for STFT
            if len(envelope_downsampled) < 512:
                # Pad if needed
                envelope_downsampled = np.pad(envelope_downsampled, 
                                            (0, 512 - len(envelope_downsampled)))
            
            # Calculate modulation spectrum
            n_fft = min(512, len(envelope_downsampled) - 1)
            hop_length = max(1, n_fft // 4)
            
            mod_S = librosa.stft(envelope_downsampled, n_fft=n_fft, hop_length=hop_length)
            
            # Check if we got valid results
            if mod_S.size == 0:
                return y
            
            mod_S_mag = np.abs(mod_S)
            mod_S_phase = np.angle(mod_S)
            
            # Calculate average modulation spectrum
            avg_mod_spectrum = np.mean(mod_S_mag, axis=1)
            
            # Create target modulation spectrum (emphasize speech modulations 4-8 Hz)
            target_mod = np.ones_like(avg_mod_spectrum)
            
            # Boost speech-relevant modulation frequencies
            freqs = np.fft.rfftfreq(n_fft, 1/target_sr)
            speech_mask = np.logical_and(freqs >= 4, freqs <= 8)
            if np.any(speech_mask):
                target_mod[speech_mask] = 2.0
            
            # Calculate modulation filter
            mod_filter = target_mod / (avg_mod_spectrum + 1e-10)
            
            # Smooth filter
            window_length = min(5, len(mod_filter))
            if window_length > 1:
                mod_filter = np.convolve(mod_filter, np.ones(window_length)/window_length, mode='same')
            
            # Limit extreme values
            mod_filter = np.clip(mod_filter, 0.2, 5.0)
            
            # Apply filter to modulation spectrum
            mod_filter = mod_filter.reshape(-1, 1)  # Reshape for broadcasting
            mod_S_eq_mag = mod_S_mag * mod_filter
            
            # Reconstruct complex modulation spectrum
            mod_S_eq = mod_S_eq_mag * np.exp(1j * mod_S_phase)
            
            # Convert back to envelope domain
            envelope_normalized = librosa.istft(mod_S_eq, hop_length=hop_length)
            
            # Check if we have a valid envelope
            if len(envelope_normalized) == 0:
                return y
            
            # Resample envelope to original length
            envelope_resampled = np.interp(
                np.linspace(0, 1, len(y)),
                np.linspace(0, 1, len(envelope_normalized)),
                envelope_normalized
            )
            
            # Apply normalized envelope
            y_norm = y * (envelope_resampled / (envelope + 1e-10))
            
            return y_norm
        except Exception as e:
            print(f"Error in modulation spectrum normalization: {e}")
            return y  # Return original signal if any error occurs



class CepstralMeanVarianceNormalizer:
    """Cepstral mean and variance normalization"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['preprocessing']['normalization']
        
        self.n_mfcc = 20
        self.window_size = 512
        self.hop_length = 256
        self.adaptive = True
        self.window_duration = 3  # seconds for sliding window
    
    def normalize_features(self, features, window_frames=None):
        """
        Apply CMVN to feature matrix
        
        Parameters:
        -----------
        features : ndarray
            Feature matrix (time x features)
        window_frames : int, optional
            Number of frames for sliding window (all features if None)
            
        Returns:
        --------
        normalized_features : ndarray
            Normalized feature matrix
        """
        if window_frames is None or not self.adaptive:
            # Global normalization
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            normalized_features = (features - mean) / (std + 1e-10)
        else:
            # Sliding window normalization
            normalized_features = np.zeros_like(features)
            
            for i in range(features.shape[0]):
                # Define window
                start = max(0, i - window_frames // 2)
                end = min(features.shape[0], i + window_frames // 2)
                window = features[start:end]
                
                # Calculate window statistics
                window_mean = np.mean(window, axis=0)
                window_std = np.std(window, axis=0)
                
                # Normalize
                normalized_features[i] = (features[i] - window_mean) / (window_std + 1e-10)
        
        return normalized_features
    
    def extract_mfcc(self, y, sr):
        """Extract MFCC features from audio"""
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=self.n_mfcc,
            n_fft=self.window_size,
            hop_length=self.hop_length
        )
        
        # Transpose to time x features
        mfccs = mfccs.T
        
        return mfccs
    
    def apply_cmvn(self, y, sr):
        """
        Apply CMVN to audio signal
        
        This extracts MFCCs, normalizes them, and reconstructs the signal.
        Note: This is a lossy transformation as perfect reconstruction
        from MFCCs is not possible.
        
        Parameters:
        -----------
        y : ndarray
            Input audio signal
        sr : int
            Sample rate
            
        Returns:
        --------
        y_norm : ndarray
            CMVN normalized audio signal (approximation)
        """
        # Extract MFCCs
        mfccs = self.extract_mfcc(y, sr)
        
        # Calculate window frames if using adaptive normalization
        window_frames = None
        if self.adaptive:
            window_frames = int(self.window_duration * sr / self.hop_length)
        
        # Normalize features
        normalized_mfccs = self.normalize_features(mfccs, window_frames)
        
        # Approximate reconstruction from MFCCs
        # Note: This is an approximation only
        return self._approximate_reconstruction(normalized_mfccs, y, sr)
    
    def _approximate_reconstruction(self, normalized_mfccs, original_y, sr):
        """
        Approximately reconstruct audio from normalized MFCCs
        
        This is an approximation as perfect reconstruction from MFCCs is not possible.
        We use the original audio's phase information.
        
        Parameters:
        -----------
        normalized_mfccs : ndarray
            Normalized MFCC features
        original_y : ndarray
            Original audio signal for phase reference
        sr : int
            Sample rate
            
        Returns:
        --------
        y_reconstructed : ndarray
            Reconstructed audio signal
        """
        # Convert MFCCs back to mel spectrogram
        mel_basis = librosa.filters.mel(
            sr=sr, 
            n_fft=self.window_size, 
            n_mels=128
        )
        
        # Original spectrogram for phase information
        S_original = librosa.stft(original_y, n_fft=self.window_size, hop_length=self.hop_length)
        phase = np.angle(S_original)
        
        # Approximate inverse of MFCC
        mel_spectrogram = librosa.feature.inverse.mfcc_to_mel(
            normalized_mfccs.T, 
            n_mels=128, 
            dct_type=2, 
            norm='ortho'
        )
        
        # Convert to power spectrogram
        S_power = librosa.feature.inverse.mel_to_stft(
            mel_spectrogram, 
            sr=sr, 
            n_fft=self.window_size, 
            power=2.0
        )
        
        # Apply original phase
        S_complex = np.sqrt(S_power) * np.exp(1j * phase[:S_power.shape[0], :S_power.shape[1]])
        
        # Inverse STFT
        y_reconstructed = librosa.istft(S_complex, hop_length=self.hop_length)
        
        # Ensure same length as input
        if len(y_reconstructed) > len(original_y):
            y_reconstructed = y_reconstructed[:len(original_y)]
        elif len(y_reconstructed) < len(original_y):
            y_reconstructed = np.pad(y_reconstructed, (0, len(original_y) - len(y_reconstructed)))
        
        return y_reconstructed


class LinearPredictiveInverseFiltering:
    """Linear Predictive Coding (LPC) for spectral flattening"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['preprocessing']['normalization']
        
        self.lpc_order = 16
        self.frame_size_ms = 25
        self.hop_size_ms = 10
    
    def apply_lpc_inverse(self, y, sr):
        """
        Apply LPC inverse filtering for spectral flattening
        
        Parameters:
        -----------
        y : ndarray
            Input audio signal
        sr : int
            Sample rate
            
        Returns:
        --------
        y_flat : ndarray
            Spectrally flattened audio signal
        """
        # Calculate frame size in samples
        frame_size = int(self.frame_size_ms * sr / 1000)
        hop_size = int(self.hop_size_ms * sr / 1000)
        
        # Frame the signal
        frames = librosa.util.frame(y, frame_length=frame_size, hop_length=hop_size)
        
        # Apply window
        window = np.hamming(frame_size)
        frames = frames.T * window
        
        # Initialize output
        residual = np.zeros_like(y)
        
        # Process each frame
        for i, frame in enumerate(frames):
            # Calculate LPC coefficients
            lpc_coeffs = self._compute_lpc(frame, self.lpc_order)
            
            # Apply inverse filtering to get residual
            res_frame = self._apply_inverse_filter(frame, lpc_coeffs)
            
            # Overlap-add
            start = i * hop_size
            end = start + frame_size
            if end <= len(residual):
                residual[start:end] += res_frame * window
        
        # Normalize energy
        energy_y = np.sum(y**2)
        energy_res = np.sum(residual**2)
        residual = residual * np.sqrt(energy_y / (energy_res + 1e-10))
        
        return residual
    
    def _compute_lpc(self, frame, order):
        """Compute LPC coefficients using autocorrelation method"""
        # Calculate autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(frame)-1:len(frame)+order]
        
        # Levinson-Durbin recursion
        if autocorr[0] == 0:
            return np.zeros(order)
        
        lpc_coeffs = np.zeros(order)
        error = autocorr[0]
        
        for i in range(order):
            # Calculate reflection coefficient
            k = -np.sum(lpc_coeffs[:i] * autocorr[i:0:-1]) / error
            
            # Update coefficients
            lpc_coeffs[i] = k
            for j in range(i//2 + 1):
                temp = lpc_coeffs[j]
                lpc_coeffs[j] = temp + k * lpc_coeffs[i-j-1]
                if j < i-j-1:
                    lpc_coeffs[i-j-1] = lpc_coeffs[i-j-1] + k * temp
            
            # Update error
            error = error * (1 - k*k)
            
            if error <= 0:
                break
        
        return np.concatenate(([1], lpc_coeffs))
    
    def _apply_inverse_filter(self, frame, lpc_coeffs):
        """Apply LPC inverse filtering to a frame"""
        # Pad frame to handle filter initialization
        padded_frame = np.pad(frame, (len(lpc_coeffs)-1, 0))
        
        # Apply filter
        residual = np.zeros_like(padded_frame)
        for i in range(len(lpc_coeffs)-1, len(padded_frame)):
            residual[i] = np.sum(lpc_coeffs * padded_frame[i-(len(lpc_coeffs)-1):i+1][::-1])
        
        # Return unpadded result
        return residual[len(lpc_coeffs)-1:]


class NormalizationFactory:
    """Factory class to create appropriate normalization objects"""
    
    @staticmethod
    def create_normalizer(method, config_path="config/processing.yaml"):
        """Create a normalizer based on method name"""
        if method == "channel_eq":
            return ChannelNormalizer(config_path)
        elif method == "cmvn":
            return CepstralMeanVarianceNormalizer(config_path)
        elif method == "lpc":
            return LinearPredictiveInverseFiltering(config_path)
        else:
            print(f"Unknown normalization method: {method}. Using channel normalization.")
            return ChannelNormalizer(config_path)
