import numpy as np
import librosa
import pywt
from scipy import signal
from typing import Tuple, List, Dict, Optional, Union

class SignalDecomposer:
    """
    Signal decomposition module for the acoustic fingerprinting system.
    
    This class provides methods to decompose audio signals into multiple
    frequency bands and extract phase information for temporal relationships.
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000, 
        n_fft: int = 1024, 
        hop_length: int = 256,
        n_mels: int = 80,
        wavelet: str = 'db4',
        decomposition_level: int = 5
    ):
        """
        Initialize the signal decomposer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            wavelet: Wavelet type for wavelet transform
            decomposition_level: Number of decomposition levels for wavelets
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.wavelet = wavelet
        self.decomposition_level = decomposition_level
        
        # Precompute mel filterbank
        self.mel_filterbank = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
        
        # Define frequency bands important for speech
        self.speech_bands = [
            (85, 255),    # Fundamental frequency range
            (255, 700),   # First formant (F1) range
            (700, 2000),  # Second formant (F2) range
            (2000, 3500), # Third formant (F3) range
            (3500, 5000)  # Fourth formant (F4) range
        ]
    
    def decompose_signal(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose the audio signal using multiple methods.
        
        Args:
            audio: Audio signal (1D numpy array)
            
        Returns:
            Dictionary containing decomposed signal representations
        """
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        # Calculate Short-Time Fourier Transform (STFT)
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude, phase = librosa.magphase(stft)
        
        # Calculate Mel spectrogram
        mel_spectrogram = np.matmul(self.mel_filterbank, magnitude)
        
        # Calculate log mel spectrogram
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        
        # Wavelet decomposition
        wavelet_coeffs = pywt.wavedec(audio, wavelet=self.wavelet, level=self.decomposition_level)
        
        # Constant-Q transform for better frequency resolution at lower frequencies
        cqt = librosa.cqt(
            audio, 
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=librosa.note_to_hz('C2'),
            bins_per_octave=24
        )
        cqt_magnitude, cqt_phase = librosa.magphase(cqt)
        
        # Filter audio into speech-specific frequency bands
        filtered_bands = self._filter_frequency_bands(audio)
        
        # Extract phase vector for capturing temporal relationships
        phase_vector = self._extract_phase_vector(phase)
        
        # Harmonic-percussive source separation for better speech component isolation
        harmonic, percussive = librosa.decompose.hpss(stft)
        
        # Package all decompositions
        return {
            'stft_magnitude': magnitude,
            'stft_phase': phase,
            'mel_spectrogram': mel_spectrogram,
            'log_mel_spectrogram': log_mel_spectrogram,
            'wavelet_coeffs': wavelet_coeffs,
            'cqt_magnitude': cqt_magnitude,
            'cqt_phase': cqt_phase,
            'filtered_bands': filtered_bands,
            'phase_vector': phase_vector,
            'harmonic': harmonic,
            'percussive': percussive
        }
    
    def _filter_frequency_bands(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Filter the audio signal into speech-specific frequency bands.
        
        Args:
            audio: Audio signal
            
        Returns:
            List of filtered signals for each frequency band
        """
        filtered_bands = []
        
        for low_freq, high_freq in self.speech_bands:
            # Design bandpass filter
            nyquist = self.sample_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            # Use a higher order filter for steeper rolloff
            b, a = signal.butter(4, [low, high], btype='bandpass')
            
            # Apply filter
            filtered = signal.filtfilt(b, a, audio)
            filtered_bands.append(filtered)
        
        return filtered_bands
    
    def _extract_phase_vector(self, phase: np.ndarray) -> np.ndarray:
        """
        Extract phase vector information to capture temporal relationships.
        
        Args:
            phase: Phase information from STFT
            
        Returns:
            Phase vector features
        """
        # Calculate phase derivative along time
        phase_diff = np.diff(phase, axis=1)
        
        # Wrap phase differences to [-pi, pi]
        phase_diff = np.angle(np.exp(1j * phase_diff))
        
        # Calculate phase acceleration (derivative of phase derivative)
        phase_accel = np.diff(phase_diff, axis=1)
        phase_accel = np.angle(np.exp(1j * phase_accel))
        
        # Pad to match original shape
        phase_diff = np.pad(phase_diff, ((0, 0), (0, 1)), mode='edge')
        phase_accel = np.pad(phase_accel, ((0, 0), (0, 2)), mode='edge')
        
        # Stack phase and its derivatives
        phase_features = np.stack([phase, phase_diff, phase_accel], axis=0)
        
        return phase_features
    
    def extract_time_frequency_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract advanced time-frequency features from audio.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of time-frequency features
        """
        # Get decompositions
        decompositions = self.decompose_signal(audio)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        spectral_contrast = librosa.feature.spectral_contrast(
            S=decompositions['stft_magnitude'], sr=self.sample_rate
        )
        
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Extract rhythmic features
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Package features
        features = {
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_contrast': spectral_contrast,
            'spectral_flatness': spectral_flatness,
            'onset_strength': onset_env,
        }
        
        return features


class SubAudioAnalyzer:
    """
    SubAudio Analysis Engine for extracting discriminative features.
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000, 
        frame_length: int = 25, 
        hop_length: int = 10,
        n_mfcc: int = 20
    ):
        """
        Initialize the SubAudio analyzer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_length: Frame length in milliseconds
            hop_length: Hop length in milliseconds
            n_mfcc: Number of MFCC coefficients
        """
        self.sample_rate = sample_rate
        self.frame_length_samples = int(frame_length * sample_rate / 1000)
        self.hop_length_samples = int(hop_length * sample_rate / 1000)
        self.n_mfcc = n_mfcc
        
        # Initialize signal decomposer
        self.decomposer = SignalDecomposer(sample_rate=sample_rate)
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive acoustic features for speaker fingerprinting.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of acoustic features
        """
        # Basic signal validation
        if audio.size == 0:
            raise ValueError("Empty audio signal provided")
        
        # Get signal decompositions
        decompositions = self.decomposer.decompose_signal(audio)
        
        # Extract MFCCs with deltas and delta-deltas
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length_samples,
            hop_length=self.hop_length_samples
        )
        
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Extract Mel-Frequency Spectral Coefficients
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.frame_length_samples,
            hop_length=self.hop_length_samples
        )
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # Extract harmonics and percussion components
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Compute chroma features for tonal content
        chroma = librosa.feature.chroma_stft(
            y=harmonic,
            sr=self.sample_rate,
            n_fft=self.frame_length_samples,
            hop_length=self.hop_length_samples
        )
        
        # Extract pitched features
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.frame_length_samples,
            hop_length=self.hop_length_samples
        )
        
        # Estimate fundamental frequency
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        # Get time-frequency features
        tf_features = self.decomposer.extract_time_frequency_features(audio)
        
        # Package all features
        features = {
            'mfcc': mfcc,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2,
            'log_mel_spectrogram': log_mel_spec,
            'chroma': chroma,
            'f0': f0,
            'voiced_flag': voiced_flag,
            'voiced_probs': voiced_probs,
            **tf_features
        }
        
        return features


def extract_formants(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extract formants from speech signal using LPC analysis.
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate of the audio
        
    Returns:
        Array of formant frequencies (F1, F2, F3, F4)
    """
    try:
        import parselmouth
        from parselmouth.praat import call
        
        # Create a Praat Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
        
        # Extract formants
        formants = call(sound, "To Formant (burg)", 0.0025, 5, 5500, 0.025, 50)
        
        # Get formant values
        num_frames = call(formants, "Get number of frames")
        f1_values = []
        f2_values = []
        f3_values = []
        f4_values = []
        
        for frame in range(1, num_frames+1):
            f1 = call(formants, "Get value at time", 1, frame * 0.0025, 'Hertz', 'Linear')
            f2 = call(formants, "Get value at time", 2, frame * 0.0025, 'Hertz', 'Linear')
            f3 = call(formants, "Get value at time", 3, frame * 0.0025, 'Hertz', 'Linear')
            f4 = call(formants, "Get value at time", 4, frame * 0.0025, 'Hertz', 'Linear')
            
            f1_values.append(f1)
            f2_values.append(f2)
            f3_values.append(f3)
            f4_values.append(f4)
        
        # Convert to numpy arrays
        f1_values = np.array(f1_values)
        f2_values = np.array(f2_values)
        f3_values = np.array(f3_values)
        f4_values = np.array(f4_values)
        
        # Replace NaN values with zeroes
        f1_values = np.nan_to_num(f1_values)
        f2_values = np.nan_to_num(f2_values)
        f3_values = np.nan_to_num(f3_values)
        f4_values = np.nan_to_num(f4_values)
        
        # Stack formants
        formants = np.stack([f1_values, f2_values, f3_values, f4_values])
        
        return formants
    
    except ImportError:
        # Fallback to LPC-based method if Praat is not available
        import scipy.signal as sig
        
        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized_signal = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Frame the signal
        frame_length = int(0.025 * sample_rate)  # 25 ms
        hop_length = int(0.010 * sample_rate)    # 10 ms
        
        # Pad signal to ensure we get full frames
        pad_length = frame_length - len(emphasized_signal) % hop_length
        if pad_length < frame_length:
            emphasized_signal = np.append(emphasized_signal, np.zeros(pad_length))
        
        # Get frames
        frames = librosa.util.frame(emphasized_signal, frame_length=frame_length, hop_length=hop_length)
        
        # Apply window
        frames = frames * np.hamming(frame_length)
        
        # LPC order
        lpc_order = 12
        
        # Lists to store formants
        formants = np.zeros((4, frames.shape[1]))
        
        for i in range(frames.shape[1]):
            # Calculate LPC coefficients
            a = librosa.lpc(frames[:, i], order=lpc_order)
            
            # Get roots of LPC polynomial
            roots = np.roots(a)
            
            # Keep only roots with positive imaginary part and inside unit circle
            roots = roots[np.imag(roots) > 0]
            roots = roots[np.abs(roots) < 0.99]
            
            # Convert to frequencies
            angles = np.arctan2(np.imag(roots), np.real(roots))
            freqs = angles * sample_rate / (2 * np.pi)
            
            # Sort by frequency
            freqs = np.sort(freqs)
            
            # Get first 4 formants if available
            for j in range(min(4, len(freqs))):
                formants[j, i] = freqs[j]
        
        return formants
