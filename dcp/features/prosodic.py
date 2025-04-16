import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import yaml
from scipy import signal
import warnings

class ProsodicFeatureExtractor:
    """Base class for prosodic feature extraction"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['features']['prosodic']
        
        self.sample_rate = 16000  # Default sample rate
        
    def extract(self, y, sr):
        """Extract basic prosodic features"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Calculate energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Basic pitch estimation
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        features = {
            'rms': rms,
            'f0': f0,
            'voiced_flag': voiced_flag
        }
        
        return features


class YAAPTPitchExtractor(ProsodicFeatureExtractor):
    """
    YAAPT pitch tracking with glottal closure modeling
    
    Note: This uses Praat as YAAPT is not readily available as a Python package
    """
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.f0_algorithm = self.config.get('f0_algorithm', 'yaapt')
        
        # Pitch settings
        f0_min, f0_max = self.config.get('f0_range', [50, 600])
        self.f0_min = f0_min
        self.f0_max = f0_max
        
    def extract(self, y, sr):
        """Extract pitch using YAAPT-like approach"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        try:
            # Create Praat Sound object
            sound = parselmouth.Sound(y, sampling_frequency=sr)
            
            # Extract pitch using Praat
            pitch = call(sound, "To Pitch", 0.0, self.f0_min, self.f0_max)
            
            # Extract pitch values
            f0_values = call(pitch, "Get values in range", 0, 0, "Hertz", "linear")
            
            # Extract pitch strength (similar to glottal closure certainty)
            strength = call(pitch, "Get strength values")
            
            # Create time axis
            times = np.arange(len(f0_values)) * call(pitch, "Get time step")
            
            # Process pitch jumps and octave errors
            f0_smooth = self._process_pitch_jumps(f0_values)
            
            # Detect glottal closure instants
            gci = self._detect_glottal_closures(y, sr, f0_smooth)
            
            features = {
                'f0': f0_smooth,
                'f0_times': times,
                'pitch_strength': strength,
                'glottal_closures': gci
            }
            
            return features
            
        except Exception as e:
            warnings.warn(f"Error in Praat-based pitch extraction: {e}. Falling back to librosa.")
            # Fallback to librosa
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=self.f0_min, 
                fmax=self.f0_max,
                sr=sr
            )
            
            return {
                'f0': f0,
                'voiced_flag': voiced_flag,
                'confidence': voiced_probs
            }
    
    def _process_pitch_jumps(self, f0):
        """Process pitch jumps and octave errors"""
        f0_smooth = np.copy(f0)
        
        # Replace NaN values with 0
        f0_smooth[np.isnan(f0_smooth)] = 0
        
        # Median filter to remove spikes
        f0_smooth = signal.medfilt(f0_smooth, kernel_size=5)
        
        # Correct octave jumps
        for i in range(1, len(f0_smooth)):
            if f0_smooth[i] > 0 and f0_smooth[i-1] > 0:
                ratio = f0_smooth[i] / f0_smooth[i-1]
                
                # If jump is close to an octave (0.5 or 2.0), correct it
                if 1.8 < ratio < 2.2:
                    f0_smooth[i] = f0_smooth[i] / 2
                elif 0.45 < ratio < 0.55:
                    f0_smooth[i] = f0_smooth[i] * 2
        
        return f0_smooth
    
    def _detect_glottal_closures(self, y, sr, f0):
        """Detect glottal closure instants"""
        # This is a simplified GCI detection
        # A full implementation would use SEDREAMS or ZFF algorithms
        
        # Create LPC residual
        order = int(sr / 1000) + 4  # LPC order rule of thumb
        lpc_coeffs = librosa.lpc(y, order=order)
        
        # Apply inverse filter to get residual
        residual = np.zeros_like(y)
        for i in range(len(y) - order):
            residual[i+order] = y[i+order] - np.sum(lpc_coeffs[1:] * y[i+order-1:i:-1])
        
        # Find positive peaks in residual
        peaks, _ = signal.find_peaks(residual, height=0.05, distance=sr/500)
        
        # Only keep peaks that are likely to be GCIs based on F0
        valid_peaks = []
        prev_peak = None
        
        for peak in peaks:
            if prev_peak is not None:
                # Calculate time difference
                delta_t = (peak - prev_peak) / sr
                
                # Estimated period from F0
                peak_time = peak / sr
                f0_idx = min(int(peak_time * len(f0) / (len(y) / sr)), len(f0) - 1)
                
                if f0[f0_idx] > 0:
                    period = 1 / f0[f0_idx]
                    
                    # Accept if within 20% of expected period
                    if 0.8 * period < delta_t < 1.2 * period:
                        valid_peaks.append(peak)
            
            prev_peak = peak
        
        return np.array(valid_peaks)


class SpeechRateEstimator(ProsodicFeatureExtractor):
    """Speech rate estimation using envelope modulation spectrum"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.estimate_speech_rate = self.config.get('speech_rate', True)
        
    def extract(self, y, sr):
        """Extract speech rate features"""
        if not self.estimate_speech_rate:
            return {}
        
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Calculate envelope
        analytic_signal = signal.hilbert(y)
        envelope = np.abs(analytic_signal)
        
        # Low-pass filter envelope to focus on syllable rate
        b, a = signal.butter(2, 10/(sr/2), 'low')
        envelope_filtered = signal.filtfilt(b, a, envelope)
        
        # Downsample envelope
        target_sr = 100  # Hz
        downsample_factor = sr // target_sr
        envelope_downsampled = envelope_filtered[::downsample_factor]
        
        # Calculate modulation spectrum
        n_fft = 1024
        mod_spec = np.abs(np.fft.rfft(envelope_downsampled, n=n_fft))
        
        # Get modulation frequencies
        mod_freqs = np.fft.rfftfreq(n_fft, 1/target_sr)
        
        # Find peak in syllable range (3-8 Hz)
        syllable_mask = np.logical_and(mod_freqs >= 3, mod_freqs <= 8)
        if np.any(syllable_mask):
            syllable_spec = mod_spec[syllable_mask]
            peak_idx = np.argmax(syllable_spec)
            syllable_rate = mod_freqs[syllable_mask][peak_idx]
        else:
            syllable_rate = 0
        
        # Estimate speech rate in syllables per minute
        speech_rate_syll_per_min = syllable_rate * 60
        
        # Calculate rhythmic features
        rhythm_features = self._calculate_rhythm_features(envelope_downsampled, target_sr)
        
        return {
            'speech_rate_syll_per_min': speech_rate_syll_per_min,
            'modulation_spectrum': mod_spec,
            'modulation_freqs': mod_freqs,
            **rhythm_features
        }
    
    def _calculate_rhythm_features(self, envelope, sr):
        """Calculate rhythm-related features"""
        # Find envelope peaks (syllable nuclei)
        peaks, properties = signal.find_peaks(
            envelope, 
            height=0.1*np.max(envelope),
            distance=sr/10  # Minimum 100ms between syllables
        )
        
        if len(peaks) < 2:
            return {
                'rhythm_nPVI': 0,
                'rhythm_varco': 0,
                'peak_count': len(peaks)
            }
        
        # Calculate intervals between peaks in seconds
        intervals = np.diff(peaks) / sr
        
        # Calculate normalized pairwise variability index (nPVI)
        # nPVI measures rhythm by comparing successive intervals
        nPVI = 0
        for i in range(len(intervals) - 1):
            nPVI += abs(intervals[i] - intervals[i+1]) / ((intervals[i] + intervals[i+1]) / 2)
        
        nPVI = nPVI * 100 / (len(intervals) - 1)
        
        # Calculate VarcoV (rate-normalized standard deviation)
        varco = np.std(intervals) / np.mean(intervals) * 100
        
        return {
            'rhythm_nPVI': nPVI,
            'rhythm_varco': varco,
            'peak_count': len(peaks)
        }


class StressPatternAnalyzer(ProsodicFeatureExtractor):
    """Stress pattern modeling through energy distribution"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.energy_bands = self.config.get('energy_bands', 5)
        
    def extract(self, y, sr):
        """Extract stress pattern features"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Calculate spectrogram
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        
        # Create frequency bands
        n_bands = self.energy_bands
        band_energies = np.zeros((n_bands, S.shape[1]))
        
        # Divide spectrum into bands
        for i in range(n_bands):
            start = int(i * S.shape[0] / n_bands)
            end = int((i + 1) * S.shape[0] / n_bands)
            band_energies[i] = np.sum(S[start:end, :], axis=0)
        
        # Normalize band energies
        band_energies = band_energies / (np.sum(band_energies, axis=0) + 1e-10)
        
        # Calculate temporal envelope
        envelope = np.sum(S, axis=0)
        envelope = envelope / np.max(envelope)
        
        # Find envelope peaks (stress points)
        peaks, _ = signal.find_peaks(envelope, height=0.1, distance=10)
        
        # Calculate stress pattern features
        if len(peaks) > 0:
            # Energy at stress points
            stress_energies = np.array([envelope[p] for p in peaks])
            
            # Intervals between stress points
            stress_intervals = np.diff(peaks)
            
            # Variability measures
            energy_variability = np.std(stress_energies) / np.mean(stress_energies)
            interval_variability = np.std(stress_intervals) / np.mean(stress_intervals) if len(stress_intervals) > 0 else 0
            
            stress_features = {
                'stress_count': len(peaks),
                'stress_energy_mean': np.mean(stress_energies),
                'stress_energy_variability': energy_variability,
                'stress_interval_variability': interval_variability
            }
        else:
            stress_features = {
                'stress_count': 0,
                'stress_energy_mean': 0,
                'stress_energy_variability': 0,
                'stress_interval_variability': 0
            }
        
        return {
            'band_energies': band_energies,
            'envelope': envelope,
            'stress_points': peaks,
            **stress_features
        }


class ProsodicFeatureFactory:
    """Factory for creating prosodic feature extractors"""
    
    @staticmethod
    def create_extractor(method, config_path="config/processing.yaml"):
        """Create feature extractor based on method name"""
        if method == "yaapt":
            return YAAPTPitchExtractor(config_path)
        elif method == "speech_rate":
            return SpeechRateEstimator(config_path)
        elif method == "stress_pattern":
            return StressPatternAnalyzer(config_path)
        else:
            print(f"Unknown feature extractor: {method}. Using basic prosodic extractor.")
            return ProsodicFeatureExtractor(config_path)
