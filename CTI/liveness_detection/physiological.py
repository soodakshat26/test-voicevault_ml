import numpy as np
import librosa
from scipy import signal
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class BreathingPatternDetector:
    """
    Detector for breathing patterns in speech based on audio envelope.
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000, 
        frame_length: int = 1024, 
        hop_length: int = 256
    ):
        """
        Initialize the breathing pattern detector.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_length: Frame length for analysis
            hop_length: Hop length between frames
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        
        # Breathing rate typically between 0.2 and 0.5 Hz
        self.breathing_band = (0.2, 0.5)
    
    def detect_breathing(self, audio: np.ndarray) -> Dict:
        """
        Detect breathing patterns in speech.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with breathing pattern features
        """
        # Calculate envelope
        envelope = self._calculate_envelope(audio)
        
        # Low-pass filter to focus on breathing frequencies
        nyquist = self.sample_rate / (2 * self.hop_length)
        low = self.breathing_band[0] / nyquist
        high = self.breathing_band[1] / nyquist
        b, a = signal.butter(2, [low, high], btype='bandpass')
        breathing_envelope = signal.filtfilt(b, a, envelope)
        
        # Find peaks (potential breathing points)
        peaks, properties = signal.find_peaks(
            breathing_envelope, 
            distance=int(self.sample_rate / self.hop_length * 2)  # Min 2 seconds between breaths
        )
        
        # Calculate breathing rate (if enough peaks detected)
        breathing_rate = None
        if len(peaks) > 1:
            # Calculate time between peaks
            peak_times = peaks * self.hop_length / self.sample_rate
            intervals = np.diff(peak_times)
            
            # Average breathing interval and rate
            avg_interval = np.mean(intervals)
            breathing_rate = 60 / avg_interval  # breaths per minute
        
        # Calculate breathing regularity
        regularity = None
        if len(peaks) > 2:
            intervals = np.diff(peaks)
            regularity = 1 - (np.std(intervals) / np.mean(intervals))
        
        # Calculate breathing depth
        depths = properties['prominences'] if 'prominences' in properties else []
        avg_depth = np.mean(depths) if len(depths) > 0 else None
        
        # Calculate breathing presence score
        if breathing_rate is not None and regularity is not None:
            # Typical human breathing is 12-20 breaths per minute
            rate_factor = 1 - abs(breathing_rate - 15) / 15
            rate_factor = max(0, min(rate_factor, 1))
            
            # Combine rate and regularity
            breathing_presence = 0.6 * rate_factor + 0.4 * regularity
        else:
            breathing_presence = 0.0
        
        # Return results
        return {
            'envelope': envelope,
            'breathing_envelope': breathing_envelope,
            'peaks': peaks,
            'breathing_rate': breathing_rate,
            'breathing_regularity': regularity,
            'breathing_depth': avg_depth,
            'breathing_presence_score': breathing_presence
        }
    
    def _calculate_envelope(self, audio: np.ndarray) -> np.ndarray:
        """
        Calculate the envelope of an audio signal.
        
        Args:
            audio: Audio signal
            
        Returns:
            Envelope of the signal
        """
        # Calculate frame RMS
        frames = librosa.util.frame(audio, frame_length=self.frame_length, hop_length=self.hop_length)
        rms = np.sqrt(np.mean(frames**2, axis=0))
        
        # Smooth envelope
        window_size = 5
        envelope = np.convolve(rms, np.ones(window_size)/window_size, mode='same')
        
        return envelope


class MicroModulationAnalyzer:
    """
    Analyzer for micro-modulations in voice that indicate liveness.
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000, 
        frame_length: int = 512, 
        hop_length: int = 128
    ):
        """
        Initialize the micro-modulation analyzer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_length: Frame length for analysis
            hop_length: Hop length between frames
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def analyze_micro_modulations(self, audio: np.ndarray) -> Dict:
        """
        Analyze micro-modulations in voice.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with micro-modulation features
        """
        # Calculate STFT
        stft = librosa.stft(
            audio, 
            n_fft=self.frame_length, 
            hop_length=self.hop_length,
            win_length=self.frame_length
        )
        magnitude, phase = librosa.magphase(stft)
        
        # Calculate spectral flux (frame-to-frame spectral difference)
        spectral_flux = librosa.onset.onset_strength(
            sr=self.sample_rate,
            S=magnitude,
            hop_length=self.hop_length
        )
        
        # Calculate smoothed flux
        window_size = 5
        smoothed_flux = np.convolve(spectral_flux, np.ones(window_size)/window_size, mode='same')
        
        # Calculate micro-variations (residual after smoothing)
        micro_variations = spectral_flux - smoothed_flux
        
        # Calculate phase variations
        phase_diff = np.diff(np.angle(phase), axis=1)
        phase_diff = np.pad(phase_diff, ((0, 0), (0, 1)), mode='constant')
        phase_variation = np.mean(np.abs(phase_diff), axis=0)
        
        # Calculate frequency stability
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(
            S=magnitude,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Get pitch stability
        pitch_stability = self._calculate_pitch_stability(pitches, magnitudes)
        
        # Calculate amplitude modulation
        # Extract envelope
        envelope = np.sqrt(np.sum(magnitude**2, axis=0))
        
        # Calculate envelope variations
        envelope_diff = np.diff(envelope)
        envelope_diff = np.pad(envelope_diff, (0, 1), mode='constant')
        
        # Normalize variations
        norm_envelope_var = np.std(envelope_diff) / (np.mean(envelope) + 1e-10)
        
        # Modulation spectrum (analyze modulation frequencies)
        mod_spec = np.abs(np.fft.rfft(envelope))
        
        # Extract modulation energy in different bands
        mod_freqs = np.fft.rfftfreq(len(envelope), d=self.hop_length/self.sample_rate)
        
        # Tremor band (4-8 Hz)
        tremor_band = np.logical_and(mod_freqs >= 4, mod_freqs <= 8)
        tremor_energy = np.sum(mod_spec[tremor_band]) / (np.sum(mod_spec) + 1e-10)
        
        # Calculate jitter and shimmer
        jitter, shimmer = self._calculate_jitter_shimmer(audio)
        
        # Calculate naturalness score based on micro-modulations
        naturalness_score = self._calculate_naturalness_score(
            pitch_stability, norm_envelope_var, tremor_energy, jitter, shimmer
        )
        
        # Return results
        return {
            'spectral_flux': spectral_flux,
            'micro_variations': micro_variations,
            'phase_variation': phase_variation,
            'pitch_stability': pitch_stability,
            'envelope_variation': norm_envelope_var,
            'tremor_energy': tremor_energy,
            'jitter': jitter,
            'shimmer': shimmer,
            'naturalness_score': naturalness_score
        }
    
    def _calculate_pitch_stability(
        self, 
        pitches: np.ndarray, 
        magnitudes: np.ndarray
    ) -> float:
        """
        Calculate pitch stability from pitch tracking results.
        
        Args:
            pitches: Pitch values from librosa.piptrack
            magnitudes: Magnitude values from librosa.piptrack
            
        Returns:
            Pitch stability score (lower is more stable)
        """
        # For each frame, find the pitch with the highest magnitude
        pitch_values = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            
            # Only add non-zero pitches (voiced frames)
            if pitch > 0 and magnitudes[index, i] > 0:
                pitch_values.append(pitch)
        
        # Calculate stability if we have enough pitch values
        if len(pitch_values) > 2:
            # Calculate coefficient of variation
            pitch_values = np.array(pitch_values)
            stability = np.std(pitch_values) / (np.mean(pitch_values) + 1e-10)
            
            # Invert so lower values mean less stable
            stability = 1 / (1 + stability)
            
            return stability
        
        return 0.0  # Default if not enough data
    
    def _calculate_jitter_shimmer(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Calculate jitter and shimmer from audio.
        
        Args:
            audio: Audio signal
            
        Returns:
            Tuple of (jitter, shimmer) values
        """
        # Try to use Praat for accurate measurements
        try:
            import parselmouth
            from parselmouth.praat import call
            
            # Create Praat sound object
            sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
            
            # Extract pitch
            pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
            
            # Get jitter
            jitter = call(pitch, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
            
            # Get shimmer
            shimmer = call([sound, pitch], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
            
            return jitter, shimmer
            
        except (ImportError, Exception):
            # Fallback to a simplified method if Praat is not available
            
            # Extract pitch
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=75,
                fmax=600,
                sr=self.sample_rate
            )
            
            # Only consider voiced segments
            voiced_indices = np.where(voiced_flag)[0]
            voiced_f0 = f0[voiced_indices]
            
            # Calculate jitter (if we have enough pitch values)
            jitter = 0.0
            if len(voiced_f0) > 2:
                # Calculate absolute differences
                pitch_diffs = np.abs(np.diff(voiced_f0))
                
                # Jitter is the average absolute difference between consecutive periods
                jitter = np.mean(pitch_diffs) / (np.mean(voiced_f0) + 1e-10)
            
            # Calculate shimmer
            shimmer = 0.0
            if len(voiced_indices) > 2:
                # Find amplitude peaks
                frames = librosa.util.frame(audio, frame_length=self.frame_length, hop_length=self.hop_length)
                amplitude_peaks = np.max(np.abs(frames), axis=0)
                
                # Calculate shimmer
                amp_diffs = np.abs(np.diff(amplitude_peaks))
                shimmer = np.mean(amp_diffs) / (np.mean(amplitude_peaks) + 1e-10)
            
            return jitter, shimmer
    
    def _calculate_naturalness_score(
        self, 
        pitch_stability: float, 
        envelope_var: float, 
        tremor_energy: float, 
        jitter: float, 
        shimmer: float
    ) -> float:
        """
        Calculate naturalness score based on micro-modulations.
        
        Args:
            pitch_stability: Pitch stability score
            envelope_var: Normalized envelope variation
            tremor_energy: Energy in tremor frequency band
            jitter: Jitter value
            shimmer: Shimmer value
            
        Returns:
            Naturalness score (0-1, higher is more natural)
        """
        # Define ideal ranges for each parameter
        # These ranges are based on typical values for natural speech
        ideal_pitch_stability = (0.7, 0.95)
        ideal_envelope_var = (0.05, 0.2)
        ideal_tremor_energy = (0.01, 0.1)
        ideal_jitter = (0.005, 0.02)
        ideal_shimmer = (0.04, 0.2)
        
        # Calculate individual scores
        pitch_score = self._range_score(pitch_stability, ideal_pitch_stability)
        envelope_score = self._range_score(envelope_var, ideal_envelope_var)
        tremor_score = self._range_score(tremor_energy, ideal_tremor_energy)
        jitter_score = self._range_score(jitter, ideal_jitter)
        shimmer_score = self._range_score(shimmer, ideal_shimmer)
        
        # Combine scores with weights
        weights = [0.3, 0.15, 0.15, 0.2, 0.2]
        scores = [pitch_score, envelope_score, tremor_score, jitter_score, shimmer_score]
        
        naturalness_score = sum(w * s for w, s in zip(weights, scores))
        
        return naturalness_score
    
    def _range_score(self, value: float, ideal_range: Tuple[float, float]) -> float:
        """
        Calculate score based on whether a value falls within an ideal range.
        
        Args:
            value: Value to score
            ideal_range: Tuple of (min, max) for ideal range
            
        Returns:
            Score between 0 and 1
        """
        if value < ideal_range[0]:
            # Below minimum
            return 1 - min(1, (ideal_range[0] - value) / ideal_range[0])
        elif value > ideal_range[1]:
            # Above maximum
            return 1 - min(1, (value - ideal_range[1]) / ideal_range[1])
        else:
            # Within range
            # Higher score if closer to middle of range
            range_mid = (ideal_range[0] + ideal_range[1]) / 2
            range_width = ideal_range[1] - ideal_range[0]
            
            return 1 - min(1, 2 * abs(value - range_mid) / range_width)


class ResonanceTracker:
    """
    Tracker for vocal tract resonance variations which indicate liveness.
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000, 
        frame_length: int = 25, 
        hop_length: int = 10
    ):
        """
        Initialize the resonance tracker.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_length: Frame length in milliseconds
            hop_length: Hop length in milliseconds
        """
        self.sample_rate = sample_rate
        self.frame_length_samples = int(frame_length * sample_rate / 1000)
        self.hop_length_samples = int(hop_length * sample_rate / 1000)
    
    def track_resonances(self, audio: np.ndarray) -> Dict:
        """
        Track vocal tract resonances over time.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with resonance tracking results
        """
        # Extract formants
        formants = self._extract_formants(audio)
        
        # Calculate formant statistics
        formant_means = np.nanmean(formants, axis=1)
        formant_stds = np.nanstd(formants, axis=1)
        
        # Calculate formant trajectories (deltas)
        formant_deltas = np.zeros_like(formants)
        formant_deltas[:, 1:] = np.diff(formants, axis=1)
        
        # Calculate formant acceleration (delta-delta)
        formant_accels = np.zeros_like(formants)
        formant_accels[:, 2:] = np.diff(formant_deltas, axis=1)
        
        # Calculate formant continuity (how smoothly formants change)
        continuity_scores = self._calculate_continuity(formants)
        
        # Calculate formant variability (natural variations)
        variability_scores = self._calculate_variability(formants, formant_deltas)
        
        # Calculate formant bandwidth estimates
        bandwidths = self._estimate_bandwidths(audio, formants)
        
        # Calculate spectral tilt
        spectral_tilt = self._calculate_spectral_tilt(audio)
        
        # Calculate resonance naturalness score
        naturalness_score = self._calculate_naturalness_score(
            formants, continuity_scores, variability_scores, bandwidths, spectral_tilt
        )
        
        # Return results
        return {
            'formants': formants,
            'formant_means': formant_means,
            'formant_stds': formant_stds,
            'formant_deltas': formant_deltas,
            'formant_accels': formant_accels,
            'continuity_scores': continuity_scores,
            'variability_scores': variability_scores,
            'bandwidths': bandwidths,
            'spectral_tilt': spectral_tilt,
            'naturalness_score': naturalness_score
        }
    
    def _extract_formants(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract formants from audio using LPC analysis.
        
        Args:
            audio: Audio signal
            
        Returns:
            Array of formant frequencies [formants, frames]
        """
        try:
            import parselmouth
            from parselmouth.praat import call
            
            # Create a Praat Sound object
            sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
            
            # Extract formants
            formants = call(sound, "To Formant (burg)", 0.0025, 5, 5500, 0.025, 50)
            
            # Extract formant values for each frame
            num_frames = call(formants, "Get number of frames")
            
            # Get times for each frame
            frame_times = []
            for frame in range(1, num_frames+1):
                time = call(formants, "Get time from frame number", frame)
                frame_times.append(time)
            
            # Initialize formant array
            formant_values = np.zeros((4, num_frames))
            
            # Extract first 4 formants for each frame
            for frame in range(1, num_frames+1):
                for formant in range(1, 5):
                    value = call(formants, "Get value at time", formant, frame_times[frame-1], 'Hertz', 'Linear')
                    if not np.isnan(value):
                        formant_values[formant-1, frame-1] = value
            
            return formant_values
            
        except (ImportError, Exception):
            # Fallback to a simplified LPC-based method
            
            # Apply pre-emphasis
            pre_emphasis = 0.97
            y_emph = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # Frame the signal
            frames = librosa.util.frame(
                y_emph, 
                frame_length=self.frame_length_samples, 
                hop_length=self.hop_length_samples
            )
            
            # Apply window
            frames = frames * np.hamming(self.frame_length_samples).reshape(-1, 1)
            
            # Number of coefficients for LPC analysis
            n_lpc = 2 + self.sample_rate // 1000
            
            # Initialize formant array
            n_frames = frames.shape[1]
            formant_values = np.zeros((4, n_frames))
            
            # Process each frame
            for i in range(n_frames):
                frame = frames[:, i]
                
                # Calculate LPC coefficients
                a_lpc = librosa.lpc(frame, n_lpc)
                
                # Get roots of LPC polynomial
                roots = np.roots(a_lpc)
                
                # Keep only roots with positive imaginary part (and inside unit circle)
                roots = roots[np.imag(roots) > 0]
                roots = roots[np.abs(roots) < 0.99]
                
                # Convert to frequencies
                angles = np.arctan2(np.imag(roots), np.real(roots))
                freqs = angles * self.sample_rate / (2 * np.pi)
                
                # Sort by frequency and take first 4 formants
                freqs = np.sort(freqs)
                for j in range(min(4, len(freqs))):
                    formant_values[j, i] = freqs[j]
            
            return formant_values
    
    def _calculate_continuity(self, formants: np.ndarray) -> np.ndarray:
        """
        Calculate continuity scores for formant trajectories.
        
        Args:
            formants: Formant frequency array [formants, frames]
            
        Returns:
            Continuity scores for each formant
        """
        # Calculate frame-to-frame changes
        deltas = np.diff(formants, axis=1)
        
        # Replace NaN values with zeros
        deltas = np.nan_to_num(deltas)
        
        # Calculate absolute changes
        abs_deltas = np.abs(deltas)
        
        # Define thresholds for natural formant movement
        max_natural_deltas = np.array([100, 150, 200, 250])  # Hz per frame
        
        # Reshape for broadcasting
        max_natural_deltas = max_natural_deltas.reshape(-1, 1)
        
        # Calculate continuity score (1 for perfect continuity, 0 for discontinuous)
        continuity = 1 - np.minimum(1, abs_deltas / max_natural_deltas)
        
        # Average over frames
        continuity_scores = np.mean(continuity, axis=1)
        
        return continuity_scores
    
    def _calculate_variability(
        self, 
        formants: np.ndarray, 
        deltas: np.ndarray
    ) -> np.ndarray:
        """
        Calculate variability scores for formant trajectories.
        
        Args:
            formants: Formant frequency array [formants, frames]
            deltas: Formant deltas array [formants, frames]
            
        Returns:
            Variability scores for each formant
        """
        # Replace NaN values with zeros
        formants_clean = np.nan_to_num(formants)
        deltas_clean = np.nan_to_num(deltas)
        
        # Calculate coefficient of variation for each formant
        cv = np.std(formants_clean, axis=1) / (np.mean(formants_clean, axis=1) + 1e-10)
        
        # Calculate variability in deltas
        delta_variability = np.std(deltas_clean, axis=1) / (np.max(np.abs(deltas_clean), axis=1) + 1e-10)
        
        # Define ideal range for natural speech
        ideal_cv = np.array([0.05, 0.08, 0.1, 0.12])
        ideal_delta_var = np.array([0.3, 0.4, 0.5, 0.6])
        
        # Calculate scores based on distance from ideal range
        cv_score = 1 - np.minimum(1, np.abs(cv - ideal_cv) / ideal_cv)
        delta_score = 1 - np.minimum(1, np.abs(delta_variability - ideal_delta_var) / ideal_delta_var)
        
        # Combine scores
        variability_scores = 0.6 * cv_score + 0.4 * delta_score
        
        return variability_scores
    
    def _estimate_bandwidths(
        self, 
        audio: np.ndarray, 
        formants: np.ndarray
    ) -> np.ndarray:
        """
        Estimate formant bandwidths.
        
        Args:
            audio: Audio signal
            formants: Formant frequency array
            
        Returns:
            Bandwidth estimates for each formant
        """
        try:
            import parselmouth
            from parselmouth.praat import call
            
            # Create a Praat Sound object
            sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
            
            # Extract formants
            formant_obj = call(sound, "To Formant (burg)", 0.0025, 5, 5500, 0.025, 50)
            
            # Get number of frames
            num_frames = call(formant_obj, "Get number of frames")
            
            # Initialize bandwidth array
            bandwidths = np.zeros((4, num_frames))
            
            # Extract bandwidths for each frame
            for frame in range(1, num_frames+1):
                time = call(formant_obj, "Get time from frame number", frame)
                
                for formant in range(1, 5):
                    try:
                        bandwidth = call(formant_obj, "Get bandwidth at time", formant, time, 'Hertz', 'Linear')
                        if not np.isnan(bandwidth):
                            bandwidths[formant-1, frame-1] = bandwidth
                    except:
                        pass
            
            return bandwidths
            
        except (ImportError, Exception):
            # Estimate bandwidths based on formant frequencies
            # This is a simplified approximation
            
            # Define typical bandwidth-to-frequency ratios
            bandwidth_ratios = np.array([0.06, 0.08, 0.1, 0.12])
            
            # Calculate bandwidths
            bandwidths = formants * bandwidth_ratios.reshape(-1, 1)
            
            return bandwidths
    
    def _calculate_spectral_tilt(self, audio: np.ndarray) -> float:
        """
        Calculate spectral tilt (spectral balance) from audio.
        
        Args:
            audio: Audio signal
            
        Returns:
            Spectral tilt value
        """
        # Calculate spectrogram
        S = librosa.stft(
            audio, 
            n_fft=self.frame_length_samples, 
            hop_length=self.hop_length_samples
        )
        
        # Convert to power
        S_power = np.abs(S)**2
        
        # Calculate frequency axis
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length_samples)
        
        # Calculate average spectrum
        avg_spectrum = np.mean(S_power, axis=1)
        
        # Create log frequency and log power arrays for linear regression
        log_freqs = np.log10(freqs + 1e-10)
        log_power = 10 * np.log10(avg_spectrum + 1e-10)
        
        # Perform linear regression to find slope
        # y = mx + b, where m is the spectral tilt
        A = np.vstack([log_freqs, np.ones(len(log_freqs))]).T
        m, b = np.linalg.lstsq(A, log_power, rcond=None)[0]
        
        # Return negative slope as spectral tilt
        return -m
    
    def _calculate_naturalness_score(
        self, 
        formants: np.ndarray, 
        continuity_scores: np.ndarray, 
        variability_scores: np.ndarray, 
        bandwidths: np.ndarray, 
        spectral_tilt: float
    ) -> float:
        """
        Calculate resonance naturalness score.
        
        Args:
            formants: Formant frequency array
            continuity_scores: Continuity scores for each formant
            variability_scores: Variability scores for each formant
            bandwidths: Bandwidth estimates for each formant
            spectral_tilt: Spectral tilt value
            
        Returns:
            Naturalness score (0-1, higher is more natural)
        """
        # Weight continuity scores (continuous formants are more natural)
        continuity_weight = 0.35
        continuity_total = np.mean(continuity_scores)
        
        # Weight variability scores (natural speech has the right amount of variability)
        variability_weight = 0.3
        variability_total = np.mean(variability_scores)
        
        # Score bandwidth-to-formant ratios (should be in natural range)
        bandwidth_weight = 0.2
        bandwidth_ratios = np.nanmean(bandwidths, axis=1) / (np.nanmean(formants, axis=1) + 1e-10)
        
        # Ideal bandwidth ratios for each formant
        ideal_ratios = np.array([0.06, 0.08, 0.1, 0.12])
        
        # Score based on distance from ideal
        bandwidth_scores = 1 - np.minimum(1, np.abs(bandwidth_ratios - ideal_ratios) / ideal_ratios)
        bandwidth_total = np.mean(bandwidth_scores)
        
        # Score spectral tilt (natural speech has moderate tilt)
        tilt_weight = 0.15
        # Ideal range for spectral tilt in natural speech
        ideal_tilt_range = (-10, -6)
        
        if spectral_tilt < ideal_tilt_range[0]:
            # Too steep (likely unnatural)
            tilt_score = 1 - min(1, (ideal_tilt_range[0] - spectral_tilt) / abs(ideal_tilt_range[0]))
        elif spectral_tilt > ideal_tilt_range[1]:
            # Too flat (likely synthetic)
            tilt_score = 1 - min(1, (spectral_tilt - ideal_tilt_range[1]) / abs(ideal_tilt_range[1]))
        else:
            # Within ideal range
            tilt_score = 1.0
        
        # Combine scores
        naturalness_score = (
            continuity_weight * continuity_total +
            variability_weight * variability_total +
            bandwidth_weight * bandwidth_total +
            tilt_weight * tilt_score
        )
        
        return naturalness_score

