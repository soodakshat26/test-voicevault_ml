import numpy as np
import librosa
from scipy import signal
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class AcousticEnvironmentAnalyzer:
    """
    Analyzer for acoustic environment characteristics to detect replay attacks.
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000, 
        frame_length: int = 1024, 
        hop_length: int = 512
    ):
        """
        Initialize the acoustic environment analyzer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_length: Frame length for analysis
            hop_length: Hop length between frames
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def analyze_environment(self, audio: np.ndarray) -> Dict:
        """
        Analyze acoustic environment characteristics.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with environment analysis results
        """
        # Calculate spectrogram
        stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Calculate room characteristics
        reverb_time = self._estimate_reverb_time(audio)
        spectral_deviation = self._calculate_spectral_deviation(magnitude)
        background_noise = self._estimate_background_noise(audio)
        
        # Analyze frequency bands for replay artifacts
        band_energies = self._analyze_frequency_bands(magnitude)
        
        # Calculate noise floor stability
        noise_floor_stability = self._calculate_noise_floor_stability(audio)
        
        # Detect unusual frequency cutoffs that might indicate speaker playback
        cutoff_score = self._detect_frequency_cutoffs(magnitude)
        
        # Calculate overall environment score
        environment_score = self._calculate_environment_score(
            reverb_time, spectral_deviation, background_noise,
            band_energies, noise_floor_stability, cutoff_score
        )
        
        # Package results
        return {
            'reverb_time': reverb_time,
            'spectral_deviation': spectral_deviation,
            'background_noise': background_noise,
            'band_energies': band_energies,
            'noise_floor_stability': noise_floor_stability,
            'cutoff_score': cutoff_score,
            'environment_score': environment_score
        }
    
    def _estimate_reverb_time(self, audio: np.ndarray) -> float:
        """
        Estimate reverberation time from audio.
        
        Args:
            audio: Audio signal
            
        Returns:
            Estimated reverberation time in seconds
        """
        # Calculate energy envelope
        frames = librosa.util.frame(audio, frame_length=self.frame_length, hop_length=self.hop_length)
        energy = np.sum(frames**2, axis=0)
        
        # Convert to dB
        energy_db = 10 * np.log10(energy + 1e-10)
        
        # Find decay slopes in energy envelope
        decay_slopes = []
        
        # Find segments with decreasing energy
        i = 0
        while i < len(energy_db) - 10:
            if energy_db[i] > energy_db[i+1]:
                # Found a potential decay
                start_idx = i
                
                # Find how long the decay continues
                j = i + 1
                while j < len(energy_db) - 1 and energy_db[j] > energy_db[j+1]:
                    j += 1
                
                decay_length = j - start_idx
                
                # Only consider decays of significant length
                if decay_length > 5:
                    # Calculate decay rate (dB per second)
                    decay_time = decay_length * self.hop_length / self.sample_rate
                    decay_db = energy_db[start_idx] - energy_db[j]
                    decay_rate = decay_db / decay_time
                    
                    # RT60 is the time to decay by 60 dB
                    if decay_rate > 0:
                        rt60 = 60 / decay_rate
                        
                        # Only consider reasonable RT60 values
                        if 0.1 < rt60 < 2.0:
                            decay_slopes.append(rt60)
                
                i = j
            else:
                i += 1
        
        # Return average RT60 if any valid decays found
        if decay_slopes:
            # Return median to avoid outliers
            return np.median(decay_slopes)
        else:
            # Default if no valid decays found
            return 0.3
    
    def _calculate_spectral_deviation(self, magnitude: np.ndarray) -> float:
        """
        Calculate spectral deviation across frames.
        
        Args:
            magnitude: Magnitude spectrogram
            
        Returns:
            Spectral deviation score
        """
        # Calculate mean spectrum
        mean_spectrum = np.mean(magnitude, axis=1)
        
        # Calculate deviation of each frame from mean
        deviations = np.zeros(magnitude.shape[1])
        
        for i in range(magnitude.shape[1]):
            frame_spectrum = magnitude[:, i]
            # Use correlation as similarity measure
            correlation = np.corrcoef(mean_spectrum, frame_spectrum)[0, 1]
            deviations[i] = 1 - max(0, correlation)
        
        # Return average deviation
        return np.mean(deviations)
    
    def _estimate_background_noise(self, audio: np.ndarray) -> Dict:
        """
        Estimate background noise characteristics.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with noise characteristics
        """
        # Calculate spectrogram
        stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
        S_db = librosa.amplitude_to_db(np.abs(stft))
        
        # Estimate noise floor as the 10th percentile of each frequency bin
        noise_floor = np.percentile(S_db, 10, axis=1)
        
        # Calculate noise color (spectral shape of noise)
        noise_slope = np.polyfit(
            np.arange(len(noise_floor)) / len(noise_floor),
            noise_floor,
            1
        )[0]
        
        # Calculate noise level (average of noise floor)
        noise_level = np.mean(noise_floor)
        
        # Calculate noise consistency (stability across time)
        frame_mins = np.min(S_db, axis=0)
        noise_consistency = 1 - np.std(frame_mins) / (np.abs(np.mean(frame_mins)) + 1e-10)
        
        return {
            'noise_floor': noise_floor,
            'noise_slope': noise_slope,
            'noise_level': noise_level,
            'noise_consistency': noise_consistency
        }
    
    def _analyze_frequency_bands(self, magnitude: np.ndarray) -> Dict:
        """
        Analyze energy in different frequency bands for replay artifacts.
        
        Args:
            magnitude: Magnitude spectrogram
            
        Returns:
            Dictionary with band energy analysis
        """
        # Calculate frequency bin indices
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
        
        # Define bands of interest
        bands = [
            (20, 150),    # Sub-bass
            (150, 300),   # Bass
            (300, 500),   # Low-mids
            (500, 2000),  # Mids
            (2000, 5000), # High-mids
            (5000, 10000), # Highs
            (10000, 20000) # Ultra-highs
        ]
        
        # Calculate energy in each band
        band_energies = {}
        
        for i, (low, high) in enumerate(bands):
            # Find bins in this frequency range
            band_bins = np.where((freqs >= low) & (freqs <= high))[0]
            
            # Calculate average energy in this band
            if len(band_bins) > 0:
                band_energy = np.mean(np.mean(magnitude[band_bins, :]))
                band_energies[f'band_{i}'] = band_energy
            else:
                band_energies[f'band_{i}'] = 0
        
        # Calculate band ratios (useful for detecting playback characteristics)
        if band_energies['band_6'] > 0:  # Avoid division by zero
            # Ratio of highs to ultra-highs (often reduced in replayed audio)
            band_energies['high_ultrahigh_ratio'] = band_energies['band_5'] / band_energies['band_6']
        else:
            band_energies['high_ultrahigh_ratio'] = 0
            
        if band_energies['band_0'] > 0:  # Avoid division by zero
            # Ratio of mids to sub-bass (often altered in replayed audio)
            band_energies['mid_subbass_ratio'] = band_energies['band_3'] / band_energies['band_0']
        else:
            band_energies['mid_subbass_ratio'] = 0
        
        return band_energies
    
    def _calculate_noise_floor_stability(self, audio: np.ndarray) -> float:
        """
        Calculate stability of the noise floor over time.
        
        Args:
            audio: Audio signal
            
        Returns:
            Noise floor stability score
        """
        # Frame the signal
        frames = librosa.util.frame(audio, frame_length=self.frame_length, hop_length=self.hop_length)
        
        # Calculate minimum energy for each frame (proxy for noise floor)
        min_energies = np.min(np.abs(frames), axis=0)
        
        # Calculate stability as inverse of coefficient of variation
        cv = np.std(min_energies) / (np.mean(min_energies) + 1e-10)
        stability = 1 / (1 + cv)
        
        return stability
    
    def _detect_frequency_cutoffs(self, magnitude: np.ndarray) -> float:
        """
        Detect frequency cutoffs characteristic of speaker playback.
        
        Args:
            magnitude: Magnitude spectrogram
            
        Returns:
            Cutoff detection score (higher means less likely to be replayed)
        """
        # Calculate average spectrum
        avg_spectrum = np.mean(magnitude, axis=1)
        
        # Normalize
        avg_spectrum = avg_spectrum / (np.max(avg_spectrum) + 1e-10)
        
        # Calculate frequency axis
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
        
        # Look for sharp cutoffs in high frequencies (common in replayed audio)
        cutoff_score = 1.0
        
        # Check high frequency region (above 10kHz)
        high_freq_idx = np.where(freqs >= 10000)[0]
        if len(high_freq_idx) > 0:
            high_spectrum = avg_spectrum[high_freq_idx]
            high_freqs = freqs[high_freq_idx]
            
            # Look for sudden drop in energy
            for i in range(1, len(high_spectrum)):
                ratio = high_spectrum[i] / (high_spectrum[i-1] + 1e-10)
                if ratio < 0.7:  # More than 30% drop
                    # Weight by frequency (higher frequency cutoffs are more suspicious)
                    freq_weight = high_freqs[i] / 20000
                    cutoff_score -= 0.2 * freq_weight
        
        # Ensure score is between 0 and 1
        cutoff_score = max(0, min(1, cutoff_score))
        
        return cutoff_score
    
    def _calculate_environment_score(
        self, 
        reverb_time: float, 
        spectral_deviation: float,
        background_noise: Dict,
        band_energies: Dict,
        noise_floor_stability: float,
        cutoff_score: float
    ) -> float:
        """
        Calculate overall environment score for replay detection.
        
        Args:
            reverb_time: Estimated reverb time
            spectral_deviation: Spectral deviation score
            background_noise: Background noise characteristics
            band_energies: Band energy analysis
            noise_floor_stability: Noise floor stability score
            cutoff_score: Frequency cutoff score
            
        Returns:
            Environment score (higher is more likely to be live)
        """
        # Score reverb time (replayed audio often has unusual reverb)
        # Natural room reverb is typically between 0.3 and 1.2 seconds
        if 0.3 <= reverb_time <= 1.2:
            reverb_score = 1.0
        else:
            reverb_score = max(0, 1 - abs(reverb_time - 0.75) / 0.75)
        
        # Score spectral deviation (live audio typically has more variation)
        # A moderate amount of spectral deviation is natural
        if 0.1 <= spectral_deviation <= 0.3:
            deviation_score = 1.0
        else:
            deviation_score = max(0, 1 - abs(spectral_deviation - 0.2) / 0.2)
        
        # Score noise characteristics
        noise_level = background_noise['noise_level']
        noise_consistency = background_noise['noise_consistency']
        
        # Extremely low or high noise levels are suspicious
        if -80 <= noise_level <= -40:
            noise_level_score = 1.0
        else:
            noise_level_score = max(0, 1 - abs(noise_level + 60) / 40)
        
        # Very consistent noise is suspicious (may indicate playback)
        noise_consistency_score = 1 - noise_consistency
        
        # Score frequency band ratios
        # Check for unusual high to ultra-high frequency ratio
        # (speakers often have reduced ultra-high frequencies)
        if 'high_ultrahigh_ratio' in band_energies:
            ratio = band_energies['high_ultrahigh_ratio']
            if 1 <= ratio <= 3:
                band_ratio_score = 1.0
            else:
                band_ratio_score = max(0, 1 - abs(ratio - 2) / 2)
        else:
            band_ratio_score = 0.5  # Default if ratio couldn't be calculated
        
        # Combine scores with weights
        weights = {
            'reverb': 0.15,
            'deviation': 0.15,
            'noise_level': 0.1,
            'noise_consistency': 0.15,
            'band_ratio': 0.15,
            'noise_floor_stability': 0.1,
            'cutoff': 0.2
        }
        
        environment_score = (
            weights['reverb'] * reverb_score +
            weights['deviation'] * deviation_score +
            weights['noise_level'] * noise_level_score +
            weights['noise_consistency'] * noise_consistency_score +
            weights['band_ratio'] * band_ratio_score +
            weights['noise_floor_stability'] * noise_floor_stability +
            weights['cutoff'] * cutoff_score
        )
        
        return environment_score


class SpectrumAnalyzer:
    """
    Analyzer for spectral characteristics to detect synthetic or replayed audio.
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000, 
        n_fft: int = 2048, 
        hop_length: int = 512
    ):
        """
        Initialize the spectrum analyzer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT size
            hop_length: Hop length between frames
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def analyze_spectrum(self, audio: np.ndarray) -> Dict:
        """
        Analyze spectral characteristics for replay/synthesis detection.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with spectrum analysis results
        """
        # Calculate STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Calculate log-magnitude spectrum
        log_magnitude = librosa.amplitude_to_db(magnitude)
        
        # Calculate frequency bins
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        
        # Calculate spectral flux (frame-to-frame spectral difference)
        spectral_flux = librosa.onset.onset_strength(
            sr=self.sample_rate,
            S=magnitude,
            hop_length=self.hop_length
        )
        
        # Calculate spectral entropy (measure of spectral flatness)
        spectral_entropy = self._calculate_spectral_entropy(magnitude)
        
        # Detect harmonic artifacts that might indicate synthesis
        harmonic_artifacts = self._detect_harmonic_artifacts(magnitude, freqs)
        
        # Calculate spectral modulation features
        modulation_features = self._calculate_modulation_features(log_magnitude)
        
        # Calculate sub-band correlation features
        subband_correlation = self._calculate_subband_correlation(magnitude)
        
        # Calculate phase coherence (can indicate synthesis)
        phase_coherence = self._calculate_phase_coherence(stft)
        
        # Calculate group delay deviation (useful for detecting unnatural phase)
        group_delay_deviation = self._calculate_group_delay_deviation(stft)
        
        # Calculate spectral contrast (measure of peak-valley relationship)
        contrast = librosa.feature.spectral_contrast(S=magnitude, sr=self.sample_rate)
        spectral_contrast_mean = np.mean(contrast, axis=1)
        
        # Calculate overall liveness score based on spectral features
        liveness_score = self._calculate_liveness_score(
            spectral_flux, spectral_entropy, harmonic_artifacts,
            modulation_features, subband_correlation, phase_coherence,
            group_delay_deviation, spectral_contrast_mean
        )
        
        # Package results
        return {
            'spectral_flux_mean': np.mean(spectral_flux),
            'spectral_flux_std': np.std(spectral_flux),
            'spectral_entropy_mean': np.mean(spectral_entropy),
            'spectral_entropy_std': np.std(spectral_entropy),
            'harmonic_artifacts': harmonic_artifacts,
            'modulation_features': modulation_features,
            'subband_correlation': subband_correlation,
            'phase_coherence': phase_coherence,
            'group_delay_deviation': group_delay_deviation,
            'spectral_contrast': spectral_contrast_mean,
            'liveness_score': liveness_score
        }
    
    def _calculate_spectral_entropy(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Calculate spectral entropy for each frame.
        
        Args:
            magnitude: Magnitude spectrogram
            
        Returns:
            Spectral entropy for each frame
        """
        # Calculate entropy for each frame
        entropy = np.zeros(magnitude.shape[1])
        
        for i in range(magnitude.shape[1]):
            # Get frame spectrum
            spectrum = magnitude[:, i]
            
            # Normalize to probability distribution
            spectrum_sum = np.sum(spectrum) + 1e-10
            p = spectrum / spectrum_sum
            
            # Calculate entropy
            entropy[i] = -np.sum(p * np.log2(p + 1e-10))
        
        return entropy
    
    def _detect_harmonic_artifacts(
        self, 
        magnitude: np.ndarray, 
        freqs: np.ndarray
    ) -> float:
        """
        Detect harmonic artifacts that might indicate synthesis.
        
        Args:
            magnitude: Magnitude spectrogram
            freqs: Frequency bins
            
        Returns:
            Harmonic artifacts score (higher means more artifacts)
        """
        # Calculate average spectrum
        avg_spectrum = np.mean(magnitude, axis=1)
        
        # Normalize
        avg_spectrum = avg_spectrum / (np.max(avg_spectrum) + 1e-10)
        
        # Calculate spectral peaks
        peaks, _ = signal.find_peaks(avg_spectrum, height=0.1, distance=5)
        
        # Check if peaks form harmonic series (equally spaced in frequency)
        harmonic_score = 0.0
        
        if len(peaks) > 3:
            # Calculate intervals between adjacent peaks
            peak_intervals = np.diff(freqs[peaks])
            
            # Calculate coefficient of variation of intervals
            # (low variation suggests evenly spaced harmonics)
            cv = np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-10)
            
            # Convert to harmonic score (higher CV means less harmonic)
            harmonic_score = 1 - min(1, cv)
        
        return harmonic_score
    
    def _calculate_modulation_features(self, log_magnitude: np.ndarray) -> Dict:
        """
        Calculate spectral modulation features.
        
        Args:
            log_magnitude: Log-magnitude spectrogram
            
        Returns:
            Dictionary with modulation features
        """
        # Calculate 2D Fourier transform of log-magnitude spectrogram
        # This captures spectral and temporal modulations
        modulation_spectrum = np.abs(np.fft.fft2(log_magnitude))
        
        # Extract relevant regions (exclude DC)
        mod_spectrum = modulation_spectrum[1:self.n_fft//4, 1:log_magnitude.shape[1]//4]
        
        # Calculate modulation energy statistics
        mod_energy_mean = np.mean(mod_spectrum)
        mod_energy_std = np.std(mod_spectrum)
        
        # Calculate modulation centroid (weighted average of modulation frequencies)
        y_indices, x_indices = np.meshgrid(
            np.arange(mod_spectrum.shape[1]),
            np.arange(mod_spectrum.shape[0])
        )
        
        # Spectral modulation centroid
        spectral_mod_centroid = np.sum(x_indices * mod_spectrum) / (np.sum(mod_spectrum) + 1e-10)
        
        # Temporal modulation centroid
        temporal_mod_centroid = np.sum(y_indices * mod_spectrum) / (np.sum(mod_spectrum) + 1e-10)
        
        return {
            'modulation_energy_mean': mod_energy_mean,
            'modulation_energy_std': mod_energy_std,
            'spectral_modulation_centroid': spectral_mod_centroid,
            'temporal_modulation_centroid': temporal_mod_centroid
        }
    
    def _calculate_subband_correlation(self, magnitude: np.ndarray) -> float:
        """
        Calculate correlation between frequency subbands.
        
        Args:
            magnitude: Magnitude spectrogram
            
        Returns:
            Subband correlation score
        """
        # Define subbands (indices in spectrum)
        n_bands = 8
        band_size = magnitude.shape[0] // n_bands
        
        # Calculate mean correlation between bands
        correlations = []
        
        for i in range(n_bands - 1):
            band1 = magnitude[i*band_size:(i+1)*band_size, :]
            band1_mean = np.mean(band1, axis=0)
            
            for j in range(i+1, n_bands):
                band2 = magnitude[j*band_size:(j+1)*band_size, :]
                band2_mean = np.mean(band2, axis=0)
                
                # Calculate correlation
                correlation = np.corrcoef(band1_mean, band2_mean)[0, 1]
                correlations.append(correlation)
        
        # Average correlation
        mean_correlation = np.mean(correlations)
        
        return mean_correlation
    
    def _calculate_phase_coherence(self, stft: np.ndarray) -> float:
        """
        Calculate phase coherence to detect synthesis.
        
        Args:
            stft: STFT complex spectrogram
            
        Returns:
            Phase coherence score
        """
        # Extract phase
        phase = np.angle(stft)
        
        # Calculate phase derivative along time
        phase_diff = np.diff(phase, axis=1)
        
        # Wrap to [-pi, pi]
        phase_diff = np.angle(np.exp(1j * phase_diff))
        
        # Calculate unwrapped phase derivative statistics
        phase_diff_std = np.std(phase_diff, axis=1)
        
        # Calculate phase coherence (inverse of variability)
        # More coherent phase (lower std) may indicate synthesis
        coherence = 1 / (1 + np.mean(phase_diff_std))
        
        return coherence
    
    def _calculate_group_delay_deviation(self, stft: np.ndarray) -> float:
        """
        Calculate group delay deviation to detect unnatural phase.
        
        Args:
            stft: STFT complex spectrogram
            
        Returns:
            Group delay deviation score
        """
        # Extract magnitude and phase
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Calculate phase derivative along frequency axis (group delay)
        group_delay = np.diff(phase, axis=0)
        
        # Wrap to [-pi, pi]
        group_delay = np.angle(np.exp(1j * group_delay))
        
        # Calculate standard deviation of group delay for each frame
        group_delay_std = np.std(group_delay, axis=0)
        
        # Calculate mean group delay deviation
        mean_deviation = np.mean(group_delay_std)
        
        # Convert to score (higher deviation is more natural)
        deviation_score = min(1, mean_deviation / np.pi)
        
        return deviation_score
    
    def _calculate_liveness_score(
        self,
        spectral_flux: np.ndarray,
        spectral_entropy: np.ndarray,
        harmonic_artifacts: float,
        modulation_features: Dict,
        subband_correlation: float,
        phase_coherence: float,
        group_delay_deviation: float,
        spectral_contrast: np.ndarray
    ) -> float:
        """
        Calculate overall liveness score based on spectral features.
        
        Args:
            spectral_flux: Spectral flux
            spectral_entropy: Spectral entropy
            harmonic_artifacts: Harmonic artifacts score
            modulation_features: Modulation features
            subband_correlation: Subband correlation score
            phase_coherence: Phase coherence score
            group_delay_deviation: Group delay deviation score
            spectral_contrast: Spectral contrast
            
        Returns:
            Liveness score (higher is more likely to be live)
        """
        # Calculate individual feature scores
        
        # Spectral flux (higher flux is more natural)
        flux_mean = np.mean(spectral_flux)
        if 0.1 <= flux_mean <= 0.5:
            flux_score = 1.0
        else:
            flux_score = max(0, 1 - abs(flux_mean - 0.3) / 0.3)
        
        # Spectral entropy (natural speech has moderate entropy)
        entropy_mean = np.mean(spectral_entropy)
        if 3.0 <= entropy_mean <= 5.0:
            entropy_score = 1.0
        else:
            entropy_score = max(0, 1 - abs(entropy_mean - 4.0) / 2.0)
        
        # Harmonic artifacts (fewer artifacts is more natural)
        harmonic_score = 1 - harmonic_artifacts
        
        # Modulation features (natural speech has certain modulation patterns)
        mod_energy = modulation_features['modulation_energy_mean']
        if 0.5 <= mod_energy <= 2.0:
            mod_score = 1.0
        else:
            mod_score = max(0, 1 - abs(mod_energy - 1.25) / 1.25)
        
        # Subband correlation (natural speech has moderate correlation)
        if 0.3 <= subband_correlation <= 0.7:
            correlation_score = 1.0
        else:
            correlation_score = max(0, 1 - abs(subband_correlation - 0.5) / 0.5)
        
        # Phase coherence (less coherence is more natural)
        coherence_score = 1 - phase_coherence
        
        # Group delay deviation (more deviation is more natural)
        delay_score = group_delay_deviation
        
        # Spectral contrast (natural speech has moderate contrast)
        contrast_mean = np.mean(spectral_contrast)
        if 1.0 <= contrast_mean <= 3.0:
            contrast_score = 1.0
        else:
            contrast_score = max(0, 1 - abs(contrast_mean - 2.0) / 2.0)
        
        # Combine scores with weights
        weights = {
            'flux': 0.15,
            'entropy': 0.1,
            'harmonic': 0.15,
            'modulation': 0.1,
            'correlation': 0.1,
            'coherence': 0.15,
            'delay': 0.15,
            'contrast': 0.1
        }
        
        liveness_score = (
            weights['flux'] * flux_score +
            weights['entropy'] * entropy_score +
            weights['harmonic'] * harmonic_score +
            weights['modulation'] * mod_score +
            weights['correlation'] * correlation_score +
            weights['coherence'] * coherence_score +
            weights['delay'] * delay_score +
            weights['contrast'] * contrast_score
        )
        
        return liveness_score
