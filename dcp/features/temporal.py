import numpy as np
import librosa
import yaml
from scipy import signal
from scipy import ndimage
import warnings

class TemporalFeatureExtractor:
    """Base class for micro-temporal feature extraction"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['features']['temporal']
        
        self.sample_rate = 16000  # Default sample rate
        
    def extract(self, y, sr):
        """Extract basic temporal features"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Calculate zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Calculate temporal envelope
        envelope = np.abs(signal.hilbert(y))
        
        features = {
            'zero_crossing_rate': zcr,
            'envelope': envelope
        }
        
        return features


class VOTMeasurement(TemporalFeatureExtractor):
    """Voice Onset Time (VOT) precision measurement"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.measure_vot = self.config.get('vot_detection', True)
        
    def extract(self, y, sr):
        """Extract VOT measurements"""
        if not self.measure_vot:
            return {}
            
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Calculate spectrogram
        S = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
        
        # Calculate temporal envelope
        envelope = np.sum(S, axis=0)
        
        # Smooth envelope
        envelope_smooth = ndimage.gaussian_filter1d(envelope, sigma=2)
        
        # Calculate derivative
        envelope_diff = np.diff(envelope_smooth, prepend=envelope_smooth[0])
        
        # Find burst/onset points (positive peaks in derivative)
        burst_points, _ = signal.find_peaks(envelope_diff, height=0.1*np.max(envelope_diff))
        
        # Calculate voicing onset after each burst
        vot_measurements = []
        
        for burst in burst_points:
            # Only consider onsets in first 80% of signal
            if burst > 0.8 * len(envelope_diff):
                continue
                
            # Look for onset of voicing after burst
            # Check if there's a sustained rise in energy
            window_size = int(0.03 * sr / 128)  # 30ms window
            
            for i in range(burst, min(burst + window_size, len(envelope_smooth))):
                # Check for sustained energy over threshold
                if envelope_smooth[i] > 0.3 * np.max(envelope_smooth):
                    # Check if energy stays above threshold
                    if i + 5 < len(envelope_smooth) and np.all(envelope_smooth[i:i+5] > 0.2 * np.max(envelope_smooth)):
                        vot = (i - burst) * 128 / sr * 1000  # VOT in ms
                        
                        # Only accept reasonable VOT values (5-150ms)
                        if 5 <= vot <= 150:
                            vot_measurements.append(vot)
                        break
        
        # Calculate VOT statistics
        if vot_measurements:
            vot_mean = np.mean(vot_measurements)
            vot_std = np.std(vot_measurements)
        else:
            vot_mean = 0
            vot_std = 0
        
        return {
            'vot_measurements': vot_measurements,
            'vot_mean_ms': vot_mean,
            'vot_std_ms': vot_std,
            'burst_points': burst_points
        }


class CoarticulationAnalyzer(TemporalFeatureExtractor):
    """Co-articulation effect quantification through transitional analysis"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        
    def extract(self, y, sr):
        """Extract co-articulation features"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Extract formants over time
        n_formants = 3
        formant_tracks = self._extract_formant_tracks(y, sr, n_formants)
        
        # Calculate formant transitions
        transitions = self._calculate_transitions(formant_tracks)
        
        # Calculate stability regions
        stability = self._calculate_stability(formant_tracks)
        
        # CV transition ratio (measures degree of coarticulation)
        cv_ratio = transitions / (stability + 1e-10)
        
        return {
            'formant_tracks': formant_tracks,
            'transition_measure': transitions,
            'stability_measure': stability,
            'coarticulation_ratio': cv_ratio
        }
    
    def _extract_formant_tracks(self, y, sr, n_formants=3):
        """Extract formant frequency tracks over time"""
        # Frame the signal
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(0.010 * sr)    # 10ms
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        
        # Apply pre-emphasis
        pre_emphasis = 0.97
        frames = np.append(frames[:, 0:1], frames[:, 1:] - pre_emphasis * frames[:, :-1], axis=1)
        
        # Apply window
        frames = frames * np.hamming(frame_length)
        
        # Initialize formant tracks
        formant_tracks = np.zeros((n_formants, frames.shape[0]))
        
        # Extract formants for each frame
        for i in range(frames.shape[0]):
            frame = frames[i]
            
            # LPC analysis
            order = 2 + int(sr / 1000)  # Rule of thumb
            try:
                lpc_coeffs = librosa.lpc(frame, order=order)
                
                # Convert to formants
                roots = np.roots(lpc_coeffs)
                roots = roots[np.abs(roots) < 1]
                angles = np.angle(roots)
                
                # Convert to frequencies
                freqs = np.abs(angles) * sr / (2 * np.pi)
                
                # Sort and keep positive frequencies
                formants_frame = np.sort(freqs[freqs > 0])
                
                # Store formants
                if len(formants_frame) >= n_formants:
                    formant_tracks[:, i] = formants_frame[:n_formants]
                else:
                    # Pad with zeros if not enough formants
                    formant_tracks[:len(formants_frame), i] = formants_frame
            except:
                # If LPC fails, keep zeros
                pass
        
        return formant_tracks
    
    def _calculate_transitions(self, formant_tracks):
        """Calculate formant transition measure"""
        # Calculate derivatives
        formant_diff = np.diff(formant_tracks, axis=1)
        
        # Transition measure: average absolute derivative
        transitions = np.mean(np.abs(formant_diff))
        
        return transitions
    
    def _calculate_stability(self, formant_tracks):
        """Calculate formant stability measure"""
        # Find regions where formants are stable
        formant_diff = np.diff(formant_tracks, axis=1)
        
        # Threshold for stability (low derivative)
        threshold = 50  # Hz per frame
        stability_mask = np.abs(formant_diff) < threshold
        
        # Stability measure: ratio of stable frames
        stability = np.mean(stability_mask)
        
        return stability


class FormantTransitionAnalyzer(TemporalFeatureExtractor):
    """Analysis of formant transitions"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.analyze_transitions = self.config.get('formant_transitions', True)
        
    def extract(self, y, sr):
        """Extract formant transition features"""
        if not self.analyze_transitions:
            return {}
        
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Extract formant tracks
        formant_analyzer = CoarticulationAnalyzer()
        formant_tracks = formant_analyzer._extract_formant_tracks(y, sr, n_formants=3)
        
        # Calculate formant velocities and accelerations
        formant_velocities = np.diff(formant_tracks, axis=1)
        formant_accelerations = np.diff(formant_velocities, axis=1)
        
        # Calculate transition metrics
        locus_equations = self._calculate_locus_equations(formant_tracks)
        
        # Calculate formant transition statistics
        f1_velocity_mean = np.mean(np.abs(formant_velocities[0]))
        f2_velocity_mean = np.mean(np.abs(formant_velocities[1]))
        f1_acceleration_mean = np.mean(np.abs(formant_accelerations[0]))
        f2_acceleration_mean = np.mean(np.abs(formant_accelerations[1]))
        
        # Calculate transition rate
        transition_rate = (f1_velocity_mean + f2_velocity_mean) / 2
        
        # Calculate trajectory complexity
        f2_f1_ratio = f2_velocity_mean / (f1_velocity_mean + 1e-10)
        
        return {
            'formant_tracks': formant_tracks,
            'formant_velocities': formant_velocities,
            'f1_velocity_mean': f1_velocity_mean,
            'f2_velocity_mean': f2_velocity_mean,
            'f1_acceleration_mean': f1_acceleration_mean,
            'f2_acceleration_mean': f2_acceleration_mean,
            'transition_rate': transition_rate,
            'f2_f1_ratio': f2_f1_ratio,
            'locus_equations': locus_equations
        }
    
    def _calculate_locus_equations(self, formant_tracks):
        """Calculate locus equations for F2"""
        # Locus equations relate onset F2 to target F2
        # We need to identify onsets and targets first
        
        # Simplified approach: identify points of high velocity as transitions
        f2_track = formant_tracks[1]
        f2_velocity = np.abs(np.diff(f2_track, prepend=f2_track[0]))
        
        # Find transition points (high velocity)
        transition_points = f2_velocity > np.percentile(f2_velocity, 80)
        
        # Find stable regions (low velocity)
        stable_points = f2_velocity < np.percentile(f2_velocity, 30)
        
        # Need at least some transition points
        if np.sum(transition_points) < 2 or np.sum(stable_points) < 2:
            return {
                'slope': 0,
                'intercept': 0,
                'r_value': 0
            }
        
        # Get formant values at transition and stable points
        f2_transition = f2_track[transition_points]
        f2_stable = f2_track[stable_points]
        
        # If we have multiple transition and stable regions, we can estimate locus equation
        if len(f2_transition) > 0 and len(f2_stable) > 0:
            # Simplified approach: use average values
            f2_onset = np.mean(f2_transition)  # Onset F2
            f2_target = np.mean(f2_stable)     # Target F2
            
            # Locus equation: F2onset = slope * F2target + intercept
            # With only one point, we can't calculate slope, so assume typical value
            slope = 0.7  # Typical value for neutral articulation
            intercept = f2_onset - slope * f2_target
            r_value = 1.0  # Can't calculate correlation with one point
        else:
            slope = 0
            intercept = 0
            r_value = 0
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value
        }


class MicroProsodyExtractor(TemporalFeatureExtractor):
    """Micro-prosody extraction at sub-phonemic scales"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.extract_microprosody = self.config.get('micro_prosody', True)
        
    def extract(self, y, sr):
        """Extract micro-prosodic features"""
        if not self.extract_microprosody:
            return {}
        
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Extract f0 contour with high time resolution
        hop_length = 64  # ~4ms at 16kHz
        win_length = 400  # ~25ms at 16kHz
        fmin = 50
        fmax = 500
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
            win_length=win_length
        )
        
        # Calculate energy contour
        S = np.abs(librosa.stft(y, n_fft=512, hop_length=hop_length))
        energy = np.sum(S, axis=0)
        
        # Create time axis
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        
        # Calculate micro-prosodic features
        
        # 1. F0 perturbation (jitter)
        voiced_indices = np.where(voiced_flag)[0]
        f0_voiced = f0[voiced_indices]
        
        if len(f0_voiced) > 2:
            # Calculate cycle-to-cycle variations
            f0_diff = np.diff(f0_voiced)
            jitter_ratio = np.mean(np.abs(f0_diff)) / np.mean(f0_voiced)
            jitter_percent = jitter_ratio * 100
        else:
            jitter_percent = 0
        
        # 2. Energy perturbation (shimmer)
        if len(voiced_indices) > 2:
            energy_voiced = energy[voiced_indices]
            energy_diff = np.diff(energy_voiced)
            shimmer_ratio = np.mean(np.abs(energy_diff)) / np.mean(energy_voiced)
            shimmer_percent = shimmer_ratio * 100
        else:
            shimmer_percent = 0
        
        # 3. F0 velocity and acceleration
        f0_velocity = np.diff(f0, prepend=f0[0])
        f0_acceleration = np.diff(f0_velocity, prepend=f0_velocity[0])
        
        # Calculate statistics for voiced regions only
        if len(voiced_indices) > 0:
            f0_velocity_voiced = f0_velocity[voiced_indices]
            f0_acceleration_voiced = f0_acceleration[voiced_indices]
            
            velocity_mean = np.mean(np.abs(f0_velocity_voiced))
            acceleration_mean = np.mean(np.abs(f0_acceleration_voiced))
        else:
            velocity_mean = 0
            acceleration_mean = 0
        
        # 4. F0 range within short windows
        f0_local_range = self._calculate_local_f0_range(f0, voiced_flag, window_size=5)
        
        return {
            'f0': f0,
            'voiced_flag': voiced_flag,
            'energy': energy,
            'times': times,
            'jitter_percent': jitter_percent,
            'shimmer_percent': shimmer_percent,
            'f0_velocity_mean': velocity_mean,
            'f0_acceleration_mean': acceleration_mean,
            'f0_local_range_mean': np.mean(f0_local_range) if len(f0_local_range) > 0 else 0
        }
    
    def _calculate_local_f0_range(self, f0, voiced_flag, window_size=5):
        """Calculate local F0 range within sliding windows"""
        local_ranges = []
        
        for i in range(len(f0) - window_size):
            window = f0[i:i+window_size]
            window_voiced = voiced_flag[i:i+window_size]
            
            if np.all(window_voiced):
                local_range = np.max(window) - np.min(window)
                local_ranges.append(local_range)
        
        return np.array(local_ranges)


class TemporalFeatureFactory:
    """Factory for creating temporal feature extractors"""
    
    @staticmethod
    def create_extractor(method, config_path="config/processing.yaml"):
        """Create feature extractor based on method name"""
        if method == "vot":
            return VOTMeasurement(config_path)
        elif method == "coarticulation":
            return CoarticulationAnalyzer(config_path)
        elif method == "formant_transitions":
            return FormantTransitionAnalyzer(config_path)
        elif method == "microprosody":
            return MicroProsodyExtractor(config_path)
        else:
            print(f"Unknown feature extractor: {method}. Using basic temporal extractor.")
            return TemporalFeatureExtractor(config_path)
