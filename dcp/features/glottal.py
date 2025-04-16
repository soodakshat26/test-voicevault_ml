import numpy as np
import librosa
import yaml
from scipy import signal
from scipy import linalg
import warnings

class GlottalFeatureExtractor:
    """Base class for glottal source feature extraction"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['features']['glottal']
        
        self.sample_rate = 16000  # Default sample rate
        
    def extract(self, y, sr):
        """Extract basic glottal features"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Basic LPC analysis
        order = 12
        lpc_coeffs = librosa.lpc(y, order=order)
        
        # Formant estimation (using LPC roots)
        roots = np.roots(lpc_coeffs)
        roots = roots[np.abs(roots) < 1]
        angles = np.angle(roots)
        
        # Convert to Hz
        formants = angles * sr / (2 * np.pi)
        formants = formants[formants > 0]
        formants = np.sort(formants)
        
        # Keep only vocal tract formants
        if len(formants) > 3:
            formants = formants[:3]
        
        features = {
            'lpc_coeffs': lpc_coeffs,
            'formants': formants
        }
        
        return features


class InverseFilteringExtractor(GlottalFeatureExtractor):
    """Glottal inverse filtering for source-filter separation"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.inverse_filtering = self.config.get('inverse_filtering', 'iaif')
        self.model_order = self.config.get('model_order', 24)
        
    def extract(self, y, sr):
        """Extract glottal source using inverse filtering"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        if self.inverse_filtering == 'iaif':
            glottal_source, vocal_tract = self._iaif(y, sr)
        else:
            # Simple LPC inverse filtering as fallback
            glottal_source, vocal_tract = self._simple_inverse_filter(y, sr)
        
        # Calculate glottal features
        glottal_features = self._extract_glottal_features(glottal_source, sr)
        
        return {
            'glottal_source': glottal_source,
            'vocal_tract_filter': vocal_tract,
            **glottal_features
        }
    
    def _simple_inverse_filter(self, y, sr):
        """Simple LPC inverse filtering"""
        # Apply window
        y_windowed = y * np.hamming(len(y))
        
        # LPC analysis
        lpc_coeffs = librosa.lpc(y_windowed, order=self.model_order)
        
        # Inverse filtering
        glottal_source = np.zeros_like(y)
        for i in range(len(y) - self.model_order):
            glottal_source[i + self.model_order] = y[i + self.model_order] - np.sum(
                lpc_coeffs[1:] * y[i + self.model_order - 1:i:-1]
            )
        
        return glottal_source, lpc_coeffs
    
    def _iaif(self, y, sr):
        """Iterative Adaptive Inverse Filtering"""
        # Implementation of IAIF algorithm
        # Based on: Alku, P. (1992). Glottal wave analysis with Pitch Synchronous Iterative Adaptive Inverse Filtering
        
        # Apply window
        y_windowed = y * np.hamming(len(y))
        
        # Initial estimate of glottal contribution (1st order LPC)
        g1_order = 1
        g1_coeffs = librosa.lpc(y_windowed, order=g1_order)
        
        # Remove initial glottal contribution
        y1 = signal.lfilter(g1_coeffs, 1, y_windowed)
        
        # Estimate vocal tract (high order LPC)
        vt_order = int(sr / 1000) + 4  # Rule of thumb
        vt_coeffs = librosa.lpc(y1, order=vt_order)
        
        # Remove vocal tract contribution to get updated glottal estimate
        y2 = signal.lfilter(vt_coeffs, 1, y_windowed)
        
        # Final glottal source model (2nd order LPC)
        g2_order = 4
        g2_coeffs = librosa.lpc(y2, order=g2_order)
        
        # Final inverse filtering - use scipy.signal.lfilter
        # First remove glottal contribution
        y_no_glottal = signal.lfilter(g2_coeffs, 1, y_windowed)
        
        # Then remove vocal tract contribution to get glottal source
        glottal_source = signal.lfilter(vt_coeffs, 1, y_no_glottal)
        
        return glottal_source, vt_coeffs

    
    def _extract_glottal_features(self, glottal_source, sr):
        """Extract features from glottal source signal"""
        # Find glottal closure instants (GCIs)
        gci = self._detect_gci(glottal_source, sr)
        
        if len(gci) < 2:
            return {
                'naq': 0,
                'oq': 0,
                'hrf': 0,
                'gci_count': 0
            }
        
        # Calculate NAQ (Normalized Amplitude Quotient)
        naq_values = []
        oq_values = []  # Open quotient
        hrf_values = []  # Harmonic richness factor
        
        for i in range(len(gci) - 1):
            start = gci[i]
            end = gci[i+1]
            
            if end - start < 5:  # Too short, skip
                continue
            
            # Extract glottal cycle
            cycle = glottal_source[start:end]
            
            # Find peak (maximum flow)
            peak_idx = np.argmax(cycle)
            peak_value = cycle[peak_idx]
            
            # Find minimum flow derivative
            cycle_diff = np.diff(cycle, prepend=cycle[0])
            min_diff_idx = np.argmin(cycle_diff)
            min_diff_value = cycle_diff[min_diff_idx]
            
            # Calculate NAQ
            if min_diff_value < 0:
                naq = peak_value / (abs(min_diff_value) * (end - start) / sr)
                naq_values.append(naq)
            
            # Calculate Open Quotient
            # Find opening instant (zero crossing before peak)
            opening_idx = 0
            for j in range(peak_idx, 0, -1):
                if cycle[j] <= 0 and cycle[j-1] >= 0:
                    opening_idx = j
                    break
            
            # Calculate OQ if opening found
            if opening_idx > 0:
                oq = (end - opening_idx) / (end - start)
                oq_values.append(oq)
            
            # Calculate Harmonic Richness Factor
            if len(cycle) >= 16:
                # Calculate spectrum
                spec = np.abs(np.fft.rfft(cycle * np.hamming(len(cycle))))
                freq = np.fft.rfftfreq(len(cycle), 1/sr)
                
                # Find f0 and harmonic peaks
                f0_estimated = sr / (end - start)
                harmonic_idxs = []
                
                for harmonic in range(1, 6):
                    target_freq = harmonic * f0_estimated
                    idx = np.argmin(np.abs(freq - target_freq))
                    harmonic_idxs.append(idx)
                
                if len(harmonic_idxs) >= 2:
                    # HRF = ratio of harmonics above f0 to f0
                    hrf = np.sum(spec[harmonic_idxs[1:]]) / (spec[harmonic_idxs[0]] + 1e-10)
                    hrf_values.append(hrf)
        
        # Calculate average values
        naq_mean = np.mean(naq_values) if naq_values else 0
        oq_mean = np.mean(oq_values) if oq_values else 0
        hrf_mean = np.mean(hrf_values) if hrf_values else 0
        
        return {
            'naq': naq_mean,
            'oq': oq_mean,
            'hrf': hrf_mean,
            'gci_count': len(gci)
        }
    
    def _detect_gci(self, glottal_source, sr):
        """Detect Glottal Closure Instants from glottal source signal"""
        # Calculate negative derivative of glottal flow
        neg_deriv = -np.diff(glottal_source, prepend=glottal_source[0])
        
        # Find peaks in negative derivative (glottal closures)
        peaks, _ = signal.find_peaks(neg_deriv, height=0.05*np.max(neg_deriv), distance=sr/500)
        
        return peaks


class ClosedPhaseAnalyzer(GlottalFeatureExtractor):
    """Quasi-closed phase analysis for glottal flow estimation"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.do_closed_phase = self.config.get('closed_phase_analysis', True)
        
    def extract(self, y, sr):
        """Extract features using closed phase analysis"""
        if not self.do_closed_phase:
            return {}
        
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Detect pitch
        f0, voiced_flag, _ = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Find GCIs using a simplified method
        # A complete implementation would use SEDREAMS or similar
        neg_diff = -np.diff(y, prepend=y[0])
        gci_candidates, _ = signal.find_peaks(neg_diff, height=0.1*np.max(neg_diff))
        
        # Refine GCIs using f0 information
        gci = self._refine_gci(gci_candidates, f0, voiced_flag, sr)
        
        if len(gci) < 2:
            return {'closed_phase_formants': []}
        
        # Perform closed phase analysis
        closed_phase_formants = []
        
        for i in range(len(gci) - 1):
            # Define closed phase region (30% after GCI)
            start = gci[i]
            end = start + int(0.3 * (gci[i+1] - gci[i]))
            
            if end - start < 20:  # Too short, skip
                continue
            
            # Extract closed phase segment
            segment = y[start:end]
            
            # Apply window
            segment = segment * np.hamming(len(segment))
            
            # LPC analysis during closed phase
            try:
                order = min(12, len(segment) - 1)
                lpc_coeffs = librosa.lpc(segment, order=order)
                
                # Convert LPC to formants
                roots = np.roots(lpc_coeffs)
                roots = roots[np.abs(roots) < 1]
                angles = np.angle(roots)
                
                # Convert to Hz and keep positive frequencies
                formants = angles[angles > 0] * sr / (2 * np.pi)
                formants = np.sort(formants)
                
                # Keep only the first 3 formants
                if len(formants) > 3:
                    formants = formants[:3]
                
                closed_phase_formants.append(formants)
            except:
                pass
        
        # Average the formants across all closed phases
        if closed_phase_formants:
            mean_formants = np.mean(np.array(closed_phase_formants), axis=0)
        else:
            mean_formants = np.array([])
        
        return {
            'closed_phase_formants': mean_formants,
            'gci': gci
        }
    
    def _refine_gci(self, gci_candidates, f0, voiced_flag, sr):
        """Refine GCI candidates using F0 information"""
        refined_gci = []
        
        # Create continuous F0 function
        f0_times = np.arange(len(f0)) * (len(voiced_flag) / sr) / len(f0)
        f0_continuous = np.copy(f0)
        f0_continuous[~voiced_flag] = 0
        
        # Interpolate to fill gaps
        for i in range(1, len(f0_continuous) - 1):
            if f0_continuous[i] == 0 and f0_continuous[i-1] > 0 and f0_continuous[i+1] > 0:
                f0_continuous[i] = (f0_continuous[i-1] + f0_continuous[i+1]) / 2
        
        # Process each candidate
        prev_gci = 0
        
        for gci in gci_candidates:
            # Skip unvoiced regions
            gci_time = gci / sr
            idx = int(gci_time * len(f0) / (len(voiced_flag) / sr))
            idx = min(idx, len(f0_continuous) - 1)
            
            if f0_continuous[idx] == 0:
                continue
            
            # Check if this GCI is consistent with local F0
            expected_period = sr / f0_continuous[idx]
            
            # Accept if first or close to expected period
            if prev_gci == 0 or 0.7 * expected_period <= gci - prev_gci <= 1.3 * expected_period:
                refined_gci.append(gci)
                prev_gci = gci
        
        return np.array(refined_gci)


class TubeModelEstimator(GlottalFeatureExtractor):
    """Tube model estimation of vocal tract configuration"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        self.tube_sections = 8  # Number of tube sections
        
    def extract(self, y, sr):
        """Extract vocal tract tube model parameters"""
        # Resample if needed
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Apply window
        y_windowed = y * np.hamming(len(y))
        
        # Perform LPC analysis
        order = 2 * self.tube_sections
        lpc_coeffs = librosa.lpc(y_windowed, order=order)
        
        # Convert LPC coefficients to reflection coefficients (PARCOR)
        # Using Levinson-Durbin recursion outcome
        reflection_coeffs = self._lpc_to_reflection(lpc_coeffs)
        
        # Convert reflection coefficients to area ratios
        area_ratios = (1 - reflection_coeffs) / (1 + reflection_coeffs)
        
        # Normalize area ratios
        area_ratios = area_ratios / area_ratios[0]
        
        # Calculate formants from LPC coefficients
        formants = self._lpc_to_formants(lpc_coeffs, sr)
        
        return {
            'tube_areas': area_ratios,
            'reflection_coeffs': reflection_coeffs,
            'formants': formants
        }
    
    def _lpc_to_reflection(self, lpc_coeffs):
        """Convert LPC coefficients to reflection coefficients"""
        # This is simplified - normally comes directly from Levinson recursion
        # Using Schur algorithm to compute reflection coefficients
        
        a = lpc_coeffs.copy()
        K = np.zeros(len(a) - 1)
        
        for i in range(len(a) - 1, 0, -1):
            K[i-1] = a[i]
            
            for j in range(1, i):
                a[j] = (a[j] - K[i-1] * a[i-j]) / (1 - K[i-1]**2)
        
        return K
    
    def _lpc_to_formants(self, lpc_coeffs, sr):
        """Convert LPC coefficients to formant frequencies"""
        # Calculate roots of the LPC polynomial
        roots = np.roots(lpc_coeffs)
        
        # Keep only roots with magnitude < 1 (stable)
        roots = roots[np.abs(roots) < 1]
        
        # Convert to frequency and bandwidth
        angles = np.angle(roots)
        frequencies = np.abs(angles) * sr / (2 * np.pi)
        
        # Sort by frequency
        frequencies = np.sort(frequencies)
        
        # Keep only positive frequencies
        frequencies = frequencies[frequencies > 0]
        
        # Keep only first few formants
        if len(frequencies) > 5:
            frequencies = frequencies[:5]
        
        return frequencies


class GlottalFeatureFactory:
    """Factory for creating glottal feature extractors"""
    
    @staticmethod
    def create_extractor(method, config_path="config/processing.yaml"):
        """Create feature extractor based on method name"""
        if method == "inverse_filtering":
            return InverseFilteringExtractor(config_path)
        elif method == "closed_phase":
            return ClosedPhaseAnalyzer(config_path)
        elif method == "tube_model":
            return TubeModelEstimator(config_path)
        else:
            print(f"Unknown feature extractor: {method}. Using basic glottal extractor.")
            return GlottalFeatureExtractor(config_path)
