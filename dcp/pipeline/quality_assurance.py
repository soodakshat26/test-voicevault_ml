import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import warnings
from scipy import stats
import librosa
import soundfile as sf
import os

class QualityAssurance:
    """Quality assurance system for voice processing"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Define thresholds for quality metrics
        self.thresholds = {
            'min_snr': 15.0,  # dB
            'min_duration': 0.5,  # seconds
            'max_duration': 30.0,  # seconds
            'max_clipping_ratio': 0.01,  # maximum ratio of clipped samples
            'min_speech_activity': 0.3,  # minimum ratio of speech vs silence
            'max_dc_offset': 0.1,  # maximum allowed DC offset
            'min_sample_rate': 16000,  # Hz
            'min_rms': 0.01,  # minimum signal RMS
            'stability_threshold': 0.7,  # feature stability threshold
        }
    
    def check_audio_quality(self, y, sr):
        """
        Check the quality of audio recording
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
            
        Returns:
        --------
        quality_metrics : dict
            Quality metrics and checks
        """
        metrics = {}
        
        # Duration check
        duration = len(y) / sr
        metrics['duration'] = duration
        metrics['duration_check'] = self.thresholds['min_duration'] <= duration <= self.thresholds['max_duration']
        
        # Sample rate check
        metrics['sample_rate'] = sr
        metrics['sample_rate_check'] = sr >= self.thresholds['min_sample_rate']
        
        # RMS energy check
        rms = np.sqrt(np.mean(y**2))
        metrics['rms'] = float(rms)
        metrics['rms_check'] = rms >= self.thresholds['min_rms']
        
        # Clipping check
        clipping_ratio = np.sum(np.abs(y) > 0.99) / len(y)
        metrics['clipping_ratio'] = float(clipping_ratio)
        metrics['clipping_check'] = clipping_ratio <= self.thresholds['max_clipping_ratio']
        
        # DC offset check
        dc_offset = np.mean(y)
        metrics['dc_offset'] = float(dc_offset)
        metrics['dc_offset_check'] = abs(dc_offset) <= self.thresholds['max_dc_offset']
        
        # SNR estimation (using percentile method)
        noise_floor = np.percentile(np.abs(y), 10)
        signal_level = np.percentile(np.abs(y), 90)
        snr_estimate = 20 * np.log10(signal_level / (noise_floor + 1e-10))
        metrics['estimated_snr'] = float(snr_estimate)
        metrics['snr_check'] = snr_estimate >= self.thresholds['min_snr']
        
        # Speech activity check
        # Use simple energy-based method
        frames = librosa.util.frame(y, frame_length=int(0.025 * sr), hop_length=int(0.01 * sr))
        frame_energies = np.sum(frames**2, axis=0)
        threshold = 0.1 * np.max(frame_energies)
        speech_ratio = np.sum(frame_energies > threshold) / len(frame_energies)
        metrics['speech_ratio'] = float(speech_ratio)
        metrics['speech_ratio_check'] = speech_ratio >= self.thresholds['min_speech_activity']
        
        # Overall quality assessment
        checks = [value for key, value in metrics.items() if key.endswith('_check')]
        metrics['overall_quality'] = float(sum(checks) / len(checks))
        metrics['pass'] = all(checks)
        
        return metrics
    
    def validate_features(self, features):
        """
        Validate extracted features for stability and sanity
        
        Parameters:
        -----------
        features : dict
            Extracted features
            
        Returns:
        --------
        validation : dict
            Validation results
        """
        validation = {}
        
        # Check for NaN or infinite values
        has_nan = self._check_nan_inf(features)
        validation['has_nan_or_inf'] = has_nan
        
        # Check feature stability where applicable
        stability_checks = {}
        
        # F0 stability check for prosodic features
        if 'prosodic' in features and 'f0' in features['prosodic']:
            f0 = features['prosodic']['f0']
            if isinstance(f0, np.ndarray) and len(f0) > 0:
                # Filter out zeros and NaN values
                valid_f0 = f0[np.logical_and(f0 > 0, ~np.isnan(f0))]
                if len(valid_f0) > 0:
                    # Stability: CV (coefficient of variation)
                    cv = np.std(valid_f0) / np.mean(valid_f0)
                    stability_checks['f0_stability'] = float(cv < 0.5)  # CV should be low for stable F0
                    stability_checks['f0_cv'] = float(cv)
        
        # Formant stability check for glottal features
        if 'glottal' in features and 'formants' in features['glottal']:
            formants = features['glottal']['formants']
            if isinstance(formants, np.ndarray) and len(formants) > 0:
                stability_checks['formants_stability'] = float(np.all(formants > 0))
        
        # MFCC stability check
        if 'acoustic' in features and 'mfcc' in features['acoustic']:
            mfcc = features['acoustic']['mfcc']
            if isinstance(mfcc, np.ndarray) and mfcc.size > 0:
                # Check standard deviation across frames
                if mfcc.ndim > 1:
                    frame_stdev = np.std(mfcc, axis=1)
                    stability_checks['mfcc_stability'] = float(np.mean(frame_stdev) < 3.0)
        
        validation['stability_checks'] = stability_checks
        
        # Calculate overall stability score
        if stability_checks:
            validation['stability_score'] = float(sum(stability_checks.values()) / len(stability_checks))
            validation['stability_pass'] = validation['stability_score'] >= self.thresholds['stability_threshold']
        else:
            validation['stability_score'] = 0.0
            validation['stability_pass'] = False
        
        # Overall validation result
        validation['pass'] = not has_nan and validation['stability_pass']
        
        return validation
    
    def _check_nan_inf(self, obj):
        """Recursively check for NaN or infinite values in a nested structure"""
        if isinstance(obj, dict):
            return any(self._check_nan_inf(v) for v in obj.values())
        elif isinstance(obj, list):
            return any(self._check_nan_inf(v) for v in obj)
        elif isinstance(obj, np.ndarray):
            return np.any(np.isnan(obj)) or np.any(np.isinf(obj))
        elif isinstance(obj, (float, np.number)):
            return np.isnan(obj) or np.isinf(obj)
        else:
            return False
    
    def generate_quality_report(self, audio_path, features, output_dir):
        """
        Generate a comprehensive quality report for an audio file and its features
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file
        features : dict
            Extracted features
        output_dir : str
            Directory to save the report
            
        Returns:
        --------
        report_path : str
            Path to the generated report
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Check audio quality
        audio_quality = self.check_audio_quality(y, sr)
        
        # Validate features
        feature_validation = self.validate_features(features)
        
        # Create report filename
        basename = os.path.basename(audio_path).split('.')[0]
        report_path = os.path.join(output_dir, f"{basename}_quality_report.txt")
        
        # Generate report
        with open(report_path, 'w') as f:
            f.write("# VoiceVault Quality Assurance Report\n\n")
            
            f.write("## Audio Quality Metrics\n\n")
            f.write(f"- File: {audio_path}\n")
            f.write(f"- Duration: {audio_quality['duration']:.2f} seconds "
                   f"({'PASS' if audio_quality['duration_check'] else 'FAIL'})\n")
            f.write(f"- Sample Rate: {audio_quality['sample_rate']} Hz "
                   f"({'PASS' if audio_quality['sample_rate_check'] else 'FAIL'})\n")
            f.write(f"- RMS Level: {audio_quality['rms']:.6f} "
                   f"({'PASS' if audio_quality['rms_check'] else 'FAIL'})\n")
            f.write(f"- Clipping Ratio: {audio_quality['clipping_ratio']:.6f} "
                   f"({'PASS' if audio_quality['clipping_check'] else 'FAIL'})\n")
            f.write(f"- DC Offset: {audio_quality['dc_offset']:.6f} "
                   f"({'PASS' if audio_quality['dc_offset_check'] else 'FAIL'})\n")
            f.write(f"- Estimated SNR: {audio_quality['estimated_snr']:.2f} dB "
                   f"({'PASS' if audio_quality['snr_check'] else 'FAIL'})\n")
            f.write(f"- Speech Activity Ratio: {audio_quality['speech_ratio']:.2f} "
                   f"({'PASS' if audio_quality['speech_ratio_check'] else 'FAIL'})\n")
            f.write(f"- Overall Audio Quality: {audio_quality['overall_quality']:.2f} "
                   f"({'PASS' if audio_quality['pass'] else 'FAIL'})\n\n")
            
            f.write("## Feature Validation\n\n")
            f.write(f"- NaN/Inf Check: {'PASS' if not feature_validation['has_nan_or_inf'] else 'FAIL'}\n")
            
            f.write("- Stability Checks:\n")
            for key, value in feature_validation['stability_checks'].items():
                f.write(f"  - {key}: {value:.2f}\n")
            
            f.write(f"- Stability Score: {feature_validation['stability_score']:.2f} "
                   f"({'PASS' if feature_validation['stability_pass'] else 'FAIL'})\n")
            
            f.write(f"\n## Overall Assessment\n\n")
            overall_pass = audio_quality['pass'] and feature_validation['pass']
            f.write(f"- Overall Quality: {'PASS' if overall_pass else 'FAIL'}\n")
            
            if not overall_pass:
                f.write("\n## Recommendations\n\n")
                if not audio_quality['pass']:
                    f.write("- Audio quality issues detected. Consider re-recording with:\n")
                    if not audio_quality['snr_check']:
                        f.write("  - Lower background noise or higher signal level\n")
                    if not audio_quality['clipping_check']:
                        f.write("  - Lower recording gain to prevent clipping\n")
                    if not audio_quality['speech_ratio_check']:
                        f.write("  - More continuous speech with fewer silences\n")
                
                if not feature_validation['pass']:
                    f.write("- Feature stability issues detected. Consider:\n")
                    f.write("  - Checking for audio artifacts or noise\n")
                    f.write("  - More consistent speaking style\n")
        
        print(f"Quality report saved to {report_path}")
        return report_path
