import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time

from .physiological import BreathingPatternDetector, MicroModulationAnalyzer, ResonanceTracker
from .replay_detection import AcousticEnvironmentAnalyzer, SpectrumAnalyzer


class LivenessDetector:
    """
    Complete liveness detection system combining multiple verification methods.
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        threshold: float = 0.7,
        confidence_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the liveness detector.
        
        Args:
            sample_rate: Audio sample rate in Hz
            threshold: Liveness threshold for verification
            confidence_weights: Custom weights for each detection method
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        
        # Default weights if not provided
        self.confidence_weights = confidence_weights or {
            'breathing': 0.15,
            'micro_modulation': 0.2,
            'resonance': 0.2,
            'environment': 0.25,
            'spectrum': 0.2
        }
        
        # Initialize detector components
        self.breathing_detector = BreathingPatternDetector(sample_rate=sample_rate)
        self.micro_modulation_analyzer = MicroModulationAnalyzer(sample_rate=sample_rate)
        self.resonance_tracker = ResonanceTracker(sample_rate=sample_rate)
        self.environment_analyzer = AcousticEnvironmentAnalyzer(sample_rate=sample_rate)
        self.spectrum_analyzer = SpectrumAnalyzer(sample_rate=sample_rate)
        
        # Store recent verification results for adaptation
        self.recent_results = []
        self.max_history = 50
    
    def detect_liveness(self, audio: np.ndarray) -> Dict:
        """
        Detect liveness in audio using multiple verification methods.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with liveness detection results
        """
        # Record start time for performance measurement
        start_time = time.time()
        
        # Run all detectors
        breathing_results = self.breathing_detector.detect_breathing(audio)
        micro_modulation_results = self.micro_modulation_analyzer.analyze_micro_modulations(audio)
        resonance_results = self.resonance_tracker.track_resonances(audio)
        environment_results = self.environment_analyzer.analyze_environment(audio)
        spectrum_results = self.spectrum_analyzer.analyze_spectrum(audio)
        
        # Extract individual scores
        breathing_score = breathing_results['breathing_presence_score']
        micro_modulation_score = micro_modulation_results['naturalness_score']
        resonance_score = resonance_results['naturalness_score']
        environment_score = environment_results['environment_score']
        spectrum_score = spectrum_results['liveness_score']
        
        # Calculate combined liveness score with weights
        liveness_score = (
            self.confidence_weights['breathing'] * breathing_score +
            self.confidence_weights['micro_modulation'] * micro_modulation_score +
            self.confidence_weights['resonance'] * resonance_score +
            self.confidence_weights['environment'] * environment_score +
            self.confidence_weights['spectrum'] * spectrum_score
        )
        
        # Determine verification result
        is_live = liveness_score >= self.threshold
        
        # Calculate confidence level
        if is_live:
            confidence = (liveness_score - self.threshold) / (1 - self.threshold)
        else:
            confidence = (self.threshold - liveness_score) / self.threshold
        
        # Clamp confidence to [0, 1]
        confidence = max(0, min(1, confidence))
        
        # Calculate computation time
        computation_time = time.time() - start_time
        
        # Prepare results
        result = {
            'is_live': is_live,
            'liveness_score': liveness_score,
            'confidence': confidence,
            'threshold': self.threshold,
            'computation_time': computation_time,
            'component_scores': {
                'breathing': breathing_score,
                'micro_modulation': micro_modulation_score,
                'resonance': resonance_score,
                'environment': environment_score,
                'spectrum': spectrum_score
            },
            'details': {
                'breathing': breathing_results,
                'micro_modulation': micro_modulation_results,
                'resonance': resonance_results,
                'environment': environment_results,
                'spectrum': spectrum_results
            }
        }
        
        # Update history and adapt if needed
        self._update_history(result)
        
        return result
    
    def _update_history(self, result: Dict):
        """
        Update history of verification results.
        
        Args:
            result: Verification result
        """
        # Add result to history
        self.recent_results.append({
            'is_live': result['is_live'],
            'liveness_score': result['liveness_score'],
            'component_scores': result['component_scores']
        })
        
        # Trim history if needed
        if len(self.recent_results) > self.max_history:
            self.recent_results = self.recent_results[-self.max_history:]
    
    def adapt_weights(self):
        """
        Adapt confidence weights based on recent verification results.
        
        This method analyzes which components are most reliable and 
        adjusts their weights accordingly.
        """
        # Need enough history to adapt
        if len(self.recent_results) < 10:
            return
        
        # Extract scores
        component_scores = {
            'breathing': [],
            'micro_modulation': [],
            'resonance': [],
            'environment': [],
            'spectrum': []
        }
        
        for result in self.recent_results:
            for component, score in result['component_scores'].items():
                component_scores[component].append(score)
        
        # Calculate variance of each component
        variances = {}
        for component, scores in component_scores.items():
            variances[component] = np.var(scores)
        
        # Lower variance indicates more consistent (possibly more reliable) scores
        total_inverse_variance = sum(1 / (v + 1e-10) for v in variances.values())
        
        # Calculate new weights based on inverse variance
        new_weights = {}
        for component, var in variances.items():
            new_weights[component] = (1 / (var + 1e-10)) / total_inverse_variance
        
        # Blend with current weights (gradual adaptation)
        for component in self.confidence_weights:
            self.confidence_weights[component] = (
                0.8 * self.confidence_weights[component] +
                0.2 * new_weights[component]
            )
    
    def set_threshold(self, threshold: float):
        """
        Set the liveness threshold.
        
        Args:
            threshold: New threshold value
        """
        self.threshold = threshold
