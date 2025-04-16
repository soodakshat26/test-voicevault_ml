import unittest
import numpy as np
import os
import tempfile
import shutil
from acoustic_fingerprinting.signal_decomposition import SignalDecomposer, SubAudioAnalyzer
from acoustic_fingerprinting.speaker_patterns import BiometricFeatureExtractor
from acoustic_fingerprinting.comparison import DTWComparator, VectorComparator
from liveness_detection.physiological import BreathingPatternDetector, MicroModulationAnalyzer
from challenge_response.challenge import ChallengeGenerator


class TestComponents(unittest.TestCase):
    """
    Test cases for individual components.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create test audio
        self.sample_rate = 16000
        duration = 2  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        
        # Create sine wave with amplitude modulation
        carrier = np.sin(2 * np.pi * 220 * t)  # 220 Hz
        modulator = 0.5 + 0.5 * np.sin(2 * np.pi * 8 * t)  # 8 Hz modulation
        self.test_audio = carrier * modulator
        
        # Add some noise
        self.test_audio += 0.05 * np.random.randn(len(self.test_audio))
    
    def test_signal_decomposer(self):
        """
        Test SignalDecomposer component.
        """
        decomposer = SignalDecomposer(sample_rate=self.sample_rate)
        result = decomposer.decompose_signal(self.test_audio)
        
        # Check result structure
        self.assertIn('stft_magnitude', result)
        self.assertIn('stft_phase', result)
        self.assertIn('mel_spectrogram', result)
        self.assertIn('log_mel_spectrogram', result)
        self.assertIn('wavelet_coeffs', result)
        self.assertIn('filtered_bands', result)
        self.assertIn('phase_vector', result)
    
    def test_sub_audio_analyzer(self):
        """
        Test SubAudioAnalyzer component.
        """
        analyzer = SubAudioAnalyzer(sample_rate=self.sample_rate)
        result = analyzer.extract_features(self.test_audio)
        
        # Check result structure
        self.assertIn('mfcc', result)
        self.assertIn('mfcc_delta', result)
        self.assertIn('mfcc_delta2', result)
        self.assertIn('log_mel_spectrogram', result)
        self.assertIn('chroma', result)
        self.assertIn('f0', result)
        self.assertIn('voiced_flag', result)
        self.assertIn('spectral_centroid', result)
    
    def test_biometric_feature_extractor(self):
        """
        Test BiometricFeatureExtractor component.
        """
        extractor = BiometricFeatureExtractor(sample_rate=self.sample_rate)
        features = extractor.extract_biometric_features(self.test_audio)
        
        # Check features shape
        self.assertEqual(features.shape, (extractor.feature_dim,))
    
    def test_dtw_comparator(self):
        """
        Test DTWComparator component.
        """
        # Create two test feature sequences
        seq1 = np.sin(np.linspace(0, 10, 100))
        seq2 = np.sin(np.linspace(0.5, 10.5, 110))
        
        comparator = DTWComparator()
        distance, path = comparator.compare(seq1, seq2)
        
        # Check results
        self.assertIsInstance(distance, float)
        self.assertGreater(len(path), 0)
        
        # Test normalized distance
        norm_distance = comparator.normalized_distance(seq1, seq2)
        self.assertIsInstance(norm_distance, float)
    
    def test_vector_comparator(self):
        """
        Test VectorComparator component.
        """
        # Create two test feature vectors
        vec1 = np.array([1, 2, 3, 4, 5])
        vec2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        vec3 = np.array([5, 4, 3, 2, 1])
        
        comparator = VectorComparator(method='cosine')
        
        # Similar vectors should have high similarity
        sim1 = comparator.compare(vec1, vec2)
        self.assertGreater(sim1, 0.9)
        
        # Dissimilar vectors should have low similarity
        sim2 = comparator.compare(vec1, vec3)
        self.assertLess(sim2, 0.5)
    
    def test_breathing_pattern_detector(self):
        """
        Test BreathingPatternDetector component.
        """
        detector = BreathingPatternDetector(sample_rate=self.sample_rate)
        result = detector.detect_breathing(self.test_audio)
        
        # Check result structure
        self.assertIn('envelope', result)
        self.assertIn('breathing_envelope', result)
        self.assertIn('peaks', result)
        self.assertIn('breathing_presence_score', result)
    
    def test_micro_modulation_analyzer(self):
        """
        Test MicroModulationAnalyzer component.
        """
        analyzer = MicroModulationAnalyzer(sample_rate=self.sample_rate)
        result = analyzer.analyze_micro_modulations(self.test_audio)
        
        # Check result structure
        self.assertIn('spectral_flux', result)
        self.assertIn('micro_variations', result)
        self.assertIn('phase_variation', result)
        self.assertIn('pitch_stability', result)
        self.assertIn('jitter', result)
        self.assertIn('shimmer', result)
        self.assertIn('naturalness_score', result)
    
    def test_challenge_generator(self):
        """
        Test ChallengeGenerator component.
        """
        generator = ChallengeGenerator()
        challenge = generator.generate_challenge()
        
        # Check challenge structure
        self.assertIn('session_id', challenge)
        self.assertIn('challenge_type', challenge)
        self.assertIn('challenge_text', challenge)
        self.assertIn('challenge_display', challenge)
        self.assertIn('verification_hash', challenge)
        
        # Check verification (simulate response)
        if challenge['challenge_type'] == 'digit_sequence':
            # Assume digits are directly in the challenge text
            digits = ''.join(c for c in challenge['challenge_display'] if c.isdigit())
            result = generator.verify_response(challenge['session_id'], digits)
            self.assertTrue(result.get('verified', False))


if __name__ == '__main__':
    unittest.main()
