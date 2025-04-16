import unittest
import numpy as np
import os
import tempfile
import shutil
from voicevault.voicevault import VoiceVault
from utils.audio_utils import load_audio


class TestVoiceVault(unittest.TestCase):
    """
    Test cases for VoiceVault system.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create temporary directory for data
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize VoiceVault with test settings
        self.voicevault = VoiceVault(
            storage_dir=self.temp_dir,
            verification_threshold=0.6,  # Lower threshold for testing
            liveness_threshold=0.5       # Lower threshold for testing
        )
        
        # Generate some test audio (sine wave with different frequencies)
        sample_rate = 16000
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Create two distinct "voices" (different frequencies)
        self.speaker1_samples = []
        self.speaker2_samples = []
        
        # Speaker 1: 200Hz sine wave
        for i in range(3):
            # Add some variation
            freq = 200 + i * 10
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
            # Add some noise for realism
            audio += 0.05 * np.random.randn(len(audio))
            self.speaker1_samples.append(audio)
        
        # Speaker 2: 400Hz sine wave
        for i in range(3):
            freq = 400 + i * 10
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
            audio += 0.05 * np.random.randn(len(audio))
            self.speaker2_samples.append(audio)
    
    def tearDown(self):
        """
        Clean up after tests.
        """
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_speaker_enrollment(self):
        """
        Test speaker enrollment.
        """
        # Enroll speaker 1
        result = self.voicevault.enroll_speaker(
            audio_samples=self.speaker1_samples,
            speaker_id="speaker1"
        )
        
        # Check result
        self.assertTrue(result['success'])
        self.assertIn('profile', result)
        self.assertIn('profile_path', result['profile'])
        
        # Check profile saved
        self.assertTrue(os.path.exists(result['profile']['profile_path']))
    
    def test_speaker_verification(self):
        """
        Test speaker verification.
        """
        # Enroll speaker 1
        enroll_result = self.voicevault.enroll_speaker(
            audio_samples=self.speaker1_samples,
            speaker_id="speaker1"
        )
        
        # Verify with same speaker
        verify_result = self.voicevault.verify_speaker(
            audio=self.speaker1_samples[0],
            profile=enroll_result['profile']
        )
        
        # Check result
        self.assertTrue(verify_result.get('verified', False))
        
        # Verify with different speaker
        verify_result2 = self.voicevault.verify_speaker(
            audio=self.speaker2_samples[0],
            profile=enroll_result['profile']
        )
        
        # Check result (should not verify)
        self.assertFalse(verify_result2.get('verified', True))
    
    def test_liveness_detection(self):
        """
        Test liveness detection.
        
        Note: This is a simplified test since synthetic test audio
        may not pass liveness detection in a real system.
        """
        # Test with speaker 1 sample
        liveness_result = self.voicevault.liveness_detector.detect_liveness(
            self.speaker1_samples[0]
        )
        
        # Check result structure
        self.assertIn('is_live', liveness_result)
        self.assertIn('liveness_score', liveness_result)
        self.assertIn('component_scores', liveness_result)
    
    def test_challenge_response(self):
        """
        Test challenge-response protocol.
        
        Note: This is a simplified test that just checks the challenge structure.
        """
        # Generate a challenge
        challenge = self.voicevault.challenge_protocol.start_verification()
        
        # Check challenge structure
        self.assertIn('session_id', challenge)
        self.assertIn('challenge_type', challenge)
        self.assertIn('challenge_text', challenge)
        self.assertIn('challenge_display', challenge)
        self.assertIn('verification_hash', challenge)
        self.assertIn('created_at', challenge)
        self.assertIn('expires_at', challenge)


if __name__ == '__main__':
    unittest.main()
