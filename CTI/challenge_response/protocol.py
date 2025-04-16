import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time
import json
import sounddevice as sd
import soundfile as sf
import tempfile
import os

from .challenge import ChallengeGenerator
from .response import SpeechRecognizer, ResponseVerifier


class ChallengeResponseProtocol:
    """
    Complete challenge-response protocol implementation.
    """
    
    def __init__(
        self,
        challenge_generator: Optional[ChallengeGenerator] = None,
        speech_recognizer: Optional[SpeechRecognizer] = None,
        response_verifier: Optional[ResponseVerifier] = None,
        challenge_difficulty: str = 'medium',
        audio_sample_rate: int = 16000
    ):
        """
        Initialize the challenge-response protocol.
        
        Args:
            challenge_generator: Challenge generator
            speech_recognizer: Speech recognizer
            response_verifier: Response verifier
            challenge_difficulty: Difficulty level ('easy', 'medium', 'hard')
            audio_sample_rate: Audio sample rate in Hz
        """
        self.challenge_generator = challenge_generator or ChallengeGenerator(
            challenge_difficulty=challenge_difficulty
        )
        
        self.speech_recognizer = speech_recognizer or SpeechRecognizer()
        
        self.response_verifier = response_verifier or ResponseVerifier(
            speech_recognizer=self.speech_recognizer
        )
        
        self.audio_sample_rate = audio_sample_rate
        
        # Store active sessions
        self.active_sessions = {}
    
    def start_verification(self, session_id: Optional[str] = None) -> Dict:
        """
        Start a verification session with a challenge.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Dictionary with challenge
        """
        # Generate challenge
        challenge = self.challenge_generator.generate_challenge(session_id)
        
        # Store in active sessions
        self.active_sessions[challenge['session_id']] = {
            'challenge': challenge,
            'start_time': time.time(),
            'status': 'pending'
        }
        
        return challenge
    
    def process_response(
        self, 
        session_id: str, 
        audio: np.ndarray
    ) -> Dict:
        """
        Process an audio response to a challenge.
        
        Args:
            session_id: Session ID
            audio: Audio response signal
            
        Returns:
            Dictionary with verification results
        """
        # Check if session exists
        if session_id not in self.active_sessions:
            return {
                'verified': False,
                'error': 'Invalid session ID',
                'session_id': session_id
            }
        
        # Get session
        session = self.active_sessions[session_id]
        
        # Get challenge
        challenge = session['challenge']
        
        # Verify response
        result = self.response_verifier.verify_response(
            audio=audio,
            sample_rate=self.audio_sample_rate,
            challenge=challenge
        )
        
        # Update session status
        if result['verified']:
            session['status'] = 'verified'
        else:
            session['status'] = 'failed'
        
        session['result'] = result
        
        # Clean up if verified or max attempts reached
        if result['verified'] or session.get('attempts', 0) >= 3:
            # In a real system, you might want to keep session data longer
            # for logging/auditing purposes
            self.active_sessions.pop(session_id, None)
        else:
            # Increment attempts
            session['attempts'] = session.get('attempts', 0) + 1
        
        return result
    
    def record_audio_response(self, duration: int = 5) -> np.ndarray:
        """
        Record audio for response.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Recorded audio signal
        """
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * self.audio_sample_rate),
            samplerate=self.audio_sample_rate,
            channels=1
        )
        sd.wait()
        
        return audio.flatten()
    
    def run_interactive_verification(self) -> Dict:
        """
        Run interactive verification with voice input.
        
        Returns:
            Dictionary with verification results
        """
        # Start verification
        challenge = self.start_verification()
        
        # Display challenge
        print("\n--- Voice Challenge ---")
        print(challenge['challenge_display'])
        print("----------------------\n")
        
        # Record response (with delay to give user time to read)
        time.sleep(1)
        print("Please speak after the beep...")
        time.sleep(0.5)
        
        # Beep
        sd.play(np.sin(2 * np.pi * 1000 * np.arange(0.1 * self.audio_sample_rate) / self.audio_sample_rate), 
                self.audio_sample_rate)
        sd.wait()
        
        # Record audio
        estimate_duration = len(challenge['challenge_text']) / 10 + 2  # Rough estimate
        audio = self.record_audio_response(duration=min(10, max(3, estimate_duration)))
        
        # Process response
        result = self.process_response(challenge['session_id'], audio)
        
        # Display result
        print("\n--- Verification Result ---")
        if result.get('verified', False):
            print("✓ Verified!")
        else:
            print("✗ Verification failed")
            
        if 'transcription' in result:
            print(f"Transcription: \"{result['transcription']}\"")
            
        if 'similarity' in result:
            print(f"Similarity: {result['similarity']:.2f} (threshold: {result.get('threshold', 0):.2f})")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        
        print("----------------------------\n")
        
        return result
