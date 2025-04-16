import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time
import os
import sys
import json
import sounddevice as sd
import soundfile as sf
# Get the absolute path to the project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Add project root to Python path
sys.path.insert(0, project_root)

from acoustic_fingerprinting.comparison import AcousticFingerprinter
from liveness_detection.liveness_detector import LivenessDetector
from challenge_response.protocol import ChallengeResponseProtocol

    
class VoiceVault:
    """
    Complete VoiceVault system combining all core technologies.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        verification_threshold: float = 0.7,
        liveness_threshold: float = 0.7,
        challenge_difficulty: str = 'medium',
        storage_dir: str = 'data'
    ):
        """
        Initialize the VoiceVault system.
        
        Args:
            sample_rate: Audio sample rate in Hz
            verification_threshold: Threshold for speaker verification
            liveness_threshold: Threshold for liveness detection
            challenge_difficulty: Difficulty level for challenges
            storage_dir: Directory for data storage
        """
        self.sample_rate = sample_rate
        self.verification_threshold = verification_threshold
        self.liveness_threshold = liveness_threshold
        self.challenge_difficulty = challenge_difficulty
        self.storage_dir = storage_dir
        
        # Create storage directories
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(os.path.join(storage_dir, 'speaker_profiles'), exist_ok=True)
        os.makedirs(os.path.join(storage_dir, 'verification_logs'), exist_ok=True)
        
        # Initialize components
        self.fingerprinter = AcousticFingerprinter(
            sample_rate=sample_rate,
            threshold=verification_threshold
        )
        
        self.liveness_detector = LivenessDetector(
            sample_rate=sample_rate,
            threshold=liveness_threshold
        )
        
        self.challenge_protocol = ChallengeResponseProtocol(
            challenge_difficulty=challenge_difficulty,
            audio_sample_rate=sample_rate
        )
        
        # Store verification sessions
        self.verification_sessions = {}
    
    def enroll_speaker(
        self, 
        audio_samples: List[np.ndarray], 
        speaker_id: str
    ) -> Dict:
        """
        Enroll a new speaker in the system.
        
        Args:
            audio_samples: List of audio samples from the speaker
            speaker_id: Unique identifier for the speaker
            
        Returns:
            Speaker profile
        """
        # Perform liveness check on samples
        liveness_results = []
        for audio in audio_samples:
            result = self.liveness_detector.detect_liveness(audio)
            liveness_results.append(result)
        
        # Check if enough samples pass liveness check
        passing_samples = [
            audio for audio, result in zip(audio_samples, liveness_results)
            if result['is_live']
        ]
        
        if len(passing_samples) < max(3, len(audio_samples) // 2):
            return {
                'success': False,
                'error': 'Not enough samples passed liveness check',
                'liveness_results': liveness_results
            }
        
        # Generate speaker profile
        profile = self.fingerprinter.enroll_speaker(
            audio_samples=passing_samples,
            speaker_id=speaker_id
        )
        
        # Return enrollment result
        return {
            'success': True,
            'profile': profile,
            'liveness_results': liveness_results
        }
    
    def verify_speaker(
        self, 
        audio: np.ndarray, 
        speaker_id: str,
        profile: Optional[Dict] = None
    ) -> Dict:
        """
        Verify if audio matches a speaker profile.
        
        Args:
            audio: Audio to verify
            speaker_id: Speaker ID to verify against
            profile: Optional speaker profile (loaded if not provided)
            
        Returns:
            Verification result
        """
        # Create session ID
        session_id = f"{speaker_id}_{int(time.time())}"
        
        # Store session
        self.verification_sessions[session_id] = {
            'speaker_id': speaker_id,
            'start_time': time.time(),
            'status': 'started'
        }
        
        # Perform liveness check
        liveness_result = self.liveness_detector.detect_liveness(audio)
        
        # Update session
        self.verification_sessions[session_id]['liveness_result'] = liveness_result
        
        # Check liveness
        if not liveness_result['is_live']:
            self.verification_sessions[session_id]['status'] = 'failed_liveness'
            
            return {
                'success': False,
                'verified': False,
                'error': 'Liveness check failed',
                'liveness_result': liveness_result,
                'session_id': session_id
            }
        
        # Get profile if not provided
        if profile is None:
            # Find latest profile for speaker_id
            try:
                profile_dir = os.path.join(self.storage_dir, 'speaker_profiles')
                profile_files = [
                    f for f in os.listdir(profile_dir) 
                    if f.startswith(f"{speaker_id}_") and f.endswith(".pkl")
                ]
                
                if not profile_files:
                    self.verification_sessions[session_id]['status'] = 'failed_profile'
                    
                    return {
                        'success': False,
                        'verified': False,
                        'error': 'Speaker profile not found',
                        'session_id': session_id
                    }
                
                # Get latest profile
                profile_files.sort(reverse=True)
                profile_path = os.path.join(profile_dir, profile_files[0])
                
                # Load profile
                profile = self.fingerprinter.profile_generator.load_profile(profile_path)
                
            except Exception as e:
                self.verification_sessions[session_id]['status'] = 'failed_profile'
                
                return {
                    'success': False,
                    'verified': False,
                    'error': f'Error loading profile: {str(e)}',
                    'session_id': session_id
                }
        
        # Verify speaker
        verification_result = self.fingerprinter.verify_speaker(audio, profile)
        
        # Update session
        self.verification_sessions[session_id]['verification_result'] = verification_result
        
        if verification_result['verified']:
            self.verification_sessions[session_id]['status'] = 'pending_challenge'
            
            # Start challenge
            challenge = self.challenge_protocol.start_verification(session_id)
            
            self.verification_sessions[session_id]['challenge'] = challenge
            
            # Return verification result with challenge
            return {
                'success': True,
                'verified': True,
                'verification_result': verification_result,
                'liveness_result': liveness_result,
                'challenge': challenge,
                'requires_challenge': True,
                'session_id': session_id
            }
        else:
            self.verification_sessions[session_id]['status'] = 'failed_verification'
            
            # Return verification result
            return {
                'success': True,
                'verified': False,
                'verification_result': verification_result,
                'liveness_result': liveness_result,
                'session_id': session_id
            }
    
    def process_challenge_response(
        self, 
        session_id: str, 
        audio: np.ndarray
    ) -> Dict:
        """
        Process a challenge response.
        
        Args:
            session_id: Session ID
            audio: Audio response
            
        Returns:
            Dictionary with verification results
        """
        # Check if session exists
        if session_id not in self.verification_sessions:
            return {
                'success': False,
                'verified': False,
                'error': 'Invalid session ID',
                'session_id': session_id
            }
        
        # Get session
        session = self.verification_sessions[session_id]
        
        # Check if session is in correct state
        if session['status'] != 'pending_challenge':
            return {
                'success': False,
                'verified': False,
                'error': f'Invalid session state: {session["status"]}',
                'session_id': session_id
            }
        
        # Process challenge response
        challenge_result = self.challenge_protocol.process_response(
            session_id=session_id,
            audio=audio
        )
        
        # Update session
        session['challenge_result'] = challenge_result
        
        if challenge_result['verified']:
            session['status'] = 'verified'
            
            # Log successful verification
            self._log_verification(session)
            
            return {
                'success': True,
                'verified': True,
                'challenge_result': challenge_result,
                'session_id': session_id
            }
        else:
            session['status'] = 'failed_challenge'
            
            return {
                'success': True,
                'verified': False,
                'challenge_result': challenge_result,
                'session_id': session_id
            }
    
    def _log_verification(self, session: Dict):
        """
        Log a successful verification.
        
        Args:
            session: Verification session
        """
        # Create log entry
        log_entry = {
            'timestamp': time.time(),
            'speaker_id': session['speaker_id'],
            'session_id': session.get('session_id', ''),
            'verification_result': session.get('verification_result', {}),
            'liveness_result': session.get('liveness_result', {}),
            'challenge_result': session.get('challenge_result', {})
        }
        
        # Generate log filename
        log_dir = os.path.join(self.storage_dir, 'verification_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_filename = f"{session['speaker_id']}_{int(time.time())}.json"
        log_path = os.path.join(log_dir, log_filename)
        
        # Save log
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
    
    def run_verification_pipeline(
        self, 
        audio: np.ndarray, 
        speaker_id: str,
        profile: Optional[Dict] = None
    ) -> Dict:
        """
        Run the complete verification pipeline.
        
        Args:
            audio: Audio to verify
            speaker_id: Speaker ID to verify against
            profile: Optional speaker profile (loaded if not provided)
            
        Returns:
            Verification result
        """
        # Step 1: Verify speaker
        verification_result = self.verify_speaker(
            audio=audio,
            speaker_id=speaker_id,
            profile=profile
        )
        
        # Return if verification failed or challenge not required
        if not verification_result.get('verified', False):
            return verification_result
        
        if not verification_result.get('requires_challenge', False):
            return verification_result
        
        # Step 2: Get challenge from result
        challenge = verification_result.get('challenge', {})
        session_id = verification_result.get('session_id', '')
        
        # Return result with challenge
        return {
            'success': True,
            'verified': True,
            'requires_challenge_response': True,
            'verification_result': verification_result,
            'challenge': challenge,
            'session_id': session_id,
            'message': 'Please complete the challenge to finish verification'
        }
    
    def get_session_status(self, session_id: str) -> Dict:
        """
        Get status of a verification session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session status
        """
        if session_id not in self.verification_sessions:
            return {
                'success': False,
                'error': 'Invalid session ID',
                'session_id': session_id
            }
        
        session = self.verification_sessions[session_id]
        
        return {
            'success': True,
            'session_id': session_id,
            'speaker_id': session['speaker_id'],
            'status': session['status'],
            'created_at': session['start_time']
        }
