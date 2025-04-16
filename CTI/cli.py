import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import json
import time
from typing import Dict, List, Optional

from .voicevault import VoiceVault
from utils.audio_utils import load_audio, save_audio, normalize_audio, trim_silence


class VoiceVaultCLI:
    """
    Command-line interface for VoiceVault.
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
        Initialize the CLI.
        
        Args:
            sample_rate: Audio sample rate in Hz
            verification_threshold: Threshold for speaker verification
            liveness_threshold: Threshold for liveness detection
            challenge_difficulty: Difficulty level for challenges
            storage_dir: Directory for data storage
        """
        # Initialize VoiceVault
        self.voicevault = VoiceVault(
            sample_rate=sample_rate,
            verification_threshold=verification_threshold,
            liveness_threshold=liveness_threshold,
            challenge_difficulty=challenge_difficulty,
            storage_dir=storage_dir
        )
        
        self.sample_rate = sample_rate
    
    def record_audio(self, duration: int = 5) -> np.ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Recorded audio
        """
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()
        
        return audio.flatten()
    
    def enroll_speaker(self, speaker_id: str, num_samples: int = 3):
        """
        Enroll a new speaker.
        
        Args:
            speaker_id: Speaker ID
            num_samples: Number of audio samples to record
        """
        print(f"Enrolling speaker: {speaker_id}")
        print(f"We'll record {num_samples} audio samples.")
        
        audio_samples = []
        
        for i in range(num_samples):
            print(f"\nSample {i+1}/{num_samples}")
            print("Please speak after the beep...")
            time.sleep(1)
            
            # Beep
            sd.play(np.sin(2 * np.pi * 1000 * np.arange(0.1 * self.sample_rate) / self.sample_rate), 
                    self.sample_rate)
            sd.wait()
            
            # Record
            audio = self.record_audio(duration=5)
            audio = normalize_audio(audio)
            audio = trim_silence(audio, sample_rate=self.sample_rate)
            
            audio_samples.append(audio)
            
            print(f"Sample {i+1} recorded.")
        
        # Enroll speaker
        result = self.voicevault.enroll_speaker(
            audio_samples=audio_samples,
            speaker_id=speaker_id
        )
        
        if result['success']:
            print(f"\nSpeaker {speaker_id} enrolled successfully!")
            print(f"Profile saved at: {result['profile'].get('profile_path', 'unknown')}")
        else:
            print(f"\nEnrollment failed: {result.get('error', 'Unknown error')}")
    
    def verify_speaker(self, speaker_id: str):
        """
        Verify a speaker.
        
        Args:
            speaker_id: Speaker ID to verify
        """
        print(f"Verifying speaker: {speaker_id}")
        print("Please speak after the beep...")
        time.sleep(1)
        
        # Beep
        sd.play(np.sin(2 * np.pi * 1000 * np.arange(0.1 * self.sample_rate) / self.sample_rate), 
                self.sample_rate)
        sd.wait()
        
        # Record
        audio = self.record_audio(duration=5)
        audio = normalize_audio(audio)
        audio = trim_silence(audio, sample_rate=self.sample_rate)
        
        # Verify speaker
        result = self.voicevault.run_verification_pipeline(
            audio=audio,
            speaker_id=speaker_id
        )
        
        if not result.get('success', False):
            print(f"\nVerification failed: {result.get('error', 'Unknown error')}")
            return
        
        if not result.get('verified', False):
            print("\nSpeaker verification failed.")
            if 'verification_result' in result:
                vr = result['verification_result']
                print(f"Similarity score: {vr.get('similarity_score', 0):.2f}")
                print(f"Threshold: {vr.get('threshold', 0):.2f}")
            return
        
        if result.get('requires_challenge_response', False):
            print("\nSpeaker verified! Please complete the challenge:")
            print(f"Challenge: {result['challenge']['challenge_display']}")
            
            print("\nRespond after the beep...")
            time.sleep(1)
            
            # Beep
            sd.play(np.sin(2 * np.pi * 1000 * np.arange(0.1 * self.sample_rate) / self.sample_rate), 
                    self.sample_rate)
            sd.wait()
            
            # Record challenge response
            response_audio = self.record_audio(duration=5)
            
            # Process challenge response
            challenge_result = self.voicevault.process_challenge_response(
                session_id=result['session_id'],
                audio=response_audio
            )
            
            if challenge_result.get('verified', False):
                print("\nChallenge completed successfully!")
                print("Verification complete!")
            else:
                print("\nChallenge failed.")
                if 'challenge_result' in challenge_result:
                    cr = challenge_result['challenge_result']
                    if 'transcription' in cr:
                        print(f"You said: \"{cr['transcription']}\"")
                    if 'similarity' in cr:
                        print(f"Similarity: {cr['similarity']:.2f}")
                        print(f"Threshold: {cr['threshold']:.2f}")
        else:
            print("\nVerification successful!")
    
    def run(self):
        """
        Run the CLI.
        """
        parser = argparse.ArgumentParser(description='VoiceVault Command-Line Interface')
        
        subparsers = parser.add_subparsers(dest='command', help='Command to run')
        
        # Enroll command
        enroll_parser = subparsers.add_parser('enroll', help='Enroll a new speaker')
        enroll_parser.add_argument('speaker_id', help='Speaker ID')
        enroll_parser.add_argument('--samples', type=int, default=3, help='Number of audio samples')
        
        # Verify command
        verify_parser = subparsers.add_parser('verify', help='Verify a speaker')
        verify_parser.add_argument('speaker_id', help='Speaker ID to verify')
        
        # Parse args
        args = parser.parse_args()
        
        if args.command == 'enroll':
            self.enroll_speaker(args.speaker_id, args.samples)
        elif args.command == 'verify':
            self.verify_speaker(args.speaker_id)
        else:
            parser.print_help()


def main():
    """
    Main entry point.
    """
    cli = VoiceVaultCLI()
    cli.run()


if __name__ == '__main__':
    main()
