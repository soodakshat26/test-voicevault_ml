import argparse
import os
import sys
from cli import VoiceVaultCLI


def main():
    """
    Main entry point for VoiceVault.
    """
    parser = argparse.ArgumentParser(description='VoiceVault - Speaker Verification System')
    
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Audio sample rate in Hz')
    
    parser.add_argument('--verification-threshold', type=float, default=0.7,
                        help='Threshold for speaker verification')
    
    parser.add_argument('--liveness-threshold', type=float, default=0.7,
                        help='Threshold for liveness detection')
    
    parser.add_argument('--challenge-difficulty', type=str, default='medium',
                        choices=['easy', 'medium', 'hard'],
                        help='Difficulty level for challenges')
    
    parser.add_argument('--storage-dir', type=str, default='data',
                        help='Directory for data storage')
    
    args = parser.parse_args()
    
    # Run CLI
    cli = VoiceVaultCLI(
        sample_rate=args.sample_rate,
        verification_threshold=args.verification_threshold,
        liveness_threshold=args.liveness_threshold,
        challenge_difficulty=args.challenge_difficulty,
        storage_dir=args.storage_dir
    )
    
    cli.run()


if __name__ == '__main__':
    main()
