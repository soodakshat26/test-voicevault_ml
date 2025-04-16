import os
import argparse
import yaml
import glob
from pipeline.processor import VoiceProcessor
from pipeline.realtime import RealtimeVoiceAnalyzer
from collection.protocol import RecordingProtocol

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='VoiceVault Voice Processing System')
    
    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Process mode parser
    process_parser = subparsers.add_parser('process', help='Process audio files')
    process_parser.add_argument('--input', '-i', required=True, help='Input audio file or directory')
    process_parser.add_argument('--output', '-o', required=True, help='Output directory')
    process_parser.add_argument('--workers', '-w', type=int, default=1, help='Number of worker processes')
    
    # Record mode parser
    record_parser = subparsers.add_parser('record', help='Record audio samples')
    record_parser.add_argument('--participant', '-p', required=True, help='Participant ID')
    record_parser.add_argument('--output', '-o', default='recordings', help='Output directory')
    
    # Realtime mode parser
    realtime_parser = subparsers.add_parser('realtime', help='Real-time audio processing')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create config directory if it doesn't exist
    os.makedirs('config', exist_ok=True)
    
    # Create default config if it doesn't exist
    if not os.path.exists('config/processing.yaml'):
        create_default_config()
    
    # Execute selected mode
    if args.mode == 'process':
        process_files(args)
    elif args.mode == 'record':
        record_samples(args)
    elif args.mode == 'realtime':
        realtime_analysis()
    else:
        parser.print_help()

def process_files(args):
    """Process audio files"""
    # Initialize processor
    processor = VoiceProcessor()
    
    # Collect input files
    if os.path.isdir(args.input):
        files = []
        for ext in ['wav', 'flac', 'mp3']:
            files.extend(glob.glob(os.path.join(args.input, f'*.{ext}')))
    else:
        files = [args.input]
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Processing {len(files)} audio files with {args.workers} workers")
    
    # Process files
    results = processor.process_batch(files, args.output, args.workers)
    
    print(f"Completed processing {len(results)} files")
    
    # Print summary
    success_count = sum(1 for r in results if 'features' in r)
    print(f"Successfully processed: {success_count}/{len(results)}")

def record_samples(args):
    """Record audio samples following protocol"""
    # Initialize recording protocol
    protocol = RecordingProtocol()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Run standard protocol
    print(f"Starting recording session for participant {args.participant}")
    protocol.run_standard_protocol(args.participant)
    
    print("Recording session completed")

def realtime_analysis():
    """Run real-time analysis with GUI"""
    analyzer = RealtimeVoiceAnalyzer()
    analyzer.start_gui()

def create_default_config():
    """Create default configuration file"""
    default_config = {
        'preprocessing': {
            'noise_suppression': {
                'method': 'spectral_subtraction',
                'bands': 64,
                'psychoacoustic_masking': True,
                'phase_preservation': True
            },
            'vad': {
                'model': 'energy',
                'frame_size': 20,
                'overlap': 10,
                'context_frames': 40,
                'decision_threshold': 0.5
            },
            'normalization': {
                'cepstral_mean_variance': True,
                'channel_equalization': True,
                'phase_normalization': True,
                'modulation_spectrum': True
            }
        },
        'features': {
            'acoustic': {
                'mfcc': {
                    'coefficients': 20,
                    'liftering': 22
                },
                'x_vectors': {
                    'embedding_size': 512,
                    'pooling': 'temporal_statistics'
                }
            },
            'spectral': {
                'reassigned_spectrogram': True,
                'multi_taper': True,
                'taper_count': 8,
                'group_delay': True
            },
            'prosodic': {
                'f0_algorithm': 'yaapt',
                'f0_range': [50, 600],
                'energy_bands': 5,
                'speech_rate': True
            },
            'glottal': {
                'inverse_filtering': 'iaif',
                'model_order': 24,
                'closed_phase_analysis': True
            },
            'temporal': {
                'vot_detection': True,
                'micro_prosody': True,
                'formant_transitions': True
            }
        }
    }
    
    # Write config file
    with open('config/processing.yaml', 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    print("Created default configuration file")

if __name__ == "__main__":
    main()
