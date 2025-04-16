# test_processing.py
import os
import numpy as np
import soundfile as sf
from pipeline.processor import VoiceProcessor

def generate_test_audio():
    """Generate a simple test audio file with a tone"""
    sr = 16000
    duration = 3  # seconds
    
    # Generate a sine wave tone
    t = np.linspace(0, duration, int(sr * duration))
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Add some noise
    noise = np.random.randn(len(tone)) * 0.05
    signal = tone + noise
    
    # Save to file
    os.makedirs("test_data", exist_ok=True)
    output_path = "test_data/test_tone.wav"
    sf.write(output_path, signal, sr)
    
    return output_path

def test_processing():
    # Generate test audio if needed
    audio_path = generate_test_audio()
    
    # Create output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Initialize processor
    processor = VoiceProcessor()
    
    # Process the file
    result = processor.process_file(audio_path, "test_output")
    
    if result and 'features' in result:
        print("Processing successful!")
        print(f"Extracted features: {list(result['features'].keys())}")
        return True
    else:
        print("Processing failed.")
        return False

if __name__ == "__main__":
    test_processing()
