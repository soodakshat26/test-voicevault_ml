import numpy as np
import soundfile as sf
import librosa
from typing import Dict, List, Tuple, Optional, Union
import os
import time


def load_audio(
    file_path: str, 
    target_sr: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with resampling if needed.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    # Load audio
    audio, sr = librosa.load(file_path, sr=None, mono=mono)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    return audio, sr


def save_audio(
    audio: np.ndarray, 
    file_path: str, 
    sample_rate: int = 16000
):
    """
    Save audio to file.
    
    Args:
        audio: Audio data
        file_path: Output file path
        sample_rate: Sample rate
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Save audio
    sf.write(file_path, audio, sample_rate)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to range [-1, 1].
    
    Args:
        audio: Audio data
        
    Returns:
        Normalized audio
    """
    # Check if audio is already normalized
    max_abs = np.max(np.abs(audio))
    if max_abs <= 1.0 and max_abs > 0.1:
        return audio
    
    # Normalize
    return audio / (max_abs + 1e-10)


def trim_silence(
    audio: np.ndarray, 
    sample_rate: int = 16000, 
    threshold: float = 0.03, 
    pad_ms: int = 100
) -> np.ndarray:
    """
    Trim silence from beginning and end of audio.
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        threshold: Energy threshold for silence detection
        pad_ms: Padding in milliseconds to keep around non-silent regions
        
    Returns:
        Trimmed audio
    """
    # Convert padding to samples
    pad_samples = int(pad_ms * sample_rate / 1000)
    
    # Trim
    trimmed, _ = librosa.effects.trim(audio, top_db=20, frame_length=1024, hop_length=256)
    
    # Add padding
    if len(trimmed) + 2 * pad_samples <= len(audio):
        # Find where trimmed audio starts in original
        for i in range(len(audio) - len(trimmed) + 1):
            if np.allclose(audio[i:i+len(trimmed)], trimmed, rtol=1e-5, atol=1e-5):
                start_idx = max(0, i - pad_samples)
                end_idx = min(len(audio), i + len(trimmed) + pad_samples)
                return audio[start_idx:end_idx]
    
    # If couldn't find exact match or padding would be out of bounds, just return trimmed
    return trimmed


def segment_audio(
    audio: np.ndarray, 
    sample_rate: int = 16000, 
    min_duration_sec: float = 1.0
) -> List[np.ndarray]:
    """
    Segment audio into phrases based on silence.
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        min_duration_sec: Minimum segment duration in seconds
        
    Returns:
        List of audio segments
    """
    # Detect non-silent intervals
    intervals = librosa.effects.split(audio, top_db=30, frame_length=1024, hop_length=256)
    
    # Convert to segments
    segments = []
    min_samples = int(min_duration_sec * sample_rate)
    
    for start, end in intervals:
        # Check minimum duration
        if end - start >= min_samples:
            segments.append(audio[start:end])
    
    return segments


def extract_audio_features(
    audio: np.ndarray, 
    sample_rate: int = 16000, 
    feature_type: str = 'mfcc',
    n_mfcc: int = 20
) -> np.ndarray:
    """
    Extract audio features.
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        feature_type: Feature type ('mfcc', 'mel', 'chroma')
        n_mfcc: Number of MFCC features (if applicable)
        
    Returns:
        Feature array
    """
    if feature_type == 'mfcc':
        # Extract MFCCs
        features = librosa.feature.mfcc(
            y=audio, 
            sr=sample_rate, 
            n_mfcc=n_mfcc
        )
    elif feature_type == 'mel':
        # Extract Mel spectrogram
        features = librosa.feature.melspectrogram(
            y=audio, 
            sr=sample_rate
        )
        features = librosa.power_to_db(features)
    elif feature_type == 'chroma':
        # Extract chroma features
        features = librosa.feature.chroma_stft(
            y=audio, 
            sr=sample_rate
        )
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    return features
