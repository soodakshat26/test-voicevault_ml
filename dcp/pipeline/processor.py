import numpy as np
import librosa
import soundfile as sf
import os
import time
import yaml
import json
from datetime import datetime
import multiprocessing as mp

# Import preprocessing components
from preprocessing.noise_suppression import NoiseSuppressionFactory
from preprocessing.vad import VADFactory
from preprocessing.normalization import NormalizationFactory

# Import feature extraction components
from features.acoustic import AcousticFeatureFactory
from features.spectral import SpectralFeatureFactory
from features.prosodic import ProsodicFeatureFactory
from features.glottal import GlottalFeatureFactory
from features.temporal import TemporalFeatureFactory

class VoiceProcessor:
    """Main processing pipeline for voice analysis"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize preprocessing components
        self._init_preprocessing()
        
        # Initialize feature extraction components
        self._init_feature_extractors()
        
        # Processing settings
        self.sample_rate = 16000  # Target sample rate
        self.debug_mode = False
        
    def _init_preprocessing(self):
        """Initialize preprocessing components based on config"""
        # Noise suppression
        noise_method = self.config['preprocessing']['noise_suppression'].get('method', 'spectral_subtraction')
        self.noise_suppressor = NoiseSuppressionFactory.create_suppressor(noise_method)
        
        # Voice activity detection
        vad_method = self.config['preprocessing']['vad'].get('model', 'energy')
        self.vad = VADFactory.create_vad(vad_method)
        
        # Normalization
        norm_method = 'channel_eq'  # Default method
        self.normalizer = NormalizationFactory.create_normalizer(norm_method)
        
    def _init_feature_extractors(self):
        """Initialize feature extraction components"""
        # Acoustic features
        self.acoustic_extractors = [
            AcousticFeatureFactory.create_extractor("mfcc"),
            AcousticFeatureFactory.create_extractor("xvector")
        ]
        
        # Spectral features
        self.spectral_extractors = [
            SpectralFeatureFactory.create_extractor("reassigned_spectrogram"),
            SpectralFeatureFactory.create_extractor("multi_taper")
        ]
        
        # Prosodic features
        self.prosodic_extractors = [
            ProsodicFeatureFactory.create_extractor("yaapt"),
            ProsodicFeatureFactory.create_extractor("speech_rate")
        ]
        
        # Glottal features
        self.glottal_extractors = [
            GlottalFeatureFactory.create_extractor("inverse_filtering")
        ]
        
        # Temporal features
        self.temporal_extractors = [
            TemporalFeatureFactory.create_extractor("vot"),
            TemporalFeatureFactory.create_extractor("microprosody")
        ]
    
    def process_file(self, file_path, output_dir=None):
        """
        Process a single audio file through the complete pipeline
        
        Parameters:
        -----------
        file_path : str
            Path to the audio file
        output_dir : str, optional
            Directory to save processed outputs
            
        Returns:
        --------
        results : dict
            Processing results including all extracted features
        """
        print(f"Processing file: {file_path}")
        
        # Create output directory if specified
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Initialize results dictionary
        results = {
            'file_path': file_path,
            'sample_rate': sr,
            'duration': len(y) / sr,
            'features': {}
        }
        
        # Step 1: Preprocessing
        y_processed = self._preprocess_audio(y, sr)
        
        # Save preprocessed audio if output directory specified
        if output_dir is not None:
            preprocessed_path = os.path.join(output_dir, 
                                            os.path.basename(file_path).replace('.', '_preprocessed.'))
            sf.write(preprocessed_path, y_processed, sr)
            results['preprocessed_path'] = preprocessed_path
        
        # Step 2: Feature extraction
        features = self._extract_features(y_processed, sr)
        results['features'] = features
        
        # Step 3: Save features if output directory specified
        if output_dir is not None:
            features_path = os.path.join(output_dir, 
                                        os.path.basename(file_path).replace('.', '_features.json'))
            # Filter out non-serializable values
            serializable_features = self._prepare_features_for_json(features)
            with open(features_path, 'w') as f:
                json.dump(serializable_features, f, indent=2)
            results['features_path'] = features_path
            
        return results
    
    def _preprocess_audio(self, y, sr):
        """Apply preprocessing chain to audio signal"""
        # Apply noise suppression if configured
        y_clean = self.noise_suppressor.suppress_noise(y, sr)
        
        # Apply voice activity detection to isolate speech
        speech_mask = self.vad.detect_speech(y_clean, sr)
        
        # Apply normalization if segments found
        y_processed = y_clean.copy()
        if np.any(speech_mask):
            # Normalize speech segments
            y_processed[speech_mask] = self.normalizer.normalize(y_clean[speech_mask], sr)
        
        return y_processed
    
    def _extract_features(self, y, sr):
        """Extract all features from preprocessed audio"""
        features = {}
        
        # Extract acoustic features
        acoustic_features = {}
        for extractor in self.acoustic_extractors:
            acoustic_features.update(extractor.extract(y, sr))
        features['acoustic'] = acoustic_features
        
        # Extract spectral features
        spectral_features = {}
        for extractor in self.spectral_extractors:
            spectral_features.update(extractor.extract(y, sr))
        features['spectral'] = spectral_features
        
        # Extract prosodic features
        prosodic_features = {}
        for extractor in self.prosodic_extractors:
            prosodic_features.update(extractor.extract(y, sr))
        features['prosodic'] = prosodic_features
        
        # Extract glottal features
        glottal_features = {}
        for extractor in self.glottal_extractors:
            glottal_features.update(extractor.extract(y, sr))
        features['glottal'] = glottal_features
        
        # Extract temporal features
        temporal_features = {}
        for extractor in self.temporal_extractors:
            temporal_features.update(extractor.extract(y, sr))
        features['temporal'] = temporal_features
        
        return features
    
    def _prepare_features_for_json(self, features):
        """Prepare features for JSON serialization (convert numpy arrays to lists)"""
        serializable = {}
        
        for category, category_features in features.items():
            serializable[category] = {}
            
            for name, value in category_features.items():
                if isinstance(value, np.ndarray):
                    # Convert 1D arrays to lists
                    if value.ndim == 1:
                        serializable[category][name] = value.tolist()
                    # Convert 2D arrays to lists of lists
                    elif value.ndim == 2:
                        serializable[category][name] = [row.tolist() for row in value]
                    # Skip higher dimensions
                    else:
                        serializable[category][name] = f"ndarray with shape {value.shape}"
                elif isinstance(value, np.number):
                    serializable[category][name] = float(value)
                elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                    serializable[category][name] = value
                else:
                    # Skip non-serializable objects
                    serializable[category][name] = str(type(value))
        
        return serializable
    
    def process_batch(self, file_paths, output_dir=None, num_workers=1):
        """
        Process multiple audio files in parallel
        
        Parameters:
        -----------
        file_paths : list
            List of paths to audio files
        output_dir : str, optional
            Directory to save processed outputs
        num_workers : int
            Number of parallel workers
            
        Returns:
        --------
        results : list
            List of processing results for each file
        """
        if num_workers <= 1 or len(file_paths) <= 1:
            # Process sequentially
            results = []
            for file_path in file_paths:
                results.append(self.process_file(file_path, output_dir))
            return results
        
        # Process in parallel using multiprocessing
        with mp.Pool(num_workers) as pool:
            # Create partial function with fixed output_dir
            from functools import partial
            process_func = partial(self.process_file, output_dir=output_dir)
            
            # Map function over all files
            results = pool.map(process_func, file_paths)
            
        return results