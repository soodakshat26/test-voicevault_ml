import json
import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import yaml
from datetime import datetime
import uuid

class MetadataSystem:
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.metadata_version = "1.0.0"
        self.iso_standard = "ISO 24617-9"  # Semantic annotation framework
    
    def generate_recording_metadata(self, audio_path, participant_info=None, recording_context=None):
        """
        Generate comprehensive metadata for a voice recording
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file
        participant_info : dict
            Information about the participant
        recording_context : dict
            Information about the recording context
            
        Returns:
        --------
        metadata : dict
            Complete metadata structure
        """
        # Create a unique ID for this metadata record
        metadata_id = str(uuid.uuid4())
        
        # Load audio to extract properties
        audio, sr = sf.read(audio_path)
        
        # Basic file metadata
        file_metadata = {
            "filename": os.path.basename(audio_path),
            "path": audio_path,
            "format": os.path.splitext(audio_path)[1].replace(".", ""),
            "created": datetime.fromtimestamp(os.path.getctime(audio_path)).isoformat(),
            "modified": datetime.fromtimestamp(os.path.getmtime(audio_path)).isoformat(),
            "filesize_bytes": os.path.getsize(audio_path)
        }
        
        # Audio technical metadata
        audio_metadata = {
            "duration_seconds": float(len(audio) / sr),
            "sample_rate": sr,
            "channels": audio.shape[1] if len(audio.shape) > 1 else 1,
            "bit_depth": 16 if audio.dtype == np.int16 else (32 if audio.dtype == np.float32 else 24),
            "codec": "PCM",  # This would need a more sophisticated detection for compressed formats
        }
        
        # Extract basic audio features for quality assessment
        audio_mono = audio if len(audio.shape) == 1 else np.mean(audio, axis=1)
        
        # RMS energy
        rms = np.sqrt(np.mean(audio_mono**2))
        
        # Signal-to-noise ratio (estimated)
        # Using percentile method to estimate noise floor
        noise_floor = np.percentile(np.abs(audio_mono), 10)
        signal_level = np.percentile(np.abs(audio_mono), 90)
        snr_estimate = 20 * np.log10(signal_level / (noise_floor + 1e-10))
        
        # Simple spectral analysis
        spec = np.abs(librosa.stft(audio_mono))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_mono, sr=sr))
        
        quality_metrics = {
            "rms_level": float(rms),
            "peak_level": float(np.max(np.abs(audio_mono))),
            "estimated_snr_db": float(snr_estimate),
            "spectral_centroid_hz": float(spectral_centroid),
            "dc_offset": float(np.mean(audio_mono))
        }
        
        # Combine all metadata
        metadata = {
            "metadata_id": metadata_id,
            "metadata_version": self.metadata_version,
            "iso_standard": self.iso_standard,
            "timestamp": datetime.now().isoformat(),
            "file": file_metadata,
            "audio_technical": audio_metadata,
            "quality_metrics": quality_metrics,
            "participant": participant_info if participant_info else {},
            "recording_context": recording_context if recording_context else {}
        }
        
        return metadata
    
    def save_metadata(self, metadata, output_path=None):
        """
        Save metadata to a JSON file
        
        Parameters:
        -----------
        metadata : dict
            Metadata to save
        output_path : str
            Path to save metadata (defaults to beside audio file)
            
        Returns:
        --------
        output_path : str
            Path where metadata was saved
        """
        if output_path is None:
            audio_path = metadata.get("file", {}).get("path", "")
            if audio_path:
                output_path = os.path.splitext(audio_path)[0] + ".metadata.json"
            else:
                output_path = f"metadata_{metadata['metadata_id']}.json"
                
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Metadata saved to {output_path}")
        return output_path
    
    def generate_hierarchical_annotation(self, audio_path, transcript, language="en-US"):
        """
        Generate hierarchical annotation structure from transcript
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file
        transcript : str
            Transcript of the audio
        language : str
            Language code
            
        Returns:
        --------
        annotation : dict
            Hierarchical annotation structure
        """
        # Load audio
        audio, sr = sf.read(audio_path)
        audio_mono = audio if len(audio.shape) == 1 else np.mean(audio, axis=1)
        
        # Generate a simple word-level segmentation
        # This is a simplified approach; real implementation would use a speech recognition system
        # with time alignments or forced alignment
        
        # Simulate word boundaries by dividing audio evenly by word count
        words = transcript.split()
        word_count = len(words)
        
        # Estimate speech rate (average English speech is ~150 words per minute)
        duration = len(audio_mono) / sr
        estimated_speech_rate = word_count / duration * 60
        
        # Create word-level annotations
        word_annotations = []
        
        for i, word in enumerate(words):
            # Simplified time alignment (evenly spaced words)
            start_time = i * duration / word_count
            end_time = (i + 1) * duration / word_count
            
            # For a real system, phoneme segmentation would be done with a forced aligner
            # This is just a placeholder simulation
            phonemes = self._simulate_phonemes(word)
            
            word_annotations.append({
                "id": f"word_{i}",
                "text": word,
                "start_time": float(start_time),
                "end_time": float(end_time),
                "confidence": 0.95,  # Placeholder confidence score
                "phonemes": phonemes
            })
        
        # Group words into phrases/sentences
        sentences = []
        current_sentence = []
        current_start = 0
        
        for i, word in enumerate(word_annotations):
            current_sentence.append(word)
            
            # Simple sentence boundary detection based on punctuation
            # A real system would use NLP tools for proper sentence segmentation
            if i == len(word_annotations) - 1 or any(word["text"].endswith(p) for p in ['.', '!', '?']):
                sentence_text = " ".join(w["text"] for w in current_sentence)
                sentence_end = word["end_time"]
                
                sentences.append({
                    "id": f"sentence_{len(sentences)}",
                    "text": sentence_text,
                    "start_time": float(current_start),
                    "end_time": float(sentence_end),
                    "words": current_sentence
                })
                
                current_sentence = []
                current_start = sentence_end
        
        # Create the complete annotation structure
        annotation = {
            "audio_path": audio_path,
            "transcript": transcript,
            "language": language,
            "duration": float(duration),
            "estimated_speech_rate_wpm": float(estimated_speech_rate),
            "sentences": sentences,
            "word_count": word_count,
            "timestamp": datetime.now().isoformat()
        }
        
        return annotation
    
    def _simulate_phonemes(self, word):
        """Simulate phoneme segmentation for a word (placeholder)"""
        # This is a very simplified approximation
        # A real system would use a phonetic dictionary or a G2P model
        
        # Rough estimate: 3 phonemes per word on average
        phoneme_count = max(1, min(len(word), 6))
        phonemes = []
        
        for i in range(phoneme_count):
            start_ratio = i / phoneme_count
            end_ratio = (i + 1) / phoneme_count
            
            # Simulate a phoneme ID (this would be a proper phoneme in reality)
            if i < len(word):
                phoneme_text = word[i]
            else:
                phoneme_text = "."
                
            phonemes.append({
                "id": f"ph_{i}",
                "phoneme": phoneme_text,
                "start_ratio": start_ratio,
                "end_ratio": end_ratio
            })
            
        return phonemes
    
    def export_dataset_catalog(self, dataset_dir, output_path):
        """
        Create a comprehensive catalog of all recordings in a dataset
        
        Parameters:
        -----------
        dataset_dir : str
            Directory containing the dataset
        output_path : str
            Path to save the catalog
            
        Returns:
        --------
        catalog : pd.DataFrame
            DataFrame containing the catalog
        """
        records = []
        
        # Walk through the dataset directory
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith(('.wav', '.flac', '.mp3')):
                    audio_path = os.path.join(root, file)
                    
                    # Check for metadata file
                    metadata_path = os.path.splitext(audio_path)[0] + ".metadata.json"
                    
                    if os.path.exists(metadata_path):
                        # Load the metadata
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            
                        # Extract key information for the catalog
                        record = {
                            "file_name": os.path.basename(audio_path),
                            "file_path": os.path.relpath(audio_path, dataset_dir),
                            "duration": metadata.get("audio_technical", {}).get("duration_seconds", 0),
                            "sample_rate": metadata.get("audio_technical", {}).get("sample_rate", 0),
                            "channels": metadata.get("audio_technical", {}).get("channels", 0),
                            "participant_id": metadata.get("participant", {}).get("id", "unknown"),
                            "quality_score": metadata.get("quality_metrics", {}).get("estimated_snr_db", 0),
                            "has_transcript": "transcript" in metadata.get("recording_context", {})
                        }
                        
                        records.append(record)
                    else:
                        # Create a minimal record if no metadata is available
                        try:
                            audio, sr = sf.read(audio_path)
                            duration = len(audio) / sr
                            channels = audio.shape[1] if len(audio.shape) > 1 else 1
                            
                            record = {
                                "file_name": os.path.basename(audio_path),
                                "file_path": os.path.relpath(audio_path, dataset_dir),
                                "duration": duration,
                                "sample_rate": sr,
                                "channels": channels,
                                "participant_id": "unknown",
                                "quality_score": 0,
                                "has_transcript": False
                            }
                            
                            records.append(record)
                        except Exception as e:
                            print(f"Error processing {audio_path}: {e}")
        
        # Create DataFrame
        catalog = pd.DataFrame(records)
        
        # Save catalog
        if output_path.endswith('.csv'):
            catalog.to_csv(output_path, index=False)
        elif output_path.endswith('.xlsx'):
            catalog.to_excel(output_path, index=False)
        else:
            catalog.to_csv(output_path + '.csv', index=False)
            
        print(f"Dataset catalog created with {len(catalog)} entries, saved to {output_path}")
        
        return catalog
