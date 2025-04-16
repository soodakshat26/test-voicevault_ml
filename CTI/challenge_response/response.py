import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time
import json
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import tempfile
import os
import soundfile as sf

class SpeechRecognizer:
    """
    Speech recognizer for processing challenge responses.
    """
    
    def __init__(
        self, 
        model_name: str = "openai/whisper-small",
        use_gpu: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the speech recognizer.
        
        Args:
            model_name: Name of the ASR model to use
            use_gpu: Whether to use GPU for inference
            verbose: Whether to print verbose output
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.verbose = verbose
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        
        # Move model to GPU if available
        if self.use_gpu:
            self.model = self.model.to("cuda")
        
        if self.verbose:
            print(f"Loaded ASR model: {model_name}")
            print(f"Using GPU: {self.use_gpu}")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Transcribe speech audio to text.
        
        Args:
            audio: Audio signal
            sample_rate: Audio sample rate in Hz
            
        Returns:
            Dictionary with transcription results
        """
        start_time = time.time()
        
        # Ensure correct sample rate
        if sample_rate != 16000:
            # Resample to 16kHz
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Save audio to temporary file (some models require file input)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            sf.write(temp.name, audio, sample_rate)
            temp_filename = temp.name
        
        try:
            # Process audio
            inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            
            # Move inputs to GPU if available
            if self.use_gpu:
                inputs = inputs.to("cuda")
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=256)
            
            # Decode transcription
            transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            computation_time = time.time() - start_time
            
            return {
                'transcription': transcription,
                'computation_time': computation_time,
                'success': True
            }
        
        except Exception as e:
            if self.verbose:
                print(f"Transcription error: {str(e)}")
            
            return {
                'transcription': '',
                'computation_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)


class ResponseVerifier:
    """
    Verifier for challenge responses.
    """
    
    def __init__(
        self, 
        speech_recognizer: SpeechRecognizer,
        similarity_threshold: float = 0.8,
        time_limit_factor: float = 1.5
    ):
        """
        Initialize the response verifier.
        
        Args:
            speech_recognizer: Speech recognizer for transcription
            similarity_threshold: Threshold for text similarity
            time_limit_factor: Factor to multiply expected time by for limit
        """
        self.speech_recognizer = speech_recognizer
        self.similarity_threshold = similarity_threshold
        self.time_limit_factor = time_limit_factor
    
    def verify_response(
        self, 
        audio: np.ndarray, 
        sample_rate: int, 
        challenge: Dict
    ) -> Dict:
        """
        Verify audio response to a challenge.
        
        Args:
            audio: Audio response signal
            sample_rate: Audio sample rate in Hz
            challenge: Challenge dictionary
            
        Returns:
            Dictionary with verification results
        """
        start_time = time.time()
        
        # Check if challenge has expired
        current_time = int(time.time())
        if current_time > challenge.get('expires_at', 0):
            return {
                'verified': False,
                'error': 'Challenge expired',
                'session_id': challenge.get('session_id', 'unknown')
            }
        
        # Transcribe response
        transcription_result = self.speech_recognizer.transcribe(audio, sample_rate)
        
        # Check if transcription was successful
        if not transcription_result['success']:
            return {
                'verified': False,
                'error': 'Transcription failed',
                'transcription_error': transcription_result.get('error', 'Unknown error'),
                'session_id': challenge.get('session_id', 'unknown')
            }
        
        transcription = transcription_result['transcription']
        
        # Request verification from challenge generator
        # This would normally be a separate API call, but we're simulating it here
        verification_result = self._verify_transcription(transcription, challenge)
        
        # Add transcription to result
        verification_result['transcription'] = transcription
        verification_result['computation_time'] = time.time() - start_time
        
        return verification_result
    
    def _verify_transcription(self, transcription: str, challenge: Dict) -> Dict:
        """
        Verify transcribed text against challenge.
        
        Args:
            transcription: Transcribed response text
            challenge: Challenge dictionary
            
        Returns:
            Dictionary with verification results
        """
        # In a real implementation, this would call the challenge generator's
        # verify_response method. For this example, we'll simulate it.
        
        # Expected response would normally be stored server-side
        # but for this example, assume it's in the challenge object
        if 'expected_response' in challenge:
            expected_response = challenge['expected_response']
        else:
            # In a real system, you would look this up using the verification_hash
            # For this example, we'll return an error
            return {
                'verified': False,
                'error': 'Expected response not found',
                'session_id': challenge.get('session_id', 'unknown')
            }
        
        # Normalize texts
        transcription_norm = self._normalize_text(transcription)
        expected_norm = self._normalize_text(expected_response)
        
        # Calculate similarity
        similarity = self._calculate_text_similarity(expected_norm, transcription_norm)
        
        # Determine verification threshold based on difficulty
        difficulty = challenge.get('difficulty', 'medium')
        if difficulty == 'easy':
            threshold = 0.7
        elif difficulty == 'medium':
            threshold = 0.8
        else:  # hard
            threshold = 0.9
        
        # Determine verification result
        verified = similarity >= threshold
        
        # Prepare result
        result = {
            'verified': verified,
            'similarity': similarity,
            'threshold': threshold,
            'session_id': challenge.get('session_id', 'unknown'),
            'challenge_type': challenge.get('challenge_type', 'unknown'),
            'response_received': transcription,
            'verification_time': int(time.time())
        }
        
        return result
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Remove punctuation and extra whitespace
        translator = str.maketrans('', '', '.,;:!?-')
        normalized = text.translate(translator)
        normalized = ' '.join(normalized.split())
        
        return normalized.lower()
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple calculation based on character-level edit distance
        from difflib import SequenceMatcher
        
        return SequenceMatcher(None, text1, text2).ratio()
