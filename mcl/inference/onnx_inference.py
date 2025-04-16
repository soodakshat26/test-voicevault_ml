# inference/onnx_inference.py
import numpy as np
import onnxruntime as ort
import time
from typing import Dict, List, Tuple, Any

class ONNXInferenceEngine:
    """
    ONNX inference engine for voice authentication
    
    This class provides optimized inference using ONNX Runtime.
    """
    def __init__(
        self,
        auth_model_path: str,
        antispoofing_model_path: str,
        device: str = 'cpu'
    ):
        # Set up ONNX Runtime session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set execution provider based on device
        providers = []
        if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Create ONNX Runtime sessions
        self.auth_session = ort.InferenceSession(
            auth_model_path, 
            sess_options=session_options,
            providers=providers
        )
        
        self.antispoofing_session = ort.InferenceSession(
            antispoofing_model_path,
            sess_options=session_options,
            providers=providers
        )
        
        # Get model metadata
        self.auth_input_name = self.auth_session.get_inputs()[0].name
        self.auth_output_names = [output.name for output in self.auth_session.get_outputs()]
        
        self.antispoofing_input_name = self.antispoofing_session.get_inputs()[0].name
        self.antispoofing_output_names = [output.name for output in self.antispoofing_session.get_outputs()]
        
        # Performance tracking
        self.inference_times = {
            'auth': [],
            'antispoofing': []
        }
    
    def authenticate(
        self,
        audio: np.ndarray,
        enrolled_embedding: np.ndarray
    ) -> Dict:
        """
        Authenticate a user based on voice
        
        Args:
            audio: Input audio waveform as numpy array
            enrolled_embedding: Enrolled user embedding
            
        Returns:
            Authentication results
        """
        # Ensure audio has correct dimensions (batch, channels, time)
        if audio.ndim == 1:
            audio = audio.reshape(1, 1, -1)
        elif audio.ndim == 2:
            audio = audio.reshape(1, *audio.shape)
        
        # Ensure correct data type
        audio = audio.astype(np.float32)
        
        # Speaker authentication
        auth_start = time.time()
        auth_outputs = self.auth_session.run(
            self.auth_output_names,
            {self.auth_input_name: audio}
        )
        auth_end = time.time()
        
        # Extract embedding from authentication outputs
        embedding_idx = self.auth_output_names.index('embedding') if 'embedding' in self.auth_output_names else 1
        auth_embedding = auth_outputs[embedding_idx]
        
        # Calculate similarity to enrolled embedding
        similarity = self._calculate_similarity(auth_embedding, enrolled_embedding)
        
        # Anti-spoofing check
        antispoofing_start = time.time()
        antispoofing_outputs = self.antispoofing_session.run(
            self.antispoofing_output_names,
            {self.antispoofing_input_name: audio}
        )
        antispoofing_end = time.time()
        
        # Extract anti-spoofing score
        antispoofing_output = antispoofing_outputs[0]
        real_prob = 1 / (1 + np.exp(-antispoofing_output[0, 0]))  # Sigmoid
        
        # Record inference times
        self.inference_times['auth'].append(auth_end - auth_start)
        self.inference_times['antispoofing'].append(antispoofing_end - antispoofing_start)
        
        # Return results
        result = {
            'similarity': float(similarity),
            'antispoofing_score': float(real_prob),
            'auth_embedding': auth_embedding,
            'inference_time_auth': auth_end - auth_start,
            'inference_time_antispoofing': antispoofing_end - antispoofing_start,
            'total_inference_time': (auth_end - auth_start) + (antispoofing_end - antispoofing_start)
        }
        
        return result
    
    def _calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embedding2, axis=1, keepdims=True)
        
        embedding1_normalized = embedding1 / norm1
        embedding2_normalized = embedding2 / norm2
        
        # Compute cosine similarity
        similarity = np.sum(embedding1_normalized * embedding2_normalized, axis=1)[0]
        
        return similarity
    
    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics"""
        stats = {}
        
        if self.inference_times['auth']:
            stats['auth_avg_time'] = np.mean(self.inference_times['auth'])
            stats['auth_min_time'] = np.min(self.inference_times['auth'])
            stats['auth_max_time'] = np.max(self.inference_times['auth'])
            
        if self.inference_times['antispoofing']:
            stats['antispoofing_avg_time'] = np.mean(self.inference_times['antispoofing'])
            stats['antispoofing_min_time'] = np.min(self.inference_times['antispoofing'])
            stats['antispoofing_max_time'] = np.max(self.inference_times['antispoofing'])
            
        if self.inference_times['auth'] and self.inference_times['antispoofing']:
            total_times = [a + b for a, b in zip(self.inference_times['auth'], self.inference_times['antispoofing'])]
            stats['total_avg_time'] = np.mean(total_times)
            stats['total_min_time'] = np.min(total_times)
            stats['total_max_time'] = np.max(total_times)
            
        return stats
