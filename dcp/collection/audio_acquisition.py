import pyaudio
import numpy as np
import soundfile as sf
import time
import yaml
import os
import threading
from queue import Queue
from datetime import datetime

class AudioAcquisitionSystem:
    def __init__(self, config_path="config/hardware.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['audio_acquisition']
        
        self.sampling_rate = self.config['sampling_rate']
        self.bit_depth = self.config['bit_depth']
        self.channels = self.config['channels']
        self.format = pyaudio.paFloat32 if self.bit_depth == 32 else pyaudio.paInt16
        self.chunk_size = 1024
        
        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.buffer_queue = Queue()
        
    def get_device_info(self):
        """Get information about available audio devices"""
        device_info = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            device_info.append(info)
        return device_info
    
    def select_device(self, device_index=None):
        """Select recording device by index or find best match for requirements"""
        if device_index is not None:
            return device_index
        
        # Find device with at least the required channels and sample rate
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if (info['maxInputChannels'] >= self.channels and 
                info['defaultSampleRate'] >= self.sampling_rate):
                return i
        
        # If no ideal device found, return default
        return self.pa.get_default_input_device_info()['index']
    
    def start_recording(self, output_path=None, duration=None, device_index=None):
        """Start recording audio from the selected device"""
        if self.recording:
            print("Already recording!")
            return False
        
        self.device_index = self.select_device(device_index)
        self.output_path = output_path or f"recordings/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Callback function for audio stream
        def callback(in_data, frame_count, time_info, status):
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.buffer_queue.put(audio_data)
            return (in_data, pyaudio.paContinue)
        
        # Open audio stream
        self.stream = self.pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.sampling_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=callback
        )
        
        self.recording = True
        
        # Start recording thread
        self.recording_thread = threading.Thread(
            target=self._recording_worker, 
            args=(duration,)
        )
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        print(f"Recording started on device {self.device_index}")
        return True
    
    def _recording_worker(self, duration=None):
        """Worker thread that processes the recording buffer"""
        buffer = []
        start_time = time.time()
        
        while self.recording:
            if not self.buffer_queue.empty():
                data = self.buffer_queue.get()
                buffer.append(data)
            
            if duration and (time.time() - start_time) >= duration:
                self.stop_recording()
        
        # Save the recording
        audio_data = np.concatenate(buffer)
        audio_data = audio_data.reshape(-1, self.channels)
        sf.write(self.output_path, audio_data, self.sampling_rate)
        print(f"Recording saved to {self.output_path}")
        
    def stop_recording(self):
        """Stop the recording process"""
        if not self.recording:
            print("Not recording!")
            return False
        
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        print("Recording stopped")
        return True
    
    def close(self):
        """Clean up resources"""
        if self.recording:
            self.stop_recording()
        self.pa.terminate()
        
    def __del__(self):
        self.close()


class BeamformingArray(AudioAcquisitionSystem):
    """Extension for beamforming microphone array processing"""
    
    def __init__(self, config_path="config/hardware.yaml"):
        super().__init__(config_path)
        self.beamforming_weights = None
        
    def set_beamforming_direction(self, azimuth, elevation):
        """Set beamforming direction with azimuth and elevation angles"""
        # Calculate delay-and-sum beamforming weights
        # Simplified implementation - a real one would account for microphone positions
        directions = np.linspace(0, 2*np.pi, self.channels, endpoint=False)
        delays = 0.5 * np.cos(directions - azimuth)
        
        # Convert delays to weights
        self.beamforming_weights = np.exp(1j * 2 * np.pi * delays)
        return self.beamforming_weights
    
    def process_beamforming(self, audio_data):
        """Apply beamforming to multichannel audio data"""
        if self.beamforming_weights is None:
            self.set_beamforming_direction(0, 0)  # Default: front direction
            
        # Reshape audio data to channels
        channels_data = audio_data.reshape(-1, self.channels)
        
        # Apply FFT to each channel
        fft_data = np.fft.rfft(channels_data, axis=0)
        
        # Apply beamforming weights
        beamformed_fft = np.sum(fft_data * self.beamforming_weights, axis=1)
        
        # Inverse FFT
        beamformed_audio = np.fft.irfft(beamformed_fft)
        
        return beamformed_audio
