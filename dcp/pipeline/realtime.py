import numpy as np
import soundfile as sf
import pyaudio
import threading
import queue
import time
import librosa
import yaml
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import os

# Import preprocessing modules
from preprocessing.noise_suppression import NoiseSuppressionFactory
from preprocessing.vad import VADFactory
from preprocessing.normalization import NormalizationFactory

# Import feature extraction modules
from features.acoustic import AcousticFeatureFactory
from features.prosodic import ProsodicFeatureFactory

class RealtimeAudioProcessor:
    """Real-time audio processing system with streaming capability"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Audio parameters
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 2048
        self.format = pyaudio.paFloat32
        
        # Processing parameters
        self.buffer_duration = 1.0  # seconds
        self.buffer_samples = int(self.buffer_duration * self.sample_rate)
        self.overlap = 0.5  # 50% overlap
        self.hop_samples = int(self.buffer_samples * (1 - self.overlap))
        
        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()
        
        # Initialize buffers
        self.audio_buffer = np.zeros(self.buffer_samples)
        self.feature_buffer = deque(maxlen=100)  # Store 100 frames of features
        
        # Initialize processing components
        self._init_processors()
        
        # Initialize streams
        self.input_stream = None
        self.output_stream = None
        
        # Processing thread and control flags
        self.processing_thread = None
        self.is_running = False
        self.is_processing = False
        self.process_output = False
        
        # Processing queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Visualization data
        self.visualization_data = {
            'waveform': np.zeros(self.buffer_samples),
            'spectrogram': np.zeros((257, 32)),  # Default sizes
            'pitch': np.zeros(100),
            'mfcc': np.zeros((20, 32)),
            'vad': np.zeros(100, dtype=bool)
        }
        
        # Event callbacks
        self.on_feature_update = None
    
    def _init_processors(self):
        """Initialize processing components"""
        # Noise suppression
        noise_method = self.config['preprocessing']['noise_suppression'].get('method', 'spectral_subtraction')
        self.noise_suppressor = NoiseSuppressionFactory.create_suppressor(noise_method)
        
        # Voice activity detection
        vad_method = self.config['preprocessing']['vad'].get('model', 'energy')
        self.vad = VADFactory.create_vad(vad_method)
        
        # Feature extractors
        self.feature_extractors = {
            'mfcc': AcousticFeatureFactory.create_extractor('mfcc'),
            'pitch': ProsodicFeatureFactory.create_extractor('yaapt')
        }
    
    def select_device(self, device_index=None):
        """Select input device by index or find suitable device"""
        if device_index is not None:
            return device_index
        
        # Find device with at least the required channels and sample rate
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if (info['maxInputChannels'] >= self.channels and 
                info['defaultSampleRate'] >= self.sample_rate):
                return i
        
                # If no ideal device found, return default
        return self.pa.get_default_input_device_info()['index']
    
    def start_streaming(self, input_device=None, output_device=None, process_output=False):
        """Start audio streaming and processing"""
        if self.is_running:
            print("Streaming already in progress!")
            return False
        
        self.is_running = True
        self.is_processing = True
        self.process_output = process_output
        
        # Select devices
        input_device_idx = self.select_device(input_device)
        
        # Input stream callback
        def input_callback(in_data, frame_count, time_info, status):
            if self.is_running:
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                self.input_queue.put(audio_data)
            return (None, pyaudio.paContinue)
        
        # Start input stream
        self.input_stream = self.pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=input_device_idx,
            frames_per_buffer=self.chunk_size,
            stream_callback=input_callback
        )
        
        # Start output stream if processing output
        if self.process_output:
            if output_device is None:
                output_device_idx = self.pa.get_default_output_device_info()['index']
            else:
                output_device_idx = output_device
            
            # Output stream callback
            def output_callback(in_data, frame_count, time_info, status):
                if not self.is_running or self.output_queue.empty():
                    # Return silence if not running or no data
                    return (np.zeros(frame_count * self.channels, dtype=np.float32), pyaudio.paContinue)
                
                try:
                    output_data = self.output_queue.get_nowait()
                    return (output_data, pyaudio.paContinue)
                except queue.Empty:
                    return (np.zeros(frame_count * self.channels, dtype=np.float32), pyaudio.paContinue)
            
            self.output_stream = self.pa.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=output_device_idx,
                frames_per_buffer=self.chunk_size,
                stream_callback=output_callback
            )
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print(f"Streaming started on device {input_device_idx}")
        return True
    
    def stop_streaming(self):
        """Stop audio streaming and processing"""
        if not self.is_running:
            print("Not streaming!")
            return False
        
        # Set flags to stop
        self.is_running = False
        self.is_processing = False
        
        # Stop streams
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
        
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
        
        # Clear queues
        while not self.input_queue.empty():
            self.input_queue.get()
        
        while not self.output_queue.empty():
            self.output_queue.get()
        
        print("Streaming stopped")
        return True
    
    def _processing_loop(self):
        """Background processing thread"""
        while self.is_processing:
            try:
                # Check if there's data in the queue
                if not self.input_queue.empty():
                    # Get chunk from queue
                    chunk = self.input_queue.get_nowait()
                    
                    # Update audio buffer with overlap
                    new_samples = len(chunk)
                    self.audio_buffer = np.roll(self.audio_buffer, -new_samples)
                    self.audio_buffer[-new_samples:] = chunk
                    
                    # Process the buffer
                    self._process_audio_buffer()
                    
                    # Update visualization data
                    self.visualization_data['waveform'] = self.audio_buffer.copy()
                    
                    # If processing output, send to output queue
                    if self.process_output:
                        self.output_queue.put(chunk)
                    
                else:
                    # No data in queue, sleep briefly
                    time.sleep(0.01)
            
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _process_audio_buffer(self):
        """Process the current audio buffer"""
        # Apply noise suppression
        clean_buffer = self.noise_suppressor.suppress_noise(self.audio_buffer, self.sample_rate)
        
        # Apply VAD
        vad_mask = self.vad.detect_speech(clean_buffer, self.sample_rate)
        self.visualization_data['vad'] = np.roll(self.visualization_data['vad'], -1)
        self.visualization_data['vad'][-1] = np.mean(vad_mask) > 0.3
        
        # Extract features
        features = {}
        
        # Extract MFCC
        mfcc_features = self.feature_extractors['mfcc'].extract(clean_buffer, self.sample_rate)
        if 'mfcc' in mfcc_features and mfcc_features['mfcc'].size > 0:
            features['mfcc'] = mfcc_features['mfcc']
            self.visualization_data['mfcc'] = features['mfcc']
        
        # Extract pitch
        pitch_features = self.feature_extractors['pitch'].extract(clean_buffer, self.sample_rate)
        if 'f0' in pitch_features and pitch_features['f0'].size > 0:
            features['pitch'] = pitch_features['f0']
            self.visualization_data['pitch'] = np.roll(self.visualization_data['pitch'], -1)
            self.visualization_data['pitch'][-1] = np.mean(pitch_features['f0'][pitch_features['f0'] > 0]) if np.any(pitch_features['f0'] > 0) else 0
        
        # Calculate spectrogram
        S = np.abs(librosa.stft(clean_buffer, n_fft=512, hop_length=128))
        self.visualization_data['spectrogram'] = librosa.amplitude_to_db(S, ref=np.max)
        
        # Store features
        if features:
            self.feature_buffer.append(features)
            
            # Trigger update callback if registered
            if self.on_feature_update is not None:
                self.on_feature_update(features)
    
    def get_visualization_data(self):
        """Get current visualization data"""
        return self.visualization_data
    
    def get_recent_features(self, n_frames=10):
        """Get the most recent N frames of features"""
        n_frames = min(n_frames, len(self.feature_buffer))
        return list(self.feature_buffer)[-n_frames:]
    
    def close(self):
        """Clean up resources"""
        self.stop_streaming()
        self.pa.terminate()


class RealtimeVoiceAnalyzer(RealtimeAudioProcessor):
    """Real-time voice analysis with GUI visualization"""
    
    def __init__(self, config_path="config/processing.yaml"):
        super().__init__(config_path)
        
        # Initialize GUI resources
        self.root = None
        self.canvas = None
        self.figure = None
        self.animation = None
        
        # Analysis thresholds
        self.voice_thresholds = {
            'pitch_min': 75,    # Hz (low male voice)
            'pitch_max': 400,   # Hz (high female voice)
            'speech_energy': 0.05  # Minimum energy to consider speech
        }
    
    def start_gui(self):
        """Initialize and start the GUI"""
        # Create Tkinter root
        self.root = tk.Tk()
        self.root.title("VoiceVault Real-time Analyzer")
        self.root.geometry("1000x600")
        
        # Create figure for plotting
        self.figure = Figure(figsize=(10, 8), dpi=100)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        self.ax_waveform = self.figure.add_subplot(4, 1, 1)
        self.ax_spectrogram = self.figure.add_subplot(4, 1, 2)
        self.ax_pitch = self.figure.add_subplot(4, 1, 3)
        self.ax_vad = self.figure.add_subplot(4, 1, 4)
        
        self.figure.tight_layout(pad=2.0)
        
        # Create animation
        self.animation = animation.FuncAnimation(
            self.figure, self._update_plots, interval=100, blit=False
        )
        
        # Create control frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add buttons
        start_button = tk.Button(control_frame, text="Start", command=self.start_streaming)
        start_button.pack(side=tk.LEFT, padx=5)
        
        stop_button = tk.Button(control_frame, text="Stop", command=self.stop_streaming)
        stop_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=5)
        
        # Register feature update callback
        self.on_feature_update = self._handle_feature_update
        
        # Start the GUI main loop
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()
    
    def _update_plots(self, frame):
        """Update all plots with current data"""
        # Get current visualization data
        data = self.get_visualization_data()
        
        # Update waveform plot
        self.ax_waveform.clear()
        self.ax_waveform.set_title("Waveform")
        self.ax_waveform.plot(data['waveform'])
        self.ax_waveform.set_ylim([-1, 1])
        self.ax_waveform.set_xlabel("Samples")
        self.ax_waveform.set_ylabel("Amplitude")
        
        # Update spectrogram plot
        self.ax_spectrogram.clear()
        self.ax_spectrogram.set_title("Spectrogram")
        self.ax_spectrogram.imshow(
            data['spectrogram'], 
            aspect='auto', 
            origin='lower',
            cmap='inferno'
        )
        self.ax_spectrogram.set_xlabel("Time")
        self.ax_spectrogram.set_ylabel("Frequency")
        
        # Update pitch plot
        self.ax_pitch.clear()
        self.ax_pitch.set_title("Pitch (F0)")
        self.ax_pitch.plot(data['pitch'])
        self.ax_pitch.set_ylim([0, 500])
        self.ax_pitch.set_xlabel("Frames")
        self.ax_pitch.set_ylabel("Frequency (Hz)")
        
        # Add pitch range reference lines
        self.ax_pitch.axhline(y=self.voice_thresholds['pitch_min'], color='g', linestyle='--', alpha=0.5)
        self.ax_pitch.axhline(y=self.voice_thresholds['pitch_max'], color='r', linestyle='--', alpha=0.5)
        
        # Update VAD plot
        self.ax_vad.clear()
        self.ax_vad.set_title("Voice Activity Detection")
        self.ax_vad.fill_between(range(len(data['vad'])), 0, data['vad'].astype(float), color='g', alpha=0.5)
        self.ax_vad.set_ylim([0, 1])
        self.ax_vad.set_xlabel("Frames")
        self.ax_vad.set_yticks([0, 1])
        self.ax_vad.set_yticklabels(["Silent", "Speech"])
        
        self.figure.tight_layout()
    
    def _handle_feature_update(self, features):
        """Handle new features, update analysis and status"""
        # Check if voice detected
        if 'pitch' in features and np.any(features['pitch'] > 0):
            valid_pitch = features['pitch'][features['pitch'] > 0]
            mean_pitch = np.mean(valid_pitch)
            
            # Update status with pitch information
            if mean_pitch < self.voice_thresholds['pitch_min']:
                self.status_var.set(f"Low pitch detected: {mean_pitch:.1f} Hz")
            elif mean_pitch > self.voice_thresholds['pitch_max']:
                self.status_var.set(f"High pitch detected: {mean_pitch:.1f} Hz")
            else:
                self.status_var.set(f"Normal pitch: {mean_pitch:.1f} Hz")
        elif self.visualization_data['vad'][-1]:
            self.status_var.set("Speech detected (No pitch)")
        else:
            self.status_var.set("No speech detected")
    
    def _on_close(self):
        """Handle window close event"""
        self.stop_streaming()
        self.close()
        self.root.destroy()

