import numpy as np
import soundfile as sf
import librosa
from scipy import signal
import yaml
import os
import random

class EnvironmentAugmenter:
    """Acoustic environment simulation for data augmentation"""
    
    def __init__(self, impulse_response_dir=None, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Default impulse response directory
        if impulse_response_dir is None:
            impulse_response_dir = "data/impulse_responses"
        
        self.impulse_response_dir = impulse_response_dir
        self.loaded_irs = {}  # Cache for loaded impulse responses
        
        # Load available impulse responses
        self.available_irs = []
        if os.path.exists(impulse_response_dir):
            for filename in os.listdir(impulse_response_dir):
                if filename.endswith(('.wav', '.flac')):
                    self.available_irs.append(os.path.join(impulse_response_dir, filename))
        
    def add_reverberation(self, y, sr, ir_path=None, wet_level=0.5):
        """
        Add reverberation using convolution with impulse response
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
        ir_path : str, optional
            Path to impulse response file (random if None)
        wet_level : float
            Mix level between dry and wet signal (0-1)
            
        Returns:
        --------
        y_reverb : ndarray
            Reverberated audio signal
        """
        # Select impulse response
        if ir_path is None and self.available_irs:
            ir_path = random.choice(self.available_irs)
        elif ir_path is None:
            # If no IR file available, create synthetic reverb
            return self._add_synthetic_reverb(y, sr, rt60=0.8, wet_level=wet_level)
        
        # Load impulse response (or use cached version)
        if ir_path in self.loaded_irs:
            ir, ir_sr = self.loaded_irs[ir_path]
        else:
            ir, ir_sr = librosa.load(ir_path, sr=None)
            self.loaded_irs[ir_path] = (ir, ir_sr)
        
        # Resample IR if needed
        if ir_sr != sr:
            ir = librosa.resample(ir, orig_sr=ir_sr, target_sr=sr)
        
        # Apply convolution for reverberation
        y_reverb = signal.convolve(y, ir, mode='full')
        
        # Trim to original length
        y_reverb = y_reverb[:len(y)]
        
        # Normalize reverberated signal
        y_reverb = y_reverb / np.max(np.abs(y_reverb)) * np.max(np.abs(y))
        
        # Mix dry and wet signals
        y_out = (1 - wet_level) * y + wet_level * y_reverb
        
        return y_out
    
    def _add_synthetic_reverb(self, y, sr, rt60=0.8, wet_level=0.5):
        """
        Add synthetic reverberation when no IR file is available
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
        rt60 : float
            Reverberation time in seconds
        wet_level : float
            Mix level between dry and wet signal (0-1)
            
        Returns:
        --------
        y_reverb : ndarray
            Reverberated audio signal
        """
        # Create synthetic impulse response
        n_samples = int(rt60 * sr)
        ir = np.random.randn(n_samples)
        ir = ir * np.exp(-6 * np.arange(n_samples) / n_samples)  # Exponential decay
        
        # Normalize IR
        ir = ir / np.sqrt(np.sum(ir**2))
        
        # Apply convolution
        y_reverb = signal.convolve(y, ir, mode='full')
        
        # Trim to original length
        y_reverb = y_reverb[:len(y)]
        
        # Normalize and mix
        y_reverb = y_reverb / np.max(np.abs(y_reverb)) * np.max(np.abs(y))
        y_out = (1 - wet_level) * y + wet_level * y_reverb
        
        return y_out
    
    def add_background_noise(self, y, sr, noise_path=None, snr_db=20):
        """
        Add background noise at specified SNR level
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
        noise_path : str, optional
            Path to noise file (colored noise if None)
        snr_db : float
            Signal-to-noise ratio in dB
            
        Returns:
        --------
        y_noisy : ndarray
            Noisy audio signal
        """
        # If no noise file specified, generate colored noise
        if noise_path is None:
            return self._add_colored_noise(y, sr, snr_db)
        
        # Load noise file
        noise, noise_sr = librosa.load(noise_path, sr=None)
        
        # Resample noise if needed
        if noise_sr != sr:
            noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=sr)
        
        # Make noise the same length as the signal
        if len(noise) < len(y):
            # Repeat noise to cover the signal
            noise = np.tile(noise, int(np.ceil(len(y) / len(noise))))
            noise = noise[:len(y)]
        elif len(noise) > len(y):
            # Take a random segment of noise
            start = np.random.randint(0, len(noise) - len(y))
            noise = noise[start:start + len(y)]
        
        # Calculate signal and noise power
        signal_power = np.mean(y**2)
        noise_power = np.mean(noise**2)
        
        # Calculate noise scale factor for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = np.sqrt(signal_power / (noise_power * snr_linear))
        
        # Scale noise and add to signal
        scaled_noise = noise_scale * noise
        y_noisy = y + scaled_noise
        
        # Normalize to avoid clipping
        y_noisy = y_noisy / np.max(np.abs(y_noisy)) * np.max(np.abs(y))
        
        return y_noisy
    
    def _add_colored_noise(self, y, sr, snr_db=20, noise_type='pink'):
        """
        Add colored noise at specified SNR level
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
        snr_db : float
            Signal-to-noise ratio in dB
        noise_type : str
            Type of colored noise ('white', 'pink', 'brown', 'blue')
            
        Returns:
        --------
        y_noisy : ndarray
            Noisy audio signal
        """
        # Generate white noise
        noise = np.random.randn(len(y))
        
        # Apply coloration if needed
        if noise_type != 'white':
            # Shape noise spectrum
            noise_fft = np.fft.rfft(noise)
            freqs = np.fft.rfftfreq(len(noise), 1/sr)
            
            if noise_type == 'pink':
                # Pink noise: 1/f spectrum
                noise_fft *= 1 / np.sqrt(freqs + 1e-10)
            elif noise_type == 'brown':
                # Brown noise: 1/f^2 spectrum
                noise_fft *= 1 / (freqs + 1e-10)
            elif noise_type == 'blue':
                # Blue noise: f^0.5 spectrum
                noise_fft *= np.sqrt(freqs + 1e-10)
            
            # Inverse FFT
            noise = np.fft.irfft(noise_fft, len(noise))
        
        # Calculate signal and noise power
        signal_power = np.mean(y**2)
        noise_power = np.mean(noise**2)
        
        # Calculate noise scale factor for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = np.sqrt(signal_power / (noise_power * snr_linear))
        
        # Scale noise and add to signal
        scaled_noise = noise_scale * noise
        y_noisy = y + scaled_noise
        
        # Normalize to avoid clipping
        y_noisy = y_noisy / np.max(np.abs(y_noisy)) * np.max(np.abs(y))
        
        return y_noisy
    
    def simulate_room_acoustics(self, y, sr, room_dim=[5, 4, 3], source_pos=None, mic_pos=None, rt60=0.5):
        """
        Simulate room acoustics using the image method
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
        room_dim : list
            Room dimensions in meters [x, y, z]
        source_pos : list
            Source position in meters [x, y, z]
        mic_pos : list
            Microphone position in meters [x, y, z]
        rt60 : float
            Desired RT60 in seconds
            
        Returns:
        --------
        y_room : ndarray
            Audio with room acoustics applied
        """
        try:
            import pyroomacoustics as pra
        except ImportError:
            print("pyroomacoustics not installed. Using simple synthetic reverb instead.")
            return self._add_synthetic_reverb(y, sr, rt60)
        
        # Default positions if not specified
        if source_pos is None:
            source_pos = [room_dim[0]/2, room_dim[1]/2, 1.7]  # Center of room, 1.7m high
        
        if mic_pos is None:
            mic_pos = [room_dim[0]/2 + 1.0, room_dim[1]/2, 1.7]  # 1m away from source
        
        # Create room
        room = pra.ShoeBox(
            room_dim,
            fs=sr,
            materials=pra.Material(1.0, rt60),
            max_order=10
        )
        
        # Add source and microphone
        room.add_source(source_pos, signal=y)
        room.add_microphone(mic_pos)
        
        # Compute room impulse response and reverberant signal
        room.compute_rir()
        room.simulate()
        
        # Get reverberant signal
        y_room = room.mic_array.signals[0, :]
        
        # Ensure same length as input
        if len(y_room) > len(y):
            y_room = y_room[:len(y)]
        elif len(y_room) < len(y):
            y_room = np.pad(y_room, (0, len(y) - len(y_room)))
        
        # Normalize
        y_room = y_room / np.max(np.abs(y_room)) * np.max(np.abs(y))
        
        return y_room
