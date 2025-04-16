import numpy as np
import soundfile as sf
import librosa
from scipy import signal
import yaml
import os
import json

class DeviceAugmenter:
    """Device characteristic emulation for data augmentation"""
    
    def __init__(self, device_profiles_path=None, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Default device profiles path
        if device_profiles_path is None:
            device_profiles_path = "data/device_profiles"
        
        self.device_profiles_path = device_profiles_path
        self.device_profiles = self._load_device_profiles()
        
        # Default device characteristics if no profiles loaded
        if not self.device_profiles:
            self.device_profiles = {
                "smartphone_basic": {
                    "sample_rate": 16000,
                    "bit_depth": 16,
                    "frequency_response": "telephone",
                    "noise_floor": -60
                },
                "telephone": {
                    "sample_rate": 8000,
                    "bit_depth": 16,
                    "frequency_response": "telephone",
                    "noise_floor": -50
                },
                "studio_mic": {
                    "sample_rate": 48000,
                    "bit_depth": 24,
                    "frequency_response": "flat",
                    "noise_floor": -80
                }
            }
    
    def _load_device_profiles(self):
        """Load device profiles from JSON files"""
        profiles = {}
        
        if os.path.exists(self.device_profiles_path):
            for filename in os.listdir(self.device_profiles_path):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(self.device_profiles_path, filename), 'r') as f:
                            profile_data = json.load(f)
                            profile_name = os.path.splitext(filename)[0]
                            profiles[profile_name] = profile_data
                    except Exception as e:
                        print(f"Error loading profile {filename}: {e}")
        
        return profiles
    
    def apply_device_profile(self, y, sr, device_name=None):
        """
        Apply device characteristics to audio signal
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
        device_name : str, optional
            Name of device profile to apply (random if None)
            
        Returns:
        --------
        y_device : ndarray
            Audio with device characteristics applied
        target_sr : int
            Target sample rate for the device
        """
        # Select device profile
        if device_name is None:
            device_name = np.random.choice(list(self.device_profiles.keys()))
        
        if device_name not in self.device_profiles:
            print(f"Device profile {device_name} not found. Using default.")
            device_name = list(self.device_profiles.keys())[0]
        
        profile = self.device_profiles[device_name]
        
        # Apply device characteristics
        # 1. Frequency response
        y_device = self._apply_frequency_response(y, sr, profile)
        
        # 2. Add device-specific noise
        noise_floor_db = profile.get('noise_floor', -60)
        y_device = self._add_device_noise(y_device, noise_floor_db)
        
        # 3. Resample to device sample rate
        target_sr = profile.get('sample_rate', 16000)
        if sr != target_sr:
            y_device = librosa.resample(y_device, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        # 4. Apply bit depth quantization
        bit_depth = profile.get('bit_depth', 16)
        y_device = self._apply_bit_depth(y_device, bit_depth)
        
        # 5. Apply compression artifacts if specified
        if 'codec' in profile:
            y_device = self._apply_codec_simulation(y_device, profile['codec'])
        
        # 6. Apply proximity effect if specified
        if 'proximity_effect' in profile:
            proximity = profile['proximity_effect']
            y_device = self._apply_proximity_effect(y_device, sr, proximity)
        
        return y_device, target_sr
    
    def _apply_frequency_response(self, y, sr, profile):
        """Apply device frequency response"""
        freq_response = profile.get('frequency_response', 'flat')
        
        if freq_response == 'flat':
            return y.copy()
        
        # Apply filter based on frequency response
        if freq_response == 'telephone':
            # Simulate telephone bandwidth (300-3400 Hz)
            b, a = signal.butter(4, [300, 3400], btype='bandpass', fs=sr)
            y_filtered = signal.filtfilt(b, a, y)
            
            # Add some "telephone" EQ
            b, a = signal.butter(2, 1800, btype='lowpass', fs=sr)
            y_filtered = signal.filtfilt(b, a, y_filtered) * 1.2  # Emphasize lows
            
        elif freq_response == 'smartphone':
            # Simulate smartphone mic (emphasis on mids, roll-off at extremes)
            b, a = signal.butter(2, [100, 8000], btype='bandpass', fs=sr)
            y_filtered = signal.filtfilt(b, a, y)
            
            # Add mid-range emphasis
            b, a = signal.butter(2, [1000, 3000], btype='bandpass', fs=sr)
            mid_emphasis = signal.filtfilt(b, a, y)
            y_filtered = y_filtered + 0.3 * mid_emphasis
            
        elif freq_response == 'laptop':
            # Simulate laptop mic (emphasis on high-mids, poor bass)
            b, a = signal.butter(2, [300, 7000], btype='bandpass', fs=sr)
            y_filtered = signal.filtfilt(b, a, y)
            
            # Reduce bass
            b, a = signal.butter(2, 600, btype='highpass', fs=sr)
            y_filtered = signal.filtfilt(b, a, y_filtered)
            
        elif isinstance(freq_response, list) and len(freq_response) > 1:
            # Custom frequency response using list of [freq, gain] pairs
            # Convert to FIR filter
            freqs = [point[0] for point in freq_response]
            gains = [point[1] for point in freq_response]
            
            # Interpolate and normalize gains
            all_freqs = np.linspace(0, sr/2, 512)
            all_gains = np.interp(all_freqs, freqs, gains)
            all_gains = all_gains / np.max(np.abs(all_gains))
            
            # Create FIR filter
            filter_taps = 128
            fir = signal.firwin2(filter_taps, all_freqs, all_gains, fs=sr)
            y_filtered = signal.filtfilt(fir, [1.0], y)
            
        else:
            # No recognized frequency response, return original
            y_filtered = y.copy()
        
        return y_filtered
    
    def _add_device_noise(self, y, noise_floor_db):
        """Add device-specific noise at specified level"""
        # Convert noise floor from dB to linear
        noise_level = 10 ** (noise_floor_db / 20)
        
        # Generate device noise (slightly colored)
        noise = np.random.randn(len(y))
        
        # Apply coloration (slight emphasis on lower frequencies)
        noise_fft = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(len(noise))
        noise_fft *= 1 / np.sqrt(freqs + 0.1)  # Colored noise
        noise = np.fft.irfft(noise_fft, len(noise))
        
        # Normalize and scale noise
        noise = noise / np.std(noise) * noise_level * np.std(y)
        
        # Add noise to signal
        y_noisy = y + noise
        
        return y_noisy
    
    def _apply_bit_depth(self, y, bit_depth):
        """Simulate lower bit depth quantization"""
        if bit_depth >= 24:
            # No need to quantize for high bit depths
            return y
        
        # Calculate quantization steps
        steps = 2 ** bit_depth
        
        # Scale to full range [-1, 1]
        y_max = np.max(np.abs(y))
        if y_max > 0:
            y_scaled = y / y_max
        else:
            return y
        
        # Quantize
        y_quantized = np.round(y_scaled * (steps-1) / 2) * 2 / (steps-1)
        
        # Scale back
        y_out = y_quantized * y_max
        
        return y_out
    
    def _apply_codec_simulation(self, y, codec_type):
        """Simulate compression codec artifacts"""
        if codec_type == 'gsm':
            # Simulate GSM codec artifacts
            # Bandlimit to 4kHz
            b, a = signal.butter(5, 4000, btype='lowpass', fs=8000)
            y_codec = signal.filtfilt(b, a, y)
            
            # Add quantization noise
            y_codec = self._apply_bit_depth(y_codec, 12)
            
            # Add mild spectral distortion
            fft = np.fft.rfft(y_codec)
            fft *= (1 + 0.05 * np.random.randn(len(fft)))
            y_codec = np.fft.irfft(fft, len(y_codec))
            
        elif codec_type == 'mp3_low':
            # Simulate low bitrate MP3 artifacts
            # Add pre-echo
            pre_echo = np.roll(y, 100) * 0.02
            y_codec = y + pre_echo
            
            # Reduce high frequencies with slight distortion
            b, a = signal.butter(3, 6000, btype='lowpass', fs=16000)
            y_codec = signal.filtfilt(b, a, y_codec)
            
            # Add some high-frequency ringing
            fft = np.fft.rfft(y_codec)
            high_freq_idx = int(len(fft) * 0.6)
            fft[high_freq_idx:] *= (1 + 0.1 * np.random.randn(len(fft) - high_freq_idx))
            y_codec = np.fft.irfft(fft, len(y_codec))
            
        else:
            # No codec simulation
            y_codec = y
        
        return y_codec
    
    def _apply_proximity_effect(self, y, sr, proximity_factor=1.0):
        """Simulate microphone proximity effect (bass boost at close distances)"""
        if proximity_factor <= 0:
            return y
        
        # Proximity effect emphasizes low frequencies
        cutoff = 600  # Hz
        gain = 1.0 + 2.0 * proximity_factor  # More boost for closer proximity
        
        # Create low-shelf filter
        b, a = self._low_shelf_filter(cutoff, gain, sr)
        
        # Apply filter
        y_proximity = signal.filtfilt(b, a, y)
        
        return y_proximity
    
    def _low_shelf_filter(self, cutoff, gain, fs):
        """Create a low-shelf filter with specified cutoff and gain"""
        # Convert gain to dB
        gain_db = 20 * np.log10(gain)
        
        # Normalized cutoff frequency
        w0 = 2 * np.pi * cutoff / fs
        
        # Filter coefficients
        alpha = np.sin(w0) / 2 * np.sqrt((gain + 1/gain) * (1/0.707 - 1) + 2)
        
        # Compute filter coefficients
        b0 = gain * ((gain + 1) - (gain - 1) * np.cos(w0) + 2 * np.sqrt(gain) * alpha)
        b1 = 2 * gain * ((gain - 1) - (gain + 1) * np.cos(w0))
        b2 = gain * ((gain + 1) - (gain - 1) * np.cos(w0) - 2 * np.sqrt(gain) * alpha)
        a0 = (gain + 1) + (gain - 1) * np.cos(w0) + 2 * np.sqrt(gain) * alpha
        a1 = -2 * ((gain - 1) + (gain + 1) * np.cos(w0))
        a2 = (gain + 1) + (gain - 1) * np.cos(w0) - 2 * np.sqrt(gain) * alpha
        
        # Normalize by a0
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1, a2]) / a0
        
        return b, a
    
    def apply_telecommunication_codec(self, y, sr, codec_type="gsm"):
        """
        Apply telecommunication codec effects
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
        codec_type : str
            Type of codec ('gsm', 'g711', 'g729', 'voip')
            
        Returns:
        --------
        y_codec : ndarray
            Audio with codec effects applied
        target_sr : int
            Target sample rate for the codec
        """
        # Resample to appropriate sample rate for codec
        target_sr = 8000  # Most voice codecs operate at 8kHz
        
        if sr != target_sr:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        else:
            y_resampled = y.copy()
        
        # Apply codec-specific effects
        if codec_type == "gsm":
            # GSM codec effects
            # Bandlimit to 300-3400 Hz
            b, a = signal.butter(4, [300, 3400], btype='bandpass', fs=target_sr)
            y_codec = signal.filtfilt(b, a, y_resampled)
            
            # Add quantization artifacts
            y_codec = self._apply_bit_depth(y_codec, 12)
            
            # Add some frame-boundary artifacts
            frame_size = 20  # ms
            samples_per_frame = int(target_sr * frame_size / 1000)
            
            for i in range(samples_per_frame, len(y_codec), samples_per_frame):
                if i + 10 < len(y_codec):
                    # Small discontinuity at frame boundaries
                    y_codec[i:i+10] *= (0.95 + 0.1 * np.random.rand())
            
        elif codec_type == "g711":
            # G.711 codec effects (mu-law or A-law companding)
            # Simulate mu-law companding
            mu = 255
            y_codec = np.sign(y_resampled) * np.log(1 + mu * np.abs(y_resampled)) / np.log(1 + mu)
            
            # Add quantization to 8 bits
            y_codec = self._apply_bit_depth(y_codec, 8)
            
            # Inverse mu-law companding
            y_codec = np.sign(y_codec) * (1/mu) * ((1 + mu)**np.abs(y_codec) - 1)
            
            # Apply bandlimiting
            b, a = signal.butter(3, [300, 3400], btype='bandpass', fs=target_sr)
            y_codec = signal.filtfilt(b, a, y_codec)
            
        elif codec_type == "g729":
            # G.729 codec effects
            # Bandlimit to 300-3400 Hz with sharper cutoff
            b, a = signal.butter(6, [300, 3400], btype='bandpass', fs=target_sr)
            y_codec = signal.filtfilt(b, a, y_resampled)
            
            # Add frame-based artifacts
            frame_size = 10  # ms
            samples_per_frame = int(target_sr * frame_size / 1000)
            
            for i in range(samples_per_frame, len(y_codec), samples_per_frame):
                if i + 5 < len(y_codec):
                    # Frame boundary effects
                    y_codec[i:i+5] *= (0.92 + 0.16 * np.random.rand())
            
            # Add some spectral distortion
            fft = np.fft.rfft(y_codec)
            # Add mild randomization to spectrum
            fft *= (1 + 0.04 * np.random.randn(len(fft)))
            y_codec = np.fft.irfft(fft, len(y_codec))
            
        elif codec_type == "voip":
            # Generic VoIP effects including packet loss
            # Apply bandlimiting
            b, a = signal.butter(3, [100, 7000], btype='bandpass', fs=target_sr)
            y_codec = signal.filtfilt(b, a, y_resampled)
            
            # Simulate packet loss
            packet_size = 20  # ms
            samples_per_packet = int(target_sr * packet_size / 1000)
            packet_loss_prob = 0.02  # 2% packet loss
            
            for i in range(0, len(y_codec), samples_per_packet):
                if np.random.rand() < packet_loss_prob:
                    # Simulate lost packet by using previous packet (simple packet loss concealment)
                    if i >= samples_per_packet:
                        prev_packet = y_codec[i-samples_per_packet:i]
                        end_idx = min(i+samples_per_packet, len(y_codec))
                        y_codec[i:end_idx] = np.resize(prev_packet, end_idx-i) * 0.8
            
            # Add some jitter-buffer artifacts
            jitter_prob = 0.05
            for i in range(0, len(y_codec), samples_per_packet):
                if np.random.rand() < jitter_prob:
                    # Small timing jitter
                    jitter_samples = int(np.random.randint(-5, 6))
                    if jitter_samples > 0 and i + samples_per_packet + jitter_samples < len(y_codec):
                        # Slight expansion
                        packet = y_codec[i:i+samples_per_packet]
                        y_codec[i:i+samples_per_packet+jitter_samples] = np.interp(
                            np.linspace(0, 1, samples_per_packet+jitter_samples),
                            np.linspace(0, 1, samples_per_packet),
                            packet
                        )
                    elif jitter_samples < 0 and i + samples_per_packet < len(y_codec):
                        # Slight compression
                        packet = y_codec[i:i+samples_per_packet]
                        y_codec[i:i+samples_per_packet+jitter_samples] = np.interp(
                            np.linspace(0, 1, samples_per_packet+jitter_samples),
                            np.linspace(0, 1, samples_per_packet),
                            packet
                        )
        else:
            # No recognized codec
            y_codec = y_resampled
        
        return y_codec, target_sr
