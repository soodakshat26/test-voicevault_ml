import torch
import torchaudio
import numpy as np
import random
from typing import Optional, Tuple, List, Union
import math

class AudioAugmenter:
    """Audio augmentation for speaker verification data"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio augmenter
        
        Args:
            sample_rate: Default sample rate for the audio
        """
        self.sample_rate = sample_rate
        
        # Load RIRs if available
        try:
            self.rir_list = self._load_rir_list()
        except:
            self.rir_list = None
        
        # Load noise samples if available
        try:
            self.noise_list = self._load_noise_list()
        except:
            self.noise_list = None
    
    def _load_rir_list(self):
        """Load room impulse responses"""
        try:
            import soundfile as sf
            import glob
            import os
            
            # Search for RIR files
            rir_path = os.environ.get('RIR_PATH', './data/rir')
            rir_files = glob.glob(f"{rir_path}/**/*.wav", recursive=True)
            
            if not rir_files:
                return None
            
            # Load a subset of RIRs for efficiency
            rir_list = []
            for file in random.sample(rir_files, min(len(rir_files), 100)):
                try:
                    rir, sr = sf.read(file)
                    # Convert to mono if stereo
                    if len(rir.shape) > 1:
                        rir = rir[:, 0]
                    # Resample if necessary
                    if sr != self.sample_rate:
                        rir = torchaudio.functional.resample(
                            torch.tensor(rir), 
                            orig_freq=sr, 
                            new_freq=self.sample_rate
                        ).numpy()
                    rir_list.append(rir)
                except Exception as e:
                    print(f"Failed to load RIR {file}: {e}")
            
            return rir_list
        except Exception as e:
            print(f"Failed to load RIR list: {e}")
            return None
    
    def _load_noise_list(self):
        """Load noise samples"""
        try:
            import soundfile as sf
            import glob
            import os
            
            # Search for noise files
            noise_path = os.environ.get('NOISE_PATH', './data/noise')
            noise_files = glob.glob(f"{noise_path}/**/*.wav", recursive=True)
            
            if not noise_files:
                return None
            
            # Load a subset of noise files for efficiency
            noise_list = []
            for file in random.sample(noise_files, min(len(noise_files), 50)):
                try:
                    noise, sr = sf.read(file)
                    # Convert to mono if stereo
                    if len(noise.shape) > 1:
                        noise = noise[:, 0]
                    # Resample if necessary
                    if sr != self.sample_rate:
                        noise = torchaudio.functional.resample(
                            torch.tensor(noise), 
                            orig_freq=sr, 
                            new_freq=self.sample_rate
                        ).numpy()
                    noise_list.append(noise)
                except Exception as e:
                    print(f"Failed to load noise {file}: {e}")
            
            return noise_list
        except Exception as e:
            print(f"Failed to load noise list: {e}")
            return None
    
    def add_noise(
        self, 
        waveform: torch.Tensor, 
        snr_db: float = 10.0,
        noise_type: str = 'random'
    ) -> torch.Tensor:
        """
        Add noise to waveform
        
        Args:
            waveform: Input waveform (channels, time) or (time,)
            snr_db: Signal-to-noise ratio in dB
            noise_type: Type of noise ('random', 'white', 'pink', 'babble')
            
        Returns:
            Noisy waveform with the same shape as input
        """
        # Check if waveform is 1D or 2D
        is_1d = waveform.dim() == 1
        if is_1d:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        
        # Get waveform properties
        channels, length = waveform.shape
        device = waveform.device
        
        # Generate noise
        if noise_type == 'white':
            # Generate white noise
            noise = torch.randn(channels, length, device=device)
        elif noise_type == 'pink':
            # Generate pink noise (1/f spectrum)
            noise = torch.randn(channels, length, device=device)
            
            # Apply 1/f filter in frequency domain
            for c in range(channels):
                noise_fft = torch.fft.rfft(noise[c])
                freqs = torch.fft.rfftfreq(length, d=1/self.sample_rate)
                
                # Avoid division by zero at DC
                freqs[0] = freqs[1]
                
                # Apply 1/f filter
                noise_fft = noise_fft / torch.sqrt(freqs)
                noise[c] = torch.fft.irfft(noise_fft, n=length)
        elif noise_type == 'babble' and self.noise_list is not None:
            # Use pre-recorded babble noise if available
            noise_idx = random.randrange(len(self.noise_list))
            noise_sample = self.noise_list[noise_idx]
            
            # Make sure noise is long enough
            if len(noise_sample) < length:
                # Repeat noise to match length
                repeats = math.ceil(length / len(noise_sample))
                noise_sample = np.tile(noise_sample, repeats)
            
            # Randomly crop noise to match length
            start = random.randint(0, len(noise_sample) - length)
            noise_sample = noise_sample[start:start+length]
            
            # Convert to torch tensor
            noise = torch.tensor(noise_sample, device=device).float()
            
            # Expand to match channels
            noise = noise.unsqueeze(0).expand(channels, -1)
        else:
            # Default to white noise if no specific type or resources available
            noise = torch.randn(channels, length, device=device)
        
        # Calculate signal and noise power
        signal_power = waveform.pow(2).mean(dim=1, keepdim=True)
        noise_power = noise.pow(2).mean(dim=1, keepdim=True)
        
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)
        
        # Scale noise to achieve desired SNR
        scale = torch.sqrt(signal_power / (noise_power * snr_linear))
        scaled_noise = noise * scale
        
        # Add noise to signal
        noisy_waveform = waveform + scaled_noise
        
        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(noisy_waveform))
        if max_val > 1.0:
            noisy_waveform = noisy_waveform / max_val
        
        # Return with original dimensions
        if is_1d:
            noisy_waveform = noisy_waveform.squeeze(0)
            
        return noisy_waveform
    
    def add_reverberation(
        self, 
        waveform: torch.Tensor, 
        sample_rate: int = None,
        reverb_intensity: float = None
    ) -> torch.Tensor:
        """
        Add reverberation to waveform using impulse responses
        
        Args:
            waveform: Input waveform (channels, time) or (time,)
            sample_rate: Sample rate of the waveform
            reverb_intensity: Reverb intensity (0 to 1). If None, random value used.
            
        Returns:
            Reverberated waveform with the same shape as input
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Check if waveform is 1D or 2D
        is_1d = waveform.dim() == 1
        if is_1d:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        
        # Get waveform properties
        channels, length = waveform.shape
        device = waveform.device
        
        # Generate synthetic RIR if no real ones available
        if self.rir_list is None or random.random() < 0.5:
            # Use synthetic reverberation
            # Generate exponentially decaying noise
            if reverb_intensity is None:
                reverb_intensity = random.uniform(0.1, 0.6)
                
            rt60 = reverb_intensity * 1.0  # RT60 in seconds
            ir_length = int(rt60 * sample_rate)
            
            # Create exponential decay
            decay = torch.exp(torch.linspace(0, -12, ir_length, device=device))
            
            # Generate random impulse response with decay
            ir = torch.randn(ir_length, device=device) * decay
            
            # Normalize IR
            ir = ir / torch.sqrt(torch.sum(ir**2))
        else:
            # Use real room impulse response
            rir_idx = random.randrange(len(self.rir_list))
            ir = torch.tensor(self.rir_list[rir_idx], device=device).float()
            
            # Trim excessively long IRs
            max_ir_length = int(1.0 * sample_rate)  # Max 1 second
            if len(ir) > max_ir_length:
                ir = ir[:max_ir_length]
            
            # Scale IR intensity if specified
            if reverb_intensity is not None:
                # Scale decay rate
                ir_length = len(ir)
                decay_factor = torch.exp(torch.linspace(0, -1, ir_length, device=device))
                decay_factor = torch.pow(decay_factor, reverb_intensity * 2)
                ir = ir * decay_factor
            
            # Normalize IR
            ir = ir / torch.sqrt(torch.sum(ir**2))
        
        # Apply convolution
        reverberated = torch.zeros_like(waveform)
        for c in range(channels):
            # Pad IR to ensure proper convolution
            ir_padded = torch.nn.functional.pad(ir, (0, length - 1))
            
            # Compute convolution in frequency domain
            waveform_fft = torch.fft.rfft(waveform[c])
            ir_fft = torch.fft.rfft(ir_padded)
            reverberated_fft = waveform_fft * ir_fft
            reverberated[c] = torch.fft.irfft(reverberated_fft, n=length + len(ir) - 1)[:length]
        
        # Mix with dry signal for natural sound
        wet_gain = random.uniform(0.4, 0.8)
        dry_gain = 1.0 - wet_gain
        mixed = dry_gain * waveform + wet_gain * reverberated
        
        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val
        
        # Return with original dimensions
        if is_1d:
            mixed = mixed.squeeze(0)
            
        return mixed
    
    def change_pitch(
        self, 
        waveform: torch.Tensor, 
        sample_rate: int = None,
        semitones: float = None
    ) -> torch.Tensor:
        """
        Change the pitch of the waveform
        
        Args:
            waveform: Input waveform (channels, time) or (time,)
            sample_rate: Sample rate of the waveform
            semitones: Pitch shift in semitones. If None, random value used.
            
        Returns:
            Pitch-shifted waveform with the same shape as input
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Random pitch shift if not specified
        if semitones is None:
            semitones = random.uniform(-2.0, 2.0)
        
        # Check if waveform is 1D or 2D
        is_1d = waveform.dim() == 1
        if is_1d:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        
        # Get device
        device = waveform.device
        
        # Convert waveform to CPU for torchaudio
        waveform_cpu = waveform.cpu()
        
        # Apply pitch shift
        try:
            shifted = torchaudio.functional.pitch_shift(
                waveform_cpu, 
                sample_rate, 
                semitones
            )
        except:
            # Fallback method if torchaudio fails
            step_ratio = 2 ** (semitones / 12)
            
            # First resample to change pitch
            shifted = torchaudio.functional.resample(
                waveform_cpu,
                orig_freq=sample_rate,
                new_freq=int(sample_rate * step_ratio)
            )
            
            # Then resample back to original length
            shifted = torchaudio.functional.resample(
                shifted,
                orig_freq=int(sample_rate * step_ratio),
                new_freq=sample_rate
            )
        
        # Move back to original device
        shifted = shifted.to(device)
        
        # Match length of original waveform
        if shifted.size(1) > waveform.size(1):
            shifted = shifted[:, :waveform.size(1)]
        elif shifted.size(1) < waveform.size(1):
            shifted = torch.nn.functional.pad(
                shifted, (0, waveform.size(1) - shifted.size(1))
            )
        
        # Return with original dimensions
        if is_1d:
            shifted = shifted.squeeze(0)
            
        return shifted
    
    def change_speed(
        self, 
        waveform: torch.Tensor, 
        speed_factor: float = None
    ) -> torch.Tensor:
        """
        Change the speed of the waveform
        
        Args:
            waveform: Input waveform (channels, time) or (time,)
            speed_factor: Speed change factor (e.g., 1.1 for 10% faster). If None, random value used.
            
        Returns:
            Speed-changed waveform with the same shape as input
        """
        # Random speed factor if not specified
        if speed_factor is None:
            speed_factor = random.uniform(0.9, 1.1)
        
        # Check if waveform is 1D or 2D
        is_1d = waveform.dim() == 1
        if is_1d:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        
        # Get waveform properties
        channels, length = waveform.shape
        device = waveform.device
        
        # Calculate new length
        new_length = int(length / speed_factor)
        
        # Apply speed change through resampling
        waveform_cpu = waveform.cpu()
        indices = torch.linspace(0, length - 1, new_length)
        
        # Resample each channel
        resampled = torch.zeros(channels, new_length, device='cpu')
        for c in range(channels):
            resampled[c] = torch.nn.functional.interpolate(
                waveform_cpu[c].unsqueeze(0).unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        # Move back to original device
        resampled = resampled.to(device)
        
        # Match length of original waveform
        if resampled.size(1) > length:
            resampled = resampled[:, :length]
        elif resampled.size(1) < length:
            resampled = torch.nn.functional.pad(
                resampled, (0, length - resampled.size(1))
            )
        
        # Return with original dimensions
        if is_1d:
            resampled = resampled.squeeze(0)
            
        return resampled
    
    def vocal_tract_length_perturbation(
        self, 
        waveform: torch.Tensor,
        sample_rate: int = None,
        alpha: float = None
    ) -> torch.Tensor:
        """
        Apply vocal tract length perturbation (VTLP)
        
        Args:
            waveform: Input waveform (channels, time) or (time,)
            sample_rate: Sample rate of the waveform
            alpha: Warping factor. If None, random value used.
            
        Returns:
            VTLP-applied waveform with the same shape as input
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Random alpha if not specified
        if alpha is None:
            alpha = random.uniform(0.8, 1.2)
        
        # Check if waveform is 1D or 2D
        is_1d = waveform.dim() == 1
        if is_1d:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        
        # Get waveform properties
        channels, length = waveform.shape
        device = waveform.device
        
        # Apply VTLP using spectral transform
        result = torch.zeros_like(waveform)
        
        for c in range(channels):
            # Compute STFT
            spec = torch.stft(
                waveform[c],
                n_fft=512,
                hop_length=128,
                win_length=512,
                window=torch.hann_window(512, device=device),
                return_complex=True
            )
            
            # Apply frequency warping
            freq_bins = spec.shape[0]
            warped_spec = torch.zeros_like(spec)
            
            for k in range(freq_bins):
                # Calculate warped frequency index
                freq = (k / freq_bins) * (sample_rate / 2)
                warped_freq = freq
                
                if alpha != 1.0:
                    warped_freq = freq * alpha
                
                # Map to new bin index
                new_bin = int((warped_freq / (sample_rate / 2)) * freq_bins)
                
                # Ensure within range
                new_bin = max(0, min(new_bin, freq_bins - 1))
                
                # Copy value to warped spectrum
                warped_spec[new_bin] = spec[k]
            
            # Inverse STFT
            result[c] = torch.istft(
                warped_spec,
                n_fft=512,
                hop_length=128,
                win_length=512,
                window=torch.hann_window(512, device=device),
                length=length
            )
        
        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(result))
        if max_val > 1.0:
            result = result / max_val
        
        # Return with original dimensions
        if is_1d:
            result = result.squeeze(0)
            
        return result
    
    def apply_specaugment(
        self, 
        spectrogram: torch.Tensor,
        time_mask_param: int = 70,
        freq_mask_param: int = 10,
        time_masks: int = 2,
        freq_masks: int = 2
    ) -> torch.Tensor:
        """
        Apply SpecAugment to a spectrogram
        
        Args:
            spectrogram: Input spectrogram (batch, freq, time)
            time_mask_param: Maximum time mask length
            freq_mask_param: Maximum frequency mask length
            time_masks: Number of time masks
            freq_masks: Number of frequency masks
            
        Returns:
            Augmented spectrogram with the same shape as input
        """
        augmented = spectrogram.clone()
        batch_size, freq_bins, time_bins = augmented.shape
        
        # Apply time masking
        for i in range(batch_size):
            for _ in range(time_masks):
                t = random.randint(0, time_mask_param)
                t0 = random.randint(0, time_bins - t)
                augmented[i, :, t0:t0+t] = 0
        
        # Apply frequency masking
        for i in range(batch_size):
            for _ in range(freq_masks):
                f = random.randint(0, freq_mask_param)
                f0 = random.randint(0, freq_bins - f)
                augmented[i, f0:f0+f, :] = 0
        
        return augmented
