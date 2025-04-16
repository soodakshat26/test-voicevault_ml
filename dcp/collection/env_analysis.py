import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy import signal
import acoustics
import yaml

class AcousticEnvironmentAnalyzer:
    def __init__(self, config_path="config/hardware.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def measure_impulse_response(self, audio_path, fs=96000, method='exponential_sweep'):
        """
        Measure room impulse response using exponential sweep or MLS method
        
        Parameters:
        -----------
        audio_path : str
            Path to the recorded sweep or MLS response
        fs : int
            Sampling rate
        method : str
            Method used for IR measurement ('exponential_sweep' or 'mls')
            
        Returns:
        --------
        ir : ndarray
            Impulse response
        """
        if method == 'exponential_sweep':
            return self._measure_ir_sweep(audio_path, fs)
        elif method == 'mls':
            return self._measure_ir_mls(audio_path, fs)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _measure_ir_sweep(self, response_path, fs=96000):
        """Measure IR using exponential sweep method"""
        # Load recorded response
        response, _ = sf.read(response_path)
        
        # Generate reference sweep
        duration = 10  # seconds
        f1 = 20
        f2 = 20000
        t = np.linspace(0, duration, int(fs * duration))
        sweep_rate = np.log(f2/f1) / duration
        sweep = np.sin(2 * np.pi * f1 * (np.exp(sweep_rate * t) - 1) / sweep_rate)
        
        # Perform deconvolution
        sweep_fft = np.fft.rfft(sweep)
        response_fft = np.fft.rfft(response)
        
        # Inverse filter
        inv_filter = np.conj(sweep_fft) / (np.abs(sweep_fft)**2 + 1e-10)
        
        # Calculate impulse response
        ir_fft = response_fft * inv_filter
        ir = np.fft.irfft(ir_fft)
        
        return ir
    
    def _measure_ir_mls(self, response_path, fs=96000):
        """Measure IR using Maximum Length Sequence method"""
        # Load recorded response
        response, _ = sf.read(response_path)
        
        # Generate MLS sequence (order 16)
        order = 16
        mls = acoustics.signal.mls(order)
        
        # Cross-correlation to get impulse response
        ir = signal.correlate(response, mls, mode='full')
        
        # Normalize
        ir = ir / np.max(np.abs(ir))
        
        return ir
    
    def calculate_rt60(self, ir, fs=96000, bands=None):
        """
        Calculate reverberation time (RT60) from impulse response
        
        Parameters:
        -----------
        ir : ndarray
            Impulse response
        fs : int
            Sampling rate
        bands : list
            Frequency bands for RT60 calculation (default: octave bands)
            
        Returns:
        --------
        rt60 : dict
            RT60 values for each frequency band
        """
        if bands is None:
            bands = [125, 250, 500, 1000, 2000, 4000, 8000]
        
        # Calculate energy decay curve
        ir_squared = ir**2
        edc = np.cumsum(ir_squared[::-1])[::-1]
        edc_db = 10 * np.log10(edc / np.max(edc) + 1e-10)
        
        rt60_values = {}
        
        for band in bands:
            # Filter IR for the frequency band
            band_low = band / np.sqrt(2)
            band_high = band * np.sqrt(2)
            sos = signal.butter(4, [band_low, band_high], btype='bandpass', 
                                fs=fs, output='sos')
            ir_filtered = signal.sosfilt(sos, ir)
            
            # Calculate energy decay curve for the band
            ir_filtered_squared = ir_filtered**2
            edc_band = np.cumsum(ir_filtered_squared[::-1])[::-1]
            edc_band_db = 10 * np.log10(edc_band / np.max(edc_band) + 1e-10)
            
            # Find -5 and -35 dB points for T30 calculation
            idx_5db = np.where(edc_band_db <= -5)[0]
            idx_35db = np.where(edc_band_db <= -35)[0]
            
            if len(idx_5db) > 0 and len(idx_35db) > 0:
                t_5db = idx_5db[0] / fs
                t_35db = idx_35db[0] / fs
                
                # Calculate RT60 by extending T30 measurement
                rt60 = 2 * (t_35db - t_5db)
                rt60_values[band] = rt60
            else:
                rt60_values[band] = None
        
        return rt60_values
    
    def measure_background_noise(self, audio_path, fs=96000):
        """
        Measure background noise level in dB(A)
        
        Parameters:
        -----------
        audio_path : str
            Path to audio recording of background noise
        fs : int
            Sampling rate
            
        Returns:
        --------
        leq : float
            A-weighted equivalent sound level (LAeq)
        spectrum : ndarray
            Noise spectrum
        """
        # Load audio
        audio, _ = sf.read(audio_path)
        
        # Apply A-weighting
        sos = signal.butter(10, [20, 20000], btype='bandpass', fs=fs, output='sos')
        audio_filtered = signal.sosfilt(sos, audio)
        
        # A-weighting filter
        b, a = signal.bilinear(*signal.A_weighting(fs))
        audio_a_weighted = signal.lfilter(b, a, audio_filtered)
        
        # Calculate LEQ
        rms = np.sqrt(np.mean(audio_a_weighted**2))
        leq = 20 * np.log10(rms/2e-5)  # Reference: 20 µPa
        
        # Calculate spectrum
        f, spectrum = signal.welch(audio, fs, nperseg=8192)
        
        return leq, (f, spectrum)
    
    def spatial_analysis(self, multi_channel_ir, mic_positions, fs=96000):
        """
        Analyze spatial acoustic characteristics using multiple IRs
        
        Parameters:
        -----------
        multi_channel_ir : ndarray
            Multi-channel impulse response
        mic_positions : ndarray
            3D positions of microphones
        fs : int
            Sampling rate
            
        Returns:
        --------
        spatial_params : dict
            Spatial acoustic parameters
        """
        # Number of channels
        n_channels = multi_channel_ir.shape[1] if len(multi_channel_ir.shape) > 1 else 1
        
        # Calculate inter-aural cross correlation (IACC) if we have at least 2 channels
        if n_channels >= 2:
            # Use first two channels as "ears"
            ir_left = multi_channel_ir[:, 0]
            ir_right = multi_channel_ir[:, 1]
            
            # Calculate cross-correlation
            corr = signal.correlate(ir_left, ir_right, mode='full')
            corr_normalized = corr / np.sqrt(np.sum(ir_left**2) * np.sum(ir_right**2))
            
            # IACC is the maximum absolute value of the normalized cross-correlation
            iacc = np.max(np.abs(corr_normalized))
            
            # Time delay between channels
            delay_idx = np.argmax(np.abs(corr_normalized)) - len(ir_left) + 1
            delay_ms = delay_idx * 1000 / fs
        else:
            iacc = None
            delay_ms = None
        
        # Calculate directional statistics if we have at least 4 channels
        if n_channels >= 4:
            # Calculate energy ratios between channels
            energy = np.sum(multi_channel_ir**2, axis=0)
            energy_ratios = energy / np.max(energy)
            
            # Estimate direction of arrival (simplified)
            # A real implementation would use beamforming or other advanced methods
            max_energy_channel = np.argmax(energy)
            direction = mic_positions[max_energy_channel]
        else:
            energy_ratios = None
            direction = None
        
        return {
            'iacc': iacc,
            'delay_ms': delay_ms,
            'energy_ratios': energy_ratios,
            'estimated_direction': direction
        }
    
    def generate_environment_report(self, output_path, rt60_values, leq, spatial_params):
        """Generate a comprehensive report on the acoustic environment"""
        with open(output_path, 'w') as f:
            f.write("# Acoustic Environment Analysis Report\n\n")
            
            f.write("## Reverberation Time (RT60)\n\n")
            f.write("| Frequency Band (Hz) | RT60 (seconds) |\n")
            f.write("|---------------------|---------------|\n")
            for band, rt60 in rt60_values.items():
                rt60_str = f"{rt60:.3f}" if rt60 is not None else "N/A"
                f.write(f"| {band} | {rt60_str} |\n")
            
            f.write("\n## Background Noise\n\n")
            f.write(f"- A-weighted Equivalent Sound Level (LAeq): {leq:.1f} dB(A)\n")
            
            f.write("\n## Spatial Characteristics\n\n")
            if spatial_params['iacc'] is not None:
                f.write(f"- Inter-Aural Cross Correlation (IACC): {spatial_params['iacc']:.3f}\n")
                f.write(f"- Inter-Aural Time Difference: {spatial_params['delay_ms']:.2f} ms\n")
            
            f.write("\n## ISO 3382 Parameters\n\n")
            # This would include additional parameters like C50, C80, etc.
            f.write("Note: Extended ISO 3382 parameters would be included in a full analysis.\n")
            
        print(f"Environment report saved to {output_path}")
