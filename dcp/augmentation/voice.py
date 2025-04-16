import numpy as np
import soundfile as sf
import librosa
import pyrubberband as pyrb
from scipy import signal
import yaml
import os

class VoiceAugmenter:
    """Voice characteristic augmentation for data augmentation"""
    
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def change_pitch(self, y, sr, semitones=0):
        """
        Change the pitch of a voice signal
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
        semitones : float
            Number of semitones to shift pitch
            
        Returns:
        --------
        y_pitch : ndarray
            Pitch-shifted audio signal
        """
        if semitones == 0:
            return y
        
        # Use pyrubberband (which uses rubberband underneath)
        try:
            y_pitch = pyrb.pitch_shift(y, sr, semitones)
        except Exception as e:
            print(f"Error with pyrubberband: {e}. Using fallback method.")
            # Fallback to librosa's pitch shift
            y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
        
        return y_pitch
    
    def change_speed(self, y, sr, speed_factor=1.0):
        """
        Change the speaking rate of a voice signal
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
        speed_factor : float
            Factor to change speed (1.0 = no change, 1.1 = 10% faster)
            
        Returns:
        --------
        y_speed : ndarray
            Speed-changed audio signal
        """
        if speed_factor == 1.0:
            return y
        
        # Use pyrubberband for high-quality time stretching
        try:
            y_speed = pyrb.time_stretch(y, sr, speed_factor)
        except Exception as e:
            print(f"Error with pyrubberband: {e}. Using fallback method.")
            # Fallback to simple resampling (changes pitch too)
            target_length = int(len(y) / speed_factor)
            y_speed = librosa.resample(y, orig_sr=len(y), target_sr=target_length)
        
        return y_speed
    
    def change_formants(self, y, sr, shift_factor=1.0):
        """
        Shift formant frequencies to simulate different vocal tract lengths
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
        shift_factor : float
            Factor to shift formants (1.0 = no change, 1.1 = 10% higher, 0.9 = 10% lower)
            
        Returns:
        --------
        y_formant : ndarray
            Formant-shifted audio signal
        """
        if shift_factor == 1.0:
            return y
        
        # Frame the signal
        frame_length = 2048
        hop_length = 512
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        
        # Apply formant shifting to each frame
        shifted_frames = np.zeros_like(frames)
        
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            
            # Apply pre-emphasis to emphasize formants
            pre_emphasized = librosa.effects.preemphasis(frame)
            
            # Compute LPC
            lpc_order = 24
            lpc_coeffs = librosa.lpc(pre_emphasized, order=lpc_order)
            
            # Find formant frequencies from LPC roots
            roots = np.roots(lpc_coeffs)
            roots = roots[np.imag(roots) > 0]  # Keep only positive imaginary parts
            
            # Sort by ascending frequency
            roots = roots[np.argsort(np.abs(np.angle(roots)))]
            
            # Shift formant frequencies
            shifted_roots = roots * np.exp(1j * np.angle(roots) * (shift_factor - 1.0))
            
            # Reconstruct LPC coefficients
            shifted_lpc = np.poly(np.concatenate([shifted_roots, shifted_roots.conj()]))
            
            # Convert back to real coefficients
            shifted_lpc = np.real(shifted_lpc)
            
            # Create impulse response of the filter
            impulse = np.zeros(frame_length)
            impulse[0] = 1
            
            # Filter the impulse to get the vocal tract impulse response
            vocal_tract = signal.lfilter([1], shifted_lpc, impulse)
            
            # Estimate excitation by inverse filtering
            excitation = signal.lfilter(lpc_coeffs, [1], pre_emphasized)
            
            # Filter excitation with new vocal tract
            shifted_frame = signal.lfilter([1], shifted_lpc, excitation)
            
            # De-emphasis
            shifted_frame = librosa.effects.deemphasis(shifted_frame)
            
            # Store shifted frame
            shifted_frames[:, i] = shifted_frame
        
        # Reconstruct signal with overlap-add
        y_formant = librosa.util.overlap_add(shifted_frames, hop_length)
        
        # Trim to original length
        if len(y_formant) > len(y):
            y_formant = y_formant[:len(y)]
        elif len(y_formant) < len(y):
            y_formant = np.pad(y_formant, (0, len(y) - len(y_formant)))
        
        # Normalize
        y_formant = y_formant / np.max(np.abs(y_formant)) * np.max(np.abs(y))
        
        return y_formant
    
    def add_voice_stress(self, y, sr, stress_level=0.0):
        """
        Add voice stress characteristics
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
        stress_level : float
            Level of stress to add (0.0-1.0)
            
        Returns:
        --------
        y_stress : ndarray
            Audio with stress characteristics
        """
        if stress_level <= 0:
            return y
        
        # Limit stress level
        stress_level = min(1.0, max(0.0, stress_level))
        
        # Apply stress characteristics:
        
        # 1. Slight pitch increase
        pitch_shift = 0.5 * stress_level  # Up to 0.5 semitones
        y_stress = self.change_pitch(y, sr, pitch_shift)
        
        # 2. Increased high frequency energy
        # Create high shelf filter
        shelf_gain = 1.0 + 0.5 * stress_level  # Up to 50% boost
        b, a = self._high_shelf_filter(1500, shelf_gain, sr)
        y_stress = signal.filtfilt(b, a, y_stress)
        
        # 3. Slightly increased speech rate
        speed_factor = 1.0 + 0.15 * stress_level  # Up to 15% faster
        y_stress = self.change_speed(y_stress, sr, speed_factor)
        
        # 4. Add slight jitter to pitch
        # Create varying pitch by modulating with an LFO
        if stress_level > 0.3:
            # Create time vector
            t = np.arange(len(y_stress)) / sr
            
            # Create subtle LFO for jitter (4-8 Hz)
            lfo_freq = 6.0
            lfo_depth = 0.03 * stress_level
            jitter = np.sin(2 * np.pi * lfo_freq * t) * lfo_depth
            
            # Apply time-varying resampling for jitter effect
            y_jitter = np.zeros_like(y_stress)
            
            # Process in small segments for efficiency
            segment_length = int(0.1 * sr)  # 100ms segments
            
            for i in range(0, len(y_stress), segment_length):
                end = min(i + segment_length, len(y_stress))
                segment = y_stress[i:end]
                
                # Calculate local jitter value
                local_jitter = np.mean(jitter[i:end])
                
                # Apply slight pitch shift based on jitter
                segment_shifted = self.change_pitch(segment, sr, local_jitter)
                
                # Ensure same length
                if len(segment_shifted) > len(segment):
                    segment_shifted = segment_shifted[:len(segment)]
                elif len(segment_shifted) < len(segment):
                    segment_shifted = np.pad(segment_shifted, (0, len(segment) - len(segment_shifted)))
                
                y_jitter[i:end] = segment_shifted
                
            y_stress = y_jitter
        
        # Normalize
        y_stress = y_stress / np.max(np.abs(y_stress)) * np.max(np.abs(y))
        
        return y_stress
    
    def _high_shelf_filter(self, cutoff, gain, fs):
        """Create a high-shelf filter with specified cutoff and gain"""
        # Convert gain to dB
        gain_db = 20 * np.log10(gain)
        
        # Normalized cutoff frequency
        w0 = 2 * np.pi * cutoff / fs
        
        # Filter coefficients
        alpha = np.sin(w0) / 2 * np.sqrt((gain + 1/gain) * (1/0.707 - 1) + 2)
        
        # Compute filter coefficients
        b0 = gain * ((gain + 1) + (gain - 1) * np.cos(w0) + 2 * np.sqrt(gain) * alpha)
        b1 = -2 * gain * ((gain - 1) + (gain + 1) * np.cos(w0))
        b2 = gain * ((gain + 1) + (gain - 1) * np.cos(w0) - 2 * np.sqrt(gain) * alpha)
        a0 = (gain + 1) - (gain - 1) * np.cos(w0) + 2 * np.sqrt(gain) * alpha
        a1 = 2 * ((gain - 1) - (gain + 1) * np.cos(w0))
        a2 = (gain + 1) - (gain - 1) * np.cos(w0) - 2 * np.sqrt(gain) * alpha
        
        # Normalize by a0
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1, a2]) / a0
        
        return b, a
    
    def simulate_age(self, y, sr, target_age):
        """
        Simulate voice characteristics for different ages
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
        target_age : int
            Target age to simulate
            
        Returns:
        --------
        y_aged : ndarray
            Age-simulated audio signal
        """
        # Define reference scales for different age groups
        reference_age = 35  # Adult reference point
        
        if target_age == reference_age:
            return y
        
        # Child voice (higher pitch, higher formants, less low end)
        if target_age < 18:
            # Calculate transformation factors based on age difference
            age_factor = (18 - target_age) / 10  # 0-1 scale (more extreme for younger)
            
            # Pitch shift (higher for younger)
            semitones = 2 + 4 * age_factor  # 2-6 semitones higher
            y_aged = self.change_pitch(y, sr, semitones)
            
            # Formant shift (higher for younger)
            formant_shift = 1.1 + 0.2 * age_factor  # 10-30% higher formants
            y_aged = self.change_formants(y_aged, sr, formant_shift)
            
            # Reduce low frequencies
            b, a = signal.butter(3, 200 * (1 + age_factor), btype='highpass', fs=sr)
            y_aged = signal.filtfilt(b, a, y_aged)
            
        # Elderly voice (lower pitch, lower formants, less high end, more tremolo)
        elif target_age > 60:
            # Calculate transformation factors
            age_factor = min(1.0, (target_age - 60) / 30)  # 0-1 scale (more extreme for older)
            
            # Pitch shift (lower for elderly)
            semitones = -1 - 2 * age_factor  # 1-3 semitones lower
            y_aged = self.change_pitch(y, sr, semitones)
            
            # Formant shift (can be lower in elderly)
            formant_shift = 1.0 - 0.1 * age_factor  # Up to 10% lower formants
            y_aged = self.change_formants(y_aged, sr, formant_shift)
            
            # Reduce high frequencies
            b, a = signal.butter(3, 7000 - 3000 * age_factor, btype='lowpass', fs=sr)
            y_aged = signal.filtfilt(b, a, y_aged)
            
            # Add tremolo (voice tremor) for elderly voices
            if age_factor > 0.3:
                # Create time vector
                t = np.arange(len(y_aged)) / sr
                
                # Tremor frequency and depth (more pronounced with age)
                tremor_freq = 5.0 + 2.0 * np.random.rand()  # 5-7 Hz
                tremor_depth = 0.02 + 0.04 * age_factor
                
                # Generate tremor modulator
                tremor = 1.0 + tremor_depth * np.sin(2 * np.pi * tremor_freq * t)
                
                # Apply amplitude modulation
                y_aged = y_aged * tremor
                
                # Apply subtle frequency modulation too
                for i in range(0, len(y_aged), int(0.2 * sr)):  # Process in 200ms segments
                    end = min(i + int(0.2 * sr), len(y_aged))
                    segment = y_aged[i:end]
                    
                    # Get local tremor factor
                    local_tremor = np.mean(tremor[i:end]) - 1.0
                    
                    # Apply tiny pitch variation
                    segment_tremor = self.change_pitch(segment, sr, local_tremor * 0.2)
                    
                    # Ensure correct length
                    segment_tremor = librosa.util.fix_length(segment_tremor, size=end-i)
                    
                    y_aged[i:end] = segment_tremor
        
        # Middle-aged adult (reference group) - mild adjustments by age
        else:
            # Very mild adjustments based on distance from reference
            age_diff = target_age - reference_age
            
            # Small pitch adjustments
            semitones = -0.05 * age_diff  # Slightly lower with age
            y_aged = self.change_pitch(y, sr, semitones)
            
            # Very mild frequency response tweaks
            if age_diff > 0:  # Older than reference
                # Slightly reduce high frequencies
                cutoff = 8000 - 50 * age_diff
                b, a = signal.butter(2, cutoff, btype='lowpass', fs=sr)
                y_aged = signal.filtfilt(b, a, y_aged)
            else:  # Younger than reference
                # No specific adjustments for slightly younger adults
                pass
        
        # Normalize
        y_aged = y_aged / np.max(np.abs(y_aged)) * np.max(np.abs(y))
        
        return y_aged
    
    def simulate_emotion(self, y, sr, emotion_type):
        """
        Simulate voice characteristics for different emotions
        
        Parameters:
        -----------
        y : ndarray
            Audio signal
        sr : int
            Sample rate
        emotion_type : str
            Emotion to simulate ('happy', 'sad', 'angry', 'neutral', 'fear')
            
        Returns:
        --------
        y_emotion : ndarray
            Emotion-simulated audio signal
        """
        if emotion_type == 'neutral':
            return y
        
        # Apply emotion-specific transformations
        if emotion_type == 'happy':
            # Happy: higher pitch, faster, more variance
            y_emotion = self.change_pitch(y, sr, 1.5)  # Higher pitch
            y_emotion = self.change_speed(y_emotion, sr, 1.1)  # Slightly faster
            
            # Enhance mid-high frequencies
            b, a = signal.butter(2, [1000, 8000], btype='bandpass', fs=sr)
            mid_high = signal.filtfilt(b, a, y_emotion)
            y_emotion = y_emotion + 0.2 * mid_high  # Add brightness
            
            # Add subtle pitch modulation (more dynamic)
            t = np.arange(len(y_emotion)) / sr
            modulation = 0.04 * np.sin(2 * np.pi * 0.5 * t)  # Slow modulation
            
            # Apply in segments
            segment_size = int(0.2 * sr)
            for i in range(0, len(y_emotion), segment_size):
                end = min(i + segment_size, len(y_emotion))
                segment = y_emotion[i:end]
                
                local_mod = np.mean(modulation[i:end])
                segment_mod = self.change_pitch(segment, sr, local_mod)
                segment_mod = librosa.util.fix_length(segment_mod, size=end-i)
                
                y_emotion[i:end] = segment_mod
                
        elif emotion_type == 'sad':
            # Sad: lower pitch, slower, less variance
            y_emotion = self.change_pitch(y, sr, -2.0)  # Lower pitch
            y_emotion = self.change_speed(y_emotion, sr, 0.9)  # Slower
            
            # Reduce high frequencies
            b, a = signal.butter(2, 5000, btype='lowpass', fs=sr)
            y_emotion = signal.filtfilt(b, a, y_emotion)
            
            # Add slight "breathiness"
            noise = np.random.randn(len(y_emotion)) * 0.05 * np.max(np.abs(y_emotion))
            hp_noise = signal.filtfilt(
                *signal.butter(2, 3000, btype='highpass', fs=sr), 
                noise
            )
            y_emotion = y_emotion + hp_noise * 0.1
            
        elif emotion_type == 'angry':
            # Angry: higher intensity, more high freq, tense voice
            # Slightly higher pitch with compression
            y_emotion = self.change_pitch(y, sr, 1.0)
            
            # Add intensity through compression/distortion
            # Simple compression
            threshold = 0.5 * np.max(np.abs(y))
            y_comp = np.copy(y)
            mask = np.abs(y_comp) > threshold
            y_comp[mask] = np.sign(y_comp[mask]) * (threshold + (np.abs(y_comp[mask]) - threshold) * 0.5)
            
            # Mix with original
            y_emotion = 0.7 * y_emotion + 0.3 * y_comp
            
            # Enhance high-mids (tension in voice)
            b, a = signal.butter(2, [1500, 6000], btype='bandpass', fs=sr)
            mid_high = signal.filtfilt(b, a, y_emotion)
            y_emotion = y_emotion + 0.3 * mid_high
            
            # Make slightly faster
            y_emotion = self.change_speed(y_emotion, sr, 1.05)
            
        elif emotion_type == 'fear':
            # Fear: trembling voice, higher pitch, jittery
            y_emotion = self.change_pitch(y, sr, 1.2)  # Higher pitch
            
            # Add trembling (faster than elderly tremor)
            t = np.arange(len(y)) / sr
            tremor_freq = 8.0  # Faster trembling
            tremor_depth = 0.06
            tremor = 1.0 + tremor_depth * np.sin(2 * np.pi * tremor_freq * t)
            
            # Apply amplitude modulation
            y_emotion = y_emotion * tremor
            
            # Add subtle pitch jitter
            segment_size = int(0.1 * sr)  # Shorter segments for more jitter
            for i in range(0, len(y_emotion), segment_size):
                end = min(i + segment_size, len(y_emotion))
                segment = y_emotion[i:end]
                
                # Random tiny pitch shift per segment
                jitter = 0.3 * np.random.randn()
                segment_jitter = self.change_pitch(segment, sr, jitter)
                segment_jitter = librosa.util.fix_length(segment_jitter, size=end-i)
                
                y_emotion[i:end] = segment_jitter
            
            # Add slight breathiness
            noise = np.random.randn(len(y_emotion)) * 0.05 * np.max(np.abs(y_emotion))
            hp_noise = signal.filtfilt(
                *signal.butter(2, 2500, btype='highpass', fs=sr), 
                noise
            )
            y_emotion = y_emotion + hp_noise * 0.15
            
        else:
            # Unknown emotion, return original
            return y
        
        # Normalize
        y_emotion = y_emotion / np.max(np.abs(y_emotion)) * np.max(np.abs(y))
        
        return y_emotion
