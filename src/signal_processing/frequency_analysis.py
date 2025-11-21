"""
Frequency Domain Analysis for Audio
FFT and Digital Filtering
"""

import numpy as np
from scipy import signal
from typing import Tuple


def compute_fft(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Fast Fourier Transform of audio signal.
    
    Args:
        audio: Time-domain audio signal
        sample_rate: Sampling rate in Hz
    
    Returns:
        Tuple of (frequencies, magnitude, phase)
    """
    # Use real FFT for efficiency
    fft_result = np.fft.rfft(audio)
    
    # Frequency bins
    frequencies = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    
    # Magnitude and phase
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    
    return frequencies, magnitude, phase


def apply_frequency_filter(audio: np.ndarray, sample_rate: int, 
                           filter_type: str = 'bandpass', 
                           cutoff: any = (300, 3400), 
                           order: int = 5) -> np.ndarray:
    """
    Apply Butterworth digital filter to audio signal.
    
    Filter Types:
        - 'lowpass': Keeps frequencies below cutoff
        - 'highpass': Keeps frequencies above cutoff
        - 'bandpass': Keeps frequencies between cutoff[0] and cutoff[1]
    
    Args:
        audio: Input audio signal
        sample_rate: Sampling rate in Hz
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
        cutoff: Cutoff frequency/frequencies in Hz
        order: Filter order (higher = sharper transition)
    
    Returns:
        Filtered audio signal
    """
    # Normalize cutoff frequency (Nyquist = 1)
    nyquist = sample_rate / 2.0
    
    if filter_type in ['lowpass', 'highpass']:
        normalized_cutoff = cutoff / nyquist
    elif filter_type == 'bandpass':
        normalized_cutoff = [cutoff[0] / nyquist, cutoff[1] / nyquist]
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Design Butterworth filter (SOS form for stability)
    sos = signal.butter(order, normalized_cutoff, btype=filter_type, output='sos')
    
    # Apply filter
    filtered = signal.sosfilt(sos, audio)
    
    return filtered
