"""
Time-Frequency Analysis for Audio
STFT and Spectrogram computation
"""

import numpy as np
from scipy import signal
from typing import Tuple


def compute_stft(audio: np.ndarray, sample_rate: int, 
                n_fft: int = 2048, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform.
    
    STFT reveals how frequency content changes over time.
    
    Args:
        audio: Input audio signal
        sample_rate: Sampling rate in Hz
        n_fft: FFT size (window length)
        hop_length: Number of samples between successive frames
    
    Returns:
        Tuple of (frequencies, times, STFT_matrix)
    """
    # Use Hann window for smooth tapering
    window = signal.get_window('hann', n_fft)
    
    # Compute STFT
    frequencies, times, stft_matrix = signal.stft(
        audio, 
        fs=sample_rate, 
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length
    )
    
    return frequencies, times, stft_matrix


def compute_spectrogram(audio: np.ndarray, sample_rate: int,
                       n_fft: int = 2048, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram (magnitude of STFT).
    
    Spectrogram is a visual representation of frequency content over time.
    
    Args:
        audio: Input audio signal
        sample_rate: Sampling rate in Hz
        n_fft: FFT size
        hop_length: Hop size between frames
    
    Returns:
        Tuple of (frequencies, times, spectrogram_matrix)
    """
    # Compute STFT
    frequencies, times, stft_matrix = compute_stft(audio, sample_rate, n_fft, hop_length)
    
    # Compute magnitude (spectrogram)
    spectrogram = np.abs(stft_matrix)
    
    return frequencies, times, spectrogram
