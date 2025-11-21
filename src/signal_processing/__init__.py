"""
Signal Processing Module
Contains frequency analysis and spectral analysis tools for audio.
"""

from .frequency_analysis import compute_fft, apply_frequency_filter
from .spectral_analysis import compute_stft, compute_spectrogram

__all__ = [
    'compute_fft',
    'apply_frequency_filter',
    'compute_stft',
    'compute_spectrogram'
]
