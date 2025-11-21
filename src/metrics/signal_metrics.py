"""
Audio Signal Quality Metrics
SNR, RMS, and error metrics
"""

import numpy as np
from typing import Dict


def calculate_snr(clean: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio.
    
    Formula:
        SNR = 10 * log10(P_signal / P_noise)
    
    Args:
        clean: Original/clean signal
        processed: Processed signal
    
    Returns:
        SNR value in dB
    """
    # Signal power
    signal_power = np.mean(clean ** 2)
    
    # Noise power (difference between signals)
    noise = clean - processed
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr


def calculate_rms(signal: np.ndarray) -> float:
    """
    Calculate Root Mean Square (RMS) of signal.
    
    RMS measures the signal power/loudness.
    
    Args:
        signal: Audio signal
    
    Returns:
        RMS value
    """
    rms = np.sqrt(np.mean(signal ** 2))
    return rms


def evaluate_audio_quality(original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
    """
    Comprehensive audio quality evaluation.
    
    Calculates:
        - SNR: Signal-to-Noise Ratio
        - RMS (original): Power of original signal
        - RMS (processed): Power of processed signal
    
    Args:
        original: Original audio signal
        processed: Processed/enhanced audio signal
    
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'snr': calculate_snr(original, processed),
        'rms_clean': calculate_rms(original),
        'rms_processed': calculate_rms(processed)
    }
    
    return metrics
