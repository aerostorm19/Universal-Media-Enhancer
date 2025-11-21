"""
Image Quality Metrics
PSNR, SSIM, MSE, MAE
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Dict


def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio.
    
    Formula:
        PSNR = 10 * log10(MAX² / MSE)
    
    Args:
        original: Original image
        processed: Processed image
    
    Returns:
        PSNR value in dB
    """
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
    
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    
    return psnr


def calculate_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index.
    
    SSIM considers luminance, contrast, and structure.
    Range: [0, 1], where 1 means identical images.
    
    Args:
        original: Original image
        processed: Processed image
    
    Returns:
        SSIM value
    """
    ssim_value = ssim(original, processed, data_range=255)
    return ssim_value


def calculate_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        original: Original image
        processed: Processed image
    
    Returns:
        MSE value
    """
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
    return mse


def calculate_mae(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        original: Original image
        processed: Processed image
    
    Returns:
        MAE value
    """
    mae = np.mean(np.abs(original.astype(float) - processed.astype(float)))
    return mae


def evaluate_image_quality(original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
    """
    Comprehensive image quality evaluation.
    
    Calculates multiple metrics:
        - PSNR: Peak Signal-to-Noise Ratio
        - SSIM: Structural Similarity Index
        - MSE: Mean Squared Error
        - MAE: Mean Absolute Error
    
    Args:
        original: Original image
        processed: Processed/enhanced image
    
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'psnr': calculate_psnr(original, processed),
        'ssim': calculate_ssim(original, processed),
        'mse': calculate_mse(original, processed),
        'mae': calculate_mae(original, processed)
    }
    
    return metrics
