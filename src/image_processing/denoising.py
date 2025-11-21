"""
Advanced Denoising Algorithms
Wiener Filter for optimal MSE-based noise reduction
"""

import numpy as np
from scipy.signal import wiener


def wiener_filter(image: np.ndarray, kernel_size: int = 5, noise_variance: float = None) -> np.ndarray:
    """
    Apply Wiener filter for optimal denoising.
    
    Mathematical Form:
        ŷ(x,y) = μ + [σ² - σ²_noise]/σ² * [y(x,y) - μ]
    
    Where:
        μ = local mean
        σ² = local variance
        σ²_noise = noise variance (estimated if not provided)
    
    Args:
        image: Input grayscale image
        kernel_size: Size of local neighborhood
        noise_variance: Noise variance (estimated if None)
    
    Returns:
        Denoised image
    """
    # Convert to float
    img_float = image.astype(np.float64)
    
    # Apply Wiener filter
    filtered = wiener(img_float, (kernel_size, kernel_size), noise=noise_variance)
    
    # Convert back to uint8
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    
    return filtered
