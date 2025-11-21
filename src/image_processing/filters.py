"""
Image Filtering Algorithms
Gaussian, Median, and Frequency Domain Filters
"""

import numpy as np
import cv2
from scipy import signal
from .convolution import convolve2d


def gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing filter for noise reduction.
    
    Mathematical Form:
        G(x,y) = (1/(2πσ²)) * exp(-(x²+y²)/(2σ²))
    
    Args:
        image: Input grayscale image
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation of Gaussian
    
    Returns:
        Smoothed image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create Gaussian kernel
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)  # Normalize
    
    # Apply convolution
    filtered = convolve2d(image, kernel, padding='reflect')
    
    return np.clip(filtered, 0, 255).astype(np.uint8)


def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filter for impulse noise removal.
    Non-linear filter that preserves edges well.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of median filter window (must be odd)
    
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Use OpenCV's optimized median blur
    filtered = cv2.medianBlur(image, kernel_size)
    
    return filtered


def apply_filter_frequency_domain(image: np.ndarray, filter_type: str = 'lowpass', 
                                  cutoff: float = 0.1) -> np.ndarray:
    """
    Apply frequency domain filtering using FFT.
    
    Args:
        image: Input grayscale image
        filter_type: 'lowpass' or 'highpass'
        cutoff: Cutoff frequency (0 to 0.5, normalized)
    
    Returns:
        Filtered image
    """
    # Compute FFT
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    # Get dimensions
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create frequency mask
    mask = np.zeros((rows, cols), dtype=np.float32)
    
    # Calculate radius for each point
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Normalize distance
    max_distance = np.sqrt(crow**2 + ccol**2)
    normalized_distance = distance / max_distance
    
    if filter_type == 'lowpass':
        # Keep low frequencies
        mask[normalized_distance <= cutoff] = 1
    elif filter_type == 'highpass':
        # Keep high frequencies
        mask[normalized_distance > cutoff] = 1
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Apply mask
    f_shift_filtered = f_shift * mask
    
    # Inverse FFT
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return np.clip(img_back, 0, 255).astype(np.uint8)
