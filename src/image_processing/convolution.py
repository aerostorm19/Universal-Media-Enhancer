"""
2D Convolution Implementation
Custom convolution for educational purposes and full control.
"""

import numpy as np


def convolve2d(image: np.ndarray, kernel: np.ndarray, padding: str = 'reflect') -> np.ndarray:
    """
    Perform 2D convolution on an image with a kernel.
    
    Args:
        image: Input 2D image array
        kernel: Convolution kernel (should be odd-sized)
        padding: Padding mode ('reflect', 'constant', 'replicate')
    
    Returns:
        Convolved image of same size as input
    """
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel dimensions must be odd")
    
    # Flip kernel for convolution (not correlation)
    kernel = np.flipud(np.fliplr(kernel))
    
    # Get dimensions
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    # Pad image
    if padding == 'reflect':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    elif padding == 'constant':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    elif padding == 'replicate':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    else:
        raise ValueError(f"Unknown padding mode: {padding}")
    
    # Output array
    output = np.zeros_like(image, dtype=np.float64)
    
    # Perform convolution
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract region
            region = padded[i:i+k_h, j:j+k_w]
            # Compute weighted sum
            output[i, j] = np.sum(region * kernel)
    
    return output


def separable_convolve2d(image: np.ndarray, kernel_1d: np.ndarray, padding: str = 'reflect') -> np.ndarray:
    """
    Optimized 2D convolution using separable 1D kernel.
    Much faster for separable kernels (e.g., Gaussian).
    
    Args:
        image: Input 2D image
        kernel_1d: 1D kernel for separable convolution
        padding: Padding mode
    
    Returns:
        Convolved image
    """
    # Convolve rows
    temp = np.apply_along_axis(lambda row: np.convolve(row, kernel_1d, mode='same'), axis=1, arr=image)
    
    # Convolve columns
    result = np.apply_along_axis(lambda col: np.convolve(col, kernel_1d, mode='same'), axis=0, arr=temp)
    
    return result
