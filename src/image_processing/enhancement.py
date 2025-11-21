"""
Image Enhancement Algorithms
Histogram Equalization (Global and CLAHE)
"""

import numpy as np
import cv2


def histogram_equalization(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """
    Apply histogram equalization for contrast enhancement.
    
    Methods:
        - 'global': Simple histogram equalization
        - 'clahe': Contrast Limited Adaptive Histogram Equalization
    
    Args:
        image: Input grayscale image
        method: Equalization method ('global' or 'clahe')
    
    Returns:
        Contrast-enhanced image
    """
    if method == 'global':
        # Global histogram equalization
        equalized = cv2.equalizeHist(image)
        
    elif method == 'clahe':
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(image)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'global' or 'clahe'")
    
    return equalized
