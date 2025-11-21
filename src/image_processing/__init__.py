"""
Image Processing Module
Contains filters, edge detection, enhancement, and denoising algorithms.
"""

from .filters import gaussian_filter, median_filter, apply_filter_frequency_domain
from .edge_detection import sobel_edge_detection, canny_edge_detection
from .enhancement import histogram_equalization
from .denoising import wiener_filter

__all__ = [
    'gaussian_filter',
    'median_filter',
    'apply_filter_frequency_domain',
    'sobel_edge_detection',
    'canny_edge_detection',
    'histogram_equalization',
    'wiener_filter'
]
