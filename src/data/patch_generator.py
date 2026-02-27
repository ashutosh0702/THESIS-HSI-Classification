"""
HSI Patch Generation Module
Extracts spatial-spectral patches from Hyperspectral Images for Deep Learning models (CNNs, ViTs).
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def pad_hsi_with_zeros(X: np.ndarray, margin: int) -> np.ndarray:
    """
    Pad the spatial dimensions of an HSI cube with zeros.
    
    Args:
        X: HSI cube of shape (Height, Width, Bands)
        margin: Number of pixels to pad on each side
        
    Returns:
        Padded HSI cube
    """
    return np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='constant', constant_values=0)

def pad_hsi_with_reflection(X: np.ndarray, margin: int) -> np.ndarray:
    """
    Pad the spatial dimensions of an HSI cube using reflection.
    Often better than zero-padding as it prevents edge-artifacts in CNNs.
    
    Args:
        X: HSI cube of shape (Height, Width, Bands)
        margin: Number of pixels to pad on each side
        
    Returns:
        Padded HSI cube
    """
    return np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')

def create_patches(
    X: np.ndarray, 
    y: np.ndarray, 
    window_size: int = 25, 
    remove_zero_labels: bool = True,
    padding_mode: str = 'reflect'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D spatial-spectral patches centered on pixels.
    
    Args:
        X: HSI data cube of shape (Height, Width, Bands)
        y: Ground truth labels of shape (Height, Width)
        window_size: Spatial size of the patch (must be odd, e.g., 5, 7, 9, 15, 25)
        remove_zero_labels: If True, only extract patches for pixels with label > 0
        padding_mode: 'reflect' or 'zero'
        
    Returns:
        Tuple of (patches, labels)
        - patches shape: (N_samples, window_size, window_size, Bands)
        - labels shape: (N_samples,)
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number to have a clear center pixel.")
        
    margin = int((window_size - 1) / 2)
    
    # Pad the original image
    if padding_mode == 'reflect':
        padded_X = pad_hsi_with_reflection(X, margin)
    else:
        padded_X = pad_hsi_with_zeros(X, margin)
        
    # Find active pixels
    if remove_zero_labels:
        # Find coordinates of all non-zero pixels
        rows, cols = np.where(y > 0)
    else:
        # Include all pixels (useful if you want to predict the entire map later)
        rows, cols = np.where(y >= 0)
        
    n_samples = len(rows)
    n_bands = X.shape[2]
    
    # Pre-allocate array for speed and memory efficiency
    patches = np.zeros((n_samples, window_size, window_size, n_bands), dtype=X.dtype)
    labels = np.zeros(n_samples, dtype=y.dtype)
    
    logger.info(f"Extracting {n_samples} patches of size {window_size}x{window_size}x{n_bands}...")
    
    for i, (r, c) in enumerate(zip(rows, cols)):
        # Because the image is padded, the center pixel (r,c) in original image 
        # is at (r + margin, c + margin) in the padded image.
        # The window goes from (r) to (r + 2*margin + 1) in the padded image.
        patch = padded_X[r : r + window_size, c : c + window_size, :]
        patches[i] = patch
        labels[i] = y[r, c]
        
    logger.info("Extraction complete.")
    return patches, labels

def extract_inference_patches(X: np.ndarray, window_size: int = 25, padding_mode: str = 'reflect') -> np.ndarray:
    """
    Extract patches for EVERY pixel in the image to generate a complete classification map.
    
    Returns:
        Patches array of shape (Height * Width, window_size, window_size, Bands)
    """
    H, W, B = X.shape
    dummy_y = np.ones((H, W))  # Dummy mask to grab every pixel
    patches, _ = create_patches(X, dummy_y, window_size=window_size, remove_zero_labels=True, padding_mode=padding_mode)
    return patches
