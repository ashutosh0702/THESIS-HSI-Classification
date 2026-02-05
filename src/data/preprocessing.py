"""
Preprocessing Module for Hyperspectral Data
Includes spectral masking, dimensionality reduction, and normalization
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)


def remove_water_absorption_bands(
    data: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    absorption_ranges: Optional[List[Tuple[float, float]]] = None,
    log_to_mlflow: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove water absorption bands from hyperspectral data.
    
    Default absorption ranges (in nm):
    - 1350-1450 nm (atmospheric water vapor)
    - 1800-1950 nm (atmospheric water vapor)
    
    Args:
        data: HSI cube of shape (H, W, B) or flattened (N, B)
        wavelengths: Array of wavelength values for each band
        absorption_ranges: List of (min_wl, max_wl) tuples to mask
        log_to_mlflow: Whether to log operation to MLflow
    
    Returns:
        Tuple of (masked_data, valid_band_indices)
    """
    if absorption_ranges is None:
        absorption_ranges = [(1350, 1450), (1800, 1950)]
    
    n_bands = data.shape[-1]
    
    if wavelengths is None:
        # If no wavelengths provided, assume default bands to remove
        # Common for datasets like Indian Pines (removes bands 104-108, 150-163)
        logger.warning("No wavelengths provided. Using default band indices for removal.")
        bands_to_remove = list(range(104, 109)) + list(range(150, 164))
        valid_bands = [i for i in range(n_bands) if i not in bands_to_remove]
    else:
        # Find bands within absorption ranges
        valid_bands = []
        for i, wl in enumerate(wavelengths):
            is_valid = True
            for min_wl, max_wl in absorption_ranges:
                if min_wl <= wl <= max_wl:
                    is_valid = False
                    break
            if is_valid:
                valid_bands.append(i)
    
    valid_bands = np.array(valid_bands)
    masked_data = data[..., valid_bands]
    
    if log_to_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_params({
            "preprocessing_water_absorption_removed": True,
            "preprocessing_original_bands": n_bands,
            "preprocessing_remaining_bands": len(valid_bands)
        })
    
    logger.info(f"Removed {n_bands - len(valid_bands)} bands, {len(valid_bands)} remaining")
    return masked_data, valid_bands


def apply_pca(
    data: np.ndarray,
    n_components: Union[int, float] = 0.99,
    log_to_mlflow: bool = True
) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA dimensionality reduction to HSI data.
    
    Args:
        data: HSI cube (H, W, B) or flattened (N, B)
        n_components: Number of components or variance ratio to retain
        log_to_mlflow: Whether to log operation to MLflow
    
    Returns:
        Tuple of (transformed_data, fitted_pca_object)
    """
    original_shape = data.shape
    
    # Flatten if 3D
    if data.ndim == 3:
        h, w, b = data.shape
        data_flat = data.reshape(-1, b)
    else:
        data_flat = data
    
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data_flat)
    
    # Reshape back if originally 3D
    if len(original_shape) == 3:
        h, w = original_shape[:2]
        transformed = transformed.reshape(h, w, -1)
    
    if log_to_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_params({
            "preprocessing_pca_applied": True,
            "preprocessing_pca_components": pca.n_components_,
            "preprocessing_pca_variance_retained": float(sum(pca.explained_variance_ratio_))
        })
    
    logger.info(f"PCA: {original_shape[-1]} -> {pca.n_components_} components "
                f"({sum(pca.explained_variance_ratio_):.2%} variance)")
    return transformed, pca


def apply_mnf(
    data: np.ndarray,
    n_components: int = 30,
    log_to_mlflow: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Apply Minimum Noise Fraction (MNF) transform for noise reduction.
    
    MNF is a two-step PCA process:
    1. Noise whitening (decorrelate noise)
    2. PCA on noise-whitened data
    
    Args:
        data: HSI cube (H, W, B) or flattened (N, B)
        n_components: Number of MNF components to retain
        log_to_mlflow: Whether to log operation to MLflow
    
    Returns:
        Tuple of (transformed_data, mnf_info_dict)
    """
    original_shape = data.shape
    
    # Flatten if 3D
    if data.ndim == 3:
        h, w, b = data.shape
        data_flat = data.reshape(-1, b)
    else:
        data_flat = data
        h, w = None, None
    
    # Estimate noise using spatial differencing (shift-difference method)
    if h is not None:
        noise_estimate = np.diff(data.reshape(h, w, -1), axis=0).reshape(-1, data.shape[-1])
    else:
        # For pre-flattened data, use simple differencing
        noise_estimate = np.diff(data_flat, axis=0)
    
    # Compute noise covariance
    noise_cov = np.cov(noise_estimate.T)
    
    # Eigendecomposition of noise covariance
    noise_eigenvalues, noise_eigenvectors = np.linalg.eigh(noise_cov)
    
    # Noise whitening transform
    noise_whitening = noise_eigenvectors @ np.diag(1.0 / np.sqrt(noise_eigenvalues + 1e-10))
    
    # Apply noise whitening
    whitened = data_flat @ noise_whitening
    
    # Standard PCA on whitened data
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(whitened)
    
    # Reshape back if originally 3D
    if h is not None:
        transformed = transformed.reshape(h, w, -1)
    
    mnf_info = {
        'noise_whitening': noise_whitening,
        'pca': pca,
        'n_components': n_components,
        'eigenvalues': pca.explained_variance_
    }
    
    if log_to_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_params({
            "preprocessing_mnf_applied": True,
            "preprocessing_mnf_components": n_components
        })
    
    logger.info(f"MNF: Reduced to {n_components} components")
    return transformed, mnf_info


def normalize_data(
    data: np.ndarray,
    method: str = 'minmax',
    per_band: bool = True,
    log_to_mlflow: bool = True
) -> Tuple[np.ndarray, object]:
    """
    Normalize hyperspectral data.
    
    Args:
        data: HSI cube (H, W, B) or flattened (N, B)
        method: 'minmax' (0-1) or 'standard' (z-score)
        per_band: If True, normalize each band independently
        log_to_mlflow: Whether to log operation to MLflow
    
    Returns:
        Tuple of (normalized_data, scaler_object)
    """
    original_shape = data.shape
    
    # Flatten spatial dimensions
    if data.ndim == 3:
        h, w, b = data.shape
        data_flat = data.reshape(-1, b)
    else:
        data_flat = data
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if per_band:
        normalized = scaler.fit_transform(data_flat)
    else:
        # Global normalization
        flat = data_flat.flatten().reshape(-1, 1)
        scaler.fit(flat)
        normalized = scaler.transform(data_flat.flatten().reshape(-1, 1)).reshape(data_flat.shape)
    
    # Reshape back
    if len(original_shape) == 3:
        normalized = normalized.reshape(original_shape)
    
    if log_to_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_params({
            "preprocessing_normalization_method": method,
            "preprocessing_normalization_per_band": per_band
        })
    
    logger.info(f"Applied {method} normalization (per_band={per_band})")
    return normalized, scaler


def preprocess_pipeline(
    data: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    remove_water: bool = True,
    reduce_dims: Optional[str] = 'pca',
    n_components: Union[int, float] = 30,
    normalize: str = 'minmax',
    log_to_mlflow: bool = True
) -> Dict:
    """
    Full preprocessing pipeline for HSI data.
    
    Args:
        data: Raw HSI cube (H, W, B)
        wavelengths: Optional wavelength array
        remove_water: Whether to remove water absorption bands
        reduce_dims: 'pca', 'mnf', or None
        n_components: Components for dimensionality reduction
        normalize: 'minmax', 'standard', or None
        log_to_mlflow: Whether to log all steps to MLflow
    
    Returns:
        Dictionary with processed data and all transformation objects
    """
    result = {
        'original_shape': data.shape,
        'data': data.copy(),
        'transforms': {}
    }
    
    # Step 1: Remove water absorption bands
    if remove_water:
        result['data'], valid_bands = remove_water_absorption_bands(
            result['data'], wavelengths, log_to_mlflow=log_to_mlflow
        )
        result['transforms']['water_absorption'] = {'valid_bands': valid_bands}
    
    # Step 2: Dimensionality reduction
    if reduce_dims == 'pca':
        result['data'], pca = apply_pca(
            result['data'], n_components, log_to_mlflow=log_to_mlflow
        )
        result['transforms']['dim_reduction'] = {'method': 'pca', 'model': pca}
    elif reduce_dims == 'mnf':
        result['data'], mnf_info = apply_mnf(
            result['data'], n_components, log_to_mlflow=log_to_mlflow
        )
        result['transforms']['dim_reduction'] = {'method': 'mnf', 'info': mnf_info}
    
    # Step 3: Normalization
    if normalize:
        result['data'], scaler = normalize_data(
            result['data'], method=normalize, log_to_mlflow=log_to_mlflow
        )
        result['transforms']['normalization'] = {'method': normalize, 'scaler': scaler}
    
    result['final_shape'] = result['data'].shape
    
    return result
