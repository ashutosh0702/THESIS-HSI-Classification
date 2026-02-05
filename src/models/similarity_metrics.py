"""
Similarity Metrics Module for Hyperspectral Image Classification

This module implements various local similarity metrics for thesis research:
- Euclidean distance-based similarity
- Spectral Angle Mapper (SAM)
- RBF kernel similarity
- Cosine similarity
- Local similarity matrices for patches
"""

import logging
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)


# ============================================================================
# Point-wise Similarity Functions
# ============================================================================

def euclidean_distance(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """
    Compute Euclidean distance between two spectral signatures.
    
    Args:
        x_i: First spectral signature (n_bands,)
        x_j: Second spectral signature (n_bands,)
    
    Returns:
        Euclidean distance (non-negative)
    """
    return np.linalg.norm(x_i - x_j)


def euclidean_similarity(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """
    Convert Euclidean distance to similarity score in [0, 1].
    
    Uses: S = 1 / (1 + d) where d is Euclidean distance
    
    Args:
        x_i: First spectral signature (n_bands,)
        x_j: Second spectral signature (n_bands,)
    
    Returns:
        Similarity score in [0, 1]
    """
    d = euclidean_distance(x_i, x_j)
    return 1.0 / (1.0 + d)


def spectral_angle_mapper(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """
    Compute Spectral Angle Mapper (SAM) distance.
    
    SAM measures the angle between two spectral vectors, making it
    invariant to illumination differences (brightness).
    
    Formula: SAM = arccos(x_i · x_j / (||x_i|| ||x_j||))
    
    Args:
        x_i: First spectral signature (n_bands,)
        x_j: Second spectral signature (n_bands,)
    
    Returns:
        Spectral angle in radians [0, π]
    """
    norm_i = np.linalg.norm(x_i)
    norm_j = np.linalg.norm(x_j)
    
    if norm_i == 0 or norm_j == 0:
        return np.pi / 2  # Orthogonal if zero vector
    
    cos_angle = np.dot(x_i, x_j) / (norm_i * norm_j)
    # Clip to handle numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def sam_similarity(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """
    Convert SAM distance to similarity in [0, 1].
    
    Uses: S = 1 - (SAM / π)
    
    Args:
        x_i: First spectral signature
        x_j: Second spectral signature
    
    Returns:
        Similarity score in [0, 1]
    """
    angle = spectral_angle_mapper(x_i, x_j)
    return 1.0 - (angle / np.pi)


def rbf_similarity(
    x_i: np.ndarray, 
    x_j: np.ndarray, 
    sigma: float = 1.0
) -> float:
    """
    Compute RBF (Radial Basis Function) kernel similarity.
    
    Formula: S(x_i, x_j) = exp(-||x_i - x_j||² / (2σ²))
    
    This is the Gaussian kernel commonly used in SVM and graph-based methods.
    
    Args:
        x_i: First spectral signature (n_bands,)
        x_j: Second spectral signature (n_bands,)
        sigma: Bandwidth parameter (kernel width)
    
    Returns:
        Similarity score in (0, 1]
    """
    sq_dist = np.sum((x_i - x_j) ** 2)
    return np.exp(-sq_dist / (2 * sigma ** 2))


def cosine_similarity(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """
    Compute cosine similarity between two spectral signatures.
    
    Formula: cos(x_i, x_j) = (x_i · x_j) / (||x_i|| ||x_j||)
    
    Range: [-1, 1], where 1 means identical direction
    
    Args:
        x_i: First spectral signature
        x_j: Second spectral signature
    
    Returns:
        Cosine similarity in [-1, 1]
    """
    norm_i = np.linalg.norm(x_i)
    norm_j = np.linalg.norm(x_j)
    
    if norm_i == 0 or norm_j == 0:
        return 0.0
    
    return np.dot(x_i, x_j) / (norm_i * norm_j)


def normalized_cosine_similarity(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """
    Compute normalized cosine similarity in [0, 1].
    
    Uses: S = (cos(x_i, x_j) + 1) / 2
    
    Args:
        x_i: First spectral signature
        x_j: Second spectral signature
    
    Returns:
        Normalized similarity in [0, 1]
    """
    return (cosine_similarity(x_i, x_j) + 1.0) / 2.0


def sid_distance(x_i: np.ndarray, x_j: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Spectral Information Divergence (SID).
    
    SID uses information theory to measure spectral similarity.
    
    Args:
        x_i: First spectral signature (must be non-negative)
        x_j: Second spectral signature (must be non-negative)
        epsilon: Small value to prevent log(0)
    
    Returns:
        SID value (non-negative, higher = more different)
    """
    # Normalize to probability distributions
    p = x_i / (np.sum(x_i) + epsilon)
    q = x_j / (np.sum(x_j) + epsilon)
    
    # Add epsilon to prevent log(0)
    p = p + epsilon
    q = q + epsilon
    
    # KL divergences
    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    
    return kl_pq + kl_qp


def sid_similarity(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """
    Convert SID distance to similarity in [0, 1].
    
    Uses: S = 1 / (1 + SID)
    """
    return 1.0 / (1.0 + sid_distance(x_i, x_j))


# ============================================================================
# Vectorized Batch Similarity Functions
# ============================================================================

def batch_euclidean_similarity(
    X: np.ndarray, 
    Y: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute pairwise Euclidean similarity matrix.
    
    Args:
        X: Array of shape (N, D)
        Y: Optional array of shape (M, D). If None, compute X vs X.
    
    Returns:
        Similarity matrix of shape (N, M) or (N, N)
    """
    if Y is None:
        Y = X
    
    # Compute squared Euclidean distances
    # ||x - y||² = ||x||² + ||y||² - 2xy
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    
    distances_sq = X_sq + Y_sq.T - 2 * np.dot(X, Y.T)
    distances_sq = np.maximum(distances_sq, 0)  # Numerical stability
    distances = np.sqrt(distances_sq)
    
    return 1.0 / (1.0 + distances)


def batch_rbf_similarity(
    X: np.ndarray, 
    Y: Optional[np.ndarray] = None,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Compute pairwise RBF kernel similarity matrix.
    
    Args:
        X: Array of shape (N, D)
        Y: Optional array of shape (M, D)
        sigma: RBF bandwidth parameter
    
    Returns:
        Similarity matrix of shape (N, M) or (N, N)
    """
    if Y is None:
        Y = X
    
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    
    distances_sq = X_sq + Y_sq.T - 2 * np.dot(X, Y.T)
    distances_sq = np.maximum(distances_sq, 0)
    
    return np.exp(-distances_sq / (2 * sigma ** 2))


def batch_sam_similarity(
    X: np.ndarray, 
    Y: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute pairwise SAM similarity matrix.
    
    Args:
        X: Array of shape (N, D)
        Y: Optional array of shape (M, D)
    
    Returns:
        Similarity matrix of shape (N, M) or (N, N)
    """
    if Y is None:
        Y = X
    
    # Normalize vectors
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
    
    # Cosine of angles
    cos_angles = np.dot(X_norm, Y_norm.T)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    
    # Convert angles to similarity
    angles = np.arccos(cos_angles)
    return 1.0 - (angles / np.pi)


# ============================================================================
# Local Similarity Matrix for Patches
# ============================================================================

def local_similarity_matrix(
    patch: np.ndarray,
    metric: str = 'rbf',
    sigma: float = 1.0,
    center_only: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Compute similarity matrix for pixels within a local patch.
    
    This is a key function for thesis research on local similarity.
    
    Args:
        patch: 3D array of shape (H, W, B) representing a local window
        metric: Similarity metric ('euclidean', 'rbf', 'sam', 'cosine')
        sigma: Bandwidth for RBF kernel
        center_only: If True, only compute similarities to center pixel
        **kwargs: Additional arguments passed to similarity function
    
    Returns:
        If center_only: (H*W-1,) array of similarities to center
        Otherwise: (H*W, H*W) similarity matrix
    """
    h, w, b = patch.shape
    n_pixels = h * w
    
    # Flatten patch to (n_pixels, bands)
    flat_patch = patch.reshape(n_pixels, b)
    
    # Select batch function based on metric
    metric_functions = {
        'euclidean': batch_euclidean_similarity,
        'rbf': lambda X, Y=None: batch_rbf_similarity(X, Y, sigma=sigma),
        'sam': batch_sam_similarity,
        'cosine': lambda X, Y=None: (np.dot(
            X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10),
            (Y if Y is not None else X).T / (np.linalg.norm(Y if Y is not None else X, axis=1, keepdims=True) + 1e-10).T
        ) + 1) / 2
    }
    
    if metric not in metric_functions:
        raise ValueError(f"Unknown metric: {metric}. Available: {list(metric_functions.keys())}")
    
    if center_only:
        center_idx = n_pixels // 2
        center_pixel = flat_patch[center_idx:center_idx+1, :]
        
        # Compute similarity to all pixels
        all_sims = metric_functions[metric](center_pixel, flat_patch).flatten()
        
        # Remove self-similarity
        mask = np.ones(n_pixels, dtype=bool)
        mask[center_idx] = False
        return all_sims[mask]
    else:
        return metric_functions[metric](flat_patch)


def compute_local_similarity_features(
    patch: np.ndarray,
    metrics: list = ['rbf', 'sam'],
    sigma_values: list = [0.1, 0.5, 1.0, 2.0],
    aggregate: str = 'mean'
) -> np.ndarray:
    """
    Extract multi-scale local similarity features from a patch.
    
    This function computes various local similarity statistics that can
    be used as features for classification.
    
    Args:
        patch: 3D array (H, W, B)
        metrics: List of similarity metrics to use
        sigma_values: RBF sigma values for multi-scale analysis
        aggregate: How to aggregate similarities ('mean', 'max', 'stats')
    
    Returns:
        Feature vector capturing local similarity characteristics
    """
    h, w, b = patch.shape
    features = []
    
    for metric in metrics:
        if metric == 'rbf':
            for sigma in sigma_values:
                sims = local_similarity_matrix(patch, metric='rbf', sigma=sigma, center_only=True)
                
                if aggregate == 'mean':
                    features.append(np.mean(sims))
                elif aggregate == 'max':
                    features.append(np.max(sims))
                elif aggregate == 'stats':
                    features.extend([np.mean(sims), np.std(sims), np.min(sims), np.max(sims)])
        else:
            sims = local_similarity_matrix(patch, metric=metric, center_only=True)
            
            if aggregate == 'mean':
                features.append(np.mean(sims))
            elif aggregate == 'max':
                features.append(np.max(sims))
            elif aggregate == 'stats':
                features.extend([np.mean(sims), np.std(sims), np.min(sims), np.max(sims)])
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# Similarity Metric Registry for MLflow Logging
# ============================================================================

SIMILARITY_METRICS: Dict[str, Callable] = {
    'euclidean': euclidean_similarity,
    'rbf': rbf_similarity,
    'sam': sam_similarity,
    'cosine': normalized_cosine_similarity,
    'sid': sid_similarity,
}

BATCH_SIMILARITY_METRICS: Dict[str, Callable] = {
    'euclidean': batch_euclidean_similarity,
    'rbf': batch_rbf_similarity,
    'sam': batch_sam_similarity,
}


def get_similarity_function(name: str, **kwargs) -> Callable:
    """
    Get a similarity function by name with optional parameters.
    
    Args:
        name: Metric name ('euclidean', 'rbf', 'sam', 'cosine', 'sid')
        **kwargs: Parameters for the metric (e.g., sigma for RBF)
    
    Returns:
        Callable similarity function
    """
    if name not in SIMILARITY_METRICS:
        raise ValueError(f"Unknown metric: {name}. Available: {list(SIMILARITY_METRICS.keys())}")
    
    base_func = SIMILARITY_METRICS[name]
    
    if kwargs:
        return lambda x, y: base_func(x, y, **kwargs)
    return base_func


def log_similarity_config_to_mlflow(metric: str, **params):
    """
    Log similarity metric configuration to MLflow.
    
    Args:
        metric: Name of the similarity metric
        **params: Additional parameters (e.g., sigma)
    """
    if MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_param("similarity_metric", metric)
        for key, value in params.items():
            mlflow.log_param(f"similarity_{key}", value)
