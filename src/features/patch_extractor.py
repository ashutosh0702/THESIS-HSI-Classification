"""
Patch Extractor Module
Extracts spatial-spectral patches for local similarity analysis
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class PatchDataset:
    """
    Container for extracted patches and labels.
    
    Attributes:
        patches: Array of shape (N, window_size, window_size, bands)
        labels: Array of shape (N,) with class labels
        positions: Array of shape (N, 2) with (row, col) positions
        window_size: Patch window size
        n_classes: Number of unique classes
    """
    patches: np.ndarray
    labels: np.ndarray
    positions: np.ndarray
    window_size: int
    
    @property
    def n_samples(self) -> int:
        return len(self.labels)
    
    @property
    def n_classes(self) -> int:
        return len(np.unique(self.labels))
    
    @property
    def spectral_dim(self) -> int:
        return self.patches.shape[-1]
    
    def get_center_pixels(self) -> np.ndarray:
        """Get center pixel spectral signatures from all patches."""
        c = self.window_size // 2
        return self.patches[:, c, c, :]
    
    def to_flat_patches(self) -> np.ndarray:
        """Flatten spatial dimensions: (N, WxWxB)"""
        return self.patches.reshape(self.n_samples, -1)


def extract_patches(
    data: np.ndarray,
    ground_truth: np.ndarray,
    window_size: int = 5,
    padding_mode: str = 'reflect',
    include_background: bool = False,
    log_to_mlflow: bool = True
) -> PatchDataset:
    """
    Extract spatial patches around labeled pixels.
    
    Args:
        data: HSI cube of shape (H, W, B)
        ground_truth: Label map of shape (H, W), 0 = background
        window_size: Patch size (must be odd for centered extraction)
        padding_mode: Numpy pad mode for edge handling
        include_background: Whether to include background (label 0) pixels
        log_to_mlflow: Whether to log to MLflow
    
    Returns:
        PatchDataset with extracted patches
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")
    
    h, w, b = data.shape
    half_size = window_size // 2
    
    # Pad the data to handle edge pixels
    padded_data = np.pad(
        data,
        [(half_size, half_size), (half_size, half_size), (0, 0)],
        mode=padding_mode
    )
    
    # Find labeled pixel positions
    if include_background:
        rows, cols = np.where(ground_truth >= 0)
    else:
        rows, cols = np.where(ground_truth > 0)
    
    n_samples = len(rows)
    patches = np.zeros((n_samples, window_size, window_size, b), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int32)
    positions = np.zeros((n_samples, 2), dtype=np.int32)
    
    for i, (r, c) in enumerate(zip(rows, cols)):
        # Extract patch from padded data
        r_pad, c_pad = r + half_size, c + half_size
        patches[i] = padded_data[
            r_pad - half_size:r_pad + half_size + 1,
            c_pad - half_size:c_pad + half_size + 1,
            :
        ]
        labels[i] = ground_truth[r, c]
        positions[i] = [r, c]
    
    if log_to_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_params({
            "patch_window_size": window_size,
            "patch_padding_mode": padding_mode,
            "patch_n_samples": n_samples,
            "patch_include_background": include_background
        })
    
    logger.info(f"Extracted {n_samples} patches of size {window_size}x{window_size}x{b}")
    
    return PatchDataset(
        patches=patches,
        labels=labels,
        positions=positions,
        window_size=window_size
    )


def split_dataset(
    dataset: PatchDataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
    spatial_aware: bool = False,
    log_to_mlflow: bool = True
) -> Tuple[PatchDataset, PatchDataset, PatchDataset]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset: PatchDataset to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        stratify: Whether to maintain class distribution
        random_state: Random seed for reproducibility
        spatial_aware: If True, ensures spatial separation between splits
        log_to_mlflow: Whether to log to MLflow
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0")
    
    indices = np.arange(dataset.n_samples)
    labels = dataset.labels
    
    if spatial_aware:
        # Use spatial blocking to prevent data leakage
        # Group pixels by spatial blocks
        positions = dataset.positions
        block_size = 10  # Group pixels in 10x10 blocks
        block_ids = (positions[:, 0] // block_size) * 1000 + (positions[:, 1] // block_size)
        unique_blocks = np.unique(block_ids)
        
        # Split blocks instead of individual samples
        n_blocks = len(unique_blocks)
        train_blocks = int(n_blocks * train_ratio)
        val_blocks = int(n_blocks * val_ratio)
        
        np.random.seed(random_state)
        shuffled_blocks = np.random.permutation(unique_blocks)
        
        train_block_set = set(shuffled_blocks[:train_blocks])
        val_block_set = set(shuffled_blocks[train_blocks:train_blocks + val_blocks])
        test_block_set = set(shuffled_blocks[train_blocks + val_blocks:])
        
        train_idx = [i for i in indices if block_ids[i] in train_block_set]
        val_idx = [i for i in indices if block_ids[i] in val_block_set]
        test_idx = [i for i in indices if block_ids[i] in test_block_set]
    else:
        # Standard random split
        stratify_labels = labels if stratify else None
        
        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=stratify_labels,
            random_state=random_state
        )
        
        # Second split: val vs test
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        temp_labels = labels[temp_idx] if stratify else None
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=relative_val_ratio,
            stratify=temp_labels,
            random_state=random_state
        )
    
    def create_subset(idx):
        return PatchDataset(
            patches=dataset.patches[idx],
            labels=dataset.labels[idx],
            positions=dataset.positions[idx],
            window_size=dataset.window_size
        )
    
    train_set = create_subset(train_idx)
    val_set = create_subset(val_idx)
    test_set = create_subset(test_idx)
    
    if log_to_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_params({
            "split_train_samples": len(train_idx),
            "split_val_samples": len(val_idx),
            "split_test_samples": len(test_idx),
            "split_stratified": stratify,
            "split_spatial_aware": spatial_aware,
            "split_random_state": random_state
        })
    
    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    return train_set, val_set, test_set


def extract_neighborhood_features(
    data: np.ndarray,
    positions: np.ndarray,
    window_size: int = 5,
    feature_type: str = 'full'
) -> np.ndarray:
    """
    Extract neighborhood-based features for given positions.
    
    Args:
        data: HSI cube (H, W, B)
        positions: Array of (row, col) positions (N, 2)
        window_size: Neighborhood window size
        feature_type: 
            'full' - Full patch flattened
            'mean' - Mean of neighborhood
            'mean_std' - Mean and std of neighborhood
            'center_neighbors' - Center pixel + mean of neighbors
    
    Returns:
        Feature array of shape (N, feature_dim)
    """
    h, w, b = data.shape
    half_size = window_size // 2
    n_samples = len(positions)
    
    # Pad data
    padded = np.pad(
        data,
        [(half_size, half_size), (half_size, half_size), (0, 0)],
        mode='reflect'
    )
    
    features_list = []
    
    for r, c in positions:
        r_pad, c_pad = r + half_size, c + half_size
        patch = padded[
            r_pad - half_size:r_pad + half_size + 1,
            c_pad - half_size:c_pad + half_size + 1,
            :
        ]
        center = patch[half_size, half_size, :]
        
        if feature_type == 'full':
            feat = patch.flatten()
        elif feature_type == 'mean':
            feat = patch.mean(axis=(0, 1))
        elif feature_type == 'mean_std':
            feat = np.concatenate([
                patch.mean(axis=(0, 1)),
                patch.std(axis=(0, 1))
            ])
        elif feature_type == 'center_neighbors':
            neighbors = patch.mean(axis=(0, 1))
            feat = np.concatenate([center, neighbors])
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        
        features_list.append(feat)
    
    return np.array(features_list, dtype=np.float32)


def create_pytorch_dataset(dataset: PatchDataset):
    """
    Convert PatchDataset to PyTorch Dataset.
    
    Returns:
        torch.utils.data.TensorDataset
    """
    try:
        import torch
        from torch.utils.data import TensorDataset
    except ImportError:
        raise ImportError("PyTorch is required for create_pytorch_dataset()")
    
    # Convert to tensors
    # Patches: (N, H, W, C) -> (N, C, H, W) for PyTorch conv layers
    patches_tensor = torch.from_numpy(
        dataset.patches.transpose(0, 3, 1, 2)
    ).float()
    labels_tensor = torch.from_numpy(dataset.labels).long()
    
    return TensorDataset(patches_tensor, labels_tensor)
