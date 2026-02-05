"""
Inference Module

Batch inference with memory management, probability map generation,
and MLflow artifact logging.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)


def batch_inference_cnn(
    model: Any,
    data: np.ndarray,
    batch_size: int = 64,
    device: str = 'cuda',
    data_format: str = '2d',
    return_probabilities: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Perform batch inference on data using a CNN model.
    
    Args:
        model: Trained PyTorch model
        data: Input data (N, H, W, B) or (N, B)
        batch_size: Batch size for inference
        device: Device to use ('cuda' or 'cpu')
        data_format: '1d', '2d', or '3d' for data reshaping
        return_probabilities: Whether to return probability maps
    
    Returns:
        Tuple of (predictions, probabilities)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for CNN inference")
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Prepare data based on format
    if data_format == '1d':
        if data.ndim == 4:
            c = data.shape[1] // 2
            data = data[:, c, c, :]
        data_tensor = torch.from_numpy(data).float()
    elif data_format == '2d':
        if data.ndim == 4:
            data = data.transpose(0, 3, 1, 2)
        data_tensor = torch.from_numpy(data).float()
    elif data_format == '3d':
        if data.ndim == 4:
            data = data.transpose(0, 3, 1, 2)
            data = data[:, np.newaxis, :, :, :]
        data_tensor = torch.from_numpy(data).float()
    else:
        raise ValueError(f"Unknown data_format: {data_format}")
    
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            outputs = model(batch)
            
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_predictions.append(preds.cpu().numpy())
            if return_probabilities:
                all_probabilities.append(probs.cpu().numpy())
    
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities) if return_probabilities else None
    
    return predictions, probabilities


def batch_inference_svm(
    model,
    data: np.ndarray,
    batch_size: int = 1000,
    return_probabilities: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Perform batch inference using SVM classifier.
    
    Args:
        model: Trained SVM classifier or sklearn pipeline
        data: Input features (N, D)
        batch_size: Batch size for memory-efficient processing
        return_probabilities: Whether to return probability estimates
    
    Returns:
        Tuple of (predictions, probabilities)
    """
    n_samples = len(data)
    all_predictions = []
    all_probabilities = []
    
    for i in range(0, n_samples, batch_size):
        batch = data[i:i + batch_size]
        
        preds = model.predict(batch)
        all_predictions.append(preds)
        
        if return_probabilities:
            probs = model.predict_proba(batch)
            all_probabilities.append(probs)
    
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities) if return_probabilities else None
    
    return predictions, probabilities


def generate_classification_map(
    predictions: np.ndarray,
    positions: np.ndarray,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Generate a 2D classification map from predictions and positions.
    
    Args:
        predictions: Array of predicted labels (N,)
        positions: Array of (row, col) positions (N, 2)
        image_shape: (height, width) of the output map
    
    Returns:
        2D classification map of shape (height, width)
    """
    h, w = image_shape
    classification_map = np.zeros((h, w), dtype=np.int32)
    
    for pred, (r, c) in zip(predictions, positions):
        classification_map[r, c] = pred
    
    return classification_map


def generate_probability_map(
    probabilities: np.ndarray,
    positions: np.ndarray,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Generate 3D probability map from class probabilities.
    
    Args:
        probabilities: Array of class probabilities (N, n_classes)
        positions: Array of (row, col) positions (N, 2)
        image_shape: (height, width) of the output map
    
    Returns:
        3D probability map of shape (height, width, n_classes)
    """
    h, w = image_shape
    n_classes = probabilities.shape[1]
    prob_map = np.zeros((h, w, n_classes), dtype=np.float32)
    
    for probs, (r, c) in zip(probabilities, positions):
        prob_map[r, c, :] = probs
    
    return prob_map


def inference_full_image(
    model: Any,
    hsi_cube: np.ndarray,
    ground_truth_mask: Optional[np.ndarray] = None,
    window_size: int = 5,
    batch_size: int = 256,
    model_type: str = 'cnn',
    data_format: str = '2d',
    device: str = 'cuda'
) -> Dict:
    """
    Perform inference on the full HSI image.
    
    Processes the image in spatial chunks to manage memory.
    
    Args:
        model: Trained model
        hsi_cube: Full HSI cube (H, W, B)
        ground_truth_mask: Optional mask (H, W) where 0 = skip
        window_size: Patch window size
        batch_size: Batch size for inference
        model_type: 'cnn' or 'svm'
        data_format: '1d', '2d', or '3d'
        device: Device for CNN inference
    
    Returns:
        Dictionary with classification_map, probability_map, and positions
    """
    h, w, b = hsi_cube.shape
    half_size = window_size // 2
    
    # Pad image
    padded = np.pad(
        hsi_cube,
        [(half_size, half_size), (half_size, half_size), (0, 0)],
        mode='reflect'
    )
    
    # Find positions to process
    if ground_truth_mask is not None:
        rows, cols = np.where(ground_truth_mask > 0)
    else:
        rows, cols = np.mgrid[0:h, 0:w]
        rows, cols = rows.flatten(), cols.flatten()
    
    n_pixels = len(rows)
    positions = np.column_stack([rows, cols])
    
    # Extract patches
    patches = np.zeros((n_pixels, window_size, window_size, b), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        r_pad, c_pad = r + half_size, c + half_size
        patches[i] = padded[
            r_pad - half_size:r_pad + half_size + 1,
            c_pad - half_size:c_pad + half_size + 1,
            :
        ]
    
    logger.info(f"Running inference on {n_pixels} pixels...")
    
    # Run inference
    if model_type == 'cnn':
        predictions, probabilities = batch_inference_cnn(
            model, patches, batch_size, device, data_format, return_probabilities=True
        )
    else:
        # Flatten patches for SVM
        if data_format == '1d':
            c = window_size // 2
            flat_data = patches[:, c, c, :]
        else:
            flat_data = patches.reshape(n_pixels, -1)
        
        predictions, probabilities = batch_inference_svm(
            model, flat_data, batch_size, return_probabilities=True
        )
    
    # Generate maps
    classification_map = generate_classification_map(predictions, positions, (h, w))
    probability_map = None
    if probabilities is not None:
        probability_map = generate_probability_map(probabilities, positions, (h, w))
    
    return {
        'classification_map': classification_map,
        'probability_map': probability_map,
        'predictions': predictions,
        'positions': positions,
        'probabilities': probabilities
    }


def save_predictions_to_mlflow(
    classification_map: np.ndarray,
    probability_map: Optional[np.ndarray] = None,
    run_id: Optional[str] = None,
    artifact_prefix: str = "predictions"
):
    """
    Save prediction maps as MLflow artifacts.
    
    Args:
        classification_map: 2D classification map
        probability_map: Optional 3D probability map
        run_id: Optional MLflow run ID (uses active run if None)
        artifact_prefix: Prefix for artifact filenames
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, skipping artifact logging")
        return
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save classification map
        class_path = Path(tmp_dir) / f"{artifact_prefix}_classification.npy"
        np.save(str(class_path), classification_map)
        mlflow.log_artifact(str(class_path))
        
        # Save probability map if available
        if probability_map is not None:
            prob_path = Path(tmp_dir) / f"{artifact_prefix}_probabilities.npy"
            np.save(str(prob_path), probability_map)
            mlflow.log_artifact(str(prob_path))
        
        logger.info(f"Saved prediction artifacts to MLflow")
