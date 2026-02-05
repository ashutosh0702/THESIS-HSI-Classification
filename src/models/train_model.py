"""
Training Module with MLflow Integration

Provides training loops for both SVM and neural network classifiers
with comprehensive experiment tracking.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    experiment_name: str = "hsi_classification"
    run_name: Optional[str] = None
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 0.001
    n_epochs: int = 100
    weight_decay: float = 1e-4
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # Device
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    # Paths
    checkpoint_dir: str = "models/checkpoints"
    
    # Logging
    log_every_n_epochs: int = 5


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_score: float, model: Optional[Any] = None) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            if model is not None:
                self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            if model is not None:
                self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        
        return self.early_stop


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: Union[str, float] = 'scale',
    experiment_name: str = "hsi_classification",
    run_name: Optional[str] = None,
    log_to_mlflow: bool = True
) -> 'SVMClassifier':
    """
    Train SVM classifier with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        kernel: SVM kernel type
        C: Regularization parameter
        gamma: Kernel coefficient
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        log_to_mlflow: Whether to log to MLflow
    
    Returns:
        Trained SVMClassifier
    """
    from .networks import SVMClassifier
    
    if log_to_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params({
                "model_type": "SVM",
                "kernel": kernel,
                "C": C,
                "gamma": str(gamma),
                "n_train_samples": len(X_train),
                "n_features": X_train.shape[1]
            })
            
            # Train
            classifier = SVMClassifier(kernel=kernel, C=C, gamma=gamma)
            classifier.fit(X_train, y_train, log_to_mlflow=False)
            
            # Evaluate
            train_acc = classifier.score(X_train, y_train)
            mlflow.log_metric("train_accuracy", train_acc)
            
            if X_val is not None and y_val is not None:
                val_acc = classifier.score(X_val, y_val)
                mlflow.log_metric("val_accuracy", val_acc)
                logger.info(f"SVM: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            else:
                logger.info(f"SVM: train_acc={train_acc:.4f}")
            
            # Log model
            mlflow.sklearn.log_model(classifier.model, "model")
    else:
        classifier = SVMClassifier(kernel=kernel, C=C, gamma=gamma)
        classifier.fit(X_train, y_train, log_to_mlflow=False)
    
    return classifier


def train_neural_network(
    model: Any,
    train_loader: Any,
    val_loader: Optional[Any] = None,
    config: Optional[TrainingConfig] = None,
    similarity_config: Optional[Dict] = None,
    log_to_mlflow: bool = True
) -> Tuple[Any, Dict]:
    """
    Train neural network with MLflow tracking.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        config: Training configuration
        similarity_config: Similarity metric configuration for logging
        log_to_mlflow: Whether to log to MLflow
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for neural network training")
    
    if config is None:
        config = TrainingConfig()
    
    device = torch.device(config.device)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    def run_epoch(loader: DataLoader, training: bool = True) -> Tuple[float, float]:
        if training:
            model.train()
        else:
            model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        context = torch.no_grad() if not training else torch.enable_grad()
        
        with context:
            for batch_data, batch_labels in loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                if training:
                    optimizer.zero_grad()
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                if training:
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item() * batch_data.size(0)
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    mlflow_context = mlflow.start_run(run_name=config.run_name) if (log_to_mlflow and MLFLOW_AVAILABLE) else None
    
    try:
        if mlflow_context:
            mlflow_context.__enter__()
            
            # Log config
            mlflow.log_params({
                "model_type": model.__class__.__name__,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "n_epochs": config.n_epochs,
                "weight_decay": config.weight_decay,
                "device": config.device
            })
            
            if similarity_config:
                for key, value in similarity_config.items():
                    mlflow.log_param(f"similarity_{key}", value)
        
        best_val_acc = 0.0
        
        for epoch in range(config.n_epochs):
            train_loss, train_acc = run_epoch(train_loader, training=True)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            if val_loader is not None:
                val_loss, val_acc = run_epoch(val_loader, training=False)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                scheduler.step(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                if early_stopping(val_acc, model):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % config.log_every_n_epochs == 0:
                val_info = f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}" if val_loader else ""
                logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                           f"train_acc={train_acc:.4f}{val_info}")
            
            if mlflow_context:
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "lr": optimizer.param_groups[0]['lr']
                }, step=epoch)
                
                if val_loader:
                    mlflow.log_metrics({
                        "val_loss": val_loss,
                        "val_acc": val_acc
                    }, step=epoch)
        
        # Restore best model
        if early_stopping.best_model_state is not None:
            model.load_state_dict(early_stopping.best_model_state)
        
        if mlflow_context:
            mlflow.log_metric("best_val_acc", best_val_acc)
            try:
                mlflow.pytorch.log_model(model, "model")
            except Exception as e:
                logger.warning(f"Could not log model artifacts to MLflow: {e}")
        
    finally:
        if mlflow_context:
            mlflow_context.__exit__(None, None, None)
    
    return model, history


def create_data_loaders(
    patches: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    data_format: str = '2d'  # '1d', '2d', '3d'
) -> Any:
    """
    Create PyTorch DataLoader from numpy arrays.
    
    Args:
        patches: Patch array (N, H, W, B) or (N, B)
        labels: Label array (N,)
        batch_size: Batch size
        shuffle: Whether to shuffle
        data_format: '1d' for spectral, '2d' or '3d' for spatial
    
    Returns:
        DataLoader
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    # Convert based on format
    if data_format == '1d':
        # Center pixel only: (N, B)
        if patches.ndim == 4:
            c = patches.shape[1] // 2
            patches = patches[:, c, c, :]
        data_tensor = torch.from_numpy(patches).float()
    
    elif data_format == '2d':
        # (N, H, W, B) -> (N, B, H, W)
        if patches.ndim == 4:
            patches = patches.transpose(0, 3, 1, 2)
        data_tensor = torch.from_numpy(patches).float()
    
    elif data_format == '3d':
        # (N, H, W, B) -> (N, 1, B, H, W)
        if patches.ndim == 4:
            patches = patches.transpose(0, 3, 1, 2)  # (N, B, H, W)
            patches = patches[:, np.newaxis, :, :, :]  # (N, 1, B, H, W)
        data_tensor = torch.from_numpy(patches).float()
    
    else:
        raise ValueError(f"Unknown data_format: {data_format}")
    
    labels_tensor = torch.from_numpy(labels).long()
    
    dataset = TensorDataset(data_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
