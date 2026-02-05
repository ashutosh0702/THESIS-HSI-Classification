"""
Neural Network and Classifier Architectures for HSI Classification

Includes:
- SVM classifier (sklearn-based)
- 1D-CNN for spectral classification
- 2D-CNN for spatial feature extraction
- 3D-CNN for spatial-spectral joint features
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)


# ============================================================================
# SVM Classifier
# ============================================================================

class SVMClassifier:
    """
    SVM classifier wrapper with preprocessing and MLflow integration.
    
    Supports RBF and linear kernels for hyperspectral classification.
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: Union[str, float] = 'scale',
        normalize: bool = True,
        random_state: int = 42
    ):
        """
        Initialize SVM classifier.
        
        Args:
            kernel: 'rbf' or 'linear'
            C: Regularization parameter
            gamma: Kernel coefficient for RBF (or 'scale'/'auto')
            normalize: Whether to apply StandardScaler
            random_state: Random seed
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for SVM classifier")
        
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.normalize = normalize
        self.random_state = random_state
        
        svm = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True  # For probability outputs
        )
        
        if normalize:
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', svm)
            ])
        else:
            self.model = svm
    
    def fit(self, X: np.ndarray, y: np.ndarray, log_to_mlflow: bool = True):
        """
        Fit the SVM classifier.
        
        Args:
            X: Training features (N, D)
            y: Training labels (N,)
            log_to_mlflow: Whether to log parameters to MLflow
        """
        self.model.fit(X, y)
        
        if log_to_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_params({
                "classifier": "SVM",
                "svm_kernel": self.kernel,
                "svm_C": self.C,
                "svm_gamma": str(self.gamma),
                "svm_normalize": self.normalize
            })
        
        logger.info(f"Trained SVM with kernel={self.kernel}, C={self.C}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score."""
        return self.model.score(X, y)


# ============================================================================
# PyTorch Neural Networks
# ============================================================================

if TORCH_AVAILABLE:
    
    class SpectralCNN1D(nn.Module):
        """
        1D CNN for spectral-only classification.
        
        Processes each pixel's spectral signature independently.
        Input: (batch, 1, n_bands)
        """
        
        def __init__(
            self,
            n_bands: int,
            n_classes: int,
            n_filters: list = [64, 128, 256],
            kernel_sizes: list = [7, 5, 3],
            dropout: float = 0.3
        ):
            super().__init__()
            
            self.n_bands = n_bands
            self.n_classes = n_classes
            
            layers = []
            in_channels = 1
            
            for n_filter, k_size in zip(n_filters, kernel_sizes):
                layers.extend([
                    nn.Conv1d(in_channels, n_filter, k_size, padding=k_size//2),
                    nn.BatchNorm1d(n_filter),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2)
                ])
                in_channels = n_filter
            
            self.features = nn.Sequential(*layers)
            
            # Calculate feature size after convolutions
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, n_bands)
                dummy_output = self.features(dummy_input)
                self.feature_size = dummy_output.view(1, -1).size(1)
            
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feature_size, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor of shape (batch, n_bands) or (batch, 1, n_bands)
            
            Returns:
                Logits of shape (batch, n_classes)
            """
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add channel dimension
            
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    
    class SpatialCNN2D(nn.Module):
        """
        2D CNN for spatial feature extraction.
        
        Processes spatial patches, treating spectral bands as channels.
        Input: (batch, n_bands, height, width)
        """
        
        def __init__(
            self,
            n_bands: int,
            n_classes: int,
            patch_size: int = 5,
            n_filters: list = [64, 128, 256],
            dropout: float = 0.3
        ):
            super().__init__()
            
            self.n_bands = n_bands
            self.n_classes = n_classes
            self.patch_size = patch_size
            
            # Initial convolution to reduce spectral channels
            self.input_conv = nn.Sequential(
                nn.Conv2d(n_bands, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            
            # Spatial feature extraction
            layers = []
            in_channels = 64
            
            for i, n_filter in enumerate(n_filters):
                layers.extend([
                    nn.Conv2d(in_channels, n_filter, 3, padding=1),
                    nn.BatchNorm2d(n_filter),
                    nn.ReLU(inplace=True),
                ])
                in_channels = n_filter
            
            self.features = nn.Sequential(*layers)
            
            # Global average pooling
            self.gap = nn.AdaptiveAvgPool2d(1)
            
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(n_filters[-1], 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor of shape (batch, n_bands, H, W)
            
            Returns:
                Logits of shape (batch, n_classes)
            """
            x = self.input_conv(x)
            x = self.features(x)
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    
    class SpatialSpectralCNN3D(nn.Module):
        """
        3D CNN for joint spatial-spectral classification.
        
        Processes patches using 3D convolutions to capture both
        spatial and spectral correlations.
        Input: (batch, 1, depth/bands, height, width)
        """
        
        def __init__(
            self,
            n_bands: int,
            n_classes: int,
            patch_size: int = 5,
            n_filters: list = [32, 64, 128],
            dropout: float = 0.3
        ):
            super().__init__()
            
            self.n_bands = n_bands
            self.n_classes = n_classes
            self.patch_size = patch_size
            
            layers = []
            in_channels = 1
            
            # First 3D conv with spectral kernel larger than spatial
            layers.extend([
                nn.Conv3d(in_channels, n_filters[0], kernel_size=(7, 3, 3), padding=(3, 1, 1)),
                nn.BatchNorm3d(n_filters[0]),
                nn.ReLU(inplace=True),
            ])
            
            # Additional 3D convolutions
            for i in range(1, len(n_filters)):
                layers.extend([
                    nn.Conv3d(n_filters[i-1], n_filters[i], kernel_size=(5, 3, 3), padding=(2, 1, 1)),
                    nn.BatchNorm3d(n_filters[i]),
                    nn.ReLU(inplace=True),
                ])
            
            self.features = nn.Sequential(*layers)
            
            # Global average pooling
            self.gap = nn.AdaptiveAvgPool3d(1)
            
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(n_filters[-1], 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor of shape (batch, 1, n_bands, H, W)
                   or (batch, n_bands, H, W) - will add channel dim
            
            Returns:
                Logits of shape (batch, n_classes)
            """
            if x.dim() == 4:
                x = x.unsqueeze(1)  # Add channel dimension
            
            x = self.features(x)
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    
    class HybridSpatialSpectralNet(nn.Module):
        """
        Hybrid network combining separate spatial and spectral branches.
        
        Uses 1D CNN for spectral features and 2D CNN for spatial features,
        then fuses them for classification.
        """
        
        def __init__(
            self,
            n_bands: int,
            n_classes: int,
            patch_size: int = 5,
            spectral_filters: list = [64, 128],
            spatial_filters: list = [32, 64],
            dropout: float = 0.3
        ):
            super().__init__()
            
            self.n_bands = n_bands
            self.n_classes = n_classes
            self.patch_size = patch_size
            
            # Spectral branch (1D CNN on center pixel)
            spectral_layers = []
            in_ch = 1
            for n_filter in spectral_filters:
                spectral_layers.extend([
                    nn.Conv1d(in_ch, n_filter, 5, padding=2),
                    nn.BatchNorm1d(n_filter),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2)
                ])
                in_ch = n_filter
            self.spectral_branch = nn.Sequential(*spectral_layers)
            self.spectral_pool = nn.AdaptiveAvgPool1d(1)
            
            # Spatial branch (2D CNN)
            spatial_layers = [
                nn.Conv2d(n_bands, 32, 1),  # 1x1 conv for channel reduction
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ]
            in_ch = 32
            for n_filter in spatial_filters:
                spatial_layers.extend([
                    nn.Conv2d(in_ch, n_filter, 3, padding=1),
                    nn.BatchNorm2d(n_filter),
                    nn.ReLU(inplace=True)
                ])
                in_ch = n_filter
            self.spatial_branch = nn.Sequential(*spatial_layers)
            self.spatial_pool = nn.AdaptiveAvgPool2d(1)
            
            # Fusion and classification
            fusion_dim = spectral_filters[-1] + spatial_filters[-1]
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor of shape (batch, n_bands, H, W)
            
            Returns:
                Logits of shape (batch, n_classes)
            """
            batch_size = x.size(0)
            
            # Extract center pixel for spectral branch
            center = self.patch_size // 2
            center_pixel = x[:, :, center, center].unsqueeze(1)  # (batch, 1, n_bands)
            
            # Spectral features
            spectral_feat = self.spectral_branch(center_pixel)
            spectral_feat = self.spectral_pool(spectral_feat).view(batch_size, -1)
            
            # Spatial features
            spatial_feat = self.spatial_branch(x)
            spatial_feat = self.spatial_pool(spatial_feat).view(batch_size, -1)
            
            # Fusion
            fused = torch.cat([spectral_feat, spatial_feat], dim=1)
            out = self.classifier(fused)
            return out


# ============================================================================
# Model Factory
# ============================================================================

def create_model(
    model_type: str,
    n_bands: int,
    n_classes: int,
    patch_size: int = 5,
    **kwargs
) -> Any:
    """
    Factory function to create classification models.
    
    Args:
        model_type: 'svm', '1d_cnn', '2d_cnn', '3d_cnn', 'hybrid'
        n_bands: Number of spectral bands
        n_classes: Number of classes
        patch_size: Spatial patch size (for CNN models)
        **kwargs: Additional model-specific parameters
    
    Returns:
        Initialized model
    """
    model_configs = {
        'svm': lambda: SVMClassifier(**kwargs),
        '1d_cnn': lambda: SpectralCNN1D(n_bands, n_classes, **kwargs),
        '2d_cnn': lambda: SpatialCNN2D(n_bands, n_classes, patch_size, **kwargs),
        '3d_cnn': lambda: SpatialSpectralCNN3D(n_bands, n_classes, patch_size, **kwargs),
        'hybrid': lambda: HybridSpatialSpectralNet(n_bands, n_classes, patch_size, **kwargs),
    }
    
    if model_type not in model_configs:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(model_configs.keys())}")
    
    return model_configs[model_type]()


def log_model_config_to_mlflow(model_type: str, model_config: Dict):
    """Log model configuration to MLflow."""
    if MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_param("model_type", model_type)
        for key, value in model_config.items():
            mlflow.log_param(f"model_{key}", value)
