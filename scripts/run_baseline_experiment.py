#!/usr/bin/env python
"""
PCA + 1D CNN Baseline Experiment

Runs a baseline hyperspectral image classification experiment using:
- PCA for dimensionality reduction
- 1D CNN (SpectralCNN1D) for spectral-only classification

Dataset: Indian Pines
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the baseline PCA + 1D CNN experiment."""
    
    # Set up MLflow to use local file storage
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hsi_baseline_pca_1dcnn")
    
    # =========================================================================
    # Configuration
    # =========================================================================
    DATASET_NAME = 'indian_pines'
    DATA_DIR = project_root / 'data' / 'external' / 'indian_pines'
    
    # PCA settings
    N_COMPONENTS = 30
    
    # Training settings
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    N_EPOCHS = 100
    PATIENCE = 15
    
    # Split ratios
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    
    RANDOM_STATE = 42
    
    logger.info("=" * 60)
    logger.info("PCA + 1D CNN Baseline Experiment")
    logger.info("=" * 60)
    
    # =========================================================================
    # Step 1: Load Dataset
    # =========================================================================
    logger.info("\n[Step 1] Loading dataset...")
    
    from src.data.hsi_loader import load_benchmark_dataset
    
    cube = load_benchmark_dataset(DATASET_NAME, DATA_DIR)
    logger.info(f"Dataset: {DATASET_NAME}")
    logger.info(f"Data shape: {cube.shape}")
    logger.info(f"Number of classes: {cube.n_classes}")
    logger.info(f"Labeled pixels: {np.sum(cube.ground_truth > 0)}")
    
    # =========================================================================
    # Step 2: Apply PCA
    # =========================================================================
    logger.info("\n[Step 2] Applying PCA dimensionality reduction...")
    
    from src.data.preprocessing import apply_pca, normalize_data
    
    # First normalize the raw data
    data_normalized, _ = normalize_data(cube.data, method='minmax', log_to_mlflow=False)
    
    # Apply PCA
    data_pca, pca = apply_pca(data_normalized, n_components=N_COMPONENTS, log_to_mlflow=False)
    logger.info(f"PCA: {cube.n_bands} -> {data_pca.shape[-1]} components")
    logger.info(f"Variance retained: {sum(pca.explained_variance_ratio_):.2%}")
    
    # =========================================================================
    # Step 3: Extract Labeled Pixels
    # =========================================================================
    logger.info("\n[Step 3] Extracting labeled pixels...")
    
    # Get positions of labeled pixels
    rows, cols = np.where(cube.ground_truth > 0)
    
    # Extract spectral signatures at labeled positions
    X = data_pca[rows, cols, :]  # Shape: (N, n_components)
    y = cube.ground_truth[rows, cols] - 1  # Convert to 0-indexed
    
    n_classes = len(np.unique(y))
    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Number of classes: {n_classes}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # =========================================================================
    # Step 4: Split Data
    # =========================================================================
    logger.info("\n[Step 4] Splitting into train/val/test...")
    
    from sklearn.model_selection import train_test_split
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=TRAIN_RATIO, stratify=y, random_state=RANDOM_STATE
    )
    
    # Second split: val vs test
    relative_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=relative_val, stratify=y_temp, random_state=RANDOM_STATE
    )
    
    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Val:   {len(X_val)} samples")
    logger.info(f"Test:  {len(X_test)} samples")
    
    # =========================================================================
    # Step 5: Create DataLoaders
    # =========================================================================
    logger.info("\n[Step 5] Creating data loaders...")
    
    from src.models.train_model import create_data_loaders
    
    train_loader = create_data_loaders(X_train, y_train, batch_size=BATCH_SIZE, 
                                       shuffle=True, data_format='1d')
    val_loader = create_data_loaders(X_val, y_val, batch_size=BATCH_SIZE, 
                                     shuffle=False, data_format='1d')
    test_loader = create_data_loaders(X_test, y_test, batch_size=BATCH_SIZE, 
                                      shuffle=False, data_format='1d')
    
    # =========================================================================
    # Step 6: Create Model
    # =========================================================================
    logger.info("\n[Step 6] Creating SpectralCNN1D model...")
    
    from src.models.networks import SpectralCNN1D
    
    model = SpectralCNN1D(
        n_bands=N_COMPONENTS,  # After PCA
        n_classes=n_classes,
        n_filters=[64, 128, 256],
        kernel_sizes=[7, 5, 3],
        dropout=0.3
    )
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Input bands: {N_COMPONENTS}, Output classes: {n_classes}")
    
    # =========================================================================
    # Step 7: Train Model
    # =========================================================================
    logger.info("\n[Step 7] Training model...")
    
    from src.models.train_model import TrainingConfig, train_neural_network
    
    config = TrainingConfig(
        experiment_name="hsi_baseline_pca_1dcnn",
        run_name=f"{DATASET_NAME}_pca{N_COMPONENTS}_1dcnn",
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        patience=PATIENCE,
        log_every_n_epochs=10
    )
    
    model, history = train_neural_network(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        log_to_mlflow=True
    )
    
    # =========================================================================
    # Step 8: Evaluate on Test Set
    # =========================================================================
    logger.info("\n[Step 8] Evaluating on test set...")
    
    import torch
    from sklearn.metrics import classification_report, accuracy_score
    
    device = torch.device(config.device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    test_accuracy = accuracy_score(all_labels, all_preds)
    
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    logger.info(f"Best Val Accuracy: {max(history['val_acc']):.4f}")
    
    logger.info("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    
    logger.info("\n" + "=" * 60)
    logger.info("Experiment Complete!")
    logger.info("View MLflow results: mlflow ui")
    logger.info("=" * 60)
    
    return {
        'test_accuracy': test_accuracy,
        'best_val_accuracy': max(history['val_acc']),
        'history': history
    }


if __name__ == '__main__':
    results = main()
