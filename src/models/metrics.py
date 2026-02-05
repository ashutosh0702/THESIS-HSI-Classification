"""
Metrics Module

Research-grade evaluation metrics for hyperspectral classification:
- Overall Accuracy (OA)
- Average Accuracy (AA)
- Kappa Coefficient
- Per-class accuracy
- Confusion matrix analysis
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        classification_report,
        f1_score,
        precision_score,
        recall_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)


def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Overall Accuracy (OA).
    
    OA = Number of correctly classified samples / Total samples
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Overall accuracy in [0, 1]
    """
    return np.mean(y_true == y_pred)


def average_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Average Accuracy (AA).
    
    AA = Mean of per-class accuracies
    
    This metric gives equal weight to each class regardless of size,
    making it more sensitive to minority class performance.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Average accuracy in [0, 1]
    """
    classes = np.unique(y_true)
    class_accuracies = []
    
    for cls in classes:
        mask = y_true == cls
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == cls)
            class_accuracies.append(class_acc)
    
    return np.mean(class_accuracies) if class_accuracies else 0.0


def kappa_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Cohen's Kappa Coefficient.
    
    Kappa measures agreement between predictions and ground truth,
    accounting for agreement by chance.
    
    Kappa = (OA - Pe) / (1 - Pe)
    
    where Pe is the expected agreement by chance.
    
    Interpretation:
    - < 0: Less than chance agreement
    - 0-0.20: Slight agreement
    - 0.21-0.40: Fair agreement
    - 0.41-0.60: Moderate agreement
    - 0.61-0.80: Substantial agreement
    - 0.81-1.00: Almost perfect agreement
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Kappa coefficient in [-1, 1]
    """
    if SKLEARN_AVAILABLE:
        return cohen_kappa_score(y_true, y_pred)
    
    # Manual implementation
    n = len(y_true)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Observed accuracy
    po = np.mean(y_true == y_pred)
    
    # Expected accuracy
    pe = 0.0
    for cls in classes:
        pe += (np.sum(y_true == cls) / n) * (np.sum(y_pred == cls) / n)
    
    if pe == 1:
        return 1.0  # Perfect agreement
    
    return (po - pe) / (1 - pe)


def per_class_accuracy(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute per-class accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
    
    Returns:
        Dictionary mapping class name/id to accuracy
    """
    classes = np.unique(y_true)
    results = {}
    
    for i, cls in enumerate(classes):
        mask = y_true == cls
        acc = np.mean(y_pred[mask] == cls) if np.sum(mask) > 0 else 0.0
        
        key = class_names[cls] if class_names else str(cls)
        results[key] = acc
    
    return results


def confusion_matrix_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False
) -> Dict:
    """
    Generate comprehensive confusion matrix analysis.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        normalize: Whether to normalize the confusion matrix
    
    Returns:
        Dictionary with confusion matrix and various statistics
    """
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Compute confusion matrix
    if SKLEARN_AVAILABLE:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            i = np.where(classes == t)[0][0]
            j = np.where(classes == p)[0][0]
            cm[i, j] += 1
    
    if normalize:
        cm = cm.astype(np.float64)
        cm = cm / cm.sum(axis=1, keepdims=True)
        np.nan_to_num(cm, copy=False)
    
    # Per-class statistics
    per_class_stats = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        key = class_names[cls] if class_names else str(cls)
        per_class_stats[key] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': int(cm[i, :].sum()) if not normalize else cm[i, :].sum()
        }
    
    return {
        'confusion_matrix': cm,
        'classes': classes.tolist(),
        'class_names': class_names or [str(c) for c in classes],
        'per_class_stats': per_class_stats
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute all standard HSI classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
    
    Returns:
        Dictionary with all metrics
    """
    oa = overall_accuracy(y_true, y_pred)
    aa = average_accuracy(y_true, y_pred)
    kappa = kappa_coefficient(y_true, y_pred)
    per_class = per_class_accuracy(y_true, y_pred, class_names)
    cm_report = confusion_matrix_report(y_true, y_pred, class_names)
    
    return {
        'overall_accuracy': oa,
        'average_accuracy': aa,
        'kappa_coefficient': kappa,
        'per_class_accuracy': per_class,
        'confusion_matrix': cm_report['confusion_matrix'],
        'per_class_stats': cm_report['per_class_stats']
    }


def log_metrics_to_mlflow(
    metrics: Dict,
    prefix: str = "",
    log_confusion_matrix: bool = True
):
    """
    Log classification metrics to MLflow.
    
    Args:
        metrics: Dictionary from compute_all_metrics
        prefix: Prefix for metric names (e.g., 'test_')
        log_confusion_matrix: Whether to log confusion matrix as artifact
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, skipping metric logging")
        return
    
    if not mlflow.active_run():
        logger.warning("No active MLflow run, skipping metric logging")
        return
    
    # Log main metrics
    mlflow.log_metrics({
        f"{prefix}overall_accuracy": metrics['overall_accuracy'],
        f"{prefix}average_accuracy": metrics['average_accuracy'],
        f"{prefix}kappa_coefficient": metrics['kappa_coefficient']
    })
    
    # Log per-class accuracies
    for class_name, acc in metrics['per_class_accuracy'].items():
        clean_name = str(class_name).replace(' ', '_').replace('-', '_')[:50]
        mlflow.log_metric(f"{prefix}class_acc_{clean_name}", acc)
    
    # Log confusion matrix as artifact
    if log_confusion_matrix:
        import tempfile
        import json
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save confusion matrix
            cm_path = Path(tmp_dir) / f"{prefix}confusion_matrix.npy"
            np.save(str(cm_path), metrics['confusion_matrix'])
            mlflow.log_artifact(str(cm_path))
            
            # Save per-class stats as JSON
            stats_path = Path(tmp_dir) / f"{prefix}per_class_stats.json"
            
            # Convert numpy types to Python types for JSON serialization
            stats_clean = {}
            for k, v in metrics['per_class_stats'].items():
                stats_clean[k] = {
                    kk: float(vv) if isinstance(vv, (np.floating, float)) else int(vv)
                    for kk, vv in v.items()
                }
            
            with open(stats_path, 'w') as f:
                json.dump(stats_clean, f, indent=2)
            mlflow.log_artifact(str(stats_path))
    
    logger.info(f"Logged metrics to MLflow: OA={metrics['overall_accuracy']:.4f}, "
                f"AA={metrics['average_accuracy']:.4f}, Kappa={metrics['kappa_coefficient']:.4f}")


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> str:
    """
    Generate a formatted classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
    
    Returns:
        Formatted report string
    """
    metrics = compute_all_metrics(y_true, y_pred, class_names)
    
    lines = [
        "=" * 60,
        "CLASSIFICATION REPORT",
        "=" * 60,
        f"Overall Accuracy (OA):    {metrics['overall_accuracy']:.4f}",
        f"Average Accuracy (AA):    {metrics['average_accuracy']:.4f}",
        f"Kappa Coefficient:        {metrics['kappa_coefficient']:.4f}",
        "",
        "-" * 60,
        "Per-Class Accuracy:",
        "-" * 60
    ]
    
    for class_name, acc in metrics['per_class_accuracy'].items():
        lines.append(f"  {class_name:30s} {acc:.4f}")
    
    lines.extend([
        "",
        "-" * 60,
        "Per-Class Statistics:",
        "-" * 60,
        f"{'Class':30s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}"
    ])
    
    for class_name, stats in metrics['per_class_stats'].items():
        lines.append(
            f"{class_name:30s} {stats['precision']:>10.4f} {stats['recall']:>10.4f} "
            f"{stats['f1_score']:>10.4f} {stats['support']:>10d}"
        )
    
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    print(report)
    return report
