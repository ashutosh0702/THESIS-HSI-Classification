"""Model architectures, training, and inference modules."""

from .similarity_metrics import (
    euclidean_similarity,
    euclidean_distance,
    spectral_angle_mapper,
    sam_similarity,
    rbf_similarity,
    cosine_similarity,
    normalized_cosine_similarity,
    sid_distance,
    sid_similarity,
    batch_euclidean_similarity,
    batch_rbf_similarity,
    batch_sam_similarity,
    local_similarity_matrix,
    compute_local_similarity_features,
    get_similarity_function,
    log_similarity_config_to_mlflow,
    SIMILARITY_METRICS,
    BATCH_SIMILARITY_METRICS
)

from .networks import (
    SVMClassifier,
    create_model,
    log_model_config_to_mlflow
)

from .train_model import (
    TrainingConfig,
    EarlyStopping,
    train_svm,
    train_neural_network,
    create_data_loaders
)

from .inference import (
    batch_inference_cnn,
    batch_inference_svm,
    generate_classification_map,
    generate_probability_map,
    inference_full_image,
    save_predictions_to_mlflow
)

from .metrics import (
    overall_accuracy,
    average_accuracy,
    kappa_coefficient,
    per_class_accuracy,
    confusion_matrix_report,
    compute_all_metrics,
    log_metrics_to_mlflow,
    print_classification_report
)

__all__ = [
    # Similarity metrics
    'euclidean_similarity', 'euclidean_distance', 'spectral_angle_mapper',
    'sam_similarity', 'rbf_similarity', 'cosine_similarity', 
    'normalized_cosine_similarity', 'sid_distance', 'sid_similarity',
    'batch_euclidean_similarity', 'batch_rbf_similarity', 'batch_sam_similarity',
    'local_similarity_matrix', 'compute_local_similarity_features',
    'get_similarity_function', 'log_similarity_config_to_mlflow',
    'SIMILARITY_METRICS', 'BATCH_SIMILARITY_METRICS',
    
    # Networks
    'SVMClassifier', 'create_model', 'log_model_config_to_mlflow',
    
    # Training
    'TrainingConfig', 'EarlyStopping', 'train_svm', 'train_neural_network',
    'create_data_loaders',
    
    # Inference
    'batch_inference_cnn', 'batch_inference_svm', 'generate_classification_map',
    'generate_probability_map', 'inference_full_image', 'save_predictions_to_mlflow',
    
    # Metrics
    'overall_accuracy', 'average_accuracy', 'kappa_coefficient',
    'per_class_accuracy', 'confusion_matrix_report', 'compute_all_metrics',
    'log_metrics_to_mlflow', 'print_classification_report'
]
