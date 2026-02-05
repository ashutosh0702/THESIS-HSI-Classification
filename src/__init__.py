"""
HSI Classification Source Package
"""

from pathlib import Path

__version__ = "0.1.0"
__author__ = "Ashutosh Baruah"

# Package root
PACKAGE_ROOT = Path(__file__).parent

# Convenience imports
from .data.hsi_loader import (
    HSICube,
    load_hsi,
    load_mat_file,
    load_tiff_file,
    load_envi_file,
    load_benchmark_dataset
)

from .data.preprocessing import (
    remove_water_absorption_bands,
    apply_pca,
    apply_mnf,
    normalize_data,
    preprocess_pipeline
)

from .features.patch_extractor import (
    PatchDataset,
    extract_patches,
    split_dataset,
    create_pytorch_dataset
)

from .models.similarity_metrics import (
    euclidean_similarity,
    spectral_angle_mapper,
    sam_similarity,
    rbf_similarity,
    cosine_similarity,
    local_similarity_matrix,
    compute_local_similarity_features,
    get_similarity_function
)

from .models.networks import (
    SVMClassifier,
    create_model
)

from .models.metrics import (
    overall_accuracy,
    average_accuracy,
    kappa_coefficient,
    per_class_accuracy,
    compute_all_metrics,
    print_classification_report
)
