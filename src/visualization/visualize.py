"""
HSI Classification Visualization Dashboard

Streamlit-based dashboard for:
- Viewing spectral bands
- Comparing ground truth vs. predictions
- Analyzing similarity metrics
- Loading MLflow experiment results
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import streamlit as st
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    raise ImportError("Streamlit and matplotlib are required for the dashboard")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def create_colormap(n_classes: int, include_background: bool = True) -> Tuple[List, Dict]:
    """Create a colormap for classification visualization."""
    # Use a perceptually distinct colormap
    base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    if n_classes > 20:
        extra = plt.cm.Set3(np.linspace(0, 1, n_classes - 20))
        colors = np.vstack([base_colors, extra])
    else:
        colors = base_colors[:n_classes]
    
    if include_background:
        # Make background black
        colors[0] = [0, 0, 0, 1]
    
    color_list = [tuple(c) for c in colors]
    cmap = mcolors.ListedColormap(color_list)
    
    return color_list, cmap


def plot_classification_map(
    classification: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Classification Map",
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True
) -> plt.Figure:
    """Plot a classification map with legend."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure
    
    n_classes = int(np.max(classification)) + 1
    colors, cmap = create_colormap(n_classes)
    
    im = ax.imshow(classification, cmap=cmap, vmin=0, vmax=n_classes - 1)
    ax.set_title(title)
    ax.axis('off')
    
    if show_legend and class_names:
        patches = [
            Patch(color=colors[i], label=class_names[i])
            for i in range(min(len(class_names), n_classes))
        ]
        ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=8)
    
    return fig


def plot_spectral_signature(
    spectra: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    title: str = "Spectral Signatures"
) -> plt.Figure:
    """Plot spectral signatures."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)
    
    n_bands = spectra.shape[1]
    x = wavelengths if wavelengths is not None else np.arange(n_bands)
    
    for i, spectrum in enumerate(spectra):
        label = labels[i] if labels else f"Spectrum {i+1}"
        ax.plot(x, spectrum, label=label)
    
    ax.set_xlabel("Wavelength (nm)" if wavelengths is not None else "Band Index")
    ax.set_ylabel("Reflectance")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


# ============================================================================
# Streamlit App
# ============================================================================

def main():
    st.set_page_config(
        page_title="HSI Classification Dashboard",
        page_icon="🛰️",
        layout="wide"
    )
    
    st.title("🛰️ Hyperspectral Image Classification Dashboard")
    st.markdown("---")
    
    # Sidebar for data loading
    with st.sidebar:
        st.header("📁 Data Loading")
        
        data_source = st.selectbox(
            "Data Source",
            ["Upload Files", "MLflow Experiment", "Sample Data"]
        )
        
        if data_source == "Upload Files":
            hsi_file = st.file_uploader("HSI Data (.npy)", type=['npy'])
            gt_file = st.file_uploader("Ground Truth (.npy)", type=['npy'])
            pred_file = st.file_uploader("Predictions (.npy)", type=['npy'])
            
            if hsi_file:
                st.session_state['hsi_data'] = np.load(hsi_file)
            if gt_file:
                st.session_state['ground_truth'] = np.load(gt_file)
            if pred_file:
                st.session_state['predictions'] = np.load(pred_file)
        
        elif data_source == "MLflow Experiment":
            if MLFLOW_AVAILABLE:
                experiment_name = st.text_input("Experiment Name", "hsi_classification")
                
                if st.button("Load Experiments"):
                    try:
                        mlflow.set_tracking_uri("mlruns")
                        experiment = mlflow.get_experiment_by_name(experiment_name)
                        if experiment:
                            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                            st.session_state['mlflow_runs'] = runs
                            st.success(f"Found {len(runs)} runs")
                        else:
                            st.warning("Experiment not found")
                    except Exception as e:
                        st.error(f"Error: {e}")
                
                if 'mlflow_runs' in st.session_state:
                    runs = st.session_state['mlflow_runs']
                    run_names = runs['tags.mlflow.runName'].fillna(runs['run_id']).tolist()
                    selected_run = st.selectbox("Select Run", run_names)
            else:
                st.warning("MLflow not installed")
        
        else:  # Sample Data
            if st.button("Generate Sample Data"):
                # Generate synthetic data for demo
                np.random.seed(42)
                h, w, b = 50, 50, 100
                n_classes = 5
                
                hsi = np.random.randn(h, w, b).astype(np.float32)
                gt = np.random.randint(0, n_classes, (h, w))
                pred = gt.copy()
                # Add some errors
                error_mask = np.random.random((h, w)) < 0.1
                pred[error_mask] = np.random.randint(0, n_classes, np.sum(error_mask))
                
                st.session_state['hsi_data'] = hsi
                st.session_state['ground_truth'] = gt
                st.session_state['predictions'] = pred
                st.session_state['class_names'] = [f"Class {i}" for i in range(n_classes)]
                st.success("Sample data generated!")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Band Viewer",
        "🗺️ Classification Maps",
        "📈 Metrics Analysis",
        "🔍 Similarity Explorer"
    ])
    
    # ========== Tab 1: Band Viewer ==========
    with tab1:
        st.header("Spectral Band Viewer")
        
        if 'hsi_data' in st.session_state:
            hsi = st.session_state['hsi_data']
            h, w, n_bands = hsi.shape
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                band_idx = st.slider("Select Band", 0, n_bands - 1, n_bands // 2)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(hsi[:, :, band_idx], cmap='viridis')
                ax.set_title(f"Band {band_idx}")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("Data Info")
                st.write(f"**Shape:** {hsi.shape}")
                st.write(f"**Bands:** {n_bands}")
                st.write(f"**Size:** {h} x {w}")
                st.write(f"**Min:** {hsi.min():.4f}")
                st.write(f"**Max:** {hsi.max():.4f}")
                
                st.subheader("Click to View Spectrum")
                st.info("Select a pixel position below")
                
                row = st.number_input("Row", 0, h-1, h//2)
                col = st.number_input("Column", 0, w-1, w//2)
                
                if st.button("Plot Spectrum"):
                    spectrum = hsi[row, col, :]
                    fig = plot_spectral_signature(spectrum, title=f"Spectrum at ({row}, {col})")
                    st.pyplot(fig)
                    plt.close()
        else:
            st.info("Please load HSI data from the sidebar")
    
    # ========== Tab 2: Classification Maps ==========
    with tab2:
        st.header("Classification Map Comparison")
        
        has_gt = 'ground_truth' in st.session_state
        has_pred = 'predictions' in st.session_state
        
        if has_gt or has_pred:
            class_names = st.session_state.get('class_names', None)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if has_gt:
                    st.subheader("Ground Truth")
                    gt = st.session_state['ground_truth']
                    n_classes = int(np.max(gt)) + 1
                    colors, cmap = create_colormap(n_classes)
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(gt, cmap=cmap, vmin=0, vmax=n_classes-1)
                    ax.set_title("Ground Truth")
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
            
            with col2:
                if has_pred:
                    st.subheader("Predictions")
                    pred = st.session_state['predictions']
                    n_classes = int(np.max(pred)) + 1
                    colors, cmap = create_colormap(n_classes)
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(pred, cmap=cmap, vmin=0, vmax=n_classes-1)
                    ax.set_title("Predicted")
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
            
            # Overlay view
            if has_gt and has_pred:
                st.subheader("Error Map")
                gt = st.session_state['ground_truth']
                pred = st.session_state['predictions']
                
                error_map = (gt != pred).astype(np.int32)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(error_map, cmap='RdYlGn_r')
                ax.set_title("Classification Errors (Red = Wrong)")
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
                
                # Error statistics
                n_errors = np.sum(error_map)
                total = error_map.size
                st.metric("Error Rate", f"{100 * n_errors / total:.2f}%")
        else:
            st.info("Please load ground truth and/or predictions from the sidebar")
    
    # ========== Tab 3: Metrics Analysis ==========
    with tab3:
        st.header("Classification Metrics")
        
        if 'ground_truth' in st.session_state and 'predictions' in st.session_state:
            gt = st.session_state['ground_truth'].flatten()
            pred = st.session_state['predictions'].flatten()
            
            # Filter out background if needed
            mask = gt > 0
            gt_filtered = gt[mask]
            pred_filtered = pred[mask]
            
            # Import metrics
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            try:
                from models.metrics import compute_all_metrics
                metrics = compute_all_metrics(gt_filtered, pred_filtered)
            except ImportError:
                # Fallback implementation
                metrics = {
                    'overall_accuracy': np.mean(gt_filtered == pred_filtered),
                    'average_accuracy': 0.0,
                    'kappa_coefficient': 0.0,
                    'per_class_accuracy': {}
                }
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Accuracy", f"{100*metrics['overall_accuracy']:.2f}%")
            with col2:
                st.metric("Average Accuracy", f"{100*metrics['average_accuracy']:.2f}%")
            with col3:
                st.metric("Kappa Coefficient", f"{metrics['kappa_coefficient']:.4f}")
            
            # Per-class accuracy chart
            st.subheader("Per-Class Accuracy")
            if metrics['per_class_accuracy']:
                classes = list(metrics['per_class_accuracy'].keys())
                accuracies = list(metrics['per_class_accuracy'].values())
                
                fig, ax = plt.subplots(figsize=(12, 5))
                bars = ax.bar(range(len(classes)), accuracies)
                ax.set_xticks(range(len(classes)))
                ax.set_xticklabels(classes, rotation=45, ha='right')
                ax.set_ylabel("Accuracy")
                ax.set_ylim(0, 1)
                ax.axhline(y=metrics['overall_accuracy'], color='r', linestyle='--', 
                          label=f"OA: {metrics['overall_accuracy']:.3f}")
                ax.legend()
                st.pyplot(fig)
                plt.close()
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            if 'confusion_matrix' in metrics:
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(metrics['confusion_matrix'], cmap='Blues')
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
                plt.close()
        else:
            st.info("Please load both ground truth and predictions to view metrics")
    
    # ========== Tab 4: Similarity Explorer ==========
    with tab4:
        st.header("Local Similarity Analysis")
        
        if 'hsi_data' in st.session_state:
            hsi = st.session_state['hsi_data']
            h, w, n_bands = hsi.shape
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Parameters")
                
                window_size = st.selectbox("Window Size", [3, 5, 7, 9], index=1)
                metric = st.selectbox("Similarity Metric", 
                                     ["RBF", "Euclidean", "SAM", "Cosine"])
                
                if metric == "RBF":
                    sigma = st.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
                
                row = st.number_input("Center Row", 0, h-1, h//2, key="sim_row")
                col_pos = st.number_input("Center Col", 0, w-1, w//2, key="sim_col")
                
                compute = st.button("Compute Similarity")
            
            with col2:
                if compute:
                    half = window_size // 2
                    
                    # Extract patch
                    r_start = max(0, row - half)
                    r_end = min(h, row + half + 1)
                    c_start = max(0, col_pos - half)
                    c_end = min(w, col_pos + half + 1)
                    
                    patch = hsi[r_start:r_end, c_start:c_end, :]
                    
                    # Compute similarity
                    try:
                        from models.similarity_metrics import local_similarity_matrix
                        
                        sim_params = {'sigma': sigma} if metric == "RBF" else {}
                        sim_matrix = local_similarity_matrix(
                            patch, 
                            metric=metric.lower(),
                            **sim_params
                        )
                        
                        # Visualize
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Show patch
                        axes[0].imshow(patch[:, :, n_bands//2], cmap='gray')
                        axes[0].set_title(f"Patch (Band {n_bands//2})")
                        axes[0].axis('off')
                        
                        # Show similarity matrix
                        im = axes[1].imshow(sim_matrix, cmap='hot', vmin=0, vmax=1)
                        axes[1].set_title(f"{metric} Similarity Matrix")
                        plt.colorbar(im, ax=axes[1])
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        # Statistics
                        st.write(f"**Mean Similarity:** {np.mean(sim_matrix):.4f}")
                        st.write(f"**Std Similarity:** {np.std(sim_matrix):.4f}")
                        
                    except ImportError:
                        st.error("Could not import similarity_metrics module")
        else:
            st.info("Please load HSI data to explore local similarity")
    
    # Footer
    st.markdown("---")
    st.markdown("*HSI Classification Dashboard | Built with Streamlit*")


if __name__ == "__main__":
    main()
