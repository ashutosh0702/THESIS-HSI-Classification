"""
HSI Data Loader Module
Handles loading of hyperspectral data from various formats (.mat, .tiff, .hdr)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class HSICube:
    """
    Dataclass representing a Hyperspectral Image Cube.
    
    Attributes:
        data: 3D numpy array of shape (height, width, bands)
        wavelengths: Optional array of wavelength values for each band
        metadata: Dictionary containing additional metadata
        ground_truth: Optional 2D array with class labels
        class_names: Optional list of class names
    """
    data: np.ndarray
    wavelengths: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    ground_truth: Optional[np.ndarray] = None
    class_names: Optional[List[str]] = None
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Returns (height, width, bands)"""
        return self.data.shape
    
    @property
    def n_bands(self) -> int:
        """Number of spectral bands"""
        return self.data.shape[2]
    
    @property
    def spatial_size(self) -> Tuple[int, int]:
        """Returns (height, width)"""
        return self.data.shape[:2]
    
    @property
    def n_pixels(self) -> int:
        """Total number of pixels"""
        return self.data.shape[0] * self.data.shape[1]
    
    @property
    def n_classes(self) -> int:
        """Number of unique classes in ground truth"""
        if self.ground_truth is None:
            return 0
        return len(np.unique(self.ground_truth)) - 1  # Exclude background (0)
    
    def get_pixel(self, row: int, col: int) -> np.ndarray:
        """Get spectral signature at (row, col)"""
        return self.data[row, col, :]
    
    def flatten(self) -> np.ndarray:
        """Flatten to (n_pixels, n_bands)"""
        h, w, b = self.data.shape
        return self.data.reshape(h * w, b)


def load_mat_file(filepath: Union[str, Path], 
                  data_key: Optional[str] = None,
                  gt_key: Optional[str] = None) -> HSICube:
    """
    Load hyperspectral data from MATLAB .mat file.
    
    Args:
        filepath: Path to .mat file
        data_key: Key for the data array in the .mat file (auto-detected if None)
        gt_key: Key for ground truth array (auto-detected if None)
    
    Returns:
        HSICube object with loaded data
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy is required for .mat file loading")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    mat_data = loadmat(str(filepath))
    
    # Filter out MATLAB metadata keys
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    
    # Auto-detect data key (typically largest array)
    if data_key is None:
        arrays = {k: mat_data[k] for k in data_keys if isinstance(mat_data[k], np.ndarray)}
        data_key = max(arrays.keys(), key=lambda k: arrays[k].size)
    
    data = mat_data[data_key].astype(np.float32)
    
    # Ensure data is 3D (height, width, bands)
    if data.ndim == 2:
        # Assume it's (pixels, bands) and needs reshape info
        raise ValueError("2D data requires spatial dimensions. Use load_flattened_mat() instead.")
    
    # Load ground truth if available
    ground_truth = None
    if gt_key is not None:
        ground_truth = mat_data[gt_key].astype(np.int32)
    else:
        # Try common GT key names
        gt_candidates = ['gt', 'ground_truth', 'labels', 'GT', 'groundTruth']
        for candidate in gt_candidates:
            if candidate in mat_data:
                ground_truth = mat_data[candidate].astype(np.int32)
                break
    
    return HSICube(
        data=data,
        ground_truth=ground_truth,
        metadata={'source_file': str(filepath), 'format': 'mat'}
    )


def load_tiff_file(filepath: Union[str, Path]) -> HSICube:
    """
    Load hyperspectral data from GeoTIFF file.
    
    Args:
        filepath: Path to .tiff/.tif file
    
    Returns:
        HSICube object with loaded data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        import rasterio
        with rasterio.open(str(filepath)) as src:
            # Read all bands: shape is (bands, height, width)
            data = src.read().astype(np.float32)
            # Transpose to (height, width, bands)
            data = np.transpose(data, (1, 2, 0))
            
            metadata = {
                'source_file': str(filepath),
                'format': 'geotiff',
                'crs': str(src.crs) if src.crs else None,
                'transform': src.transform,
                'bounds': src.bounds
            }
            
            return HSICube(data=data, metadata=metadata)
            
    except ImportError:
        # Fallback to tifffile
        try:
            import tifffile
            data = tifffile.imread(str(filepath)).astype(np.float32)
            
            # Handle different axis orderings
            if data.ndim == 3:
                # Assume (bands, height, width) or check shape
                if data.shape[0] < min(data.shape[1], data.shape[2]):
                    data = np.transpose(data, (1, 2, 0))
            
            return HSICube(
                data=data,
                metadata={'source_file': str(filepath), 'format': 'tiff'}
            )
        except ImportError:
            raise ImportError("rasterio or tifffile is required for TIFF loading")


def load_envi_file(header_path: Union[str, Path], 
                   data_path: Optional[Union[str, Path]] = None) -> HSICube:
    """
    Load hyperspectral data from ENVI format (.hdr + binary).
    
    Args:
        header_path: Path to .hdr header file
        data_path: Optional path to binary data file (auto-detected if None)
    
    Returns:
        HSICube object with loaded data
    """
    try:
        import spectral.io.envi as envi
    except ImportError:
        raise ImportError("spectral library is required for ENVI file loading")
    
    header_path = Path(header_path)
    if not header_path.exists():
        raise FileNotFoundError(f"Header file not found: {header_path}")
    
    # Auto-detect data file
    if data_path is None:
        stem = header_path.stem
        parent = header_path.parent
        for ext in ['', '.raw', '.img', '.dat', '.bil', '.bip', '.bsq']:
            candidate = parent / f"{stem}{ext}"
            if candidate.exists() and candidate != header_path:
                data_path = candidate
                break
    
    img = envi.open(str(header_path), str(data_path) if data_path else None)
    data = img.load().astype(np.float32)
    
    # Extract wavelengths if available
    wavelengths = None
    if hasattr(img, 'bands') and hasattr(img.bands, 'centers'):
        wavelengths = np.array(img.bands.centers)
    
    metadata = {
        'source_file': str(header_path),
        'format': 'envi',
        'interleave': img.metadata.get('interleave', 'unknown'),
        'description': img.metadata.get('description', '')
    }
    
    return HSICube(data=data, wavelengths=wavelengths, metadata=metadata)


def load_hsi(filepath: Union[str, Path], **kwargs) -> HSICube:
    """
    Universal HSI loader that auto-detects file format.
    
    Args:
        filepath: Path to HSI file (.mat, .tiff, .tif, .hdr)
        **kwargs: Additional arguments passed to specific loaders
    
    Returns:
        HSICube object with loaded data
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.mat':
        return load_mat_file(filepath, **kwargs)
    elif suffix in ['.tiff', '.tif']:
        return load_tiff_file(filepath, **kwargs)
    elif suffix == '.hdr':
        return load_envi_file(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. "
                        f"Supported formats: .mat, .tiff, .tif, .hdr")


def load_benchmark_dataset(name: str, data_dir: Union[str, Path]) -> HSICube:
    """
    Load a standard benchmark HSI dataset.
    
    Args:
        name: Dataset name ('indian_pines', 'pavia_university', 'salinas')
        data_dir: Directory containing the dataset files
    
    Returns:
        HSICube with data and ground truth
    """
    data_dir = Path(data_dir)
    
    dataset_info = {
        'indian_pines': {
            'data_file': 'Indian_pines_corrected.mat',
            'gt_file': 'Indian_pines_gt.mat',
            'data_key': 'indian_pines_corrected',
            'gt_key': 'indian_pines_gt',
            'class_names': [
                'Background', 'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                'Stone-Steel-Towers'
            ]
        },
        'pavia_university': {
            'data_file': 'PaviaU.mat',
            'gt_file': 'PaviaU_gt.mat',
            'data_key': 'paviaU',
            'gt_key': 'paviaU_gt',
            'class_names': [
                'Background', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                'Painted metal sheets', 'Bare Soil', 'Bitumen',
                'Self-Blocking Bricks', 'Shadows'
            ]
        },
        'salinas': {
            'data_file': 'Salinas_corrected.mat',
            'gt_file': 'Salinas_gt.mat',
            'data_key': 'salinas_corrected',
            'gt_key': 'salinas_gt',
            'class_names': [
                'Background', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2',
                'Fallow', 'Fallow_rough_plow', 'Fallow_smooth', 'Stubble',
                'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
                'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
                'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
                'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis'
            ]
        }
    }
    
    if name.lower() not in dataset_info:
        raise ValueError(f"Unknown dataset: {name}. "
                        f"Available: {list(dataset_info.keys())}")
    
    info = dataset_info[name.lower()]
    
    # Load data
    data_path = data_dir / info['data_file']
    cube = load_mat_file(data_path, data_key=info['data_key'])
    
    # Load ground truth
    gt_path = data_dir / info['gt_file']
    if gt_path.exists():
        from scipy.io import loadmat
        gt_data = loadmat(str(gt_path))
        cube.ground_truth = gt_data[info['gt_key']].astype(np.int32)
    
    cube.class_names = info['class_names']
    cube.metadata['dataset_name'] = name
    
    return cube
