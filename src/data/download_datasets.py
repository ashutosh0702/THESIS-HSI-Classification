#!/usr/bin/env python
"""
Download Benchmark Hyperspectral Datasets

Downloads and prepares standard HSI benchmark datasets:
- Indian Pines (AVIRIS sensor)
- Pavia University (ROSIS sensor)
- Salinas (AVIRIS sensor)

Data source: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
"""

import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

# Try to import scipy for .mat file handling
try:
    from scipy.io import loadmat, savemat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed. Install with: pip install scipy")


# Dataset URLs (from the GIC - University of the Basque Country)
DATASETS = {
    'indian_pines': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        'gt_url': 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
        'data_key': 'indian_pines_corrected',
        'gt_key': 'indian_pines_gt',
        'description': 'Indian Pines scene (AVIRIS sensor, 145x145 pixels, 200 bands after water removal)',
        'classes': [
            'Background', 'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
            'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
            'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
            'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
            'Stone-Steel-Towers'
        ]
    },
    'pavia_university': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
        'gt_url': 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
        'data_key': 'paviaU',
        'gt_key': 'paviaU_gt',
        'description': 'Pavia University scene (ROSIS sensor, 610x340 pixels, 103 bands)',
        'classes': [
            'Background', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
            'Painted metal sheets', 'Bare Soil', 'Bitumen',
            'Self-Blocking Bricks', 'Shadows'
        ]
    },
    'salinas': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
        'gt_url': 'http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
        'data_key': 'salinas_corrected',
        'gt_key': 'salinas_gt',
        'description': 'Salinas scene (AVIRIS sensor, 512x217 pixels, 204 bands after water removal)',
        'classes': [
            'Background', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2',
            'Fallow', 'Fallow_rough_plow', 'Fallow_smooth', 'Stubble',
            'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
            'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
            'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
            'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis'
        ]
    }
}


def download_file(url: str, dest_path: Path, show_progress: bool = True, max_retries: int = 3) -> bool:
    """Download a file with progress indication and retry logic."""
    for attempt in range(max_retries):
        try:
            if show_progress:
                print(f"  Downloading: {url}" + (f" (attempt {attempt + 1}/{max_retries})" if attempt > 0 else ""))
                
            def report_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, block_num * block_size * 100 // total_size)
                    sys.stdout.write(f"\r  Progress: {percent}%")
                    sys.stdout.flush()
            
            urllib.request.urlretrieve(url, str(dest_path), reporthook=report_hook if show_progress else None)
            
            if show_progress:
                print()  # New line after progress
                
            return True
        except Exception as e:
            print(f"\n  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                import time
                print(f"  Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"  Error: All {max_retries} download attempts failed")
                return False
    return False


def download_dataset(name: str, output_dir: Path, force: bool = False) -> bool:
    """
    Download a single dataset.
    
    Args:
        name: Dataset name ('indian_pines', 'pavia_university', 'salinas')
        output_dir: Directory to save the dataset
        force: If True, re-download even if files exist
    
    Returns:
        True if successful
    """
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        print(f"Available: {list(DATASETS.keys())}")
        return False
    
    info = DATASETS[name]
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"Description: {info['description']}")
    print(f"{'='*60}")
    
    # Download data file
    data_filename = info['data_url'].split('/')[-1]
    data_path = dataset_dir / data_filename
    
    if data_path.exists() and not force:
        print(f"  Data file already exists: {data_path}")
    else:
        if not download_file(info['data_url'], data_path):
            return False
    
    # Download ground truth file
    gt_filename = info['gt_url'].split('/')[-1]
    gt_path = dataset_dir / gt_filename
    
    if gt_path.exists() and not force:
        print(f"  Ground truth file already exists: {gt_path}")
    else:
        if not download_file(info['gt_url'], gt_path):
            return False
    
    # Verify and save metadata
    if SCIPY_AVAILABLE:
        print("  Verifying data...")
        try:
            data = loadmat(str(data_path))[info['data_key']]
            gt = loadmat(str(gt_path))[info['gt_key']]
            
            print(f"  ✓ Data shape: {data.shape}")
            print(f"  ✓ Ground truth shape: {gt.shape}")
            print(f"  ✓ Number of classes: {len(np.unique(gt)) - 1} (excluding background)")
            print(f"  ✓ Labeled samples: {np.sum(gt > 0)}")
            
            # Save class names
            classes_path = dataset_dir / 'class_names.txt'
            with open(classes_path, 'w') as f:
                for i, cls in enumerate(info['classes']):
                    f.write(f"{i}: {cls}\n")
            print(f"  ✓ Saved class names to {classes_path}")
            
        except Exception as e:
            print(f"  Warning: Could not verify data: {e}")
    
    print(f"  ✓ Dataset saved to: {dataset_dir}")
    return True


def download_all(output_dir: Path, datasets: list = None, force: bool = False):
    """Download all or specified datasets."""
    if datasets is None:
        datasets = list(DATASETS.keys())
    
    print("\n" + "="*60)
    print("HSI Benchmark Dataset Downloader")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Datasets to download: {datasets}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful = []
    failed = []
    
    for name in datasets:
        if download_dataset(name, output_dir, force):
            successful.append(name)
        else:
            failed.append(name)
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    print(f"Successful: {successful}")
    if failed:
        print(f"Failed: {failed}")
    
    # Usage instructions
    print("\n" + "-"*60)
    print("Usage in Python:")
    print("-"*60)
    print("""
from src.data import load_benchmark_dataset

# Load Indian Pines
cube = load_benchmark_dataset('indian_pines', 'data/external')
print(cube.shape)  # (145, 145, 200)
print(cube.n_classes)  # 16
""")
    
    return len(failed) == 0


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download HSI benchmark datasets')
    parser.add_argument('--output', '-o', type=str, default='data/external',
                        help='Output directory')
    parser.add_argument('--datasets', '-d', nargs='+', 
                        choices=list(DATASETS.keys()) + ['all'],
                        default=['all'],
                        help='Datasets to download')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force re-download even if files exist')
    
    args = parser.parse_args()
    
    # Get project root (assuming script is in src/data/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    output_dir = project_root / args.output
    
    datasets = None if 'all' in args.datasets else args.datasets
    
    success = download_all(output_dir, datasets, args.force)
    sys.exit(0 if success else 1)
