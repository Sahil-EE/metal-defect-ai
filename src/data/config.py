# src/data/config.py
"""
Central configuration file.
All settings in ONE place → easy to change, nothing hardcoded.

Rule: If you find yourself writing the same number/string twice,
it belongs in a config file.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Dict


# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
class Paths:
    """All project paths defined in one place."""
    
    ROOT         = Path("metal_defect_system")
    DATA_RAW     = ROOT / "data" / "raw" / "NEU-DET"
    DATA_PROC    = ROOT / "data" / "processed"
    DATA_AUG     = ROOT / "data" / "augmented"
    MODELS_CKPT  = ROOT / "models" / "checkpoints"
    MODELS_EXP   = ROOT / "models" / "exported"
    LOGS_TB      = ROOT / "logs" / "tensorboard"
    LOGS_EVAL    = ROOT / "logs" / "evaluation"
    RESULTS      = ROOT / "results"
    DEBUG        = ROOT / "results" / "debug"
    CONFIGS      = ROOT / "configs"


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class DatasetConfig:
    """
    Everything about your NEU-DET dataset.
    
    WHY map class names to numbers?
    → Neural networks work with numbers, not strings
    → "crazing" → 0, "inclusion" → 1, etc.
    """
    
    # Exact folder names in your NEU-DET dataset
    CLASS_NAMES: List[str] = [
        "crazing",
        "inclusion", 
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]
    
    # Number → class name (for showing predictions)
    IDX_TO_CLASS: Dict[int, str] = {
        0: "crazing",
        1: "inclusion",
        2: "patches",
        3: "pitted_surface",
        4: "rolled-in_scale",
        5: "scratches",
    }
    
    # Class name → number (for training)
    CLASS_TO_IDX: Dict[str, int] = {
        v: k for k, v in IDX_TO_CLASS.items()
    }
    
    NUM_CLASSES: int = 6
    
    # Short names for charts/tables
    CLASS_SHORT: Dict[str, str] = {
        "crazing":          "Cr",
        "inclusion":        "In",
        "patches":          "Pa",
        "pitted_surface":   "PS",
        "rolled-in_scale":  "RS",
        "scratches":        "Sc",
    }
    
    # Colors for visualization (one per class)
    CLASS_COLORS: Dict[str, str] = {
        "crazing":          "#FF6B6B",   # Red
        "inclusion":        "#4ECDC4",   # Teal
        "patches":          "#45B7D1",   # Blue
        "pitted_surface":   "#96CEB4",   # Green
        "rolled-in_scale":  "#FFEAA7",   # Yellow
        "scratches":        "#DDA0DD",   # Purple
    }
    
    # Original image size in NEU-DET
    ORIGINAL_SIZE: Tuple[int, int] = (200, 200)  # H, W
    
    # Both available splits in NEU-DET
    SPLITS = ["train", "validation"]


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
@dataclass
class PreprocessConfig:
    """
    Settings for industrial image preprocessing.
    Each parameter explained with WHY it's set to that value.
    """
    
    # Input size for the neural network
    # 224×224 = standard size for ImageNet-pretrained models
    # WHY ImageNet size? We'll use pretrained weights → need matching size
    target_size: Tuple[int, int] = (224, 224)
    
    # Bilateral Filter (noise removal)
    # d=9 : looks at 9-pixel diameter neighborhood
    # sigma_color=75 : pixels within 75 intensity units are "similar"
    # sigma_space=75 : spatial influence of nearby pixels
    # Higher sigma = stronger filtering (but can blur defects)
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    
    # CLAHE (contrast enhancement for glare/dark areas)
    # clip_limit=2.0 : caps histogram amplification (prevents noise boost)
    # tile_grid=(8,8) : divides image into 8×8 local regions
    # WHY local? Different parts of image have different brightness
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    
    # Bbox crop padding
    # 0.20 = expand the defect bounding box by 20% on each side
    # WHY? Model needs context around defect to classify correctly
    crop_padding: float = 0.20
    
    # ImageNet normalization values
    # WHY these specific numbers?
    # → Pretrained models were trained with images normalized this way
    # → Using same normalization = better transfer learning
    normalize_mean: float = 0.485
    normalize_std:  float = 0.229


# ─────────────────────────────────────────────────────────────
# TRAINING (used in Phase 3)
# ─────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """Will be fully used in Phase 3 — defined here for reference."""
    
    batch_size: int   = 32     # Images processed together
    num_epochs: int   = 50     # Full passes through dataset  
    learning_rate: float = 1e-4
    num_workers: int  = 4      # CPU threads for data loading
    device: str       = "auto" # "auto" picks GPU if available
    seed: int         = 42     # For reproducibility


# Create one instance to import everywhere
PATHS       = Paths()
DATASET_CFG = DatasetConfig()
PREPROCESS  = PreprocessConfig()
TRAINING    = TrainingConfig()
