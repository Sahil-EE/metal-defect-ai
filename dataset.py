# metal_defect_system/src/data/dataset.py
"""
PyTorch Dataset and DataLoader for NEU-DET

BEGINNER: What is a DataLoader?
────────────────────────────────────────────────────────
Training neural networks requires feeding data in BATCHES.

Without DataLoader (bad):
  for image in all_1440_images:    ← loads one by one, slow
      train(image)

With DataLoader (good):
  for batch in dataloader:         ← loads 32 images at once
      train(batch)                   ← uses multiple CPU threads
                                      ← shuffles every epoch
                                      ← handles memory efficiently

Dataset class = defines HOW to load one sample
DataLoader    = handles batching, shuffling, multi-threading
"""

import torch
import numpy as np
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metal_defect_system.src.data.config       import (
    DATASET_CFG, PREPROCESS, TRAINING, PATHS
)
from metal_defect_system.src.data.parser       import DefectSample
from metal_defect_system.src.data.preprocessor import IndustrialPreprocessor


# ─────────────────────────────────────────────────────────────
# AUGMENTATION TRANSFORMS
# ─────────────────────────────────────────────────────────────

# REPLACE WITH:
def get_train_transforms() -> T.Compose:
    """
    Training transforms WITH augmentation.

    Order matters:
    1. ToTensor first    → converts numpy [0,1] → tensor [0,1]
    2. Augmentations     → applied on [0,1] tensor (correct range)
    3. Normalize LAST    → applies ImageNet mean/std AFTER augmentation
       WHY normalize last? ColorJitter expects [0,1] range.
       Normalizing first → values go to [-2,+2] → ColorJitter breaks.
    """
    return T.Compose([
        # Step 1: numpy float32 [0,1] → torch tensor [0,1]
        T.ToTensor(),

        # Step 2: Augmentations (all expect [0,1] tensor)
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.RandomErasing(
            p=0.1,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value=0
        ),

        # Step 3: ImageNet normalization LAST
        # mean=[0.485] std=[0.229] for grayscale
        # Result: values now in range ~[-2.1, +2.1]
        T.Normalize(mean=[0.485], std=[0.229]),
    ])


def get_val_transforms() -> T.Compose:
    """
    Validation transforms — NO augmentation.
    Same normalization as train (MUST match or model gets confused).
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229]),
    ])

def get_val_transforms() -> T.Compose:
    """
    Transforms for VALIDATION and TEST — NO augmentation.
    Only the essential tensor conversion.
    """
    return T.Compose([
        T.ToTensor(),
    ])


# ─────────────────────────────────────────────────────────────
# PYTORCH DATASET CLASS
# ─────────────────────────────────────────────────────────────

class DefectDataset(Dataset):
    """
    PyTorch Dataset for NEU-DET defect samples.

    BEGINNER: A PyTorch Dataset must implement 3 methods:
      __init__   → setup/store the data list
      __len__    → return how many samples we have
      __getitem__ → return ONE sample given an index

    The DataLoader calls __getitem__ repeatedly and
    collects the results into batches automatically.

    How DataLoader uses Dataset:
      dataset[0]    → first sample
      dataset[100]  → 101st sample
      dataset[1439] → last training sample

    DataLoader(dataset, batch_size=32):
      → calls __getitem__ 32 times in parallel
      → stacks results into one batch tensor
    """

    def __init__(
        self,
        samples:     List[DefectSample],
        split:       str = "train",      # "train" or "val"
        use_crop:    bool = True,
        transforms:  Optional[T.Compose] = None,
    ):
        """
        Args:
            samples   : List of DefectSample objects
            split     : "train" (with augmentation) or "val" (without)
            use_crop  : Use bounding box crop mode
            transforms: Optional custom transforms (auto-set if None)
        """
        self.samples      = samples
        self.split        = split
        self.preprocessor = IndustrialPreprocessor(use_crop=use_crop)

        # Set transforms based on split
        if transforms is not None:
            self.transforms = transforms
        elif split == "train":
            self.transforms = get_train_transforms()
        else:
            self.transforms = get_val_transforms()

        print(f"  📦 {split} dataset: {len(samples)} samples, "
              f"augmentation={'ON' if split=='train' else 'OFF'}")

    def __len__(self) -> int:
        """Returns total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns ONE preprocessed sample as (image_tensor, label).

        Args:
            idx: Index of the sample (0 to len-1)

        Returns:
            image  : torch.Tensor of shape (1, 224, 224) float32
            label  : int class index (0-5)

        WHAT HAPPENS HERE:
        1. Get sample at index `idx`
        2. Run preprocessing pipeline (bilateral, CLAHE, crop, etc.)
        3. Convert numpy array → torch tensor
        4. Apply augmentation (train only)
        5. Add channel dimension: (224,224) → (1, 224, 224)
        6. Return (image_tensor, class_index)
        """
        sample = self.samples[idx]

        # ── Preprocess ────────────────────────────────────────
        result = self.preprocessor.process(sample, debug=False)

        if result is None:
            # If preprocessing fails, return next sample
            # (rare edge case — corrupt image)
            return self.__getitem__((idx + 1) % len(self.samples))

        img = result["processed_image"]  # numpy (224, 224) float32

        # ── Apply Augmentation Transforms ─────────────────────
        # T.ToTensor() expects:
        #   HxW (grayscale) or HxWxC (color) numpy array
        # Outputs:
        #   CxHxW torch tensor
        # So (224,224) → (1, 224, 224) automatically

        img_tensor = self.transforms(img)

        # ── Get Label ─────────────────────────────────────────
        label = sample.class_idx  # integer 0-5

        return img_tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculates class weights for weighted loss function.

        WHY? If classes are imbalanced (e.g., 10× more scratches than crazing),
        the model will just predict "scratches" for everything to get
        high accuracy. Class weights force it to learn all classes equally.

        Formula: weight[class] = total_samples / (n_classes × count[class])
        → Rare classes get HIGH weight (model penalized more for missing them)
        → Common classes get LOW weight (mistakes less penalized)

        For NEU-DET our classes ARE balanced, but good practice to include.
        """
        counts = torch.zeros(DATASET_CFG.NUM_CLASSES)
        for sample in self.samples:
            counts[sample.class_idx] += 1

        total   = counts.sum()
        weights = total / (DATASET_CFG.NUM_CLASSES * counts)

        # Replace inf/nan (for missing classes)
        weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0)

        return weights


# ─────────────────────────────────────────────────────────────
# DATALOADER BUILDER
# ─────────────────────────────────────────────────────────────

def build_dataloaders(
    train_samples:  List[DefectSample],
    val_samples:    List[DefectSample],
    batch_size:     int = 32,
    num_workers:    int = 0,        # 0 = main thread (safe on Windows)
    use_crop:       bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation DataLoaders.

    BEGINNER: What do DataLoader parameters mean?
    ──────────────────────────────────────────────────────────
    batch_size=32:
      → Process 32 images at once
      → Larger = faster training but more GPU/RAM needed
      → 32 is safe for 4GB+ RAM

    shuffle=True (train only):
      → Randomize sample order every epoch
      → WHY? Prevents model from memorizing order
      → If always [crazing, inclusion, patches, ...] repeating,
        model learns the pattern, not the content

    num_workers=0:
      → 0 = load data in main process (slower but safe on Windows)
      → WHY 0 for Windows? Python multiprocessing has issues on Windows
      → Mac/Linux can use num_workers=4 for 4× faster loading

    pin_memory=True:
      → Pre-loads batch into GPU-accessible memory
      → Speeds up CPU→GPU data transfer
      → Only beneficial if using GPU
    """
    train_dataset = DefectDataset(
        samples=train_samples,
        split="train",
        use_crop=use_crop,
    )

    val_dataset = DefectDataset(
        samples=val_samples,
        split="val",
        use_crop=use_crop,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Randomize every epoch
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,         # Drop incomplete last batch
        # WHY drop_last? BatchNorm fails with batch_size=1
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # Keep validation order fixed
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,        # Keep all validation samples
    )

    return train_loader, val_loader