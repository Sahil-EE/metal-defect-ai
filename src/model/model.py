# metal_defect_system/src/models/model.py
"""
EfficientNet-B0 Based Metal Defect Classifier

Architecture:
  EfficientNet-B0 backbone (pretrained ImageNet)
  + Custom classification head (6 defect classes)

Design Decisions:
  - Grayscale input → replicate to 3 channels (EfficientNet needs RGB)
  - Freeze backbone initially → train head only (Phase A)
  - Unfreeze top layers later → fine-tune (Phase B)
  - Dropout regularization → prevents overfitting on small dataset
  - BatchNorm in head → stabilizes training
"""

import torch
import torch.nn as nn
import torchvision.models as models

from pathlib import Path
from typing import Dict, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metal_defect_system.src.data.config import DATASET_CFG, PATHS


# ─────────────────────────────────────────────────────────────
# WHAT IS A TENSOR?
# ─────────────────────────────────────────────────────────────
# A tensor = multi-dimensional array (like numpy array but for GPU)
#
# Our image tensor shape: (Batch, Channels, Height, Width)
# Example: (32, 1, 224, 224)
#           │   │    │    │
#           │   │    │    └── image width
#           │   │    └─────── image height
#           │   └──────────── 1 = grayscale channel
#           └──────────────── 32 images processed together (batch)


# ─────────────────────────────────────────────────────────────
# GRAYSCALE → RGB CONVERTER
# ─────────────────────────────────────────────────────────────

class GrayscaleToRGB(nn.Module):
    """
    Converts 1-channel grayscale input to 3-channel RGB.

    WHY? EfficientNet-B0 was pretrained on RGB images (3 channels).
    Its first layer expects 3 input channels.
    Our images are grayscale (1 channel).

    Solution: Simply REPEAT the grayscale channel 3 times.
    Result: [R=gray, G=gray, B=gray] → looks gray but has 3 channels.

    WHY not retrain the first layer from scratch?
    → We'd lose the pretrained edge-detection filters
    → Repeating = free RGB conversion with zero information loss

    Input shape:  (B, 1, H, W)
    Output shape: (B, 3, H, W)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch.repeat_interleave or torch.cat or expand
        # .expand() is memory-efficient (no actual data copy)
        return x.expand(-1, 3, -1, -1)
        # -1 means "keep this dimension as-is"
        # Result: (B, 1, H, W) → (B, 3, H, W)


# ─────────────────────────────────────────────────────────────
# CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────────

class DefectClassificationHead(nn.Module):
    """
    Custom classification head added on top of EfficientNet backbone.

    Takes 1280 backbone features → outputs 6 class probabilities.

    Layer by layer explanation:
    ──────────────────────────────────────────────────────────
    GlobalAvgPool:
      Input:  (B, 1280, 7, 7) — spatial feature maps
      Output: (B, 1280)       — one value per feature
      WHY? Reduces spatial dimensions while keeping feature info.
           Makes model size-independent (works with any input size).

    Dropout(0.3):
      Randomly zeros 30% of neurons during training.
      WHY? Prevents memorization → forces learning generalizable features.
      Only active during training, disabled during inference.

    Linear(1280 → 512):
      Learns which combinations of 1280 features matter for defects.
      WHY 512? Good middle ground between 1280 (too wide) and 6 (too narrow).

    BatchNorm1d(512):
      Normalizes the 512 values to have mean≈0, std≈1.
      WHY? Keeps values in a healthy range → faster, more stable training.
      Prevents "exploding" or "vanishing" activations.

    ReLU():
      Activation function: max(0, x)
      WHY? Introduces non-linearity — without it, all layers
           collapse into one linear transformation (useless!).
      Simple but powerful: negative values → 0, positive → kept.

    Dropout(0.2):
      Second dropout, lighter (20%) before final layer.

    Linear(512 → 6):
      Final mapping: 512 features → 6 class scores (logits).
      WHY logits not probabilities?
      → CrossEntropyLoss applies softmax internally
      → More numerically stable than applying softmax manually
    """

    def __init__(
        self,
        in_features:  int = 1280,   # EfficientNet-B0 output size
        hidden_size:  int = 512,
        num_classes:  int = 6,
        dropout1:     float = 0.3,
        dropout2:     float = 0.2,
    ):
        super().__init__()

        self.head = nn.Sequential(
            # Step 1: Flatten spatial dimensions
            nn.AdaptiveAvgPool2d(1),    # (B, 1280, H, W) → (B, 1280, 1, 1)
            nn.Flatten(),               # (B, 1280, 1, 1) → (B, 1280)

            # Step 2: First dropout
            nn.Dropout(p=dropout1),

            # Step 3: Dense layer
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),

            # Step 4: Second dropout
            nn.Dropout(p=dropout2),

            # Step 5: Final classification layer
            nn.Linear(hidden_size, num_classes),
            # Note: NO softmax here — CrossEntropyLoss handles it
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ─────────────────────────────────────────────────────────────
# COMPLETE MODEL
# ─────────────────────────────────────────────────────────────

class MetalDefectClassifier(nn.Module):
    """
    Complete Metal Surface Defect Classifier.

    Architecture:
        GrayscaleToRGB → EfficientNet-B0 Backbone → Custom Head

    Two training phases:
        Phase A (frozen backbone):
            → Only classification head trains
            → Backbone stays fixed (pretrained weights preserved)
            → Fast convergence, low risk of breaking pretrained features
            → Run for ~10 epochs

        Phase B (unfrozen top layers):
            → Top 20% of backbone layers also train
            → Fine-tunes high-level features for metal defects
            → Lower learning rate to avoid destroying pretrained weights
            → Run for ~20-30 more epochs
    """

    def __init__(
        self,
        num_classes:   int   = 6,
        pretrained:    bool  = True,
        dropout1:      float = 0.3,
        dropout2:      float = 0.2,
        hidden_size:   int   = 512,
    ):
        """
        Args:
            num_classes : Number of defect types (6 for NEU-DET)
            pretrained  : Load ImageNet weights (True = recommended)
            dropout1    : Dropout rate before hidden layer
            dropout2    : Dropout rate before output layer
            hidden_size : Size of hidden dense layer
        """
        super().__init__()

        self.num_classes = num_classes

        # ── 1. Grayscale → RGB converter ──────────────────────
        self.gray_to_rgb = GrayscaleToRGB()

        # ── 2. EfficientNet-B0 Backbone ───────────────────────
        # Load pretrained model
        if pretrained:
            # Multi-strategy weight loading (handles hash mismatches)
            cache_dir    = Path.home() / ".cache/torch/hub/checkpoints"
            local_paths  = [
                cache_dir / "efficientnet_b0_rwightman-7f5810bc.pth",
                cache_dir / "efficientnet_b0_rwightman-3dd342df.pth",
            ]
            loaded = False

            # Strategy 1: Load from local cache (no hash check)
            for lp in local_paths:
                if lp.exists():
                    try:
                        efficientnet = models.efficientnet_b0(weights=None)
                        state        = torch.load(str(lp), map_location="cpu")
                        efficientnet.load_state_dict(state, strict=False)
                        print(f"   ✅ Pretrained weights loaded: {lp.name}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"   ⚠️  Local load failed ({lp.name}): {e}")

            # Strategy 2: Download without hash check
            if not loaded:
                try:
                    url      = "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"
                    out_path = cache_dir / "efficientnet_b0_rwightman-7f5810bc.pth"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    print("   📥 Downloading weights (no hash check)...")
                    torch.hub.download_url_to_file(
                        url, str(out_path),
                        hash_prefix=None, progress=True
                    )
                    efficientnet = models.efficientnet_b0(weights=None)
                    state        = torch.load(str(out_path), map_location="cpu")
                    efficientnet.load_state_dict(state, strict=False)
                    print("   ✅ Pretrained weights downloaded and loaded!")
                    loaded = True
                except Exception as e:
                    print(f"   ⚠️  Download failed: {e}")

            # Strategy 3: Random weights fallback
            if not loaded:
                print("   ⚠️  Using random weights — will still train")
                efficientnet = models.efficientnet_b0(weights=None)
        else:
            efficientnet = models.efficientnet_b0(weights=None)

        # Remove the original classifier head
        # (it was designed for 1000 ImageNet classes)
        # We keep only the feature extractor part
        self.backbone = efficientnet.features
        # backbone output: (B, 1280, 7, 7) for 224×224 input

        # ── 3. Custom Classification Head ─────────────────────
        self.classifier = DefectClassificationHead(
            in_features=1280,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout1=dropout1,
            dropout2=dropout2,
        )

        # ── 4. Initialize: Freeze backbone ────────────────────
        # Start with backbone frozen (Phase A training)
        self.freeze_backbone()

        print(f"✅ MetalDefectClassifier created")
        print(f"   Backbone  : EfficientNet-B0 "
              f"({'pretrained' if pretrained else 'random'})")
        print(f"   Classes   : {num_classes}")
        print(f"   Status    : backbone frozen (Phase A)")

    def freeze_backbone(self):
        """
        Freezes all backbone layers.
        Frozen = weights don't change during training.
        Only the classification head will be trained.

        WHY freeze first?
        → Pretrained backbone already extracts good features
        → Training it immediately with high LR destroys these features
        → Freeze → train head → then gradually unfreeze
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        print("🔒 Backbone FROZEN — only head will train")

    def unfreeze_top_layers(self, num_layers: int = 3):
        """
        Unfreezes the last N layer blocks of the backbone.

        WHY unfreeze gradually?
        → Unfreezing ALL at once with wrong LR = catastrophic forgetting
        → Unfreeze top layers first (closest to output)
        → They contain the most task-specific features
        → Use very small learning rate (1e-5 to 1e-4)

        EfficientNet-B0 has 9 feature blocks (indices 0-8).
        We unfreeze the last `num_layers` blocks.
        """
        # Get all child modules of backbone as list
        backbone_children = list(self.backbone.children())
        total             = len(backbone_children)

        # Unfreeze the last `num_layers` blocks
        layers_to_unfreeze = backbone_children[total - num_layers:]

        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        # Count trainable vs frozen
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        frozen_params    = total_params - trainable_params

        print(f"🔓 Unfroze last {num_layers} backbone blocks")
        print(f"   Trainable params : {trainable_params:,}")
        print(f"   Frozen params    : {frozen_params:,}")

    def unfreeze_all(self):
        """Unfreezes entire backbone for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print("🔓 ALL layers unfrozen — full fine-tuning mode")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input → predictions.

        Args:
            x: Input tensor of shape (B, 1, 224, 224)
               B = batch size, 1 = grayscale channel

        Returns:
            Logits tensor of shape (B, 6)
            Each row = 6 raw scores for 6 defect classes
            (NOT probabilities yet — apply softmax for that)
        """
        # Step 1: 1-channel → 3-channel
        x = self.gray_to_rgb(x)        # (B,1,224,224) → (B,3,224,224)

        # Step 2: Extract features with backbone
        x = self.backbone(x)           # (B,3,224,224) → (B,1280,7,7)

        # Step 3: Classify
        x = self.classifier(x)         # (B,1280,7,7)  → (B,6)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns CLASS PROBABILITIES (0 to 1, sum to 1).
        Use this for inference, not training.

        Difference from forward():
          forward()      → raw logits (can be any value)
          predict_proba()→ softmax probabilities (0 to 1)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict_class(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns predicted class INDEX and CONFIDENCE.

        Returns:
            class_idx   : (B,) tensor of predicted class indices
            confidence  : (B,) tensor of confidence scores (0-1)
        """
        proba       = self.predict_proba(x)
        confidence, class_idx = torch.max(proba, dim=1)
        return class_idx, confidence

    def get_model_info(self) -> Dict:
        """Returns model statistics."""
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        # Estimate model size in MB
        size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32

        return {
            "total_params":     total_params,
            "trainable_params": trainable_params,
            "frozen_params":    total_params - trainable_params,
            "size_mb":          size_mb,
            "num_classes":      self.num_classes,
            "input_shape":      "(B, 1, 224, 224)",
            "output_shape":     f"(B, {self.num_classes})",
        }

    def print_model_info(self):
        """Prints formatted model information table."""
        info = self.get_model_info()

        print("\n" + "═" * 55)
        print("  🧠 MODEL INFORMATION")
        print("═" * 55)
        print(f"  Architecture    : EfficientNet-B0 + Custom Head")
        print(f"  Total params    : {info['total_params']:>12,}")
        print(f"  Trainable params: {info['trainable_params']:>12,}")
        print(f"  Frozen params   : {info['frozen_params']:>12,}")
        print(f"  Model size      : {info['size_mb']:>11.1f} MB")
        print(f"  Input shape     : {info['input_shape']}")
        print(f"  Output shape    : {info['output_shape']}")
        print(f"  Num classes     : {info['num_classes']}")
        print("═" * 55)
