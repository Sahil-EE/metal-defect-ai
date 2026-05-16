# metal_defect_system/src/data/preprocessor.py
"""
Industrial Image Preprocessor for Metal Surface Defect Detection

Beginner Note:
  Every function here solves ONE specific real-world problem.
  Read the WHY comments — they explain the engineering decisions.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metal_defect_system.src.data.config import PATHS, PREPROCESS
from metal_defect_system.src.data.parser import DefectSample, BoundingBox


# ─────────────────────────────────────────────────────────────
# STEP 1: BILATERAL FILTER — Noise Removal
# ─────────────────────────────────────────────────────────────

def apply_bilateral_filter(image: np.ndarray) -> np.ndarray:
    """
    Removes sensor noise while preserving defect edges.

    WHY bilateral and not Gaussian blur?
    ──────────────────────────────────────────────────────────
    Gaussian blur = blurs EVERYTHING including defect edges.
    Bilateral     = blurs ONLY flat regions, keeps edges sharp.

    How it works:
    For each pixel, it averages nearby pixels BUT gives less
    weight to pixels that are very different in brightness.

    Result:
    → Smooth metal background (noise removed)
    → Sharp defect boundaries (edges preserved)

    Parameters explained:
    d=9         → look at a 9×9 pixel neighborhood
    sigmaColor  → pixels within 75 brightness units = "similar"
    sigmaSpace  → how far away pixels still influence each other

    WHY these values?
    → d=9 is strong enough to remove noise
    → sigmaColor=75 keeps defect edges (usually >75 contrast)
    → Too high = over-smoothing, defects disappear
    → Too low  = noise not removed
    """
    return cv2.bilateralFilter(
        src=image,
        d=PREPROCESS.bilateral_d,
        sigmaColor=PREPROCESS.bilateral_sigma_color,
        sigmaSpace=PREPROCESS.bilateral_sigma_space
    )


# ─────────────────────────────────────────────────────────────
# STEP 2: BACKGROUND SUBTRACTION — Fix Uneven Lighting
# ─────────────────────────────────────────────────────────────

def apply_background_subtraction(image: np.ndarray) -> np.ndarray:
    """
    Removes lighting gradients (one side bright, other side dark).

    WHY does this happen in factories?
    → Light source not perfectly centered over conveyor
    → Camera angle creates shadows on one side
    → Different batches have different ambient lighting

    How it works:
    ──────────────────────────────────────────────────────────
    1. Create a HEAVILY blurred copy of the image
       → The blur is so strong that defects disappear
       → What's left = only the background lighting gradient

                  Original:          Background estimate:
                  ████▓▒░            ████▓▓▒▒░░
                  (defect visible)   (defect gone, only gradient)

    2. Subtract background from original
       → The lighting gradient cancels out
       → Defects pop out clearly

    3. Add 128 to shift result back to valid range
       → Subtraction creates negative values
       → Adding 128 centers around middle gray

    WHY kernel size (51, 51)?
    → Must be LARGER than the biggest defect
    → If defect is 100px wide, use kernel > 100px
    → Our images are 200×200, defects up to ~70% = 140px
    → 51×51 works because NEU-DET defects are mostly < 50px wide
    """
    # Create background estimate (just the lighting, no defects)
    background = cv2.GaussianBlur(image, (51, 51), sigmaX=0)

    # Subtract lighting from image
    # cv2.subtract handles underflow (negative values → 0)
    corrected = cv2.subtract(image, background)

    # Shift to center around 128 (so details aren't lost in darkness)
    corrected = cv2.add(corrected, np.full(corrected.shape, 128, dtype=np.uint8))

    return corrected


# ─────────────────────────────────────────────────────────────
# STEP 3: CLAHE — Fix Glare and Low Contrast
# ─────────────────────────────────────────────────────────────

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    CLAHE = Contrast Limited Adaptive Histogram Equalization

    WHY do we need this?
    ──────────────────────────────────────────────────────────
    Metallic surfaces reflect light → white glare blobs
    These blobs hide defects underneath them.

    What is Histogram Equalization?
    → A histogram shows how many pixels are at each brightness
    → Equalization spreads them evenly → increases contrast
    → Dark regions become brighter, glare gets toned down

    WHY "Adaptive"?
    → Regular equalization = one setting for WHOLE image
    → But top-left might be glare, bottom-right might be dark
    → Adaptive = works on small LOCAL tiles separately
    → Each tile gets its own equalization

    WHY "Contrast Limited"?
    → If we amplify contrast too much → noise gets amplified too
    → clipLimit = max amplification allowed
    → clipLimit=2.0 = safe boost without noise amplification

    tileGridSize=(8,8):
    → Divides image into 8×8 = 64 local regions
    → Each region gets equalized separately
    → Smooth blending between regions
    """
    clahe = cv2.createCLAHE(
        clipLimit=PREPROCESS.clahe_clip_limit,
        tileGridSize=PREPROCESS.clahe_tile_grid
    )
    return clahe.apply(image)


# ─────────────────────────────────────────────────────────────
# STEP 4: SHARPENING — Recover Edge Detail
# ─────────────────────────────────────────────────────────────

def apply_sharpening(image: np.ndarray) -> np.ndarray:
    """
    Sharpens defect edges (cracks, scratches, boundaries).

    WHY do we need this?
    ──────────────────────────────────────────────────────────
    Bilateral filter + blur can soften edges slightly.
    Factory cameras may have slight motion blur (conveyor speed).
    Sharpening recovers the fine crack/scratch detail.

    How does the kernel work?
    ──────────────────────────────────────────────────────────
    kernel = [ 0, -1,  0]
             [-1,  5, -1]
             [ 0, -1,  0]

    For each pixel, we compute a weighted sum of itself
    and its 4 neighbors.

    The center pixel gets weight 5 (amplified).
    The 4 neighbors get weight -1 (subtracted).

    Example:
    If pixel = 100, neighbors = 100 (flat region):
      result = 5×100 - 4×100 = 100 (no change → flat stays flat)

    If pixel = 200, neighbors = 100 (edge!):
      result = 5×200 - 4×100 = 600 (clamped to 255 → edge enhanced)

    WHY apply AFTER denoising?
    → Sharpening amplifies EVERYTHING including noise
    → Denoise first → remove noise → THEN sharpen edges
    → Order matters!

    np.clip(result, 0, 255):
    → Sharpening can produce values outside 0-255
    → clip keeps them in valid uint8 range
    """
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# STEP 5A: BBOX CROP — Focus on Defect Region
# ─────────────────────────────────────────────────────────────

def crop_defect_region(
    image:         np.ndarray,
    bbox:          BoundingBox,
    padding_ratio: float = 0.20
) -> np.ndarray:
    """
    Crops the defect region from the full image.

    WHY crop instead of using full image?
    ──────────────────────────────────────────────────────────
    Full image = 200×200 pixels
    Defect may occupy only 13% (26×26 pixels for scratches!)

    If we feed the full image to the model:
    → 87% of image = normal metal (no useful info)
    → Model gets confused by background variations
    → Harder to learn defect patterns

    If we crop the defect region:
    → 100% of input = relevant defect area
    → Model focuses entirely on defect texture
    → Higher accuracy, especially for small defects

    WHY padding_ratio=0.20?
    ──────────────────────────────────────────────────────────
    Defect alone (no context):         Defect with padding:
    ┌──────────┐                       ┌──────────────────┐
    │ //scratch│                       │  metal  metal     │
    └──────────┘                       │  metal  metal     │
                                       │  //scratch        │
    Model can't tell:                  │  metal  metal     │
    "is this a scratch or              └──────────────────┘
    just a white line?"
                                       Now model sees CONTEXT:
                                       scratch surrounded by
                                       normal metal = defect!

    Edge clamping:
    → max(0, xmin - pad) → prevents going below image boundary
    → min(img_w, xmax + pad) → prevents going past image edge
    """
    img_h, img_w = image.shape[:2]

    # Calculate padding in pixels
    pad_x = int(bbox.width  * padding_ratio)
    pad_y = int(bbox.height * padding_ratio)

    # Expand bbox with padding, clamped to image boundaries
    x1 = max(0,     bbox.xmin - pad_x)
    y1 = max(0,     bbox.ymin - pad_y)
    x2 = min(img_w, bbox.xmax + pad_x)
    y2 = min(img_h, bbox.ymax + pad_y)

    # Crop
    crop = image[y1:y2, x1:x2]

    # Safety: if crop is empty (bad bbox), return full image
    if crop.size == 0:
        return image

    return crop


# ─────────────────────────────────────────────────────────────
# STEP 5B: RESIZE + NORMALIZE
# ─────────────────────────────────────────────────────────────

# REPLACE WITH THIS:
def resize_and_normalize(image: np.ndarray) -> np.ndarray:
    """
    Resizes to model input size.
    Returns float32 in range [0, 1].

    WHY remove ImageNet normalization from here?
    → T.ToTensor() in dataset.py clips float values to [0,1]
    → This destroyed our [-2, +2] ImageNet normalization
    → FIX: normalize AFTER ToTensor, inside dataset transforms
    → T.Normalize() in dataset.py handles ImageNet stats correctly
    """
    target_h, target_w = PREPROCESS.target_size
    resized = cv2.resize(
        image,
        (target_w, target_h),           # cv2 takes (W, H)
        interpolation=cv2.INTER_LANCZOS4
    )
    # Only divide by 255 → range [0, 1]
    # T.Normalize in dataset.py will apply ImageNet mean/std
    return resized.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────
# COMPLETE PIPELINE — Ties All Steps Together
# ─────────────────────────────────────────────────────────────

class IndustrialPreprocessor:
    """
    Complete preprocessing pipeline for one DefectSample.

    Usage:
        preprocessor = IndustrialPreprocessor()
        result = preprocessor.process(sample)
        # result.processed_image → float32 numpy array
        # result.debug_steps     → dict of intermediate images
    """

    def __init__(self, use_crop: bool = True):
        """
        Args:
            use_crop: If True, crop defect region using bbox.
                      If False, use full image.
                      use_crop=True → higher accuracy
                      use_crop=False → simpler (no bbox needed)
        """
        self.use_crop = use_crop

    def process(
        self,
        sample:     DefectSample,
        debug:      bool = False
    ) -> Optional[Dict]:
        """
        Runs full preprocessing pipeline on one sample.

        Args:
            sample: DefectSample object
            debug:  If True, saves intermediate step images

        Returns dict with:
            processed_image : float32 numpy array (H, W)
                              ready for neural network input
            original_shape  : (H, W) before resize
            class_name      : string class label
            class_idx       : integer class index
            had_bbox        : whether bounding box was used
            debug_steps     : dict of intermediate images
                              (only if debug=True)
        """
        # ── Load image ────────────────────────────────────────
        img = sample.load_image()
        if img is None:
            return None

        debug_steps = {}
        if debug:
            debug_steps["0_original"] = img.copy()

        original_shape = img.shape

        # ── Step 1: Denoise ───────────────────────────────────
        img = apply_bilateral_filter(img)
        if debug:
            debug_steps["1_bilateral"] = img.copy()

        # ── Step 2: Fix Lighting ──────────────────────────────
        img = apply_background_subtraction(img)
        if debug:
            debug_steps["2_bg_subtract"] = img.copy()

        # ── Step 3: Enhance Contrast ──────────────────────────
        img = apply_clahe(img)
        if debug:
            debug_steps["3_clahe"] = img.copy()

        # ── Step 4: Sharpen Edges ─────────────────────────────
        img = apply_sharpening(img)
        if debug:
            debug_steps["4_sharpened"] = img.copy()

        # ── Step 5a: Crop Defect Region ───────────────────────
        bbox     = sample.primary_bbox()
        had_bbox = False

        if self.use_crop and bbox is not None:
            img      = crop_defect_region(
                           img, bbox,
                           PREPROCESS.crop_padding
                       )
            had_bbox = True
            if debug:
                debug_steps["5_cropped"] = img.copy()

        # ── Step 5b: Resize + Normalize ───────────────────────
        processed = resize_and_normalize(img)
        if debug:
            debug_steps["6_normalized"] = processed.copy()

        return {
            "processed_image": processed,
            "original_shape":  original_shape,
            "class_name":      sample.class_name,
            "class_idx":       sample.class_idx,
            "had_bbox":        had_bbox,
            "debug_steps":     debug_steps,
        }

    def process_batch(
        self,
        samples:    List[DefectSample],
        debug:      bool = False,
        verbose:    bool = True
    ) -> List[Dict]:
        """
        Processes multiple samples.

        Returns list of result dicts (None results are filtered out).
        """
        results = []
        failed  = 0

        for i, sample in enumerate(samples):
            result = self.process(sample, debug=(debug and i < 6))

            if result is None:
                failed += 1
                continue

            results.append(result)

            if verbose and (i + 1) % 100 == 0:
                print(f"    Processed {i+1}/{len(samples)}...")

        if verbose:
            print(f"    ✅ {len(results)} processed, {failed} failed")

        return results


# ─────────────────────────────────────────────────────────────
# DEBUG VISUALIZER
# ─────────────────────────────────────────────────────────────

class PreprocessingVisualizer:
    """
    Generates visual reports of the preprocessing pipeline.
    Use this to VERIFY each step is working correctly.
    Always visualize before training — never trust blindly!
    """

    @staticmethod
    def show_pipeline_steps(
        sample:     DefectSample,
        save_path:  Optional[Path] = None
    ):
        """
        Shows all pipeline steps side by side for one sample.

        Layout:
        [Original] [Bilateral] [BG Sub] [CLAHE] [Sharp] [Crop] [Norm]
        """
        preprocessor = IndustrialPreprocessor(use_crop=True)
        result = preprocessor.process(sample, debug=True)

        if result is None:
            print(f"❌ Failed to process: {sample.image_path.name}")
            return

        steps = result["debug_steps"]

        n_steps = len(steps)
        fig, axes = plt.subplots(
            1, n_steps,
            figsize=(n_steps * 3.2, 4)
        )

        step_titles = {
            "0_original":   "Original\n(raw input)",
            "1_bilateral":  "Step 1\nBilateral\n(denoise)",
            "2_bg_subtract":"Step 2\nBG Subtract\n(fix lighting)",
            "3_clahe":      "Step 3\nCLAHE\n(fix glare)",
            "4_sharpened":  "Step 4\nSharpened\n(edges)",
            "5_cropped":    "Step 5a\nCropped\n(defect region)",
            "6_normalized": "Step 5b\nNormalized\n(model input)",
        }

        for ax, (step_name, step_img) in zip(axes, steps.items()):

            # Denormalize the normalized image for display
            if step_img.dtype == np.float32:
                # Reverse: (x - mean) / std → x * std + mean → * 255
                display = (
                    step_img * PREPROCESS.normalize_std +
                    PREPROCESS.normalize_mean
                ) * 255.0
                display = np.clip(display, 0, 255).astype(np.uint8)
            else:
                display = step_img

            ax.imshow(display, cmap="gray", aspect="auto")
            ax.set_title(
                step_titles.get(step_name, step_name),
                fontsize=8.5, pad=4
            )
            ax.set_xticks([])
            ax.set_yticks([])

            # Show image stats below
            ax.set_xlabel(
                f"shape: {step_img.shape}\n"
                f"min:{step_img.min():.1f} "
                f"max:{step_img.max():.1f}",
                fontsize=7, color="#555555"
            )

        fig.suptitle(
            f"Preprocessing Pipeline — {sample.class_name} "
            f"({sample.image_path.name})",
            fontsize=11, fontweight="bold"
        )

        plt.tight_layout()

        if save_path is None:
            save_path = (PATHS.RESULTS / "debug" /
                         f"pipeline_{sample.class_name}.png")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  🖼️  Pipeline saved: {save_path.name}")
        plt.close()

    @staticmethod
    def compare_all_classes(
        samples_per_class: Dict[str, DefectSample],
        save_path: Optional[Path] = None
    ):
        """
        Shows Original vs Final preprocessed image for each class.
        Grid: 6 rows (classes) × 2 columns (before/after)

        This is the most important visualization:
        → Confirms preprocessing HELPS each defect type
        → Reveals if any class needs different settings
        """
        from metal_defect_system.src.data.config import DATASET_CFG

        n_classes = len(samples_per_class)
        fig, axes = plt.subplots(
            n_classes, 2,
            figsize=(7, n_classes * 2.8)
        )

        fig.suptitle(
            "Before vs After Preprocessing\n"
            "(Left: Raw  |  Right: Preprocessed + Cropped)",
            fontsize=12, fontweight="bold"
        )

        preprocessor = IndustrialPreprocessor(use_crop=True)

        for row, (class_name, sample) in \
                enumerate(samples_per_class.items()):

            color = DATASET_CFG.CLASS_COLORS.get(class_name, "#FFFFFF")

            # ── LEFT: Original ─────────────────────────────────
            ax_orig = axes[row, 0]
            orig_img = sample.load_image()

            if orig_img is not None:
                ax_orig.imshow(orig_img, cmap="gray", aspect="auto")

                # Draw bbox on original
                bbox = sample.primary_bbox()
                if bbox:
                    import matplotlib.patches as mp
                    rect = mp.Rectangle(
                        (bbox.xmin, bbox.ymin),
                        bbox.width, bbox.height,
                        linewidth=2,
                        edgecolor=color,
                        facecolor="none"
                    )
                    ax_orig.add_patch(rect)

            ax_orig.set_title(
                f"{DATASET_CFG.CLASS_SHORT[class_name]} — Original",
                fontsize=9, color=color, fontweight="bold"
            )
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])

            # ── RIGHT: Preprocessed ────────────────────────────
            ax_proc = axes[row, 1]
            result = preprocessor.process(sample, debug=False)

            if result is not None:
                proc_img = result["processed_image"]
                # Denormalize for display
                display = (
                    proc_img * PREPROCESS.normalize_std +
                    PREPROCESS.normalize_mean
                ) * 255.0
                display = np.clip(display, 0, 255).astype(np.uint8)
                ax_proc.imshow(display, cmap="gray", aspect="auto")

            ax_proc.set_title(
                f"{DATASET_CFG.CLASS_SHORT[class_name]} — Preprocessed",
                fontsize=9, color=color, fontweight="bold"
            )
            ax_proc.set_xticks([])
            ax_proc.set_yticks([])

            # Color borders
            for ax in [ax_orig, ax_proc]:
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(1.8)

        plt.tight_layout()

        if save_path is None:
            save_path = (PATHS.RESULTS / "plots" /
                         "before_after_preprocessing.png")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  🖼️  Before/After saved: {save_path.name}")
        plt.close()