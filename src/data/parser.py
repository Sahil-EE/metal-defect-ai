# metal_defect_system/src/data/parser.py
"""
NEU-DET Dataset Parser — FIXED for Flat Annotation Structure

YOUR dataset structure:
  train/
    images/
      crazing/        ← class subfolders here
      inclusion/
      ...
    annotations/
      crazing_1.xml   ← ALL xml files flat here (no subfolders)
      inclusion_1.xml
      ...

  validation/
    images/
      crazing/
      ...
    annotations/
      crazing_240.xml ← flat here too
      ...
"""

import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metal_defect_system.src.data.config import DATASET_CFG, PATHS


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """
    One bounding box annotation.

    Coordinate system (pixels):
    (0,0)──────────────► X
      │
      │  (xmin,ymin)─────────┐
      │  │                   │
      │  │   DEFECT REGION   │
      │  │                   │
      │  └─────────(xmax,ymax)
      ▼
      Y
    """
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.xmin + self.xmax) // 2,
                (self.ymin + self.ymax) // 2)

    def is_valid(self, img_w: int, img_h: int) -> bool:
        """
        Returns True if bbox makes geometric sense.
        Catches corrupt annotations before they cause training errors.
        """
        return (
            0 <= self.xmin < self.xmax <= img_w and
            0 <= self.ymin < self.ymax <= img_h and
            self.area > 0
        )

    def __repr__(self):
        return (f"BBox(x:{self.xmin}→{self.xmax}, "
                f"y:{self.ymin}→{self.ymax}, "
                f"size:{self.width}×{self.height})")


@dataclass
class DefectSample:
    """
    One complete sample = one image + its XML annotation.
    This is the core data unit used everywhere in the system.
    """
    image_path:   Path
    xml_path:     Path
    class_name:   str
    class_idx:    int
    bboxes:       List[BoundingBox] = field(default_factory=list)
    image_width:  int = 200
    image_height: int = 200
    split:        str = "train"

    def load_image(self) -> Optional[np.ndarray]:
        """
        Loads pixel data as grayscale numpy array.
        Shape: (H, W)  dtype: uint8  values: 0-255

        WHY grayscale?
        → Metal defects are texture-based, not color-based
        → 1 channel = 3× faster than RGB on mobile
        """
        img = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ❌ Cannot load: {self.image_path.name}")
        return img

    def primary_bbox(self) -> Optional[BoundingBox]:
        """Returns the largest bounding box (main defect)."""
        if not self.bboxes:
            return None
        return max(self.bboxes, key=lambda b: b.area)

    def __repr__(self):
        return (f"DefectSample("
                f"class='{self.class_name}', "
                f"bboxes={len(self.bboxes)}, "
                f"split='{self.split}')")


# ─────────────────────────────────────────────────────────────
# XML PARSER
# ─────────────────────────────────────────────────────────────

class XMLParser:
    """
    Parses Pascal VOC XML annotation files.

    Pascal VOC format:
    <annotation>
      <filename>crazing_1.jpg</filename>
      <size><width>200</width><height>200</height></size>
      <object>
        <name>crazing</name>
        <bndbox>
          <xmin>45</xmin><ymin>30</ymin>
          <xmax>160</xmax><ymax>170</ymax>
        </bndbox>
      </object>
    </annotation>
    """

    @staticmethod
    def parse(xml_path: Path) -> dict:
        """
        Reads one XML file and returns structured data.

        Returns:
            dict with keys: filename, width, height, objects
            dict with key:  error  (if something went wrong)
        """
        if not xml_path.exists():
            return {"error": f"XML not found: {xml_path.name}"}

        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
        except ET.ParseError as e:
            return {"error": f"Corrupt XML {xml_path.name}: {e}"}

        result = {
            "filename": root.findtext("filename", default="unknown"),
            "width":    int(root.findtext("size/width",  default="200")),
            "height":   int(root.findtext("size/height", default="200")),
            "objects":  []
        }

        for obj_tag in root.findall("object"):
            name     = obj_tag.findtext("name", default="unknown")
            name     = name.strip().lower()
            bbox_tag = obj_tag.find("bndbox")

            if bbox_tag is None:
                continue

            try:
                bbox = BoundingBox(
                    xmin=int(float(bbox_tag.findtext("xmin", "0"))),
                    ymin=int(float(bbox_tag.findtext("ymin", "0"))),
                    xmax=int(float(bbox_tag.findtext("xmax", "0"))),
                    ymax=int(float(bbox_tag.findtext("ymax", "0"))),
                )
            except (ValueError, TypeError) as e:
                print(f"  ⚠️  Bad coordinate in {xml_path.name}: {e}")
                continue

            result["objects"].append({
                "class_name": name,
                "bbox":       bbox,
                "difficult":  int(obj_tag.findtext("difficult", "0")),
            })

        return result


# ─────────────────────────────────────────────────────────────
# ANNOTATION INDEX BUILDER
# ─────────────────────────────────────────────────────────────

class AnnotationIndex:
    """
    Builds a fast lookup dictionary for flat annotation folders.

    YOUR structure:
      annotations/
        crazing_1.xml      ← flat, no subfolders
        inclusion_5.xml
        scratches_200.xml
        ...

    This class reads ALL xml filenames once and builds:
      index = {
        "crazing_1":   Path("annotations/crazing_1.xml"),
        "inclusion_5": Path("annotations/inclusion_5.xml"),
        ...
      }

    Then given image "crazing_1.jpg" → stem = "crazing_1"
    → look up index["crazing_1"] → found immediately.

    WHY build an index?
    → Without it, for every image we'd scan 1439 files = very slow
    → With index, lookup = instant (dictionary = O(1) speed)
    """

    def __init__(self, annotations_dir: Path):
        self.anno_dir = annotations_dir
        self.index: Dict[str, Path] = {}
        self._build()

    def _build(self):
        """Scans annotations folder and builds stem → path mapping."""
        if not self.anno_dir.exists():
            print(f"  ❌ Annotations folder missing: {self.anno_dir}")
            return

        xml_files = list(self.anno_dir.glob("*.xml"))

        if not xml_files:
            print(f"  ❌ No XML files found in: {self.anno_dir}")
            return

        for xml_file in xml_files:
            # stem = filename without extension
            # "crazing_1.xml" → stem = "crazing_1"
            self.index[xml_file.stem] = xml_file

        print(f"  📑 Annotation index built: "
              f"{len(self.index)} XMLs in {self.anno_dir.name}/")

    def find(self, image_stem: str) -> Optional[Path]:
        """
        Finds XML for a given image stem.

        Example:
            image file: "crazing_1.jpg"
            image stem: "crazing_1"
            returns:    Path("annotations/crazing_1.xml")
            or None if not found
        """
        return self.index.get(image_stem, None)

    def __len__(self):
        return len(self.index)


# ─────────────────────────────────────────────────────────────
# MAIN DATASET PARSER
# ─────────────────────────────────────────────────────────────

class NEUDETParser:
    """
    Main class that reads the entire NEU-DET dataset.

    Handles YOUR exact structure:
      images/   → class subfolders → .jpg files
      annotations/ → ALL .xml files flat (no subfolders)

    Usage:
        parser  = NEUDETParser(dataset_root=PATHS.DATA_RAW)
        samples = parser.parse_all()
        train   = parser.get_split("train")
        val     = parser.get_split("validation")
    """

    def __init__(self, dataset_root: Path):
        self.root       = Path(dataset_root)
        self.xml_parser = XMLParser()
        self.samples:   List[DefectSample] = []
        self.stats      = defaultdict(lambda: defaultdict(int))
        self._validate_structure()

    def _validate_structure(self):
        """Checks expected top-level folders exist."""
        print("\n🔍 Validating dataset structure...")

        required = [
            self.root / "train"      / "images",
            self.root / "train"      / "annotations",
            self.root / "validation" / "images",
            self.root / "validation" / "annotations",
        ]

        all_ok = True
        for path in required:
            if path.exists():
                print(f"  ✅ {path.relative_to(self.root)}")
            else:
                print(f"  ❌ MISSING: {path.relative_to(self.root)}")
                all_ok = False

        if not all_ok:
            raise FileNotFoundError(
                f"\n❌ Dataset structure wrong at: {self.root.absolute()}"
            )

        print(f"\n✅ Dataset structure OK at:\n   {self.root.absolute()}\n")

    def _normalize_class_name(self, folder_name: str) -> Optional[str]:
        """
        Converts folder name variations to our standard class names.
        Handles capitalization, underscores, dashes, abbreviations.
        """
        name = folder_name.strip().lower()

        if name in DATASET_CFG.CLASS_TO_IDX:
            return name

        aliases = {
            "rolled_in_scale":   "rolled-in_scale",
            "rolled in scale":   "rolled-in_scale",
            "rolledinscale":     "rolled-in_scale",
            "rolled-in scale":   "rolled-in_scale",
            "pitted":            "pitted_surface",
            "patch":             "patches",
            "crack":             "crazing",
            "craze":             "crazing",
            "cracking":          "crazing",
            "scratch":           "scratches",
        }

        if name in aliases:
            mapped = aliases[name]
            print(f"  ℹ️  '{folder_name}' → '{mapped}'")
            return mapped

        print(f"  ⚠️  Unknown class folder: '{folder_name}' — skipping")
        return None

    def parse_all(self) -> List[DefectSample]:
        """
        Parses entire dataset.

        KEY CHANGE from old parser:
        → Builds AnnotationIndex per split (flat XML lookup)
        → No longer looks for class subfolders in annotations/
        → Matches image stem → xml stem directly

        Example match:
          images/crazing/crazing_1.jpg
          stem = "crazing_1"
          → index["crazing_1"] = annotations/crazing_1.xml ✅
        """
        self.samples   = []
        total_skipped  = 0
        total_no_xml   = 0

        for split in ["train", "validation"]:
            images_dir = self.root / split / "images"
            annots_dir = self.root / split / "annotations"

            print(f"\n📂 Parsing '{split}' split...")
            print("  " + "─" * 45)

            # ── BUILD ANNOTATION INDEX (flat folder lookup) ────
            anno_index = AnnotationIndex(annots_dir)

            if len(anno_index) == 0:
                print(f"  ❌ No annotations found for split: {split}")
                continue

            # ── LOOP THROUGH CLASS SUBFOLDERS IN images/ ───────
            class_folders = sorted([
                f for f in images_dir.iterdir() if f.is_dir()
            ])

            if not class_folders:
                print(f"  ❌ No class folders found in: {images_dir}")
                continue

            for class_folder in class_folders:
                class_name = self._normalize_class_name(class_folder.name)
                if class_name is None:
                    continue

                class_idx = DATASET_CFG.CLASS_TO_IDX[class_name]

                # Find all images in this class folder
                image_files = sorted(
                    list(class_folder.glob("*.jpg")) +
                    list(class_folder.glob("*.jpeg")) +
                    list(class_folder.glob("*.png")) +
                    list(class_folder.glob("*.bmp"))
                )

                if not image_files:
                    print(f"  ⚠️  No images in: {class_folder.name}/")
                    continue

                loaded  = 0
                skipped = 0
                no_xml  = 0

                for img_path in image_files:

                    # ── FLAT XML LOOKUP ────────────────────────
                    # img_path.stem = "crazing_1" (no extension)
                    xml_path = anno_index.find(img_path.stem)

                    if xml_path is None:
                        # XML missing — use image-only sample
                        # (still useful for classification training)
                        no_xml += 1
                        sample = DefectSample(
                            image_path   = img_path,
                            xml_path     = Path(""),   # no xml
                            class_name   = class_name,
                            class_idx    = class_idx,
                            bboxes       = [],          # no bbox
                            image_width  = 200,
                            image_height = 200,
                            split        = split,
                        )
                        self.samples.append(sample)
                        self.stats[split][class_name] += 1
                        loaded += 1
                        continue

                    # ── PARSE XML ──────────────────────────────
                    anno = self.xml_parser.parse(xml_path)

                    if "error" in anno:
                        skipped += 1
                        total_skipped += 1
                        continue

                    # ── VALIDATE BBOXES ────────────────────────
                    valid_bboxes = []
                    for obj in anno["objects"]:
                        bbox   = obj["bbox"]
                        img_w  = anno["width"]
                        img_h  = anno["height"]

                        if bbox.is_valid(img_w, img_h):
                            valid_bboxes.append(bbox)
                        else:
                            print(f"    ⚠️  Invalid bbox in "
                                  f"{img_path.name}: {bbox}")

                    # ── CREATE SAMPLE ──────────────────────────
                    sample = DefectSample(
                        image_path   = img_path,
                        xml_path     = xml_path,
                        class_name   = class_name,
                        class_idx    = class_idx,
                        bboxes       = valid_bboxes,
                        image_width  = anno["width"],
                        image_height = anno["height"],
                        split        = split,
                    )

                    self.samples.append(sample)
                    self.stats[split][class_name] += 1
                    loaded += 1

                # ── PER-CLASS REPORT ───────────────────────────
                status = "✅" if skipped == 0 else "⚠️ "
                no_xml_note = f"  ({no_xml} without XML)" if no_xml else ""
                print(f"  {status} {class_name:<22} "
                      f"{loaded:>4} loaded, "
                      f"{skipped:>2} skipped"
                      f"{no_xml_note}")

                total_no_xml  += no_xml
                total_skipped += skipped

        # ── FINAL SUMMARY ──────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"  ✅ Total loaded   : {len(self.samples)}")
        print(f"  ⚠️  Total skipped  : {total_skipped}")
        if total_no_xml:
            print(f"  ℹ️  Without XML    : {total_no_xml} "
                  f"(loaded as class-only samples)")

        return self.samples

    def get_split(self, split: str) -> List[DefectSample]:
        """Returns samples for one split: 'train' or 'validation'."""
        return [s for s in self.samples if s.split == split]

    def print_full_statistics(self):
        """Prints formatted statistics table."""
        print("\n" + "═" * 65)
        print("  📊 NEU-DET DATASET STATISTICS")
        print("═" * 65)

        print(f"\n  {'Class':<22} {'Short':>5} "
              f"{'Train':>8} {'Val':>8} {'Total':>8}  Bar")
        print("  " + "─" * 60)

        total_train = total_val = 0

        for class_name in DATASET_CFG.CLASS_NAMES:
            short   = DATASET_CFG.CLASS_SHORT[class_name]
            train_n = self.stats["train"].get(class_name, 0)
            val_n   = self.stats["validation"].get(class_name, 0)
            total_n = train_n + val_n

            total_train += train_n
            total_val   += val_n

            bar = "█" * (train_n // 30)
            print(f"  {class_name:<22} {short:>5} "
                  f"{train_n:>8} {val_n:>8} {total_n:>8}  {bar}")

        print("  " + "─" * 60)
        print(f"  {'TOTAL':<22} {'':>5} "
              f"{total_train:>8} {total_val:>8} "
              f"{total_train + total_val:>8}")
        print("═" * 65)

        # Imbalance check
        train_counts = [
            self.stats["train"].get(c, 0)
            for c in DATASET_CFG.CLASS_NAMES
        ]
        if max(train_counts, default=0) > 0:
            ratio = max(train_counts) / (min(train_counts) + 1e-8)
            if ratio > 2.0:
                print(f"\n  ⚠️  IMBALANCE RATIO: {ratio:.1f}x "
                      f"→ weighted loss needed in Phase 3")
            else:
                print(f"\n  ✅ Classes balanced (ratio: {ratio:.1f}x)")

        # BBox coverage
        print("\n  📐 BOUNDING BOX COVERAGE PER CLASS")
        print(f"\n  {'Class':<22} {'Avg Coverage':>14}  Scale")
        print("  " + "─" * 48)

        for class_name in DATASET_CFG.CLASS_NAMES:
            samples_with_bbox = [
                s for s in self.samples
                if s.class_name == class_name and s.primary_bbox()
            ]
            if not samples_with_bbox:
                print(f"  {class_name:<22} {'no bbox data':>14}")
                continue

            coverages = [
                s.primary_bbox().area / (s.image_width * s.image_height) * 100
                for s in samples_with_bbox
            ]
            avg = np.mean(coverages)
            scale = ("🔴 Small" if avg < 20
                     else "🟡 Medium" if avg < 50
                     else "🟢 Large")
            print(f"  {class_name:<22} {avg:>13.1f}%  {scale}")

        print()

    def visualize_all_classes(
        self,
        n_per_class: int = 3,
        show_bbox: bool = True,
        save_path: Optional[Path] = None
    ):
        """
        Creates a grid: rows = defect classes, columns = sample images.
        Draws bounding boxes in class-specific colors.
        """
        n_classes = len(DATASET_CFG.CLASS_NAMES)
        fig, axes = plt.subplots(
            n_classes, n_per_class,
            figsize=(n_per_class * 3.5, n_classes * 3.5)
        )
        fig.suptitle(
            "NEU-DET — Sample Images with Bounding Box Annotations",
            fontsize=13, fontweight="bold", y=1.01
        )

        for row, class_name in enumerate(DATASET_CFG.CLASS_NAMES):
            train_samples = [
                s for s in self.samples
                if s.class_name == class_name and s.split == "train"
            ]
            short = DATASET_CFG.CLASS_SHORT[class_name]
            color = DATASET_CFG.CLASS_COLORS[class_name]

            for col in range(n_per_class):
                ax = axes[row, col]
                ax.set_xticks([])
                ax.set_yticks([])

                # Class label on left column
                if col == 0:
                    ax.set_ylabel(
                        f"{class_name}\n({short})",
                        fontsize=9, fontweight="bold",
                        color=color, rotation=90, labelpad=5
                    )

                if col >= len(train_samples):
                    ax.text(0.5, 0.5, "N/A",
                            transform=ax.transAxes,
                            ha="center", va="center",
                            color="gray", fontsize=10)
                    continue

                sample = train_samples[col]
                img    = sample.load_image()

                if img is None:
                    ax.text(0.5, 0.5, "Load Error",
                            transform=ax.transAxes,
                            ha="center", va="center",
                            color="red", fontsize=9)
                    continue

                ax.imshow(img, cmap="gray", aspect="auto")

                # Draw bounding boxes
                if show_bbox and sample.bboxes:
                    for bbox in sample.bboxes:
                        rect = mpatches.FancyBboxPatch(
                            (bbox.xmin, bbox.ymin),
                            bbox.width, bbox.height,
                            boxstyle="round,pad=1",
                            linewidth=2,
                            edgecolor=color,
                            facecolor="none"
                        )
                        ax.add_patch(rect)
                        ax.text(
                            bbox.xmin,
                            max(bbox.ymin - 4, 8),
                            short,
                            fontsize=7.5,
                            color=color,
                            fontweight="bold",
                            bbox=dict(facecolor="black",
                                      alpha=0.6, pad=1,
                                      boxstyle="round")
                        )

                ax.set_title(sample.image_path.name[:18],
                             fontsize=7, color="#555555", pad=2)

                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(1.5)

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        if save_path is None:
            save_path = PATHS.RESULTS / "plots" / "dataset_overview.png"

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  🖼️  Saved: {save_path.absolute()}")
        plt.close()

    def visualize_bbox_size_distribution(
        self,
        save_path: Optional[Path] = None
    ):
        """Bar chart: average defect size per class."""
        names, avgs, colors = [], [], []

        for class_name in DATASET_CFG.CLASS_NAMES:
            samps = [
                s for s in self.samples
                if s.class_name == class_name and s.primary_bbox()
            ]
            if not samps:
                continue

            coverages = [
                s.primary_bbox().area /
                (s.image_width * s.image_height) * 100
                for s in samps
            ]
            names.append(DATASET_CFG.CLASS_SHORT[class_name])
            avgs.append(np.mean(coverages))
            colors.append(DATASET_CFG.CLASS_COLORS[class_name])

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, avgs, color=colors,
                      alpha=0.85, edgecolor="white", linewidth=1.5)

        for bar, val in zip(bars, avgs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold"
            )

        ax.set_title(
            "Average Defect Coverage per Class\n"
            "(% of total image area)",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Defect Class", fontsize=11)
        ax.set_ylabel("Avg Coverage (%)", fontsize=11)
        ax.set_ylim(0, max(avgs) * 1.3 if avgs else 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()

        if save_path is None:
            save_path = PATHS.RESULTS / "plots" / "bbox_distribution.png"

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Saved: {save_path.absolute()}")
        plt.close()
