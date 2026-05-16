# run_phase4.py
"""
PHASE 4: Model Training
Run: python run_phase4.py

NOTE FOR CPU USERS:
  Phase A (15 epochs): ~45-60 minutes on CPU
  Phase B (25 epochs): ~75-90 minutes on CPU
  TOTAL: ~2-2.5 hours

  For faster training:
  → Reduce epochs in TrainConfig (PHASE_A=5, PHASE_B=10)
  → Or use Google Colab (free GPU, 10× faster)

  We'll add a QUICK MODE option below to test everything works
  before doing the full run.
"""

import sys
import time
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from metal_defect_system.src.data.config   import PATHS, DATASET_CFG
from metal_defect_system.src.data.parser   import NEUDETParser
from metal_defect_system.src.data.dataset  import build_dataloaders
from metal_defect_system.src.models.model  import MetalDefectClassifier
from metal_defect_system.src.training.trainer import (
    DefectTrainer, TrainConfig
)


def run_training(quick_mode: bool = False):

    start = time.time()

    print("\n")
    print("╔" + "═"*56 + "╗")
    print("║   🔥  METAL DEFECT DETECTION — PHASE 4            ║")
    print("║        Model Training Pipeline                    ║")
    print("╚" + "═"*56 + "╝")

    # ── Quick Mode vs Full Mode ────────────────────────────────
    if quick_mode:
        print("\n  ⚡ QUICK MODE: 2+3 epochs (test run only)")
        print("     Run without --quick for full training\n")
        TrainConfig.PHASE_A_EPOCHS = 2
        TrainConfig.PHASE_B_EPOCHS = 3
        TrainConfig.PATIENCE       = 99   # no early stop in quick mode
    else:
        print("\n  🏋️  FULL TRAINING MODE")
        print(f"     Phase A: {TrainConfig.PHASE_A_EPOCHS} epochs")
        print(f"     Phase B: {TrainConfig.PHASE_B_EPOCHS} epochs")
        print(f"     Total  : {TrainConfig.PHASE_A_EPOCHS + TrainConfig.PHASE_B_EPOCHS} epochs\n")

    # ── STEP 1: Device ────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    if device.type == "cpu" and not quick_mode:
        total_epochs = TrainConfig.PHASE_A_EPOCHS + TrainConfig.PHASE_B_EPOCHS
        est_minutes  = total_epochs * 3  # ~3 min/epoch on CPU
        print(f"\n  ⏱️  CPU estimated time: ~{est_minutes} minutes")
        print(f"  💡 Tip: Use --quick flag to test (5 epochs)")
        print(f"  💡 Tip: Google Colab for free GPU\n")

    # ── STEP 2: Load Data ─────────────────────────────────────
    print("═" * 58)
    print("  Loading dataset...")
    print("═" * 58)

    parser  = NEUDETParser(dataset_root=PATHS.DATA_RAW)
    samples = parser.parse_all()

    train_samples = parser.get_split("train")
    val_samples   = parser.get_split("validation")

    train_loader, val_loader = build_dataloaders(
        train_samples = train_samples,
        val_samples   = val_samples,
        batch_size    = TrainConfig.BATCH_SIZE,
        num_workers   = 0,
        use_crop      = True,
    )

    print(f"\n  ✅ Train: {len(train_samples)} samples → "
          f"{len(train_loader)} batches")
    print(f"  ✅ Val:   {len(val_samples)} samples → "
          f"{len(val_loader)} batches")

    # ── STEP 3: Build Model ───────────────────────────────────
    print("\n" + "═" * 58)
    print("  Building model...")
    print("═" * 58 + "\n")

    model = MetalDefectClassifier(
        num_classes = DATASET_CFG.NUM_CLASSES,
        pretrained  = True,
        dropout1    = 0.3,
        dropout2    = 0.2,
        hidden_size = 512,
    )

    # ── STEP 4: Create Trainer ────────────────────────────────
    trainer = DefectTrainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
    )

    # ── STEP 5: PHASE A TRAINING ──────────────────────────────
    trainer.train_phase_a()

    # ── STEP 6: PHASE B FINE-TUNING ───────────────────────────
    trainer.train_phase_b()

    # ── STEP 7: Save Results ──────────────────────────────────
    trainer.save_training_history()
    trainer.plot_training_curves()

    # ── FINAL SUMMARY ─────────────────────────────────────────
    elapsed = time.time() - start
    mins    = elapsed / 60

    print("\n")
    print("╔" + "═"*56 + "╗")
    print("║           ✅  PHASE 4 COMPLETE                    ║")
    print("╠" + "═"*56 + "╣")
    print(f"║  Best Val Accuracy : {trainer.best_val_acc:.2%}                        ║")
    print(f"║  Best Epoch        : {trainer.metrics.best_epoch:<34} ║")
    print(f"║  Total Time        : {mins:.1f} minutes                     ║")
    print(f"║  Device Used       : {str(device):<34} ║")
    print("╠" + "═"*56 + "╣")
    print("║  📁 Saved files:                                   ║")
    print("║    models/checkpoints/best_model.pth              ║")
    print("║    models/checkpoints/final_model.pth             ║")
    print("║    logs/evaluation/training_history.json          ║")
    print("║    results/plots/training_curves.png              ║")
    print("╠" + "═"*56 + "╣")
    print("║  ► Type 'Phase 5 ready' for Evaluation            ║")
    print("╚" + "═"*56 + "╝\n")

    return trainer.best_val_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run 2+3 epochs only (test that everything works)"
    )
    args = parser.parse_args()

    run_training(quick_mode=args.quick)