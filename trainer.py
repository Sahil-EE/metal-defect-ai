# metal_defect_system/src/training/trainer.py
"""
Complete Training Pipeline for Metal Defect Classifier

Handles:
- Phase A: Train head only (frozen backbone)
- Phase B: Fine-tune top layers (unfrozen)
- Loss function + optimizer + scheduler
- Validation loop
- Model checkpointing (saves best model)
- Early stopping (stops when not improving)
- Training history logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time

from pathlib import Path
from typing  import Dict, List, Tuple, Optional
from copy    import deepcopy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metal_defect_system.src.data.config  import PATHS, DATASET_CFG
from metal_defect_system.src.models.model import MetalDefectClassifier


# ─────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────

class TrainConfig:
    """
    All training hyperparameters in one place.

    BEGINNER: What is a hyperparameter?
    → Settings YOU choose before training starts
    → The model cannot learn these automatically
    → Examples: learning rate, batch size, num epochs
    → Wrong hyperparameters = bad model, no matter how good the data
    """

    # ── Phase A: Train head only ──────────────────────────────
    PHASE_A_EPOCHS   = 15       # epochs to train with frozen backbone
    PHASE_A_LR       = 1e-3     # 0.001 — higher LR for random head
    PHASE_A_LR_MIN   = 1e-5     # minimum LR during scheduling

    # ── Phase B: Fine-tune top layers ────────────────────────
    PHASE_B_EPOCHS   = 25       # additional epochs after unfreezing
    PHASE_B_LR       = 1e-4     # 0.0001 — lower LR to protect backbone
    PHASE_B_LR_MIN   = 1e-6
    PHASE_B_UNFREEZE = 3        # how many backbone blocks to unfreeze

    # ── General ───────────────────────────────────────────────
    BATCH_SIZE       = 32
    WEIGHT_DECAY     = 1e-4     # L2 regularization (prevents overfitting)

    # ── Early Stopping ────────────────────────────────────────
    # Stops training if val accuracy doesn't improve for N epochs
    PATIENCE         = 10       # wait 10 epochs before stopping
    MIN_DELTA        = 0.001    # minimum improvement to count as "better"

    # ── Checkpointing ─────────────────────────────────────────
    SAVE_TOP_K       = 1        # save only the best model


# ─────────────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation accuracy stops improving.

    WHY early stopping?
    ──────────────────────────────────────────────────────────
    Without it:
      Epoch 30: val_acc = 98% ← BEST
      Epoch 31: val_acc = 97% ← getting worse (overfitting!)
      Epoch 40: val_acc = 93% ← much worse
      (but training loss keeps going down — model memorizing!)

    Overfitting = model memorizes training data but
    fails on new (validation) data.

    With early stopping:
      Epoch 30: val_acc = 98% ← BEST, save model
      Epoch 31: val_acc = 97% ← patience counter: 1/10
      ...
      Epoch 40: val_acc = 97% ← patience counter: 10/10 → STOP
      → We use the epoch 30 checkpoint (best model)
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0          # epochs without improvement
        self.best_score = None       # best val accuracy seen
        self.stop       = False      # flag: should we stop?

    def __call__(self, val_accuracy: float) -> bool:
        """
        Call after each epoch with current val accuracy.
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = val_accuracy

        elif val_accuracy > self.best_score + self.min_delta:
            # Improved! Reset counter
            self.best_score = val_accuracy
            self.counter    = 0

        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

        return self.stop


# ─────────────────────────────────────────────────────────────
# METRICS TRACKER
# ─────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Tracks training and validation metrics across epochs.
    Saves history for plotting later.
    """

    def __init__(self):
        self.history = {
            "train_loss": [],
            "train_acc":  [],
            "val_loss":   [],
            "val_acc":    [],
            "lr":         [],
            "epoch_time": [],
        }
        self.best_val_acc   = 0.0
        self.best_epoch     = 0

    def update(
        self,
        epoch:      int,
        train_loss: float,
        train_acc:  float,
        val_loss:   float,
        val_acc:    float,
        lr:         float,
        epoch_time: float,
    ):
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["lr"].append(lr)
        self.history["epoch_time"].append(epoch_time)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch   = epoch

    def save(self, path: Path):
        """Saves history to JSON for later plotting."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def print_epoch(
        self,
        epoch:      int,
        total:      int,
        train_loss: float,
        train_acc:  float,
        val_loss:   float,
        val_acc:    float,
        lr:         float,
        epoch_time: float,
        is_best:    bool,
    ):
        """Prints one clean epoch summary line."""
        best_mark = " ⭐ BEST" if is_best else ""
        print(
            f"  Ep {epoch:>3}/{total} │ "
            f"Loss: {train_loss:.4f}→{val_loss:.4f} │ "
            f"Acc: {train_acc:.1%}→{val_acc:.1%} │ "
            f"LR: {lr:.2e} │ "
            f"{epoch_time:.0f}s"
            f"{best_mark}"
        )


# ─────────────────────────────────────────────────────────────
# CORE TRAINING FUNCTIONS
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
) -> Tuple[float, float]:
    """
    Runs ONE complete pass through the training data.

    BEGINNER: What happens in one epoch?
    ──────────────────────────────────────────────────────────
    for each batch of 32 images:
      1. Send images to GPU/CPU
      2. Forward pass: model makes predictions
      3. Calculate loss: how wrong were the predictions?
      4. Backward pass: which weights caused the error?
         (this is called "backpropagation")
      5. Optimizer step: nudge weights to reduce error
      6. Zero gradients: clear old gradient info
         (WHY? PyTorch accumulates gradients by default)

    Returns:
        avg_loss : average loss across all batches
        accuracy : fraction of correct predictions
    """
    model.train()   # Training mode ON
    # (enables dropout, batch norm in train mode)

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):

        # Move data to device (GPU if available)
        images = images.to(device)
        labels = labels.to(device)

        # ── Forward Pass ──────────────────────────────────────
        # model(images) calls model.forward(images)
        logits = model(images)          # shape: (32, 6)

        # ── Calculate Loss ────────────────────────────────────
        # CrossEntropyLoss:
        #   - Applies softmax to logits → probabilities
        #   - Measures how far probabilities are from true labels
        #   - Loss = 0 if prediction is perfect
        #   - Loss → ∞ if completely wrong
        loss = criterion(logits, labels)

        # ── Backward Pass ─────────────────────────────────────
        # Step 1: Clear old gradients
        optimizer.zero_grad()
        # Step 2: Compute gradients (backpropagation)
        loss.backward()
        # Step 3: Clip gradients (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Step 4: Update weights
        optimizer.step()

        # ── Track Metrics ─────────────────────────────────────
        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)     # predicted class
        correct    += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples

    return avg_loss, accuracy


@torch.no_grad()
def validate_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    """
    Runs ONE pass through validation data (no weight updates).

    BEGINNER:
    @torch.no_grad() = "don't track gradients"
    WHY? During validation:
      → We're just MEASURING accuracy, not learning
      → No need to compute gradients (saves memory + time)
      → model.eval() disables dropout (consistent predictions)
    """
    model.eval()    # Evaluation mode ON (disables dropout)

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits  = model(images)
        loss    = criterion(logits, labels)

        total_loss    += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples

    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────
# MAIN TRAINER CLASS
# ─────────────────────────────────────────────────────────────

class DefectTrainer:
    """
    Manages the complete training process.

    Usage:
        trainer = DefectTrainer(model, train_loader, val_loader, device)
        trainer.train_phase_a()   # frozen backbone
        trainer.train_phase_b()   # fine-tune
        trainer.save_final()
    """

    def __init__(
        self,
        model:        MetalDefectClassifier,
        train_loader: torch.utils.data.DataLoader,
        val_loader:   torch.utils.data.DataLoader,
        device:       torch.device,
        config:       TrainConfig = None,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.cfg          = config or TrainConfig()

        # Loss function
        # CrossEntropyLoss = softmax + negative log likelihood
        # Perfect for multi-class classification
        self.criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self.metrics   = MetricsTracker()
        self.epoch_num = 0   # global epoch counter across phases

        # Best model weights (saved in memory)
        self.best_weights = None
        self.best_val_acc = 0.0

        # Create checkpoint directory
        PATHS.MODELS_CKPT.mkdir(parents=True, exist_ok=True)
        PATHS.LOGS_EVAL.mkdir(parents=True, exist_ok=True)

    def _make_optimizer(self, lr: float) -> optim.Optimizer:
        """
        Creates Adam optimizer for trainable parameters only.

        WHY Adam?
        ──────────────────────────────────────────────────────
        Adam = Adaptive Moment Estimation

        Old approach (SGD):
          nudge = learning_rate × gradient
          (same size nudge for all parameters)

        Adam:
          nudge size adapts per parameter
          → Parameters that haven't moved much → larger nudge
          → Parameters that changed a lot → smaller nudge
          → Converges faster, more stable training

        weight_decay = L2 regularization
          → Adds small penalty for large weights
          → Prevents overfitting (keeps weights small)

        only trainable params:
          → Frozen backbone params have requires_grad=False
          → Filter(None, params) keeps only trainable ones
        """
        trainable = filter(
            lambda p: p.requires_grad,
            self.model.parameters()
        )
        return optim.Adam(
            trainable,
            lr=lr,
            weight_decay=self.cfg.WEIGHT_DECAY
        )

    def _make_scheduler(
        self,
        optimizer: optim.Optimizer,
        lr_min:    float,
        epochs:    int,
    ) -> optim.lr_scheduler.CosineAnnealingLR:
        """
        Creates learning rate scheduler.

        WHY use a scheduler?
        ──────────────────────────────────────────────────────
        Fixed LR problems:
          Too high → training unstable, bounces around optimal
          Too low  → training too slow, gets stuck

        CosineAnnealingLR:
          Starts at lr_max, smoothly decreases to lr_min
          following a cosine curve.

          Epoch  1: LR = 0.001  (start)
          Epoch  5: LR = 0.0007 (decreasing)
          Epoch 10: LR = 0.0001 (getting small)
          Epoch 15: LR = 0.00001 (near minimum)

          WHY cosine? Smooth decay → no sudden drops
          → Large steps early (fast progress)
          → Small steps late (fine-tuning the solution)
        """
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr_min
        )

    def _run_phase(
        self,
        phase_name:  str,
        num_epochs:  int,
        lr:          float,
        lr_min:      float,
        patience:    int = None,
    ):
        """
        Generic training phase runner.
        Used by both phase A and phase B.
        """
        patience    = patience or self.cfg.PATIENCE
        optimizer   = self._make_optimizer(lr)
        scheduler   = self._make_scheduler(optimizer, lr_min, num_epochs)
        early_stop  = EarlyStopping(patience=patience)

        # Count trainable params for this phase
        trainable = sum(
            p.numel() for p in self.model.parameters()
            if p.requires_grad
        )

        print(f"\n  Trainable params : {trainable:,}")
        print(f"  Learning rate    : {lr}")
        print(f"  Epochs           : {num_epochs}")
        print(f"  Patience         : {patience}")
        print(f"\n  {'Ep':>5} │ {'TrainLoss':>9} {'ValLoss':>9} │ "
              f"{'TrainAcc':>9} {'ValAcc':>9} │ "
              f"{'LR':>9} │ {'Time':>5}")
        print("  " + "─" * 72)

        for epoch in range(1, num_epochs + 1):
            self.epoch_num += 1
            t_start = time.time()

            # ── Train ─────────────────────────────────────────
            train_loss, train_acc = train_one_epoch(
                self.model, self.train_loader,
                optimizer, self.criterion, self.device
            )

            # ── Validate ──────────────────────────────────────
            val_loss, val_acc = validate_one_epoch(
                self.model, self.val_loader,
                self.criterion, self.device
            )

            # ── Scheduler step ────────────────────────────────
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            epoch_time = time.time() - t_start

            # ── Check if best model ───────────────────────────
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                # Save best weights in memory
                self.best_weights = deepcopy(self.model.state_dict())
                # Save best checkpoint to disk
                self._save_checkpoint(
                    epoch=self.epoch_num,
                    val_acc=val_acc,
                    filename="best_model.pth"
                )

            # ── Log metrics ───────────────────────────────────
            self.metrics.update(
                epoch=self.epoch_num,
                train_loss=train_loss, train_acc=train_acc,
                val_loss=val_loss, val_acc=val_acc,
                lr=current_lr, epoch_time=epoch_time
            )

            # ── Print epoch summary ───────────────────────────
            best_mark = " ⭐" if is_best else ""
            print(
                f"  {epoch:>5} │ "
                f"{train_loss:>9.4f} {val_loss:>9.4f} │ "
                f"{train_acc:>9.1%} {val_acc:>9.1%} │ "
                f"{current_lr:>9.2e} │ "
                f"{epoch_time:>4.0f}s"
                f"{best_mark}"
            )

            # ── Early stopping check ──────────────────────────
            if early_stop(val_acc):
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                print(f"      No improvement for {patience} epochs")
                break

        # ── Phase complete ────────────────────────────────────
        print(f"\n  ✅ {phase_name} complete!")
        print(f"     Best val accuracy : {self.best_val_acc:.2%}")
        print(f"     Best epoch        : {self.metrics.best_epoch}")

    def _save_checkpoint(
        self,
        epoch:    int,
        val_acc:  float,
        filename: str
    ):
        """Saves model weights + metadata to disk."""
        checkpoint = {
            "epoch":        epoch,
            "val_acc":      val_acc,
            "model_state":  self.model.state_dict(),
            "num_classes":  self.model.num_classes,
            "class_names":  DATASET_CFG.CLASS_NAMES,
        }
        path = PATHS.MODELS_CKPT / filename
        torch.save(checkpoint, path)

    def train_phase_a(self):
        """
        Phase A: Train classification head only.
        Backbone is frozen.
        """
        print("\n" + "═" * 75)
        print(f"  🔒 PHASE A — HEAD TRAINING (backbone frozen)")
        print("═" * 75)

        self.model.freeze_backbone()
        self._run_phase(
            phase_name = "Phase A",
            num_epochs = self.cfg.PHASE_A_EPOCHS,
            lr         = self.cfg.PHASE_A_LR,
            lr_min     = self.cfg.PHASE_A_LR_MIN,
        )

        # Save phase A checkpoint
        self._save_checkpoint(
            epoch    = self.epoch_num,
            val_acc  = self.best_val_acc,
            filename = "phase_a_final.pth"
        )

    def train_phase_b(self):
        """
        Phase B: Unfreeze top backbone layers + fine-tune.
        Uses lower learning rate to protect pretrained weights.
        """
        print("\n" + "═" * 75)
        print(f"  🔓 PHASE B — FINE TUNING "
              f"(top {self.cfg.PHASE_B_UNFREEZE} backbone blocks unfrozen)")
        print("═" * 75)

        self.model.unfreeze_top_layers(
            num_layers=self.cfg.PHASE_B_UNFREEZE
        )

        # Reload best weights from Phase A before fine-tuning
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)
            print("  ✅ Loaded best Phase A weights")

        self._run_phase(
            phase_name = "Phase B",
            num_epochs = self.cfg.PHASE_B_EPOCHS,
            lr         = self.cfg.PHASE_B_LR,
            lr_min     = self.cfg.PHASE_B_LR_MIN,
        )

        # Save final checkpoint
        self._save_checkpoint(
            epoch    = self.epoch_num,
            val_acc  = self.best_val_acc,
            filename = "final_model.pth"
        )

    def save_training_history(self):
        """Saves full training history to JSON."""
        path = PATHS.LOGS_EVAL / "training_history.json"
        self.metrics.save(path)
        print(f"\n  📊 Training history saved: {path}")

    def plot_training_curves(self):
        """
        Plots training and validation accuracy/loss curves.
        Save as PNG — use to diagnose training issues.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        history   = self.metrics.history
        n_epochs  = len(history["train_loss"])
        epochs    = range(1, n_epochs + 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            "Training Progress — Metal Defect Classifier",
            fontsize=13, fontweight="bold"
        )

        # ── Plot 1: Accuracy ──────────────────────────────────
        ax = axes[0]
        ax.plot(epochs, history["train_acc"],
                label="Train", color="#4ECDC4", linewidth=2)
        ax.plot(epochs, history["val_acc"],
                label="Val",   color="#FF6B6B", linewidth=2)
        ax.axhline(y=0.99, color="gray",
                   linestyle="--", alpha=0.5, label="99% target")

        # Mark best epoch
        best_ep  = self.metrics.best_epoch
        best_acc = self.best_val_acc
        ax.scatter([best_ep], [best_acc],
                   color="gold", s=100, zorder=5, label=f"Best: {best_acc:.1%}")

        ax.set_title("Accuracy", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.0%}')
        )

        # ── Plot 2: Loss ──────────────────────────────────────
        ax = axes[1]
        ax.plot(epochs, history["train_loss"],
                label="Train", color="#4ECDC4", linewidth=2)
        ax.plot(epochs, history["val_loss"],
                label="Val",   color="#FF6B6B", linewidth=2)
        ax.set_title("Loss", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend()
        ax.grid(alpha=0.3)

        # ── Plot 3: Learning Rate ─────────────────────────────
        ax = axes[2]
        ax.plot(epochs, history["lr"],
                color="#45B7D1", linewidth=2)
        ax.set_title("Learning Rate Schedule",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_yscale("log")   # log scale for LR
        ax.grid(alpha=0.3)

        plt.tight_layout()

        save_path = PATHS.RESULTS / "plots" / "training_curves.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📈 Training curves saved: {save_path}")
        plt.close()