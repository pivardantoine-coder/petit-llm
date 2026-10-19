from __future__ import annotations

from pathlib import Path
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import BinaryTokenDataset, get_batch

# ============================================================
# PATHS
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "model"
sys.path.insert(0, str(MODEL_DIR))
from transformer import TransformerLM, count_parameters

TRAIN_PATH = PROJECT_DIR / "training" / "data" / "train_large.bin"
VAL_PATH = PROJECT_DIR / "training" / "data" / "val_large.bin"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest_checkpoint.pt"

# ============================================================
# MODEL CONFIG
# ============================================================
CONTEXT_LENGTH = 2048
MODEL_CONFIG = {
    "vocab_size": 16000,
    "d_model": 768,
    "num_heads": 12,
    "num_layers": 24,
    "ffn_hidden_dim": 2176,
    "max_seq_len": CONTEXT_LENGTH,
}

# ============================================================
# TRAINING CONFIG
# ============================================================
MICRO_BATCH_SIZE = 1
GRAD_ACCUMULATION_STEPS = 8
# Final absolute step, not the number of extra steps.
# A checkpoint at step 2000 resumes at 2001 and trains through TOTAL_STEPS.
TOTAL_STEPS = 10_000
MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
VAL_EVERY = 500
SAVE_EVERY = 1000
VAL_BATCHES = 4
SEED = 42
# Set True only when intentionally starting from random weights.
START_FROM_SCRATCH = False

# ============================================================
# REPRODUCIBILITY / DEVICE
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")


def get_learning_rate(step: int) -> float:
    """Cosine learning-rate schedule indexed by the absolute training step."""
    if step <= WARMUP_STEPS:
        return MAX_LR * step / WARMUP_STEPS
    if TOTAL_STEPS <= WARMUP_STEPS:
        return MIN_LR
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    progress = max(0.0, min(1.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (MAX_LR - MIN_LR) * cosine


def validate_checkpoint_config(saved_config: dict) -> None:
    if not isinstance(saved_config, dict):
        raise ValueError("Le checkpoint ne contient pas de configuration valide.")
    keys = ("vocab_size", "d_model", "num_heads", "num_layers", "ffn_hidden_dim", "max_seq_len")
    errors = []
    for key in keys:
        saved = saved_config.get(key)
        if key == "max_seq_len" and saved is None:
            saved = saved_config.get("context_length")
        current = MODEL_CONFIG.get(key)
        if saved is not None and current is not None and saved != current:
            errors.append(f"{key}: checkpoint={saved}, script={current}")
    if errors:
        raise ValueError(
            "Configuration incompatible entre checkpoint et script :\n  - "
            + "\n  - ".join(errors)
            + "\nUtilise la configuration exacte du modèle entraîné."
        )


@torch.no_grad()
def evaluate(model: nn.Module, dataset: BinaryTokenDataset, batches: int = VAL_BATCHES) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(batches):
        x, y = get_batch(dataset, batch_size=MICRO_BATCH_SIZE, device=device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
        else:
            logits = model(x)
        loss = F.cross_entropy(
            logits.float().reshape(-1, MODEL_CONFIG["vocab_size"]), y.reshape(-1)
        )
        total_loss += float(loss.item())
    model.train()
    return total_loss / batches


def atomic_save(checkpoint: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)


def save_checkpoint(model, optimizer, scaler, step, train_loss, val_loss, best_val_loss) -> None:
    checkpoint = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": dict(MODEL_CONFIG),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    numbered_path = CHECKPOINT_DIR / f"V2_step_{step}.pt"
    atomic_save(checkpoint, numbered_path)
    atomic_save(checkpoint, LATEST_CHECKPOINT)
    print(f"Checkpoint sauvegardé : {numbered_path}")
    print(f"Checkpoint latest mis à jour : {LATEST_CHECKPOINT}")


def restore_rng_states(checkpoint: dict) -> None:
    try:
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        cuda_states = checkpoint.get("cuda_rng_state_all")
        if torch.cuda.is_available() and cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    except Exception as exc:
        print(f"AVERTISSEMENT : impossible de restaurer tous les RNG : {exc}")


def load_checkpoint(model, optimizer, scaler):
    if START_FROM_SCRATCH:
        print("START_FROM_SCRATCH=True : checkpoint ignoré.")
        return 1, float("inf")
    if not LATEST_CHECKPOINT.exists():
        print(f"Aucun checkpoint trouvé : {LATEST_CHECKPOINT}")
        print("Démarrage à l'étape 1.")
        return 1, float("inf")

    print(f"Chargement du checkpoint : {LATEST_CHECKPOINT}")
    # Use weights_only=False only for a checkpoint you trust. Pickle loading can execute code.
    checkpoint = torch.load(LATEST_CHECKPOINT, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Le checkpoint doit être un dictionnaire.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint sans 'model_state_dict' : ce n'est pas un checkpoint complet.")

    validate_checkpoint_config(checkpoint.get("config", {}))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        print("État de l'optimiseur restauré.")
    else:
        print("AVERTISSEMENT : état de l'optimiseur absent.")

    if "scaler_state_dict" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("État du GradScaler restauré.")
        except Exception as exc:
            print(f"AVERTISSEMENT : GradScaler non restauré : {exc}")

    restore_rng_states(checkpoint)
    saved_step = int(checkpoint.get("step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    print(f"Checkpoint chargé à l'étape : {saved_step}")
    print(f"Train loss enregistrée : {checkpoint.get('train_loss', 'inconnue')}")
    print(f"Val loss enregistrée : {checkpoint.get('val_loss', 'inconnue')}")
    print(f"Reprise à l'étape : {saved_step + 1}")
    return saved_step + 1, best_val_loss


def main():
    print("=" * 72)
    print("TRAINING — TRANSFORMER V2")
    print("=" * 72)
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"VRAM   : {vram:.2f} GB")

    print("\nConfiguration modèle :")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key:18s}: {value}")
    print(f"\nMicro batch          : {MICRO_BATCH_SIZE}")
    print(f"Gradient accumulation: {GRAD_ACCUMULATION_STEPS}")
    print(f"Effective batch      : {MICRO_BATCH_SIZE * GRAD_ACCUMULATION_STEPS}")
    print(f"Context              : {CONTEXT_LENGTH}")
    print(f"Total target steps   : {TOTAL_STEPS}")
    print(f"Max LR               : {MAX_LR}")
    print(f"Min LR               : {MIN_LR}")
    print(f"Warmup steps         : {WARMUP_STEPS}")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Train introuvable : {TRAIN_PATH}")
    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Validation introuvable : {VAL_PATH}")

    print("\n" + "=" * 72 + "\nCHARGEMENT DES DATASETS\n" + "=" * 72)
    train_dataset = BinaryTokenDataset(TRAIN_PATH, context_length=CONTEXT_LENGTH)
    val_dataset = BinaryTokenDataset(VAL_PATH, context_length=CONTEXT_LENGTH)
    print(f"Train tokens : {train_dataset.num_tokens:,}")
    print(f"Val tokens   : {val_dataset.num_tokens:,}")

    print("\n" + "=" * 72 + "\nCREATION DU MODELE\n" + "=" * 72)
    model = TransformerLM(**MODEL_CONFIG).to(device)
    parameters = count_parameters(model)
    print(f"Paramètres    : {parameters:,}")
    print(f"Paramètres (M): {parameters / 1_000_000:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Restore checkpoint before the training loop.
    start_step, best_val_loss = load_checkpoint(model, optimizer, scaler)
    if start_step > TOTAL_STEPS:
        print(
            f"\nLe checkpoint est à l'étape {start_step - 1}; TOTAL_STEPS={TOTAL_STEPS} est déjà atteint."
        )
        print("Augmente TOTAL_STEPS pour continuer l'entraînement.")
        return

    print("\n" + "=" * 72)
    print(f"TRAINING — reprise à {start_step}, cible {TOTAL_STEPS}")
    print("=" * 72)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_train_loss = float("nan")
    last_val_loss = float("nan")

    for step in range(start_step, TOTAL_STEPS + 1):
        accumulated_loss = 0.0
        for _ in range(GRAD_ACCUMULATION_STEPS):
            x, y = get_batch(train_dataset, batch_size=MICRO_BATCH_SIZE, device=device)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, MODEL_CONFIG["vocab_size"]), y.reshape(-1)
                    )
                scaler.scale(loss / GRAD_ACCUMULATION_STEPS).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, MODEL_CONFIG["vocab_size"]), y.reshape(-1)
                )
                (loss / GRAD_ACCUMULATION_STEPS).backward()
            accumulated_loss += float(loss.detach().item())

        if device.type == "cuda":
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        lr = get_learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if device.type == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        train_loss = accumulated_loss / GRAD_ACCUMULATION_STEPS
        last_train_loss = train_loss
        if step == start_step or step % 10 == 0:
            print(
                f"Step {step:5d}/{TOTAL_STEPS} | train loss = {train_loss:.4f} | lr = {lr:.2e}"
            )

        val_loss = float("nan")
        if step % VAL_EVERY == 0 or step == TOTAL_STEPS:
            val_loss = evaluate(model, val_dataset, batches=VAL_BATCHES)
            last_val_loss = val_loss
            print(f"                 val loss = {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                best_path = CHECKPOINT_DIR / "best_weights.pt"
                torch.save(model.state_dict(), best_path)
                print(f"Nouvelle meilleure val loss : {best_val_loss:.4f}")
                print(f"Poids sauvegardés : {best_path}")

        if step % SAVE_EVERY == 0 or step == TOTAL_STEPS:
            save_checkpoint(
                model, optimizer, scaler, step, train_loss, val_loss, best_val_loss
            )

    print("\n" + "=" * 72)
    print("TRAINING TERMINE")
    print(f"Dernière étape : {TOTAL_STEPS}")
    print(f"Dernière train loss : {last_train_loss:.4f}")
    print(f"Dernière val loss : {last_val_loss:.4f}")
    print(f"Meilleure val loss : {best_val_loss:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
