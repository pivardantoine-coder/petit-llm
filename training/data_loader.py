from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\2629495\Downloads\petit-llm\V2"
)

DATA_DIR = (
    PROJECT_DIR
    / "training"
    / "data"
)

TRAIN_PATH = DATA_DIR / "train.bin"
VAL_PATH = DATA_DIR / "val.bin"

CONTEXT_LENGTH = 512


# ============================================================
# DATASET
# ============================================================

class BinaryTokenDataset:

    def __init__(
        self,
        path: str | Path,
        context_length: int = CONTEXT_LENGTH,
    ) -> None:

        self.path = Path(
            path
        ).expanduser().resolve()

        self.context_length = (
            context_length
        )

        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset introuvable : {self.path}"
            )

        # Lecture mémoire-mappée :
        # le fichier reste sur le disque.
        self.tokens = np.memmap(
            self.path,
            dtype=np.uint16,
            mode="r",
        )

        self.num_tokens = len(
            self.tokens
        )

        if self.num_tokens <= (
            self.context_length + 1
        ):
            raise ValueError(
                "Dataset trop petit pour "
                "le contexte demandé."
            )

        self.max_start = (
            self.num_tokens
            - self.context_length
            - 1
        )

    # ========================================================
    # UN EXEMPLE
    # ========================================================

    def get_example(
        self,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if (
            start < 0
            or start > self.max_start
        ):
            raise IndexError(
                f"Position invalide : {start}"
            )

        sequence = self.tokens[
            start:
            start + self.context_length + 1
        ]

        x = torch.from_numpy(
            sequence[:-1].astype(
                np.int64,
                copy=True,
            )
        )

        y = torch.from_numpy(
            sequence[1:].astype(
                np.int64,
                copy=True,
            )
        )

        return x, y


# ============================================================
# BATCH
# ============================================================

def get_batch(
    dataset: BinaryTokenDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:

    starts = np.random.randint(
        0,
        dataset.max_start + 1,
        size=batch_size,
    )

    x_batch = np.empty(
        (
            batch_size,
            dataset.context_length,
        ),
        dtype=np.int64,
    )

    y_batch = np.empty(
        (
            batch_size,
            dataset.context_length,
        ),
        dtype=np.int64,
    )

    for i, start in enumerate(starts):

        sequence = dataset.tokens[
            start:
            start + dataset.context_length + 1
        ]

        x_batch[i] = sequence[:-1]

        y_batch[i] = sequence[1:]

    x = torch.from_numpy(
        x_batch
    ).to(
        device=device,
        dtype=torch.long,
    )

    y = torch.from_numpy(
        y_batch
    ).to(
        device=device,
        dtype=torch.long,
    )

    return x, y


# ============================================================
# TEST
# ============================================================

def main():

    print("=" * 70)
    print("TEST DATA LOADER — V2")
    print("=" * 70)

    train_dataset = (
        BinaryTokenDataset(
            TRAIN_PATH,
            context_length=CONTEXT_LENGTH,
        )
    )

    val_dataset = (
        BinaryTokenDataset(
            VAL_PATH,
            context_length=CONTEXT_LENGTH,
        )
    )

    print()
    print(
        "Train tokens :",
        f"{train_dataset.num_tokens:,}",
    )

    print(
        "Validation tokens :",
        f"{val_dataset.num_tokens:,}",
    )

    print(
        "Context length :",
        CONTEXT_LENGTH,
    )

    # --------------------------------------------------------
    # CPU pour le test
    # --------------------------------------------------------

    device = torch.device(
        "cpu"
    )

    batch_size = 4

    x, y = get_batch(
        train_dataset,
        batch_size=batch_size,
        device=device,
    )

    print()
    print(
        "X shape :",
        tuple(x.shape),
    )

    print(
        "Y shape :",
        tuple(y.shape),
    )

    print()
    print(
        "Premier X :",
        x[0, :20].tolist(),
    )

    print(
        "Premier Y :",
        y[0, :20].tolist(),
    )

    print()

    # Vérification fondamentale :
    # Y doit être X décalé d'un token.
    correct = torch.equal(
        x[:, 1:],
        y[:, :-1],
    )

    print(
        "Décalage X/Y correct :",
        correct,
    )

    if not correct:
        raise RuntimeError(
            "Erreur dans la construction de X/Y."
        )

    print()
    print("=" * 70)
    print("DATA LOADER OK")
    print("=" * 70)


if __name__ == "__main__":
    main()