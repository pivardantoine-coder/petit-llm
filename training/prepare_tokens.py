from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import sentencepiece as spm


# ============================================================
# CHEMINS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\2629495\Downloads\petit-llm\V2"
)

CORPUS_PATH = (
    PROJECT_DIR
    / "corpus"
    / "fineweb_100M.txt"
)

TOKENIZER_PATH = (
    PROJECT_DIR
    / "tokenizer"
    / "tokenizer.model"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "training"
    / "data"
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_RATIO = 0.90

CHUNK_SIZE = 1024 * 1024  # 1 MB de texte


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer() -> spm.SentencePieceProcessor:

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer introuvable : {TOKENIZER_PATH}"
        )

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(TOKENIZER_PATH)
    )

    print(
        "Vocab size :",
        tokenizer.get_piece_size()
    )

    return tokenizer


# ============================================================
# TOKENISATION
# ============================================================

def tokenize_corpus(
    tokenizer: spm.SentencePieceProcessor,
) -> Path:

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus introuvable : {CORPUS_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_tokens_path = (
        OUTPUT_DIR / "all_tokens.bin"
    )

    total_tokens = 0
    total_chunks = 0

    print()
    print("=" * 70)
    print("TOKENISATION DU CORPUS")
    print("=" * 70)

    print()
    print("Corpus :")
    print(CORPUS_PATH)

    print()
    print("Sortie :")
    print(all_tokens_path)

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8",
    ) as source, open(
        all_tokens_path,
        "wb",
    ) as output:

        while True:

            text = source.read(
                CHUNK_SIZE
            )

            if not text:
                break

            ids = tokenizer.encode(
                text,
                out_type=int,
            )

            if not ids:
                continue

            # 16 000 tokens < 65 535,
            # donc uint16 suffit largement.
            array = np.asarray(
                ids,
                dtype=np.uint16,
            )

            array.tofile(output)

            total_tokens += len(ids)
            total_chunks += 1

            if total_chunks % 10 == 0:

                print(
                    f"Chunks : {total_chunks:,} | "
                    f"Tokens : {total_tokens:,}"
                )

    print()
    print(
        f"Tokens totaux : {total_tokens:,}"
    )

    print(
        f"Taille fichier : "
        f"{all_tokens_path.stat().st_size / (1024 * 1024):.2f} MB"
    )

    return all_tokens_path


# ============================================================
# TRAIN / VALIDATION
# ============================================================

def split_dataset(
    all_tokens_path: Path,
    total_tokens: int,
) -> None:

    train_tokens = int(
        total_tokens * TRAIN_RATIO
    )

    val_tokens = (
        total_tokens - train_tokens
    )

    train_path = (
        OUTPUT_DIR / "train.bin"
    )

    val_path = (
        OUTPUT_DIR / "val.bin"
    )

    print()
    print("=" * 70)
    print("SPLIT TRAIN / VALIDATION")
    print("=" * 70)

    print()
    print(
        f"Train : {train_tokens:,} tokens"
    )

    print(
        f"Val   : {val_tokens:,} tokens"
    )

    with open(
        all_tokens_path,
        "rb",
    ) as source:

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_bytes = (
            train_tokens * 2
        )

        with open(
            train_path,
            "wb",
        ) as train_file:

            remaining = train_bytes

            while remaining > 0:

                chunk = source.read(
                    min(
                        16 * 1024 * 1024,
                        remaining,
                    )
                )

                if not chunk:
                    raise RuntimeError(
                        "Fin inattendue du fichier."
                    )

                train_file.write(
                    chunk
                )

                remaining -= len(chunk)

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        with open(
            val_path,
            "wb",
        ) as val_file:

            while True:

                chunk = source.read(
                    16 * 1024 * 1024
                )

                if not chunk:
                    break

                val_file.write(
                    chunk
                )

    print()
    print("Train :", train_path)
    print("Val   :", val_path)


# ============================================================
# METADATA
# ============================================================

def save_metadata(
    total_tokens: int,
) -> None:

    train_tokens = int(
        total_tokens * TRAIN_RATIO
    )

    val_tokens = (
        total_tokens - train_tokens
    )

    metadata = {
        "tokenizer": str(
            TOKENIZER_PATH
        ),
        "vocab_size": 16000,
        "total_tokens": total_tokens,
        "train_tokens": train_tokens,
        "validation_tokens": val_tokens,
        "train_ratio": TRAIN_RATIO,
        "dtype": "uint16",
        "bytes_per_token": 2,
        "context_length": 512,
    }

    path = (
        OUTPUT_DIR
        / "dataset_info.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print()
    print(
        f"Metadata : {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    tokenizer = load_tokenizer()

    all_tokens_path = tokenize_corpus(
        tokenizer
    )

    file_size = (
        all_tokens_path.stat().st_size
    )

    if file_size % 2 != 0:
        raise RuntimeError(
            "La taille du fichier n'est pas "
            "compatible avec uint16."
        )

    total_tokens = (
        file_size // 2
    )

    split_dataset(
        all_tokens_path,
        total_tokens,
    )

    save_metadata(
        total_tokens
    )

    print()
    print("=" * 70)
    print("DATASET PRÊT")
    print("=" * 70)


if __name__ == "__main__":
    main()