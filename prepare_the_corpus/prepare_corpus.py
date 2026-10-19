
from pathlib import Path
import hashlib
import json
import sqlite3
import re
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Les fichiers Parquet sont dans ce dossier ou ses sous-dossiers.
INPUT_DIR = SCRIPT_DIR

# Cherche le tokenizer localement, puis à son emplacement V2 connu.
TOKENIZER_CANDIDATES = [
    SCRIPT_DIR / "tokenizer.model",
    SCRIPT_DIR / "tokenizer" / "tokenizer.model",
    Path.home() / "Downloads" / "petit-llm" / "V2"
    / "tokenizer" / "tokenizer.model",
]

OUTPUT_DIR = SCRIPT_DIR / "prepared_output"

# Taille cible : 200 MiB par shard, sauf le dernier de chaque split.
SHARD_SIZE_BYTES = 200 * 1024 * 1024
BYTES_PER_TOKEN = 2
TOKENS_PER_SHARD = SHARD_SIZE_BYTES // BYTES_PER_TOKEN

# Doit correspondre au modèle V2 existant.
EXPECTED_VOCAB_SIZE = 16000

# Filtrage modéré, à ajuster après inspection.
MIN_CHARS = 100
PARQUET_BATCH_SIZE = 128

# Environ 2 % des documents pour la validation.
VALIDATION_PERCENT = 2

# Colonne de texte FineWeb habituelle.
TEXT_COLUMN = "text"


# ============================================================
# UTILITAIRES
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def find_tokenizer() -> Path:
    for path in TOKENIZER_CANDIDATES:
        if path.is_file():
            return path

    searched = "\n".join(str(p) for p in TOKENIZER_CANDIDATES)

    raise FileNotFoundError(
        "Impossible de trouver tokenizer.model.\n"
        "Emplacements vérifiés :\n" + searched
    )


def clean_text(value) -> str | None:
    """Nettoyage prudent : ne pas supprimer le contenu technique utile."""

    if not isinstance(value, str):
        return None

    # Retirer les caractères de contrôle, sauf tabulations et retours ligne.
    value = "".join(
        char for char in value
        if char in "\n\r\t" or ord(char) >= 32
    )

    # Normaliser les espaces sans supprimer les paragraphes.
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = value.strip()

    if len(value) < MIN_CHARS:
        return None

    # Rejeter uniquement les textes presque entièrement non alphanumériques.
    # Le seuil reste volontairement modéré pour préserver maths et code.
    meaningful = sum(c.isalnum() for c in value)

    if meaningful / max(len(value), 1) < 0.10:
        return None

    return value


def get_split(text_hash: bytes) -> str:
    """Affectation stable : le même document va toujours dans le même split."""

    number = int.from_bytes(text_hash[:8], "big") % 100

    if number < VALIDATION_PERCENT:
        return "validation"

    return "train"


# ============================================================
# ÉCRITURE DES SHARDS
# ============================================================

class ShardWriter:
    def __init__(self, folder: Path, split: str):
        self.folder = folder
        self.split = split
        self.folder.mkdir(parents=True, exist_ok=True)

        self.file = None
        self.path = None
        self.shard_index = 0
        self.tokens_in_shard = 0
        self.total_tokens = 0
        self.shards = []

    def open_shard(self):
        name = f"fineweb_{self.split}_{self.shard_index:05d}.bin"
        self.path = self.folder / name
        self.file = self.path.open("wb")
        self.tokens_in_shard = 0

    def close_shard(self):
        if self.file is None:
            return

        self.file.close()
        self.file = None

        self.shards.append({
            "file": self.path.name,
            "split": self.split,
            "tokens": self.tokens_in_shard,
            "bytes": self.path.stat().st_size,
            "sha256": sha256_file(self.path),
        })

        print(
            f"  Créé {self.path.name}: "
            f"{self.tokens_in_shard:,} tokens"
        )

        self.shard_index += 1
        self.path = None
        self.tokens_in_shard = 0

    def write_ids(self, ids):
        """Écrit les tokens par blocs sans garder un shard entier en RAM."""

        array = np.asarray(ids, dtype=np.int64)

        if array.size == 0:
            return

        if array.min() < 0 or array.max() >= 65536:
            raise ValueError(
                "Un identifiant de token ne tient pas dans uint16."
            )

        position = 0

        while position < len(array):
            if self.file is None:
                self.open_shard()

            capacity = TOKENS_PER_SHARD - self.tokens_in_shard
            count = min(capacity, len(array) - position)

            chunk = array[position:position + count].astype(
                np.uint16, copy=False
            )
            chunk.tofile(self.file)

            position += count
            self.tokens_in_shard += count
            self.total_tokens += count

            if self.tokens_in_shard == TOKENS_PER_SHARD:
                self.close_shard()

    def finish(self):
        # Termine le dernier shard, même s'il est plus petit que la cible.
        self.close_shard()


# ============================================================
# PRÉPARATION DU CORPUS
# ============================================================

def main():
    tokenizer_path = find_tokenizer()

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(tokenizer_path)
    )

    vocab_size = tokenizer.get_piece_size()

    print("=" * 60)
    print("PRÉPARATION DU CORPUS FINEWEB — CPU")
    print("=" * 60)
    print(f"Dossier des données : {INPUT_DIR}")
    print(f"Tokenizer : {tokenizer_path}")
    print(f"Vocabulaire : {vocab_size}")

    if vocab_size != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"Le tokenizer contient {vocab_size} tokens, "
            f"mais le script attend {EXPECTED_VOCAB_SIZE}. "
            "Vérifie la configuration réelle de ton modèle V2."
        )

    if vocab_size > 65536:
        raise ValueError("Le vocabulaire dépasse la capacité de uint16.")

    parquet_files = sorted(
        path for path in INPUT_DIR.rglob("*.parquet")
        if OUTPUT_DIR not in path.parents
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"Aucun fichier .parquet trouvé dans {INPUT_DIR}"
        )

    print(f"Fichiers Parquet trouvés : {len(parquet_files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # La base SQLite stocke les empreintes sur disque : contrairement à
    # un set Python, elle ne conserve pas tous les hashes en RAM.
    database_path = OUTPUT_DIR / "dedup.sqlite3"

    if database_path.exists():
        print(
            "\nUne base de déduplication existe déjà :\n"
            f"{database_path}\n"
            "Pour éviter de mélanger deux exécutions, ce script refuse "
            "de continuer avec une ancienne base. Supprime le dossier "
            "prepared_output si tu veux recommencer entièrement."
        )
        return

    connection = sqlite3.connect(database_path)

    connection.execute(
        "CREATE TABLE seen (digest BLOB PRIMARY KEY)"
    )

    writers = {
        "train": ShardWriter(OUTPUT_DIR / "train", "train"),
        "validation": ShardWriter(
            OUTPUT_DIR / "validation", "validation"
        ),
    }

    kept = {"train": 0, "validation": 0}
    skipped = 0
    duplicate_count = 0
    processed = 0

    eos_id = tokenizer.eos_id()

    try:
        for parquet_path in parquet_files:
            print(f"\nLecture : {parquet_path.name}")

            parquet = pq.ParquetFile(parquet_path)

            if TEXT_COLUMN not in parquet.schema.names:
                raise ValueError(
                    f"Colonne '{TEXT_COLUMN}' absente de "
                    f"{parquet_path.name}. Colonnes disponibles : "
                    f"{parquet.schema.names}"
                )

            for batch in tqdm(
                parquet.iter_batches(
                    batch_size=PARQUET_BATCH_SIZE,
                    columns=[TEXT_COLUMN],
                ),
                desc=parquet_path.name,
                unit="batch",
            ):
                texts = batch.column(0).to_pylist()

                for value in texts:
                    processed += 1

                    text = clean_text(value)

                    if text is None:
                        skipped += 1
                        continue

                    # Déduplication exacte sur le texte nettoyé.
                    digest = hashlib.sha256(
                        text.encode("utf-8")
                    ).digest()

                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO seen (digest) VALUES (?)",
                        (digest,),
                    )

                    if cursor.rowcount == 0:
                        duplicate_count += 1
                        continue

                    split = get_split(digest)

                    # Utiliser l'EOS existant pour séparer les documents.
                    ids = tokenizer.encode(text, out_type=int)

                    if eos_id >= 0:
                        ids.append(eos_id)

                    if not ids:
                        skipped += 1
                        continue

                    writers[split].write_ids(ids)
                    kept[split] += 1

                    # Commit périodique pour limiter le journal SQLite.
                    if processed % 1000 == 0:
                        connection.commit()

        connection.commit()

        for writer in writers.values():
            writer.finish()

    finally:
        connection.commit()
        connection.close()

    all_shards = (
        writers["train"].shards + writers["validation"].shards
    )

    manifest = {
        "format_version": 1,
        "source": str(INPUT_DIR),
        "parquet_files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
            }
            for path in parquet_files
        ],
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "dtype": "uint16",
        "shard_target_bytes": SHARD_SIZE_BYTES,
        "tokens_per_full_shard": TOKENS_PER_SHARD,
        "validation_percent_target": VALIDATION_PERCENT,
        "documents_processed": processed,
        "documents_kept": kept,
        "documents_skipped": skipped,
        "exact_duplicates_removed": duplicate_count,
        "total_train_tokens": writers["train"].total_tokens,
        "total_validation_tokens": writers["validation"].total_tokens,
        "shards": all_shards,
    }

    manifest_path = OUTPUT_DIR / "manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("PRÉPARATION TERMINÉE")
    print("=" * 60)
    print(f"Documents parcourus : {processed:,}")
    print(f"Documents conservés : {kept}")
    print(f"Documents ignorés : {skipped:,}")
    print(f"Doublons exacts : {duplicate_count:,}")
    print(f"Tokens train : {writers['train'].total_tokens:,}")
    print(f"Tokens validation : {writers['validation'].total_tokens:,}")
    print(f"Nombre total de shards : {len(all_shards)}")
    print(f"Manifeste : {manifest_path}")
    print("\nLe dossier dedup.sqlite3 contient les empreintes de déduplication.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterruption demandée. Le corpus peut être incomplet. "
            "Vérifie les fichiers avant toute réutilisation."
        )
        sys.exit(130)
