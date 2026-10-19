from __future__ import annotations

import argparse
from pathlib import Path
import sentencepiece as spm


# ============================================================
# CONFIGURATION PAR DÉFAUT
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "V2" / "tokenizer"
)

DEFAULT_INPUT_FILES = [
    PROJECT_DIR / "corpus" / "train-00001-of-00004.parquet","train-00000-of-00004.parquet", "train-00002-of-00004.parquet", "train-00003-of-00004.parquet",
    PROJECT_DIR / "corpus" / "TinyStories-valid.txt",
    PROJECT_DIR / "corpus.txt",
]


# ============================================================
# TROUVER LE CORPUS
# ============================================================

def find_default_corpus() -> Path:
    for path in DEFAULT_INPUT_FILES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "\nAucun corpus trouvé automatiquement.\n\n"
        "Place ton corpus dans l'un de ces emplacements :\n"
        + "\n".join(f"  - {path}" for path in DEFAULT_INPUT_FILES)
        + "\n\n"
        "Ou utilise --input chemin\\vers\\ton_corpus.txt"
    )


# ============================================================
# ENTRAÎNEMENT DU TOKENIZER
# ============================================================

def train_tokenizer(
    input_path: Path,
    output_dir: Path,
    vocab_size: int,
) -> None:

    if not input_path.exists():
        raise FileNotFoundError(
            f"Corpus introuvable : {input_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_prefix = output_dir / "tokenizer"

    print("=" * 70)
    print("TRAINING BPE TOKENIZER")
    print("=" * 70)

    print(f"Corpus       : {input_path}")
    print(f"Sortie       : {output_dir}")
    print(f"Vocab size   : {vocab_size}")
    print()

    file_size_mb = input_path.stat().st_size / (
        1024 * 1024
    )

    print(
        f"Taille corpus : {file_size_mb:.2f} MB"
    )
    print()

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",

        # Tokens spéciaux
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,

        pad_piece="<PAD>",
        unk_piece="<UNK>",
        bos_piece="<BOS>",
        eos_piece="<EOS>",

        # Garde les caractères nécessaires
        character_coverage=1.0,

        # Permet de conserver les espaces
        add_dummy_prefix=True,

        # Normalisation Unicode standard
        normalization_rule_name="nmt_nfkc",

        # Réduit les surprises liées aux très petits corpus
        hard_vocab_limit=False,

        # Ne supprime pas les espaces
        remove_extra_whitespaces=False,
    )

    print()
    print("=" * 70)
    print("TOKENIZER TERMINÉ")
    print("=" * 70)

    print()
    print("Fichiers créés :")
    print(f"  {model_prefix}.model")
    print(f"  {model_prefix}.vocab")


# ============================================================
# TEST DU TOKENIZER
# ============================================================

def test_tokenizer(model_path: Path) -> None:

    print()
    print("=" * 70)
    print("TEST DU TOKENIZER")
    print("=" * 70)

    sp = spm.SentencePieceProcessor(
        model_file=str(model_path)
    )

    test_sentences = [
        "Hello, my name is Antoine.",
        "I am interested in astrophysics and artificial intelligence.",
        "A black hole is a region of spacetime from which nothing can escape.",
        "Bonjour, je construis mon propre modèle de langage.",
        "2 + 2 = 4",
    ]

    for text in test_sentences:

        ids = sp.encode(
            text,
            out_type=int,
        )

        tokens = sp.encode(
            text,
            out_type=str,
        )

        decoded = sp.decode(ids)

        print()
        print("TEXT:")
        print(text)

        print()
        print("TOKENS:")
        print(tokens)

        print()
        print("IDS:")
        print(ids)

        print()
        print("DECODED:")
        print(decoded)

        print()
        print(
            "IDENTIQUE :",
            decoded == text
        )

        print("-" * 70)

    print()
    print(
        f"Vocabulaire réel : {sp.get_piece_size()} tokens"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Entraîne le tokenizer BPE de la V2."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Chemin vers le corpus .txt",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Dossier de sortie",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16000,
        help="Taille du vocabulaire",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Corpus
    # --------------------------------------------------------

    if args.input:
        input_path = Path(args.input).expanduser().resolve()
    else:
        input_path = find_default_corpus()

    # --------------------------------------------------------
    # Sortie
    # --------------------------------------------------------

    output_dir = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    # --------------------------------------------------------
    # Entraînement
    # --------------------------------------------------------

    train_tokenizer(
        input_path=input_path,
        output_dir=output_dir,
        vocab_size=args.vocab_size,
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    model_path = (
        output_dir / "tokenizer.model"
    )

    test_tokenizer(model_path)


if __name__ == "__main__":
    main()