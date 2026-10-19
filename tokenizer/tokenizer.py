from __future__ import annotations

from pathlib import Path
from typing import List

import sentencepiece as spm


class BPETokenizer:
    """
    Tokenizer BPE basé sur SentencePiece.

    Utilisation :

        tokenizer = BPETokenizer("tokenizer.model")

        ids = tokenizer.encode("Hello world")
        text = tokenizer.decode(ids)
    """

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:

        self.model_path = Path(
            model_path
        ).expanduser().resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Tokenizer introuvable : "
                f"{self.model_path}"
            )

        self.sp = spm.SentencePieceProcessor(
            model_file=str(
                self.model_path
            )
        )

        self.vocab_size = (
            self.sp.get_piece_size()
        )

        self.pad_id = (
            self.sp.pad_id()
        )

        self.unk_id = (
            self.sp.unk_id()
        )

        self.bos_id = (
            self.sp.bos_id()
        )

        self.eos_id = (
            self.sp.eos_id()
        )

    # ========================================================
    # ENCODE
    # ========================================================

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:

        if not isinstance(text, str):
            raise TypeError(
                "text doit être une chaîne de caractères."
            )

        ids = self.sp.encode(
            text,
            out_type=int,
        )

        if add_bos:
            ids.insert(
                0,
                self.bos_id,
            )

        if add_eos:
            ids.append(
                self.eos_id,
            )

        return ids

    # ========================================================
    # DECODE
    # ========================================================

    def decode(
        self,
        ids: List[int],
    ) -> str:

        if not isinstance(ids, list):
            ids = list(ids)

        return self.sp.decode(
            ids
        )

    # ========================================================
    # TOKENS
    # ========================================================

    def encode_pieces(
        self,
        text: str,
    ) -> List[str]:

        return self.sp.encode(
            text,
            out_type=str,
        )

    # ========================================================
    # INFORMATIONS
    # ========================================================

    def piece_to_id(
        self,
        piece: str,
    ) -> int:

        return self.sp.piece_to_id(
            piece
        )

    def id_to_piece(
        self,
        token_id: int,
    ) -> str:

        return self.sp.id_to_piece(
            token_id
        )

    # ========================================================
    # TEST
    # ========================================================

    def test(
        self,
        text: str,
    ) -> None:

        ids = self.encode(text)

        pieces = self.encode_pieces(
            text
        )

        decoded = self.decode(ids)

        print("=" * 70)
        print("TOKENIZER TEST")
        print("=" * 70)

        print()
        print("Original :")
        print(text)

        print()
        print("Pieces :")
        print(pieces)

        print()
        print("IDs :")
        print(ids)

        print()
        print("Decoded :")
        print(decoded)

        print()
        print(
            "Round-trip :",
            decoded == text
        )

        print()
        print(
            "Nombre de tokens :",
            len(ids)
        )

        print(
            "Vocab size :",
            self.vocab_size
        )


# ============================================================
# TEST DIRECT
# ============================================================

if __name__ == "__main__":

    tokenizer_path = (
        Path(__file__).resolve().parent
        / "tokenizer.model"
    )

    tokenizer = BPETokenizer(
        tokenizer_path
    )

    tests = [
        
        "Bonjour, je construis un modèle qui va révolutionner le monde.",
    ]

    for text in tests:
        tokenizer.test(text)
        print()