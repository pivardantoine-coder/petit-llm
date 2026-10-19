from __future__ import annotations

import torch
import torch.nn as nn

from rmsnorm import RMSNorm
from block import TransformerBlock


class TransformerLM(nn.Module):
    """
    Transformer decoder-only pour next-token prediction.

    Architecture :

        token IDs
             ↓
        token embedding
             ↓
        24 Transformer Blocks
             ↓
          RMSNorm
             ↓
        output head
             ↓
        vocab logits
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 24,
        ffn_hidden_dim: int = 2176,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ) -> None:

        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model doit être divisible par num_heads."
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_hidden_dim = ffn_hidden_dim
        self.max_seq_len = max_seq_len

        # ----------------------------------------------------
        # TOKEN EMBEDDING
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        # ----------------------------------------------------
        # TRANSFORMER BLOCKS
        # ----------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    max_seq_len=max_seq_len,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )

        # ----------------------------------------------------
        # FINAL NORMALIZATION
        # ----------------------------------------------------

        self.final_norm = RMSNorm(
            d_model
        )

        # ----------------------------------------------------
        # LANGUAGE MODEL HEAD
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        # ----------------------------------------------------
        # WEIGHT TYING
        # ----------------------------------------------------
        #
        # Même matrice pour :
        #   token embedding
        #   output projection
        #
        # Cela économise beaucoup de paramètres.
        # ----------------------------------------------------

        self.lm_head.weight = (
            self.token_embedding.weight
        )

        # ----------------------------------------------------
        # INITIALISATION
        # ----------------------------------------------------

        self.apply(
            self._init_weights
        )

    def _init_weights(
        self,
        module: nn.Module,
    ) -> None:

        if isinstance(
            module,
            nn.Linear,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:

        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids doit avoir la forme "
                "[batch, sequence]."
            )

        batch_size, seq_len = (
            token_ids.shape
        )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Séquence trop longue : "
                f"{seq_len} > {self.max_seq_len}"
            )

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        x = self.token_embedding(
            token_ids
        )

        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        for block in self.blocks:
            x = block(x)

        # ----------------------------------------------------
        # FINAL NORM
        # ----------------------------------------------------

        x = self.final_norm(x)

        # ----------------------------------------------------
        # LOGITS
        # ----------------------------------------------------

        logits = self.lm_head(x)

        return logits


def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main() -> None:

    print("=" * 70)
    print("TRANSFORMER V2")
    print("=" * 70)

    config = {
        "vocab_size": 16000,
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 24,
        "ffn_hidden_dim": 2176,
        "max_seq_len": 512,
    }

    print()
    print("Configuration :")

    for key, value in config.items():
        print(
            f"  {key:18s}: {value}"
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = TransformerLM(
        **config
    )

    parameters = count_parameters(
        model
    )

    print()
    print(
        "Paramètres totaux :",
        f"{parameters:,}",
    )

    print(
        "Paramètres (M) :",
        f"{parameters / 1_000_000:.2f}M",
    )

    # --------------------------------------------------------
    # TEST FORWARD
    # --------------------------------------------------------

    batch_size = 2
    sequence_length = 128

    x = torch.randint(
        0,
        config["vocab_size"],
        (
            batch_size,
            sequence_length,
        ),
        dtype=torch.long,
    )

    print()
    print(
        "Input shape :",
        tuple(x.shape),
    )

    logits = model(x)

    print(
        "Logits shape:",
        tuple(logits.shape),
    )

    expected_shape = (
        batch_size,
        sequence_length,
        config["vocab_size"],
    )

    assert logits.shape == expected_shape

    assert torch.isfinite(
        logits
    ).all()

    # --------------------------------------------------------
    # TEST LOSS
    # --------------------------------------------------------

    targets = torch.randint(
        0,
        config["vocab_size"],
        (
            batch_size,
            sequence_length,
        ),
        dtype=torch.long,
    )

    loss = nn.functional.cross_entropy(
        logits.reshape(
            -1,
            config["vocab_size"],
        ),
        targets.reshape(-1),
    )

    print(
        "Loss test :",
        f"{loss.item():.6f}",
    )

    # --------------------------------------------------------
    # BACKWARD
    # --------------------------------------------------------

    loss.backward()

    gradients_ok = all(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Gradients :",
        gradients_ok,
    )

    assert gradients_ok

    print()
    print("=" * 70)
    print("TRANSFORMER V2 OK")
    print("=" * 70)


if __name__ == "__main__":
    main()