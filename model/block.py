from __future__ import annotations

import torch
import torch.nn as nn

from rmsnorm import RMSNorm
from attention import MultiHeadSelfAttention
from swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """
    Un bloc Transformer complet :

        x
        │
        ├── RMSNorm
        │
        ├── Self-Attention + RoPE
        │
        └── Residual Add
                │
                ├── RMSNorm
                │
                ├── SwiGLU
                │
                └── Residual Add
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_hidden_dim: int,
        max_seq_len: int,
        rope_base: float = 10000.0,
    ) -> None:

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.max_seq_len = max_seq_len

        # ----------------------------------------------------
        # NORMALISATION
        # ----------------------------------------------------

        self.norm1 = RMSNorm(
            d_model
        )

        self.norm2 = RMSNorm(
            d_model
        )

        # ----------------------------------------------------
        # ATTENTION
        # ----------------------------------------------------

        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )

        # ----------------------------------------------------
        # FEED-FORWARD
        # ----------------------------------------------------

        self.feed_forward = SwiGLU(
            d_model=d_model,
            hidden_dim=ffn_hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Attention block
        # ----------------------------------------------------

        x = (
            x
            + self.attention(
                self.norm1(x)
            )
        )

        # ----------------------------------------------------
        # Feed-forward block
        # ----------------------------------------------------

        x = (
            x
            + self.feed_forward(
                self.norm2(x)
            )
        )

        return x


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
    print("TEST TRANSFORMER BLOCK")
    print("=" * 70)

    # Configuration réelle de notre V2
    batch_size = 2
    sequence_length = 128
    d_model = 768
    num_heads = 12
    ffn_hidden_dim = 2176
    max_seq_len = 512

    print()
    print("Configuration :")
    print("  d_model       :", d_model)
    print("  heads         :", num_heads)
    print(
        "  head_dim      :",
        d_model // num_heads,
    )
    print(
        "  FFN hidden    :",
        ffn_hidden_dim,
    )
    print(
        "  context       :",
        max_seq_len,
    )

    # --------------------------------------------------------
    # Modèle
    # --------------------------------------------------------

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        ffn_hidden_dim=ffn_hidden_dim,
        max_seq_len=max_seq_len,
    )

    # --------------------------------------------------------
    # Entrée
    # --------------------------------------------------------

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    print()
    print(
        "Input shape :",
        tuple(x.shape),
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    y = block(x)

    print(
        "Output shape:",
        tuple(y.shape),
    )

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    assert x.shape == y.shape

    assert torch.isfinite(
        y
    ).all()

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    loss = y.pow(2).mean()

    loss.backward()

    gradients_ok = all(
        p.grad is not None
        for p in block.parameters()
        if p.requires_grad
    )

    assert gradients_ok

    # --------------------------------------------------------
    # Résultats
    # --------------------------------------------------------

    print()
    print(
        "Paramètres :",
        f"{count_parameters(block):,}",
    )

    print(
        "Loss test  :",
        f"{loss.item():.6f}",
    )

    print(
        "Gradients  :",
        gradients_ok,
    )

    print()
    print("=" * 70)
    print("TRANSFORMER BLOCK OK")
    print("=" * 70)


if __name__ == "__main__":
    main()