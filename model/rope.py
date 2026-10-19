from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    x doit avoir la forme :

        [batch, heads, sequence, head_dim]

    head_dim doit être pair.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
    ) -> None:

        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                "head_dim doit être pair pour RoPE."
            )

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Fréquences :
        #
        # 1 / base^(2i / head_dim)
        #
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        freqs = torch.outer(
            positions,
            inv_freq,
        )

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # [1, 1, seq, head_dim/2]
        self.register_buffer(
            "cos_cached",
            cos.unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            sin.unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                "RoPE attend un tenseur "
                "[batch, heads, sequence, head_dim]."
            )

        batch_size, num_heads, seq_len, head_dim = (
            x.shape
        )

        if head_dim != self.head_dim:
            raise ValueError(
                f"head_dim incorrect : "
                f"{head_dim} != {self.head_dim}"
            )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Séquence trop longue : "
                f"{seq_len} > {self.max_seq_len}"
            )

        cos = self.cos_cached[
            :, :, :seq_len, :
        ].to(
            device=x.device,
            dtype=x.dtype,
        )

        sin = self.sin_cached[
            :, :, :seq_len, :
        ].to(
            device=x.device,
            dtype=x.dtype,
        )

        # Séparer les dimensions paires / impaires.
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = (
            x_even * cos
            - x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        output = torch.empty_like(x)

        output[..., 0::2] = rotated_even
        output[..., 1::2] = rotated_odd

        return output


def main() -> None:

    print("=" * 70)
    print("TEST ROPE")
    print("=" * 70)

    batch_size = 2
    num_heads = 12
    sequence_length = 512
    head_dim = 64

    rope = RotaryEmbedding(
        head_dim=head_dim,
        max_seq_len=512,
    )

    x = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
    )

    y = rope(x)

    print(
        "Input shape :",
        tuple(x.shape),
    )

    print(
        "Output shape:",
        tuple(y.shape),
    )

    assert x.shape == y.shape

    assert torch.isfinite(y).all()

    difference = (
        x - y
    ).abs().mean().item()

    print(
        "Différence moyenne :",
        difference,
    )

    assert difference > 0

    print()
    print("ROPE OK")


if __name__ == "__main__":
    main()