from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    x
      ├── gate ── SiLU ──┐
      │                   ×
      └── up ─────────────┘
                          ↓
                        down
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
    ) -> None:

        super().__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.gate_proj = nn.Linear(
            d_model,
            hidden_dim,
            bias=False,
        )

        self.up_proj = nn.Linear(
            d_model,
            hidden_dim,
            bias=False,
        )

        self.down_proj = nn.Linear(
            hidden_dim,
            d_model,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        gate = F.silu(
            self.gate_proj(x)
        )

        up = self.up_proj(x)

        hidden = (
            gate * up
        )

        return self.down_proj(
            hidden
        )


def main() -> None:

    print("=" * 70)
    print("TEST SWIGLU")
    print("=" * 70)

    batch_size = 2
    sequence_length = 512
    d_model = 768
    hidden_dim = 2176

    model = SwiGLU(
        d_model=d_model,
        hidden_dim=hidden_dim,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    y = model(x)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Input shape :",
        tuple(x.shape),
    )

    print(
        "Output shape:",
        tuple(y.shape),
    )

    print(
        "Paramètres :",
        f"{parameters:,}",
    )

    assert x.shape == y.shape

    assert torch.isfinite(y).all()

    print()
    print("SWIGLU OK")


if __name__ == "__main__":
    main()