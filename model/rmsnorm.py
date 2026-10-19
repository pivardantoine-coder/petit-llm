from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.

    Entrée :
        (..., dim)

    Sortie :
        (..., dim)
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:

        super().__init__()

        self.dim = dim
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # RMS = sqrt(mean(x²))
        rms = torch.rsqrt(
            x.pow(2).mean(
                dim=-1,
                keepdim=True,
            )
            + self.eps
        )

        return (
            x
            * rms
            * self.weight
        )


def main() -> None:

    print("=" * 70)
    print("TEST RMSNORM")
    print("=" * 70)

    batch_size = 2
    sequence_length = 512
    dim = 768

    x = torch.randn(
        batch_size,
        sequence_length,
        dim,
    )

    norm = RMSNorm(dim)

    y = norm(x)

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print(
        "Paramètres :",
        sum(
            p.numel()
            for p in norm.parameters()
        ),
    )

    assert x.shape == y.shape

    assert torch.isfinite(y).all()

    print()
    print("RMSNORM OK")


if __name__ == "__main__":
    main()