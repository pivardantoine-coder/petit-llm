from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RotaryEmbedding


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention avec RoPE et
    scaled_dot_product_attention de PyTorch.

    Entrée :
        [batch, sequence, d_model]

    Sortie :
        [batch, sequence, d_model]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_base: float = 10000.0,
    ) -> None:

        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model doit être divisible par num_heads."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len

        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim doit être pair pour RoPE."
            )

        # ----------------------------------------------------
        # QKV
        # ----------------------------------------------------

        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        # ----------------------------------------------------
        # RoPE
        # ----------------------------------------------------

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

    def _split_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        [B, S, D]
        →
        [B, H, S, Hd]
        """

        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def _merge_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        [B, H, S, Hd]
        →
        [B, S, D]
        """

        batch_size, _, seq_len, _ = x.shape

        x = x.transpose(
            1,
            2,
        ).contiguous()

        return x.view(
            batch_size,
            seq_len,
            self.d_model,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "L'entrée doit avoir la forme "
                "[batch, sequence, d_model]."
            )

        batch_size, seq_len, d_model = x.shape

        if d_model != self.d_model:
            raise ValueError(
                f"d_model incorrect : "
                f"{d_model} != {self.d_model}"
            )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Séquence trop longue : "
                f"{seq_len} > {self.max_seq_len}"
            )

        # ----------------------------------------------------
        # Projections Q / K / V
        # ----------------------------------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ----------------------------------------------------
        # Multi-head
        # ----------------------------------------------------

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # ----------------------------------------------------
        # RoPE
        # ----------------------------------------------------

        q = self.rope(q)
        k = self.rope(k)

        # ----------------------------------------------------
        # Scaled Dot-Product Attention
        # ----------------------------------------------------
        #
        # is_causal=True :
        # le token t ne peut pas regarder t+1, t+2, ...
        #
        # PyTorch choisit automatiquement un kernel
        # optimisé lorsque cela est possible.
        # ----------------------------------------------------

        attention_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        # ----------------------------------------------------
        # Fusion des heads
        # ----------------------------------------------------

        output = self._merge_heads(
            attention_output
        )

        # ----------------------------------------------------
        # Projection finale
        # ----------------------------------------------------

        output = self.out_proj(
            output
        )

        return output


def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:

    print("=" * 70)
    print("TEST MULTI-HEAD SELF-ATTENTION")
    print("=" * 70)

    batch_size = 2
    sequence_length = 128
    d_model = 768
    num_heads = 12
    max_seq_len = 512

    print()
    print("Configuration :")
    print(
        "  d_model    :",
        d_model,
    )
    print(
        "  heads      :",
        num_heads,
    )
    print(
        "  head_dim   :",
        d_model // num_heads,
    )
    print(
        "  context    :",
        max_seq_len,
    )

    model = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
    )

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

    y = model(x)

    print(
        "Output shape:",
        tuple(y.shape),
    )

    # --------------------------------------------------------
    # Tests de forme
    # --------------------------------------------------------

    assert x.shape == y.shape

    # --------------------------------------------------------
    # Test valeurs
    # --------------------------------------------------------

    assert torch.isfinite(y).all()

    # --------------------------------------------------------
    # Test backward
    # --------------------------------------------------------

    loss = y.pow(2).mean()

    loss.backward()

    # --------------------------------------------------------
    # Vérifier qu'au moins un gradient existe
    # --------------------------------------------------------

    gradients_ok = all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert gradients_ok

    print()
    print(
        "Paramètres :",
        f"{count_parameters(model):,}",
    )

    print(
        "Loss test :",
        float(loss.item()),
    )

    print(
        "Gradients :",
        gradients_ok,
    )

    print()
    print("=" * 70)
    print("ATTENTION OK")
    print("=" * 70)


if __name__ == "__main__":
    main()