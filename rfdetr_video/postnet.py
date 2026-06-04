"""Temporal post-net: per-slot self-attention over frozen RF-DETR decoder
hidden states, mixing across the T frames of a window. The only trainable
part of the late-temporal-modeling baseline; out_proj is zero-init so it
starts as identity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalPostNet(nn.Module):
    """Per-slot temporal self-attention. hs (B*T, Q, D) -> (B*T, Q, D)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        n_layers: int = 1,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.d_model = d_model
        self.n_layers = n_layers

        self.norms = nn.ModuleList(
            nn.LayerNorm(d_model) for _ in range(n_layers)
        )
        self.attns = nn.ModuleList(
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(n_layers)
        )
        for attn in self.attns:
            nn.init.zeros_(attn.out_proj.weight)
            nn.init.zeros_(attn.out_proj.bias)

    def forward(self, hs: torch.Tensor, B: int, T: int) -> torch.Tensor:
        if hs.dim() != 3:
            raise ValueError(
                f"TemporalPostNet expects (B*T, Q, D), got {tuple(hs.shape)}",
            )
        BT, Q, D = hs.shape
        if BT != B * T:
            raise ValueError(
                f"hs.shape[0]={BT} does not match B*T={B * T}",
            )
        if D != self.d_model:
            raise ValueError(
                f"hs feature dim {D} does not match d_model {self.d_model}",
            )

        x = hs.reshape(B, T, Q, D).permute(0, 2, 1, 3).contiguous()
        x = x.reshape(B * Q, T, D)
        for norm, attn in zip(self.norms, self.attns):
            x_n = norm(x)
            attn_out, _ = attn(x_n, x_n, x_n)
            x = x + attn_out
        x = x.reshape(B, Q, T, D).permute(0, 2, 1, 3).contiguous()
        return x.reshape(B * T, Q, D)
