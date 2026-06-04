"""Global Feature Enhancement block: 3-frame self-attention window + DynamicConv
fusion on the FPN top-level feature map. Output keeps the input shape (replaces P5).
"""

from typing import Tuple

import torch
import torch.nn as nn

from .dynamic_conv import DynamicConv


class GFE(nn.Module):
    """3-frame MHA window + DynamicConv residual on a per-frame feature map."""

    def __init__(
        self,
        channels: int = 256,
        heads: int = 8,
        dc_kernels: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.heads = heads

        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_mha = nn.LayerNorm(channels)

        # in is concat([f_n, f'_n]) = 2C channels
        self.dc = DynamicConv(
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=3,
            K=dc_kernels,
        )
        self.fc = nn.Linear(channels, channels)
        self.ln_dc = nn.LayerNorm(channels)

    @staticmethod
    def _build_window(feat: torch.Tensor) -> torch.Tensor:
        """Replicate-padded 3-frame window. (B,T,...) -> (B,T,3,...) with dim2 = [prev,curr,next]."""
        if feat.dim() < 2:
            raise ValueError("feat must have at least (B, T, ...) shape")
        T = feat.shape[1]
        prev = torch.cat([feat[:, :1], feat[:, :-1]], dim=1) if T > 1 else feat
        nxt = torch.cat([feat[:, 1:], feat[:, -1:]], dim=1) if T > 1 else feat
        return torch.stack([prev, feat, nxt], dim=2)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, T, C, H, W) -> (B, T, C, H, W)."""
        if feat.dim() != 5:
            raise ValueError(f"expected (B,T,C,H,W); got {tuple(feat.shape)}")
        B, T, C, H, W = feat.shape
        if C != self.channels:
            raise ValueError(f"channel mismatch: got C={C}, expected {self.channels}")

        # MHA: query = current frame, kv = 3-frame window of spatial tokens
        feat_tok = feat.permute(0, 1, 3, 4, 2).contiguous()       # (B, T, H, W, C)
        feat_tok = feat_tok.view(B, T, H * W, C)                  # (B, T, HW, C)
        win = self._build_window(feat_tok)                        # (B, T, 3, HW, C)
        kv = win.reshape(B * T, 3 * H * W, C)
        q = feat_tok.reshape(B * T, H * W, C)
        attn_out, _ = self.mha(q, kv, kv, need_weights=False)     # (B*T, HW, C)
        v_prime = self.ln_mha(attn_out + q)                       # (B*T, HW, C)

        f_prime = v_prime.view(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

        # DynamicConv on concat(f_n, f'_n)
        concat_in = torch.cat([feat, f_prime], dim=2)             # (B,T,2C,H,W)
        concat_in = concat_in.view(B * T, 2 * C, H, W)
        dc_out = self.dc(concat_in)                               # (B*T, C, H, W)
        dc_out = dc_out.view(B, T, C, H, W)

        dc_tok = dc_out.permute(0, 1, 3, 4, 2).contiguous()       # (B,T,H,W,C)
        dc_tok = self.fc(dc_tok)
        feat_tok_perm = feat.permute(0, 1, 3, 4, 2)
        f_dprime = self.ln_dc(dc_tok + feat_tok_perm)             # (B,T,H,W,C)

        return f_dprime.permute(0, 1, 4, 2, 3).contiguous()       # (B,T,C,H,W)
