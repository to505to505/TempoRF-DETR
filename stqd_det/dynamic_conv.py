"""Dynamic Convolution (Chen et al., CVPR 2020): K kernels mixed by
attention weights from a GAP descriptor.

Not the same as DiffusionDet's DynamicHead in stqd_det.decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicConv(nn.Module):
    """K-kernel attention-mixed convolution. padding defaults to
    kernel_size // 2 (same spatial size)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        K: int = 4,
        reduction: int = 4,
        bias: bool = True,
        stride: int = 1,
        padding: int = None,
        groups: int = 1,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K = K
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # K kernels: (out, in/groups, kh, kw) each
        weight = torch.empty(
            K, out_channels, in_channels // groups, kernel_size, kernel_size
        )
        for k in range(K):
            nn.init.kaiming_uniform_(weight[k], a=5 ** 0.5)
        self.weight = nn.Parameter(weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_channels))
        else:
            self.register_parameter("bias", None)

        hidden = max(in_channels // reduction, K)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, K),
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """x: (B, C_in, H, W) -> (B, C_out, H', W')."""
        B = x.shape[0]
        logits = self.attention(x) / max(temperature, 1e-6)        # (B, K)
        pi = F.softmax(logits, dim=-1)

        # Per-sample kernel, then grouped conv (groups=B) does all samples
        # in one F.conv2d call.
        w = torch.einsum("bk,koihw->boihw", pi, self.weight)
        w = w.reshape(B * self.out_channels, self.in_channels // self.groups,
                      self.kernel_size, self.kernel_size)

        if self.bias is not None:
            b = torch.einsum("bk,ko->bo", pi, self.bias).reshape(-1)
        else:
            b = None

        x = x.reshape(1, B * self.in_channels, x.shape[2], x.shape[3])
        out = F.conv2d(
            x,
            w,
            bias=b,
            stride=self.stride,
            padding=self.padding,
            groups=B * self.groups,
        )
        out = out.reshape(B, self.out_channels, out.shape[2], out.shape[3])
        return out
