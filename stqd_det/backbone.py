"""ResNet-50 + FPN backbone. C2..C5 from torchvision ResNet-50, top-down FPN to P2..P5 (256ch)."""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import FeaturePyramidNetwork


_RESNET50_CHANNELS = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
_STAGE_TO_LEVEL = {"layer1": "P2", "layer2": "P3", "layer3": "P4", "layer4": "P5"}
FPN_LEVELS: List[str] = ["P2", "P3", "P4", "P5"]


def _freeze_batchnorm_stats(model: nn.Module) -> None:
    """Put every BatchNorm in eval mode with requires_grad=False; conv weights stay trainable."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


class ResNet50FPN(nn.Module):
    """ResNet-50 trunk + top-down FPN.

    forward: (B,3,H,W) -> {P2: /4, P3: /8, P4: /16, P5: /32}, all 256ch.
    """

    def __init__(
        self,
        out_channels: int = 256,
        pretrained: bool = True,
        frozen_bn: bool = True,
    ):
        super().__init__()
        weights: Optional[ResNet50_Weights] = ResNet50_Weights.DEFAULT if pretrained else None
        r = resnet50(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3
        self.layer4 = r.layer4

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[_RESNET50_CHANNELS[k] for k in ("layer1", "layer2", "layer3", "layer4")],
            out_channels=out_channels,
        )
        self.out_channels = out_channels

        self._frozen_bn = frozen_bn
        if frozen_bn:
            _freeze_batchnorm_stats(self)

    def train(self, mode: bool = True):  # type: ignore[override]
        # super().train() un-freezes the BN children, so re-freeze after.
        super().train(mode)
        if self._frozen_bn:
            _freeze_batchnorm_stats(self)
        return self

    @property
    def level_names(self) -> List[str]:
        return list(FPN_LEVELS)

    @property
    def level_strides(self) -> Dict[str, int]:
        return {"P2": 4, "P3": 8, "P4": 16, "P5": 32}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        feats = OrderedDict()
        for stage_name, feat in (("layer1", c2), ("layer2", c3), ("layer3", c4), ("layer4", c5)):
            feats[_STAGE_TO_LEVEL[stage_name]] = feat
        pyramid = self.fpn(feats)
        return pyramid


def build_backbone(out_channels: int = 256, pretrained: bool = True, frozen_bn: bool = True) -> ResNet50FPN:
    return ResNet50FPN(out_channels=out_channels, pretrained=pretrained, frozen_bn=frozen_bn)
