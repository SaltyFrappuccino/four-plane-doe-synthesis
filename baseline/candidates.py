from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from baseline.models import ConvBlock, DirectUNet, Fresnel, LearnedUnrolled
from baseline.robust_unrolled import RobustTailUnrolled
from baseline.ssim_unrolled import AdamSSIMUnrolled, SSIMUnrolled


class FourierUNet(DirectUNet):

    def __init__(self, width=12):
        super().__init__(width=width)
        self.physics = Fresnel()
        generator = torch.Generator().manual_seed(76017)
        self.register_buffer("carrier", torch.exp(2j * torch.pi * torch.rand(
            (1, 4, 128, 128), generator=generator)))
        self.enc1 = ConvBlock(10, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, target):
        fields = self.physics.adjoint(target * self.carrier)
        initial = torch.angle(fields.sum(dim=1) + 1e-7)
        features = torch.cat((fields.real, fields.imag), dim=1)
        return initial + super().forward(features)


class ImageUnrolled(LearnedUnrolled):

    def __init__(self, stages=16, width=12):
        super().__init__(stages=stages, width=width)
        del self.corrector
        self.corrector = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )
        nn.init.zeros_(self.corrector[-1].weight)
        nn.init.zeros_(self.corrector[-1].bias)

    def forward(self, target):
        phase = self.initial(target)
        velocity = torch.zeros_like(phase)
        batch, planes, height, width = target.shape
        for index in range(self.stages):
            fields = self.physics(phase)
            magnitude = fields.abs().clamp_min(1e-7)
            residual = magnitude - target
            features = torch.stack((magnitude, target, residual), dim=2)
            adjustment = self.corrector(features.reshape(-1, 3, height, width))
            adjustment = adjustment.reshape(batch, planes, height, width).tanh()
            residual = residual + self.correction_scale[index].tanh() * adjustment
            backward = self.physics.adjoint(residual * fields / magnitude).mean(dim=1)
            gradient = 2 * (torch.exp(-1j * phase) * backward).imag
            velocity = self.momentum_logits[index].sigmoid() * velocity + gradient
            phase = phase - F.softplus(self.step_logits[index]) * velocity
        return phase


class CarrierUnrolled(LearnedUnrolled):

    def __init__(self, stages=16, width=12):
        super().__init__(stages=stages, width=width)
        self.carrier_phase = nn.Parameter(self.carrier.angle())
        del self.carrier

    def initial(self, target):
        fields = self.physics.adjoint(target * torch.exp(1j * self.carrier_phase))
        return torch.angle(fields.sum(dim=1) + 1e-7)


def build_model(config: dict) -> nn.Module:
    name = config.get("name", "learned_unrolled")
    width = int(config.get("width", 12))
    stages = int(config.get("stages", 8))
    if name == "learned_unrolled":
        return LearnedUnrolled(stages=stages, width=width)
    if name == "image_unrolled":
        return ImageUnrolled(stages=stages, width=width)
    if name == "carrier_unrolled":
        return CarrierUnrolled(stages=stages, width=width)
    if name == "ssim_unrolled":
        return SSIMUnrolled(stages=stages, ssim_weight=config.get("ssim_weight", 0.1))
    if name == "adam_ssim_unrolled":
        return AdamSSIMUnrolled(stages=stages, ssim_weight=config.get("ssim_weight", 1.0),
                                step_size=config.get("step_size", 0.08))
    if name == "robust_tail_unrolled":
        return RobustTailUnrolled(stages=stages, prefix_stages=config["prefix_stages"],
                                  ssim_weight=config["ssim_weight"], step_size=config["step_size"],
                                  minmax_weight=config["minmax_weight"])
    if name == "direct_unet":
        model = DirectUNet(width=width)
        model.physics = Fresnel()
        return model
    if name == "fourier_unet":
        return FourierUNet(width=width)
    raise ValueError(f"Unknown architecture: {name}")
