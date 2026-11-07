from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class Fresnel(nn.Module):
    def __init__(self, size=128, pitch=10e-6, wavelength=532e-9,
                 distances=(0.0241, 0.0253, 0.0265, 0.0277)):
        super().__init__()
        axis = (torch.arange(size, dtype=torch.float64) - size // 2) * pitch
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        z = torch.tensor(distances, dtype=torch.float64)[:, None, None]
        q = math.pi * (xx.square() + yy.square())[None] / (wavelength * z)
        self.register_buffer("chirp", torch.exp(1j * q).to(torch.complex64))
        self.size = size

    @staticmethod
    def fft(x):
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(
            x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))

    @staticmethod
    def ifft(x):
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(
            x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))

    def forward(self, phase):
        return self.fft(torch.exp(1j * phase[:, None]) * self.chirp[None])

    def adjoint(self, fields):
        return self.ifft(fields) * self.chirp.conj()[None]

    def phase_gradient(self, phase, amplitude):
        fields = self(phase)
        magnitude = fields.abs().clamp_min(1e-7)
        residual = (magnitude - amplitude) * fields / magnitude
        backward = self.adjoint(residual).mean(dim=1)
        return 2 * (torch.exp(-1j * phase) * backward).imag


def normalize_targets(targets):
    rms = targets.square().mean(dim=(-2, -1), keepdim=True).sqrt()
    scale = torch.where(rms < 1e-7, torch.ones_like(rms), rms)
    return targets / scale, scale


def quantize_phase(phase):
    wrapped = torch.remainder(phase + math.pi, 2 * math.pi) - math.pi
    code = torch.round((wrapped + math.pi) * (255 / (2 * math.pi)))
    return code.to(torch.uint8), code * (2 * math.pi / 255) - math.pi


class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, padding=1),
                                 nn.SiLU(),
                                 nn.Conv2d(outgoing, outgoing, 3, padding=1),
                                 nn.SiLU())

    def forward(self, x):
        return self.net(x)


class DirectUNet(nn.Module):
    def __init__(self, size=128, width=12):
        super().__init__()
        coords = torch.linspace(-1, 1, size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("coords", torch.stack((xx, yy))[None])
        self.enc1 = ConvBlock(6, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.middle = ConvBlock(width * 4, width * 8)
        self.dec3 = ConvBlock(width * 12, width * 4)
        self.dec2 = ConvBlock(width * 6, width * 2)
        self.dec1 = ConvBlock(width * 3, width)
        self.output = nn.Conv2d(width, 1, 1)

    def forward(self, target):
        x = torch.cat((target, self.coords.expand(target.shape[0], -1, -1, -1)), 1)
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.enc3(F.avg_pool2d(b, 2))
        d = self.middle(F.avg_pool2d(c, 2))
        d = self.dec3(torch.cat((F.interpolate(d, size=c.shape[-2:], mode="bilinear", align_corners=False), c), 1))
        d = self.dec2(torch.cat((F.interpolate(d, size=b.shape[-2:], mode="bilinear", align_corners=False), b), 1))
        d = self.dec1(torch.cat((F.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False), a), 1))
        return self.output(d)[:, 0]


class LearnedUnrolled(nn.Module):
    def __init__(self, physics=None, size=128, stages=8, width=12):
        super().__init__()
        self.physics = physics if physics is not None else Fresnel(size=size)
        self.stages = stages
        generator = torch.Generator().manual_seed(76017)
        carrier_phase = 2 * math.pi * torch.rand((1, 4, size, size), generator=generator)
        self.register_buffer("carrier", torch.exp(1j * carrier_phase))
        self.step_logits = nn.Parameter(torch.full((stages,), 0.5))
        self.momentum_logits = nn.Parameter(torch.full((stages,), -1.5))
        self.correction_scale = nn.Parameter(torch.full((stages,), 0.05))
        self.corrector = nn.Sequential(nn.Conv2d(4, width, 3, padding=1), nn.SiLU(),
                                       nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                       nn.Conv2d(width, 1, 3, padding=1))
        nn.init.zeros_(self.corrector[-1].weight)
        nn.init.zeros_(self.corrector[-1].bias)

    def initial(self, target):
        field = self.physics.adjoint(target * self.carrier).sum(dim=1)
        return torch.angle(field + 1e-7)

    def forward(self, target):
        phase = self.initial(target)
        velocity = torch.zeros_like(phase)
        for index in range(self.stages):
            gradient = self.physics.phase_gradient(phase, target)
            features = torch.stack((phase.sin(), phase.cos(), gradient, velocity), 1)
            correction = self.corrector(features)[:, 0].tanh()
            velocity = torch.sigmoid(self.momentum_logits[index]) * velocity + gradient
            phase = phase - F.softplus(self.step_logits[index]) * velocity
            phase = phase + self.correction_scale[index].tanh() * correction
        return phase
