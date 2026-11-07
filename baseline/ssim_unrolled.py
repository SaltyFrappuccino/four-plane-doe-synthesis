from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from baseline.models import Fresnel, LearnedUnrolled


def ssim_input_gradient(prediction: torch.Tensor,
                        target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Expected equal [batch,planes,height,width] arrays")
    batch, planes, height, width = prediction.shape
    if min(height, width) < 11:
        raise ValueError("Gaussian11 SSIM requires spatial dimensions >=11")
    axis = torch.arange(-5, 6, dtype=prediction.dtype, device=prediction.device)
    kernel = torch.exp(-0.5 * (axis / 1.5).square())
    kernel = kernel / kernel.sum()
    horizontal, vertical = kernel.reshape(1, 1, 1, 11), kernel.reshape(1, 1, 11, 1)
    moments = torch.stack((prediction, target, prediction.square(),
                           target.square(), prediction * target), dim=2)
    moments = moments.reshape(batch * planes * 5, 1, height, width)
    moments = F.conv2d(F.conv2d(moments, horizontal), vertical)
    moments = moments.reshape(batch, planes, 5, height - 10, width - 10)
    mean_p, mean_t, second_p, second_t, cross = moments.unbind(dim=2)
    raw_variance_p = second_p - mean_p.square()
    variance_p = raw_variance_p.clamp_min(0)
    variance_t = (second_t - mean_t.square()).clamp_min(0)
    covariance = cross - mean_p * mean_t
    a = 2 * mean_p * mean_t + 0.0001
    b = 2 * covariance + 0.0009
    c = mean_p.square() + mean_t.square() + 0.0001
    d = variance_p + variance_t + 0.0009
    inverse = (c * d).reciprocal()
    values = a * b * inverse

    derivative_covariance = 2 * a * inverse
    derivative_variance = -values / d * (raw_variance_p >= 0).to(prediction.dtype)
    derivative_mean = (2 * mean_t * b * inverse - 2 * mean_p * values / c
                       - 2 * mean_p * derivative_variance - mean_t * derivative_covariance)
    derivatives = torch.stack((derivative_mean, derivative_variance, derivative_covariance), dim=2)
    derivatives = derivatives.reshape(batch * planes * 3, 1, height - 10, width - 10)
    derivatives = derivatives / ((height - 10) * (width - 10))
    adjoint = F.conv_transpose2d(F.conv_transpose2d(derivatives, vertical), horizontal)
    adjoint = adjoint.reshape(batch, planes, 3, height, width)
    derivative = adjoint[:, :, 0] + 2 * prediction * adjoint[:, :, 1] + target * adjoint[:, :, 2]
    return values.mean(dim=(-2, -1)), derivative


def target_peak_scale(target: torch.Tensor) -> torch.Tensor:
    maximum = target.amax(dim=(-2, -1), keepdim=True)
    return torch.where(maximum > 1e-7, maximum, torch.ones_like(maximum)).reciprocal()


def ssim_phase_gradient(phase: torch.Tensor, target: torch.Tensor,
                        physics: Fresnel, weight: torch.Tensor | float,
                        scale: torch.Tensor | None = None) -> torch.Tensor:
    fields = physics(phase)
    amplitude = fields.abs()
    if scale is None:
        scale = target_peak_scale(target)
    _, derivative = ssim_input_gradient(amplitude * scale, target * scale)
    pixels = phase.shape[-2] * phase.shape[-1]
    amplitude_derivative = 2 * (amplitude - target) - weight * pixels * scale * derivative
    residual = amplitude_derivative * fields / amplitude.clamp_min(1e-7)
    backward = physics.adjoint(residual).mean(dim=1)
    return (torch.exp(-1j * phase) * backward).imag


class SSIMUnrolled(LearnedUnrolled):

    def __init__(self, stages: int = 8, ssim_weight: float = 0.1):
        if stages < 1 or not math.isfinite(ssim_weight) or ssim_weight <= 0:
            raise ValueError("stages and initial SSIM weight must be positive")
        super().__init__(stages=stages, width=1)
        del self.corrector
        del self.correction_scale
        self.ssim_logits = nn.Parameter(torch.full((stages,), math.log(math.expm1(ssim_weight))))

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        phase = self.initial(target)
        velocity = torch.zeros_like(phase)
        scale = target_peak_scale(target)
        for index in range(self.stages):
            gradient = ssim_phase_gradient(phase, target, self.physics,
                                           F.softplus(self.ssim_logits[index]), scale)
            velocity = self.momentum_logits[index].sigmoid() * velocity + gradient
            phase = phase - F.softplus(self.step_logits[index]) * velocity
        return phase


class AdamSSIMUnrolled(SSIMUnrolled):

    def __init__(self, stages: int = 32, ssim_weight: float = 1.0, step_size: float = 0.08):
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("Initial step size must be positive")
        super().__init__(stages=stages, ssim_weight=ssim_weight)
        del self.momentum_logits
        nn.init.constant_(self.step_logits, math.log(math.expm1(step_size)))

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        phase = self.initial(target)
        first_moment = torch.zeros_like(phase)
        second_moment = torch.zeros_like(phase)
        scale = target_peak_scale(target)
        phase, _, _ = self.advance(phase, first_moment, second_moment, target, scale, 0, self.stages)
        return phase

    def stage_gradient(self, phase, target, scale, index):
        return ssim_phase_gradient(phase, target, self.physics, F.softplus(self.ssim_logits[index]), scale)

    def advance(self, phase, first_moment, second_moment, target, scale, start, stop):
        for index in range(start, stop):
            gradient = self.stage_gradient(phase, target, scale, index)
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.99 * second_moment + 0.01 * gradient.square()
            corrected_first = first_moment / (1 - 0.9 ** (index + 1))
            corrected_second = second_moment / (1 - 0.99 ** (index + 1))
            denominator = corrected_second.clamp_min(1e-12).sqrt() + 1e-6
            phase = phase - F.softplus(self.step_logits[index]) * corrected_first / denominator
        return phase, first_moment, second_moment
