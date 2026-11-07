from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from baseline.evaluation import normalize_minmax
from baseline.models import Fresnel
from baseline.ssim_unrolled import AdamSSIMUnrolled, ssim_input_gradient, ssim_phase_gradient, target_peak_scale


def ssim7_input_gradient(prediction: torch.Tensor,
                        target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Expected equal [batch,planes,height,width] arrays")
    batch, planes, height, width = prediction.shape
    if min(height, width) < 7:
        raise ValueError("Uniform7 SSIM requires spatial dimensions >=7")
    moments = torch.stack((prediction, target, prediction.square(), target.square(), prediction*target), dim=2)
    moments = F.avg_pool2d(moments.reshape(batch, planes*5, height, width), 7, stride=1)
    moments = moments.reshape(batch, planes, 5, height-6, width-6)
    mean_p, mean_t, second_p, second_t, cross = moments.unbind(dim=2)
    raw_variance_p = second_p-mean_p.square()
    correction = 49/48
    variance_p = raw_variance_p.clamp_min(0)*correction
    variance_t = (second_t-mean_t.square()).clamp_min(0)*correction
    covariance = (cross-mean_p*mean_t)*correction
    a, b = 2*mean_p*mean_t+0.0001, 2*covariance+0.0009
    c, d = mean_p.square()+mean_t.square()+0.0001, variance_p+variance_t+0.0009
    inverse = (c*d).reciprocal()
    values = a*b*inverse
    derivative_cross = 2*a*inverse*correction
    derivative_second = -values/d*correction*(raw_variance_p >= 0).to(prediction.dtype)
    derivative_mean = (2*mean_t*b*inverse-2*mean_p*values/c
                       -2*mean_p*derivative_second-mean_t*derivative_cross)
    derivatives = torch.stack((derivative_mean, derivative_second, derivative_cross), dim=2)
    derivatives = derivatives.reshape(batch*planes*3, 1, height-6, width-6)/((height-6)*(width-6))
    horizontal = torch.full((1, 1, 1, 7), 1/7, dtype=prediction.dtype, device=prediction.device)
    adjoint = F.conv_transpose2d(F.conv_transpose2d(derivatives, horizontal.transpose(-2, -1)), horizontal)
    adjoint = adjoint.reshape(batch, planes, 3, height, width)
    gradient = adjoint[:, :, 0]+2*prediction*adjoint[:, :, 1]+target*adjoint[:, :, 2]
    return values.mean((-2, -1)), gradient


def minmax_input_gradient(values: torch.Tensor, output_gradient: torch.Tensor) -> torch.Tensor:
    minimum = values.amin((-2, -1), keepdim=True)
    maximum = values.amax((-2, -1), keepdim=True)
    span = maximum-minimum
    denominator = span.clamp_min(1e-12)
    minimum_mask = (values == minimum).to(values.dtype)
    maximum_mask = (values == maximum).to(values.dtype)
    minimum_weights = minimum_mask/minimum_mask.sum((-2, -1), keepdim=True)
    maximum_weights = maximum_mask/maximum_mask.sum((-2, -1), keepdim=True)
    normalized = (values-minimum)/denominator
    direct = output_gradient/denominator
    span_term = (output_gradient*normalized).sum((-2, -1), keepdim=True)/denominator
    span_term = span_term*(span >= 1e-12).to(values.dtype)
    return (direct-output_gradient.sum((-2, -1), keepdim=True)/denominator*minimum_weights
            -span_term*(maximum_weights-minimum_weights))


def robust_phase_gradient(phase: torch.Tensor, target: torch.Tensor, physics: Fresnel,
                          ssim_weight: torch.Tensor | float, minmax_weight: torch.Tensor | float = 0.1,
                          scale: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(minmax_weight, (int, float)) and minmax_weight == 0:
        return ssim_phase_gradient(phase, target, physics, ssim_weight, scale)
    if scale is None:
        scale = target_peak_scale(target)
    fields = physics(phase)
    amplitude = fields.abs()
    _, rms_derivative = ssim_input_gradient(amplitude*scale, target*scale)
    _, normalized_derivative = ssim7_input_gradient(normalize_minmax(amplitude), normalize_minmax(target))
    minmax_derivative = minmax_input_gradient(amplitude, normalized_derivative)
    pixels = phase.shape[-2]*phase.shape[-1]
    derivative = (2*(amplitude-target)-pixels*(ssim_weight*scale*rms_derivative
                                              +minmax_weight*minmax_derivative))
    backward = physics.adjoint(derivative*fields/amplitude.clamp_min(1e-7)).mean(dim=1)
    return (torch.exp(-1j*phase)*backward).imag


class RobustTailUnrolled(AdamSSIMUnrolled):

    def __init__(self, stages: int, prefix_stages: int, ssim_weight: float, step_size: float,
                 minmax_weight: float):
        if not 0 < prefix_stages < stages:
            raise ValueError("The fixed prefix must be shorter than the full network")
        if not math.isfinite(minmax_weight) or minmax_weight <= 0:
            raise ValueError("Initial tail minmax weight must be positive")
        super().__init__(stages=stages, ssim_weight=ssim_weight, step_size=step_size)
        self.prefix_stages = prefix_stages
        self.minmax_logits = torch.nn.Parameter(
            torch.full((stages - prefix_stages,), math.log(math.expm1(minmax_weight))))

    def stage_gradient(self, phase, target, scale, index):
        if index < self.prefix_stages:
            return super().stage_gradient(phase, target, scale, index)
        return robust_phase_gradient(
            phase, target, self.physics, F.softplus(self.ssim_logits[index]),
            F.softplus(self.minmax_logits[index - self.prefix_stages]), scale)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        phase = self.initial(target)
        scale = target_peak_scale(target)
        with torch.no_grad():
            phase, first, second = self.advance(
                phase, torch.zeros_like(phase), torch.zeros_like(phase), target, scale, 0, self.prefix_stages)
        phase, _, _ = self.advance(phase, first, second, target, scale, self.prefix_stages, self.stages)
        return phase
