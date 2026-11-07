from __future__ import annotations

import torch
from torch.nn import functional as F


def normalize_minmax(values: torch.Tensor) -> torch.Tensor:
    minimum = values.amin(dim=(-2, -1), keepdim=True)
    span = values.amax(dim=(-2, -1), keepdim=True) - minimum
    return (values - minimum) / span.clamp_min(1e-12)


def ssim7(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM expects equal [batch, planes, height, width] arrays")
    if min(prediction.shape[-2:]) < 7:
        raise ValueError("SSIM requires spatial dimensions of at least 7")
    moments = torch.cat((prediction, target, prediction.square(),
                         target.square(), prediction * target), dim=1)
    moments = F.avg_pool2d(moments, kernel_size=7, stride=1)
    mean_p, mean_t, second_p, second_t, cross = moments.chunk(5, dim=1)
    correction = 49 / 48
    var_p = (second_p - mean_p.square()).clamp_min(0) * correction
    var_t = (second_t - mean_t.square()).clamp_min(0) * correction
    covariance = (cross - mean_p * mean_t) * correction
    values = ((2 * mean_p * mean_t + 0.0001) * (2 * covariance + 0.0009) /
              ((mean_p.square() + mean_t.square() + 0.0001) *
               (var_p + var_t + 0.0009)))
    return values.mean(dim=(-2, -1))
