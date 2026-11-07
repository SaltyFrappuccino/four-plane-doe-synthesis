from __future__ import annotations

import torch
from torch.nn import functional as F


def local_ssim(prediction, target, data_range=1.0):
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM expects equal [batch, planes, height, width] arrays")
    if min(prediction.shape[-2:]) < 11:
        raise ValueError("SSIM requires spatial dimensions of at least 11")
    axis = torch.arange(-5, 6, dtype=prediction.dtype, device=prediction.device)
    kernel = torch.exp(-0.5 * (axis / 1.5).square())
    kernel = kernel / kernel.sum()
    batch, planes, height, width = prediction.shape
    moments = torch.stack((prediction, target, prediction.square(),
                           target.square(), prediction * target), dim=2)
    moments = moments.reshape(batch * planes * 5, 1, height, width)
    moments = F.conv2d(moments, kernel.reshape(1, 1, 1, 11))
    moments = F.conv2d(moments, kernel.reshape(1, 1, 11, 1))
    moments = moments.reshape(batch, planes, 5, height - 10, width - 10)
    mu_p, mu_t, second_p, second_t, cross = moments.unbind(dim=2)
    var_p = (second_p - mu_p.square()).clamp_min(0)
    var_t = (second_t - mu_t.square()).clamp_min(0)
    covariance = cross - mu_p * mu_t
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    values = ((2 * mu_p * mu_t + c1) * (2 * covariance + c2) /
              ((mu_p.square() + mu_t.square() + c1) * (var_p + var_t + c2)))
    return values.mean(dim=(-2, -1))


def pearson(prediction, target):
    p = prediction - prediction.mean(dim=(-2, -1), keepdim=True)
    t = target - target.mean(dim=(-2, -1), keepdim=True)
    numerator = (p * t).sum(dim=(-2, -1))
    denominator = (p.square().sum(dim=(-2, -1)) *
                   t.square().sum(dim=(-2, -1))).clamp_min(1e-24).sqrt()
    return numerator / denominator


def loss_terms(phase, target_normalized, target_rms, physics):
    amplitude = physics(phase).abs()
    raw_prediction = amplitude * target_rms
    raw_target = target_normalized * target_rms
    return {
        "mse": (amplitude - target_normalized).square().mean(dim=(-3, -2, -1)),
        "pearson": pearson(raw_prediction, raw_target).mean(dim=1),
        "ssim": local_ssim(raw_prediction, raw_target).mean(dim=1),
    }


def joint_loss(phase, target_normalized, target_rms, physics,
               weight_pcc=0.1, weight_ssim=0.1):
    amplitude = physics(phase).abs()
    result = (amplitude - target_normalized).square().mean(dim=(-3, -2, -1))
    if weight_pcc == 0 and weight_ssim == 0:
        return result
    raw_prediction = amplitude * target_rms
    raw_target = target_normalized * target_rms
    if weight_pcc:
        result = result + weight_pcc * (1 - pearson(raw_prediction, raw_target).mean(dim=1))
    if weight_ssim:
        result = result + weight_ssim * (1 - local_ssim(raw_prediction, raw_target).mean(dim=1))
    return result
