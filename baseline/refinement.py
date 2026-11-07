from __future__ import annotations

from baseline.evaluation import normalize_minmax, ssim7
from baseline.objectives import joint_loss, local_ssim, pearson


LOSS_WEIGHTS = {
    "legacy": {"mse": 1.0, "pcc": 0.1, "ssim": 0.1, "minmax": 0.0},
    "ssim": {"mse": 1.0, "pcc": 0.1, "ssim": 1.0, "minmax": 0.0},
    "robust": {"mse": 1.0, "pcc": 0.1, "ssim": 1.0, "minmax": 0.1},
}
LOSS_FORMULAS = {
    "legacy": "amplitude_MSE + 0.1*(1-PCC) + 0.1*(1-SSIM)",
    "ssim": "amplitude_MSE + 0.1*(1-PCC) + 1.0*(1-SSIM)",
    "robust": "amplitude_MSE + 0.1*(1-PCC) + 1.0*(1-SSIM) + 0.1*(1-minmax_SSIM7)",
}


def refinement_terms(phase, target_normalized, target_rms, physics, loss_name="legacy"):
    weights = LOSS_WEIGHTS[loss_name]
    amplitude = physics(phase).abs()
    raw_prediction = amplitude * target_rms
    raw_target = target_normalized * target_rms
    terms = {
        "mse": (amplitude - target_normalized).square().mean(dim=(-3, -2, -1)),
        "pearson": pearson(raw_prediction, raw_target).mean(dim=1),
        "ssim": local_ssim(raw_prediction, raw_target).mean(dim=1),
    }
    if weights["minmax"]:
        terms["minmax_ssim"] = ssim7(normalize_minmax(amplitude), normalize_minmax(raw_target)).mean(dim=1)
    return terms


def combine_terms(terms, loss_name="legacy"):
    weights = LOSS_WEIGHTS[loss_name]
    objective = terms["mse"] + weights["pcc"] * (1 - terms["pearson"]) + weights["ssim"] * (1 - terms["ssim"])
    if weights["minmax"]:
        objective = objective + weights["minmax"] * (1 - terms["minmax_ssim"])
    return objective


def refinement_loss(phase, target_normalized, target_rms, physics, loss_name="legacy"):
    weights = LOSS_WEIGHTS[loss_name]
    if not weights["minmax"]:
        return joint_loss(phase, target_normalized, target_rms, physics,
                          weight_pcc=weights["pcc"], weight_ssim=weights["ssim"])
    return combine_terms(refinement_terms(phase, target_normalized, target_rms, physics, loss_name), loss_name)
