from __future__ import annotations

import time
_PROGRAM_STARTED = time.perf_counter()

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

from baseline.candidates import build_model
from baseline.models import LearnedUnrolled, normalize_targets, quantize_phase
from baseline.refinement import (LOSS_FORMULAS, LOSS_WEIGHTS, combine_terms,
                                 refinement_loss, refinement_terms)

SIZE = 128
DISTANCES = (0.0241, 0.0253, 0.0265, 0.0277)
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "submission_v2.pt"
DEFAULT_STEPS = {"fast": 0, "balanced": 50, "quality": 200}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_gray_png(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(33)
    if (len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n" or
            struct.unpack(">I", header[8:12])[0] != 13 or header[12:16] != b"IHDR"):
        raise ValueError(f"{path}: expected a PNG with an IHDR header")
    width, height = struct.unpack(">II", header[16:24])
    if (width, height) != (SIZE, SIZE) or header[24:26] != bytes((8, 0)):
        raise ValueError(f"{path}: expected 128x128, 8-bit grayscale PNG (IHDR color type 0)")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if (image.format != "PNG" or image.mode != "L" or image.size != (SIZE, SIZE)
                or getattr(image, "is_animated", False)):
            raise ValueError(f"{path}: expected one non-animated 128x128 PNG in mode L")
        pixels = np.array(image, copy=True)
    if pixels.dtype != np.uint8 or pixels.shape != (SIZE, SIZE):
        raise ValueError(f"{path}: decoded pixel format is not uint8 [128,128]")
    return pixels


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinity")


def load_model(path: Path, device: torch.device) -> tuple[LearnedUnrolled, dict]:
    if not path.is_file():
        raise ValueError(f"Required learned checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("name") not in (
            "learned_unrolled", "image_unrolled", "direct_unet", "fourier_unet",
            "carrier_unrolled", "ssim_unrolled", "adam_ssim_unrolled", "robust_tail_unrolled"):
        raise ValueError("Expected a supported trained checkpoint; random weights are not a fallback")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no model state dictionary")
    architecture = checkpoint.get("architecture", {"name": "learned_unrolled", "stages": 8, "width": 12})
    if architecture.get("name") != checkpoint["name"]:
        raise ValueError("Checkpoint architecture and name disagree")
    model = build_model(architecture)
    expected_chirp = model.physics.chirp.detach().clone()
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Invalid checkpoint tensor: {name}")
        require_finite(value, f"checkpoint tensor {name}")
    model.load_state_dict(state, strict=True)
    if not torch.allclose(model.physics.chirp, expected_chirp, atol=1e-6, rtol=1e-6):
        raise ValueError("Checkpoint diffraction configuration differs from the documented 4-plane configuration")
    model.to(device=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    info = {key: checkpoint.get(key) for key in
            ("name", "seed", "step", "validation_loss", "training_steps_budget")}
    info["sha256"] = sha256(path)
    info["architecture"] = architecture
    return model, info


@torch.no_grad()
def score_quantized(phase: torch.Tensor, target: torch.Tensor,
                    target_rms: torch.Tensor, model: LearnedUnrolled,
                    loss_name: str = "legacy") -> tuple[torch.Tensor, dict]:
    require_finite(phase, "phase checkpoint")
    codes, _ = quantize_phase(phase)
    decoded = codes.to(dtype=phase.dtype) * (2 * math.pi / 255) - math.pi
    terms = refinement_terms(decoded, target, target_rms, model.physics, loss_name)
    values = {name: float(value.item()) for name, value in terms.items()}
    values["objective"] = combine_terms(values, loss_name)
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Quantized checkpoint objective is not finite")
    return codes.detach().clone(), values


def synthesize(model: LearnedUnrolled, target: torch.Tensor, target_rms: torch.Tensor,
               mode: str, steps: int, device: torch.device,
               loss_name: str = "legacy") -> tuple[np.ndarray, dict]:
    synchronize(device)
    synthesis_start = time.perf_counter()
    with torch.inference_mode():
        initial = model(target)
        require_finite(initial, "network phase")
    synchronize(device)
    forward_seconds = time.perf_counter() - synthesis_start
    trace = []
    selected_step = 0
    selected_terms = None
    refinement_start = time.perf_counter()
    if mode == "fast":
        with torch.inference_mode():
            codes, _ = quantize_phase(initial)
    else:
        phase = torch.nn.Parameter(initial.detach().clone())
        optimizer = torch.optim.Adam([phase], lr=0.08)
        codes, selected_terms = score_quantized(phase, target, target_rms, model, loss_name)
        trace.append({"step": 0, **selected_terms})
        with torch.enable_grad():
            for step in range(1, steps + 1):
                optimizer.zero_grad(set_to_none=True)
                objective = refinement_loss(phase, target, target_rms, model.physics, loss_name).mean()
                require_finite(objective, "continuous optimization objective")
                objective.backward()
                require_finite(phase.grad, "phase gradient")
                optimizer.step()
                with torch.no_grad():
                    require_finite(phase, "updated phase")
                    if step % 5 == 0 or step == steps:
                        phase.copy_(torch.remainder(phase + math.pi, 2 * math.pi) - math.pi)
                candidate_codes, candidate_terms = score_quantized(phase, target, target_rms, model, loss_name)
                trace.append({"step": step, **candidate_terms})
                if candidate_terms["objective"] < selected_terms["objective"]:
                    codes, selected_terms = candidate_codes, candidate_terms
                    selected_step = step
    pixels = codes[0].detach().cpu().numpy().copy()
    synchronize(device)
    synthesis_seconds = time.perf_counter() - synthesis_start
    info = {
        "learned_forward_count": 1,
        "refinement_steps": steps,
        "refinement_loss": loss_name if mode != "fast" else None,
        "selected_step": selected_step,
        "selection": ("none: quantized learned output" if mode == "fast" else
                      "minimum internal mixed objective among decoded uint8 phase checkpoints, including step 0"),
        "selected_internal_objective_terms": selected_terms,
        "checkpoint_trace": trace,
        "learned_forward_seconds": forward_seconds,
        "post_forward_seconds": time.perf_counter() - refinement_start,
        "synthesis_seconds": synthesis_seconds,
    }
    return pixels, info


def save_mask(path: Path, pixels: np.ndarray) -> None:
    if pixels.shape != (SIZE, SIZE) or pixels.dtype != np.uint8:
        raise ValueError("Internal output is not one uint8 [128,128] mask")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".phase-", suffix=".png", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        Image.fromarray(pixels).save(temporary_path, format="PNG")
        if not np.array_equal(read_gray_png(temporary_path), pixels):
            raise ValueError("PNG read-back did not preserve phase codes")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json(path: Path, value: dict) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".metadata-", suffix=".json", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description='Synthesize one phase mask from four grayscale PNGs.')
    result.add_argument("inputs", type=Path, nargs=4, metavar="TARGET.png",
                        help="four ordered amplitude PNGs, one for each increasing propagation distance")
    result.add_argument("--output", type=Path, required=True, help="output phase PNG (128x128, L, uint8)")
    result.add_argument("--mode", choices=tuple(DEFAULT_STEPS), default="fast",
                        help="fast: one learned forward (default); balanced/quality: experimental phase refinement")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu",
                        help="execution device (default cpu; auto selects CUDA when available)")
    result.add_argument("--steps", type=int, help="refinement steps; defaults fast=0, balanced=50, quality=200")
    result.add_argument("--refinement-loss", choices=tuple(LOSS_WEIGHTS), default="robust",
                        help="experimental refinement objective (default robust); unused in fast mode")
    result.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="trained checkpoint; defaults to the selected bundled weights")
    result.add_argument("--metadata", type=Path, help="optional separate JSON with provenance, internal objective and timings")
    result.add_argument("--threads", type=int, default=2, help="CPU intra-op threads (default 2; inter-op is 1)")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    steps = DEFAULT_STEPS[args.mode] if args.steps is None else args.steps
    if steps < 0 or (args.mode == "fast" and steps != 0):
        raise ValueError("--steps must be nonnegative; fast requires zero refinement steps")
    args.inputs = [path.expanduser().resolve() for path in args.inputs]
    args.output = args.output.expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    args.metadata = args.metadata.expanduser().resolve() if args.metadata else None
    if args.output.suffix.lower() != ".png":
        raise ValueError("--output must have the .png extension")
    protected = set(args.inputs + [args.weights])
    if args.output in protected:
        raise ValueError("Output must not overwrite an input or the learned weights")
    if args.metadata and (args.metadata in protected or args.metadata == args.output):
        raise ValueError("Metadata must have a separate path and must not overwrite inputs or weights")
    if args.metadata and args.metadata.suffix.lower() != ".json":
        raise ValueError("--metadata must have the .json extension")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu or auto")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    load_start = time.perf_counter()
    arrays = [read_gray_png(path) for path in args.inputs]
    raw = torch.from_numpy(np.stack(arrays).astype(np.float32) / 255.0)[None].to(device)
    target, target_rms = normalize_targets(raw)
    require_finite(target, "normalized targets")
    model, checkpoint_info = load_model(args.weights, device)
    synchronize(device)
    loading_seconds = time.perf_counter() - load_start
    pixels, synthesis_info = synthesize(model, target, target_rms, args.mode, steps, device,
                                       args.refinement_loss)
    export_start = time.perf_counter()
    save_mask(args.output, pixels)
    if not np.array_equal(read_gray_png(args.output), pixels):
        raise ValueError("Final phase PNG failed read-back verification")
    export_seconds = time.perf_counter() - export_start
    if args.metadata:
        information = {
            "schema_version": 1,
            "status": "research_baseline_not_official_judge_score",
            "mode": args.mode,
            "test_time_optimization": args.mode != "fast",
            "eligibility_note": "Permission for test-time optimization must be confirmed before using balanced/quality in a submission.",
            "inputs": [{"path": str(path), "sha256": sha256(path)} for path in args.inputs],
            "output": {"path": str(args.output), "sha256": sha256(args.output),
                       "shape": [SIZE, SIZE], "mode": "L", "bit_depth": 8,
                       "decode_radians": "2*pi*uint8_code/255 - pi", "read_back_verified": True},
            "checkpoint": checkpoint_info,
            "physical_model": {"size": SIZE, "wavelength_m": 532e-9, "pitch_m": 10e-6,
                               "distances_m": list(DISTANCES), "illumination": "uniform unit amplitude",
                               "propagator": "centered unitary single-FFT Fresnel, input chirp; output chirp omitted for magnitudes"},
            "normalization": {"input": "uint8/255 treated as target amplitude; no resize or minmax",
                              "network": "each plane divided by its RMS; RMS below 1e-7 uses scale 1 (including a completely black target)",
                              "plane_rms": raw.square().mean(dim=(-2, -1)).sqrt().detach().cpu().reshape(4).tolist(),
                              "plane_scale": target_rms.detach().cpu().reshape(4).tolist(),
                              "refinement_reconstruction_scale": "unitary reconstructed amplitude multiplied by the corresponding target RMS scale (fallback 1 for zero); no clipping",
                              "official_judge_normalization": "unknown and not imitated"},
            "candidate_objective": {"name": args.refinement_loss, "active": args.mode != "fast",
                                    "formula": LOSS_FORMULAS[args.refinement_loss],
                                    "weights": LOSS_WEIGHTS[args.refinement_loss],
                                    "mse_scale": "per-plane RMS-normalized target amplitude; RMS below 1e-7 uses scale 1",
                                    "ssim": "Gaussian 11x11, sigma=1.5, population covariance, valid windows, fixed data_range=1",
                                    "minmax_ssim": ("independent per-plane minmax, span floor 1e-12; uniform7, sample covariance, valid windows, data_range=1"
                                                    if args.refinement_loss == "robust" else None),
                                    "pcc_constant_policy": "zero when centered numerator is zero; denominator has 1e-24 squared floor",
                                    "learning_rate": 0.08, "phase_wrap_every_steps": 5,
                                    "selection_frequency_steps": 1,
                                    "official_score": False},
            "runtime": {"device": str(device), "cpu_intra_op_threads": args.threads,
                        "cpu_inter_op_threads": 1, "torch_version": torch.__version__,
                        "numpy_version": np.__version__, "cuda_synchronized": device.type == "cuda"},
            "synthesis": synthesis_info,
            "timings": {"loading_seconds": loading_seconds, "synthesis_seconds": synthesis_info["synthesis_seconds"],
                        "png_export_verify_seconds": export_seconds,
                        "program_seconds_before_metadata_write": time.perf_counter() - _PROGRAM_STARTED,
                        "scope": "program elapsed includes library imports; excludes interpreter startup and the final metadata write; no warm-up forward"},
        }
        write_json(args.metadata, information)
    print(f"Saved {args.output} | mode={args.mode} | synthesis={synthesis_info['synthesis_seconds']:.6f}s | device={device}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
