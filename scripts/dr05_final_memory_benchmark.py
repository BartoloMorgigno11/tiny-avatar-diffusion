"""Future-only DR05-D13 canonical sampling-memory benchmark.

Invoking this script is intentionally a separate authorization from this
implementation task.  The parent command spawns one fresh Python process per
canonical condition so CUDA allocator history cannot cross conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.dr05_materialize_canonical_checkpoints import (
        _conditional_models_from_artifact,
        _unconditional_model_from_artifact,
    )
except ModuleNotFoundError:
    from dr05_materialize_canonical_checkpoints import (  # type: ignore[no-redef]
        _conditional_models_from_artifact,
        _unconditional_model_from_artifact,
    )
from tiny_avatar_diffusion.conditional_sampling import conditional_sample
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.dr05_efficiency import (
    CANONICAL_CONDITIONS,
    MemoryCondition,
    canonical_condition,
    measure_sampling_memory,
    require_cuda,
    result_record,
)
from tiny_avatar_diffusion.tokenizer import WhitespaceTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "runs" / "dr05_final_efficiency_v1"
VOCAB = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "vocab.json"
CHECKPOINTS = {
    "conditional": (ROOT / "artifacts" / "checkpoints" / "conditional_main_ema_step20000.pt", "a1abad595c53887b621021b67154a04af2505a4b68e6b9229841601250ca14fb"),
    "unconditional": (ROOT / "artifacts" / "checkpoints" / "unconditional_main_ema_step20000.pt", "70def31c417a3aabe2fedbc355c2b6fae8478d1f878690f9bbf2d7263cf9fb3e"),
}
PROMPT = "avatar face_shape_wide glasses_thick_rectangular_glasses hair_color_silver face_color_deep_brown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokenizer() -> WhitespaceTokenizer:
    return WhitespaceTokenizer(json.loads(VOCAB.read_text(encoding="utf-8"))["token_to_id"])


def _sample_unconditional(schedule: LinearDDPMSchedule, model: torch.nn.Module, *, batch_size: int) -> torch.Tensor:
    x_t = torch.randn((batch_size, 3, 32, 32), device="cuda:0", dtype=torch.float32)
    with torch.inference_mode():
        for value in range(schedule.betas.shape[0] - 1, -1, -1):
            timesteps = torch.full((batch_size,), value, dtype=torch.long, device="cuda:0")
            noise = torch.randn(x_t.shape, device="cuda:0", dtype=torch.float32)
            x_t = schedule.p_sample_from_noise(
                x_t, timesteps, model(x_t, timesteps), noise=noise, clip_denoised=True
            )["sample"]
    return x_t


def _sample_conditional(
    schedule: LinearDDPMSchedule, text: torch.nn.Module, model: torch.nn.Module,
    tokenizer: WhitespaceTokenizer, *, batch_size: int,
) -> torch.Tensor:
    encoded = tokenizer.encode(PROMPT)
    input_ids = torch.tensor([encoded.token_ids] * batch_size, dtype=torch.long, device="cuda:0")
    mask = torch.tensor([encoded.attention_mask] * batch_size, dtype=torch.long, device="cuda:0")
    return conditional_sample(schedule, text, model, input_ids, mask, clip_denoised=True)


def _provenance(device: torch.device) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def run_condition(condition: MemoryCondition, output_dir: Path) -> Path:
    device = require_cuda()
    checkpoint, expected_sha = CHECKPOINTS[condition.model_type]
    if sha256_file(checkpoint) != expected_sha:
        raise ValueError(f"unexpected frozen checkpoint SHA256: {checkpoint}")
    schedule = LinearDDPMSchedule().to(device).eval()
    if condition.conditional:
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        text, model = _conditional_models_from_artifact(artifact)
        text, model = text.to(device).eval(), model.to(device).eval()
        tokenizer = _tokenizer()
        sample = lambda: _sample_conditional(schedule, text, model, tokenizer, batch_size=condition.batch_size)
    else:
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = _unconditional_model_from_artifact(artifact).to(device).eval()
        sample = lambda: _sample_unconditional(schedule, model, batch_size=condition.batch_size)

    # All setup above is deliberately outside the reset-to-peak sampling window.
    measurement = measure_sampling_memory(sample, device=device)
    output = measurement.pop("_output")
    if output.shape != (condition.batch_size, 3, 32, 32):
        raise RuntimeError("invalid canonical sampling output shape")
    del output
    record = result_record(
        condition, device=device, gpu_name=torch.cuda.get_device_name(device),
        checkpoint_path=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=expected_sha,
        provenance=_provenance(device), measurement=measurement,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{condition.key}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark record: {path}")
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def dispatch_conditions(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse canonical output root: {output_dir}")
    for condition in CANONICAL_CONDITIONS:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--run-condition", condition.key, "--output", str(output_dir)],
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-condition", choices=[item.key for item in CANONICAL_CONDITIONS])
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_condition:
        run_condition(canonical_condition(args.run_condition), args.output)
    else:
        dispatch_conditions(args.output)


if __name__ == "__main__":
    main()
