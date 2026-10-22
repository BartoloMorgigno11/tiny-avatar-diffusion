"""Fail-closed CPU release package for the completed STEP 12 EMA-60k."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from tiny_avatar_diffusion.v2a_unconditional_native64 import EXPECTED_PARAMETERS, V2AUnconditionalNative64UNet

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/runs/v2a_unconditional_native64_60k_seed20261021"
SOURCE, TELEMETRY = RUN / "ema_step60000.pt", RUN / "training_telemetry.json"
OUTPUT = ROOT / "artifacts/checkpoints/v2_a_unconditional_ema_step60000.pt"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""): digest.update(part)
    return digest.hexdigest()

def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    telemetry = json.loads(TELEMETRY.read_text(encoding="utf-8"))
    if telemetry.get("status") != "TARGET_REACHED" or telemetry.get("successful_updates") != 60_000: raise ValueError("STEP 12 did not reach exactly 60k updates")
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if source.get("artifact_type") != "v2a_unconditional_native64_ema" or source.get("selected_step") != 60_000 or source.get("parameter_count") != EXPECTED_PARAMETERS: raise ValueError("unexpected STEP 12 EMA artifact")
    model = V2AUnconditionalNative64UNet(); model.load_state_dict(source["model_state_dict"], strict=True)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != EXPECTED_PARAMETERS: raise ValueError("parameter mismatch")
    package = {**source, "artifact_format_version": 1, "artifact_type": "canonical_inference_checkpoint", "model_role": "v2a_unconditional_native64_60k", "weight_variant": "EMA", "source_ema_package_path": SOURCE.relative_to(ROOT).as_posix(), "source_ema_package_sha256": sha256(SOURCE), "source_terminal_path": (RUN / "latest.pt").relative_to(ROOT).as_posix(), "source_terminal_sha256": telemetry["terminal_sha256"]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True); temporary = OUTPUT.with_suffix(".tmp"); torch.save(package, temporary)
    check = torch.load(temporary, map_location="cpu", weights_only=False)
    if check["source_terminal_sha256"] != telemetry["terminal_sha256"] or any(not torch.equal(source["model_state_dict"][k], check["model_state_dict"][k]) for k in source["model_state_dict"]): raise ValueError("release verification failed")
    temporary.replace(OUTPUT); print(json.dumps({"path": str(OUTPUT.relative_to(ROOT)), "sha256": sha256(OUTPUT), "source_ema_sha256": sha256(SOURCE)}, indent=2))

if __name__ == "__main__": main()
