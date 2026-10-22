"""Package the authenticated V2-A conditional EMA-60k state for public inference."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/v2_a_matched_budget_continuation_40k_to_60k_seed20261017/checkpoint_success_060000.pt"
DESTINATION = ROOT / "artifacts/checkpoints/v2_a_conditional_ema_step60000.pt"
SOURCE_SHA = "f47132dc9e247861c6697e41b5d352ccfa347ba6a19e48dcee7f061206cb0f14"

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()

def main() -> None:
    if sha(SOURCE) != SOURCE_SHA: raise ValueError("source SHA mismatch")
    state = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if state.get("schema") != "tiny_avatar_diffusion.v2a_matched_budget_training_v1" or state.get("progress", {}).get("successful_updates") != 60000 or set(state.get("ema", {})) != {"model", "text"}:
        raise ValueError("source EMA-60k identity mismatch")
    artifact = {"artifact_format_version": 1, "artifact_type": "v2a_conditional_ema_inference", "model_role": "v2a_conditional_60k", "weight_variant": "EMA", "selected_step": 60000, "source_checkpoint_path": SOURCE.relative_to(ROOT).as_posix(), "source_checkpoint_sha256": SOURCE_SHA, "architecture": {"variant": "v2_a", "resolution": 64, "text_cross_attention": ["CA8", "CA16"], "text_encoder": "TextConditioningEncoder"}, "expected_parameter_count": 5230403, "expected_denoiser_parameter_count": 5127427, "expected_text_parameter_count": 102976, "model_state_dict": state["ema"]["model"], "text_state_dict": state["ema"]["text"]}
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists(): raise FileExistsError(DESTINATION)
    torch.save(artifact, DESTINATION)
    print(sha(DESTINATION))

if __name__ == "__main__": main()
