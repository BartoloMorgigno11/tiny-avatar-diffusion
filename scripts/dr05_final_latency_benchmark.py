"""Dedicated DR05-D12 final latency producer; execute only with separate authorization."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch

from scripts.dr05_materialize_canonical_checkpoints import _conditional_models_from_artifact, _unconditional_model_from_artifact
from tiny_avatar_diffusion.conditional_sampling import conditional_sample
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.dr05_latency import CANONICAL_CONDITIONS, benchmark_condition, condition_record, write_latency_summary
from tiny_avatar_diffusion.tokenizer import WhitespaceTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "runs" / "dr05_final_efficiency_v1"
PROTOCOL = ROOT / "data" / "processed" / "dr05_final_eval_v1" / "dr05_final_protocol.json"
CONDITIONAL = ROOT / "artifacts" / "checkpoints" / "conditional_main_ema_step20000.pt"
UNCONDITIONAL = ROOT / "artifacts" / "checkpoints" / "unconditional_main_ema_step20000.pt"
VOCAB = ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / "vocab.json"
PROMPT_ID = "cs10002930376453395152"
PROMPT = "avatar face_shape_wide glasses_thick_rectangular_glasses hair_color_silver face_color_deep_brown"
PROTOCOL_SHA = "706a191fff8f2fba1b263919385a6e55ad4579a5285a189cab5acf20ae38549f"
CHECKPOINTS = {"conditional": (CONDITIONAL, "a1abad595c53887b621021b67154a04af2505a4b68e6b9229841601250ca14fb"), "unconditional": (UNCONDITIONAL, "70def31c417a3aabe2fedbc355c2b6fae8478d1f878690f9bbf2d7263cf9fb3e")}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("DR05-D12 canonical latency benchmark requires CUDA; CPU is not supported")
    if sha256(PROTOCOL) != PROTOCOL_SHA or any(sha256(path) != digest for path, digest in CHECKPOINTS.values()):
        raise ValueError("frozen protocol or checkpoint identity mismatch")
    device = torch.device("cuda:0")
    schedule = LinearDDPMSchedule().to(device).eval()
    vocabulary = json.loads(VOCAB.read_text(encoding="utf-8"))["token_to_id"]
    tokenizer = WhitespaceTokenizer(vocabulary)
    records = []
    for condition in CANONICAL_CONDITIONS:
        checkpoint, digest = CHECKPOINTS[condition.model_type]
        if condition.conditional:
            text, model = _conditional_models_from_artifact(torch.load(checkpoint, map_location="cpu", weights_only=False))
            text, model = text.to(device).eval(), model.to(device).eval()
            def sample(seed: int, text=text, model=model, batch=condition.batch_size):
                encoded = tokenizer.encode(PROMPT)
                ids = torch.tensor([encoded.token_ids] * batch, device=device)
                mask = torch.tensor([encoded.attention_mask] * batch, device=device)
                return conditional_sample(schedule, text, model, ids, mask, generator=torch.Generator(device=device).manual_seed(seed), clip_denoised=True)
        else:
            model = _unconditional_model_from_artifact(torch.load(checkpoint, map_location="cpu", weights_only=False)).to(device).eval()
            def sample(seed: int, model=model, batch=condition.batch_size):
                generator = torch.Generator(device=device).manual_seed(seed)
                x = torch.randn((batch, 3, 32, 32), device=device, generator=generator)
                with torch.inference_mode():
                    for value in range(999, -1, -1):
                        timestep = torch.full((batch,), value, dtype=torch.long, device=device)
                        x = schedule.p_sample_from_noise(x, timestep, model(x, timestep), noise=torch.randn(x.shape, device=device, generator=generator), clip_denoised=True)["sample"]
                return x
        records.append(condition_record(condition, benchmark_condition(sample, condition=condition, device=device), checkpoint_path=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=digest))
        del model
        if condition.conditional: del text
        torch.cuda.empty_cache()
    artifact = {"schema": "dr05_final_latency_v1", "protocol_sha256": PROTOCOL_SHA, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "environment": {"device": str(device), "gpu_name": torch.cuda.get_device_name(device), "python_version": platform.python_version(), "pytorch_version": torch.__version__, "cuda_version": torch.version.cuda}, "prompt": {"id": PROMPT_ID, "caption": PROMPT, "source": "data/processed/dr01_v1/text/10k/id_val.jsonl"}, "timing": {"warmups": 3, "measured_runs": 10, "method": "cuda_synchronize + time.perf_counter + cuda_synchronize", "std_convention": "population"}, "conditions": records}
    write_latency_summary(OUTPUT_ROOT, artifact)
    return artifact

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
