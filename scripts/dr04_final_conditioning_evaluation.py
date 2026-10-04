"""Guarded production entry point for the frozen DR04 conditioning evaluation."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tiny_avatar_diffusion.dr04_final_conditioning_executor import (
    FinalConditioningExecutor,
    build_release_addendum,
    canonical_sha256,
    collect_parameter_counts,
    expand_plan,
    materialize_release_addendum,
    validate_panel,
)

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/dr04_final_eval_v1/final_prompt_panel_manifest.json"
CONTROLS = ROOT / "data/processed/dr04_final_eval_v1/matched_controls_manifest.json"
FREEZE = ROOT / "data/processed/dr04_final_eval_v1/dr04_freeze_manifest.json"
CONDITIONAL = ROOT / "artifacts/checkpoints/conditional_main_ema_step20000.pt"
EVALUATOR = ROOT / "artifacts/checkpoints/dr04_evaluator_epoch41.pt"
FINAL_ROOT = ROOT / "results/runs/dr04_final_conditioning_v1"
VOCAB = ROOT / "data/processed/dr01_v1/text/10k/vocab.json"
RELEASE_ADDENDUM = ROOT / "data/processed/dr04_final_eval_v1/dr04_release_provenance_addendum.json"
EXPECTED = {PANEL: "102c76b5281119df066cc83e59ca7dfd836aa99bb3ab511b0628f3d750224e60", CONTROLS: "c955c39bb5debd201a72ecd65aa7e2a66de7407f6d38515ba52c13d2d99749c7", FREEZE: "c849a50ae342803707c20090d8ed5cd1383ceac9854d2e6b035a2519ee6b4af6", CONDITIONAL: "a1abad595c53887b621021b67154a04af2505a4b68e6b9229841601250ca14fb", EVALUATOR: "414d2e0bb597c53041f6cf695d965231ab1c4f38c9080b68c06f9c09dcf001dc", VOCAB: "15f2feb538e6e13389bf6ff113c48ed7a9663e0aefd8416dcd4c6fc760d7041a"}


def preflight() -> dict[str, object]:
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): canonical_sha256(path) for path in EXPECTED}
    if any(hashes[str(path.relative_to(ROOT)).replace("\\", "/")] != wanted for path, wanted in EXPECTED.items()): raise RuntimeError("frozen identity mismatch")
    rows = validate_panel(json.loads(PANEL.read_text(encoding="utf-8")), json.loads(CONTROLS.read_text(encoding="utf-8")))
    plan = expand_plan(rows, str(CONDITIONAL.relative_to(ROOT)).replace("\\", "/"), EXPECTED[CONDITIONAL])
    return {"status": "preflight_pass", "hashes": hashes, "prompts": len(rows), "slots": len(plan), "id_slots": sum(r["panel_split"] == "ID" for r in plan), "ood_slots": sum(r["panel_split"] == "OOD" for r in plan), "final_output_touched": False}


def execution_context() -> tuple[Path, list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Bind the one permitted production root, plan, identities, and environment."""
    preflight()
    rows = validate_panel(json.loads(PANEL.read_text(encoding="utf-8")), json.loads(CONTROLS.read_text(encoding="utf-8")))
    plan = expand_plan(rows, str(CONDITIONAL.relative_to(ROOT)).replace("\\", "/"), EXPECTED[CONDITIONAL])
    identities = {"conditional_checkpoint_path": str(CONDITIONAL.relative_to(ROOT)).replace("\\", "/"), "conditional_checkpoint_sha256": EXPECTED[CONDITIONAL], "evaluator_checkpoint_path": str(EVALUATOR.relative_to(ROOT)).replace("\\", "/"), "evaluator_checkpoint_sha256": EXPECTED[EVALUATOR], "vocabulary_path": str(VOCAB.relative_to(ROOT)).replace("\\", "/"), "vocabulary_sha256": EXPECTED[VOCAB], "freeze_manifest_sha256": EXPECTED[FREEZE], "panel_manifest_sha256": EXPECTED[PANEL], "matched_controls_manifest_sha256": EXPECTED[CONTROLS]}
    import torch
    environment = {"python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))}
    return FINAL_ROOT, plan, identities, environment


def _load_conditional_generator() -> tuple[Callable[[dict[str, Any]], np.ndarray], tuple[Any, Any, Any]]:
    """Load canonical EMA conditional inference components once for phase 1."""
    import torch
    from tiny_avatar_diffusion.conditional_sampling import conditional_sample
    from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
    from tiny_avatar_diffusion.text_conditioning import TextConditioningConfig, TextConditioningEncoder
    from tiny_avatar_diffusion.tokenizer import WhitespaceTokenizer
    from tiny_avatar_diffusion.unet import ConditionalTinyUNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(CONDITIONAL, map_location="cpu", weights_only=False)
    if artifact.get("artifact_type") != "canonical_inference_checkpoint" or artifact.get("model_role") != "conditional_main" or artifact.get("selected_step") != 20_000 or artifact.get("weight_variant") != "EMA" or artifact.get("expected_parameter_count") != 4_723_907: raise ValueError("unexpected canonical conditional checkpoint")
    architecture, state = artifact.get("architecture"), artifact.get("model_state_dict")
    if not isinstance(architecture, dict) or not isinstance(state, dict): raise ValueError("invalid canonical conditional checkpoint schema")
    text = TextConditioningEncoder(TextConditioningConfig(**architecture["text_encoder"]["config"])).to(device).eval()
    unet = ConditionalTinyUNet(dropout=architecture["conditional_unet"]["dropout"]).to(device).eval()
    text.load_state_dict(state["text_encoder"]); unet.load_state_dict(state["conditional_unet"])
    schedule = LinearDDPMSchedule(num_timesteps=1_000).to(device)
    tokenizer = WhitespaceTokenizer(json.loads(VOCAB.read_text(encoding="utf-8"))["token_to_id"])
    def generate(slot: dict[str, Any]) -> np.ndarray:
        encoded = tokenizer.encode(slot["prompt"]); generator = torch.Generator(device=device).manual_seed(slot["seed"])
        sample = conditional_sample(schedule, text, unet, torch.tensor([encoded.token_ids], dtype=torch.long, device=device), torch.tensor([encoded.attention_mask], dtype=torch.long, device=device), generator=generator, clip_denoised=True)[0]
        return ((sample.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    class _BackboneParameters:
        def parameters(self) -> Any:
            return (parameter for name, parameter in unet.named_parameters() if not name.startswith("image_text_cross_attention."))
    return generate, (_BackboneParameters(), unet.image_text_cross_attention, text)


def _load_evaluator() -> tuple[Callable[[np.ndarray], dict[str, int]], Any]:
    """Load the canonical eval-only evaluator once for phase 3."""
    import torch
    from PIL import Image
    from tiny_avatar_diffusion.evaluator import ARCHITECTURE_ID, CLASS_COUNTS, AttributeEvaluator, preprocess_cached
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(EVALUATOR, map_location="cpu", weights_only=False)
    if artifact.get("artifact_type") != "canonical_evaluator_inference_checkpoint" or artifact.get("model_role") != "dr04_evaluator" or artifact.get("selected_epoch") != 41 or artifact.get("expected_parameter_count") != 293_064: raise ValueError("unexpected canonical evaluator checkpoint")
    architecture = artifact.get("architecture")
    if not isinstance(architecture, dict) or architecture.get("architecture_id") != ARCHITECTURE_ID or architecture.get("class_counts") != CLASS_COUNTS: raise ValueError("invalid canonical evaluator checkpoint schema")
    model = AttributeEvaluator().to(device); model.load_state_dict(artifact["model_state_dict"]); model.eval()
    def evaluate(pixels: np.ndarray) -> dict[str, int]:
        with torch.no_grad():
            tensor = preprocess_cached(Image.fromarray(pixels, mode="RGB")).unsqueeze(0).to(device)
            return {name: int(values.argmax(dim=1).item()) for name, values in model(tensor).items()}
    return evaluate, model


def execute_final_scientific_run() -> dict[str, Any]:
    """Run phases 0--5 through the durable executor after both CLI guards pass."""
    root, plan, identities, environment = execution_context()
    summary = root / "final_conditioning_summary.json"
    if summary.is_file():
        return json.loads(summary.read_text(encoding="utf-8"))
    executor = FinalConditioningExecutor(root, plan, identities); state = executor.state()
    generator, (unet, cross_attention, text_encoder) = _load_conditional_generator()
    evaluator, evaluator_model = _load_evaluator()
    executor.record_parameter_counts(collect_parameter_counts(unet, cross_attention, text_encoder, evaluator_model))
    if state.get("phase", 0) < 1: executor.generate(generator)
    if executor.state().get("phase", 0) < 2: executor.audit()
    if executor.state().get("phase", 0) < 3: executor.predict(evaluator)
    if executor.state().get("phase", 0) < 4: executor.metrics()
    if executor.state().get("phase", 0) < 5: return executor.finalize(environment)
    raise RuntimeError("inconsistent final conditioning execution state")


def materialize_dr04_release_addendum() -> dict[str, Any]:
    """Write the immutable reporting addendum from already-finalized DR04 evidence."""
    preflight()
    payload = build_release_addendum(
        FINAL_ROOT,
        vocabulary_path=str(VOCAB.relative_to(ROOT)).replace("\\", "/"),
        vocabulary_sha256=EXPECTED[VOCAB],
    )
    return materialize_release_addendum(RELEASE_ADDENDUM, payload)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("status", "final"), default="status"); parser.add_argument("--status-only", action="store_true"); parser.add_argument("--confirm-final-conditioning-eval", action="store_true"); parser.add_argument("--materialize-release-addendum", action="store_true"); args = parser.parse_args(argv)
    if args.confirm_final_conditioning_eval and args.mode != "final": parser.error("confirmation is valid only with --mode final")
    if args.materialize_release_addendum and args.mode != "status": parser.error("release addendum is valid only in status mode")
    if args.mode == "final" and not args.confirm_final_conditioning_eval: parser.error("final mode requires --confirm-final-conditioning-eval")
    if args.mode == "final": execute_final_scientific_run()
    result = materialize_dr04_release_addendum() if args.materialize_release_addendum else preflight()
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__": main()
