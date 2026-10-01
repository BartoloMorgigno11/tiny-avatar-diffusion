"""Render the three canonical, evidence-only DR03 closeout figures.

Run from the repository root with ``python scripts/render_dr03_figures.py``.
This utility reads frozen CSV/JSON metadata and saved CPU preview tensors only;
it does not train, sample, evaluate KID, or require CUDA.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "reports" / "dr03_canonical"
PREVIEWS = (
    ROOT
    / "reports"
    / "dr03_main_training"
    / "dr03_conditional_main_10k_seed2026_r2"
    / "previews"
)
MILESTONES = (5000, 10000, 15000, 20000)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render_training_curve(summary: dict[str, object]) -> None:
    rows = read_csv(CANONICAL / "training_validation_metrics.csv")
    train = [row for row in rows if row["train_epsilon_mse_raw"]]
    validation = [row for row in rows if row["id_val_epsilon_mse_raw"]]
    steps = [int(row["step"]) for row in train]
    raw = [float(row["train_epsilon_mse_raw"]) for row in train]
    smooth = [float(row["train_epsilon_mse_smooth_100"]) for row in train]

    fig, axis = plt.subplots(figsize=(8.7, 5.0), layout="constrained")
    axis.plot(steps, raw, color="#8aa7bd", linewidth=0.65, alpha=0.55, label="Train (raw)")
    axis.plot(steps, smooth, color="#174f73", linewidth=1.6, label="Train (trailing 100-observation mean)")
    axis.scatter(
        [int(row["step"]) for row in validation],
        [float(row["id_val_epsilon_mse_raw"]) for row in validation],
        color="#bd5b36", marker="o", s=34, zorder=3, label="ID_VAL (RAW weights)",
    )
    axis.scatter(
        [int(row["step"]) for row in validation],
        [float(row["comp_val_epsilon_mse_raw"]) for row in validation],
        color="#5e7d31", marker="s", s=30, zorder=3, label="COMP_VAL (RAW weights)",
    )
    axis.set_title("DR03 Conditional Training and Validation epsilon-MSE")
    axis.set_xlabel("Successful optimizer step")
    axis.set_ylabel("epsilon-MSE")
    axis.set_xlim(left=0)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axis.legend(frameon=False, fontsize=8, loc="upper right")
    axis.text(
        0.01, 0.97,
        summary["training"]["metrics_export"]["smoothing"],
        transform=axis.transAxes, va="top", fontsize=8, color="#333333",
    )
    fig.savefig(CANONICAL / "training_validation_epsilon_mse.png", dpi=180)
    plt.close(fig)


def render_kid_curve() -> None:
    rows = read_csv(CANONICAL / "checkpoint_selection_kid.csv")
    steps = [int(row["step"]) for row in rows]
    fig, axis = plt.subplots(figsize=(8.7, 4.6), layout="constrained")
    axis.plot(steps, [float(row["id_val_kid"]) for row in rows], "o-", color="#bd5b36", linewidth=1.7, markersize=5, label="ID_VAL")
    axis.plot(steps, [float(row["comp_val_kid"]) for row in rows], "s-", color="#5e7d31", linewidth=1.7, markersize=5, label="COMP_VAL")
    axis.set_title("DR03 Validation-Only KID Checkpoint Guardrail")
    axis.set_xlabel("EMA milestone")
    axis.set_ylabel("KID (lower is better)")
    axis.set_xticks(steps, [f"{step // 1000}k" for step in steps])
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axis.legend(frameon=False, fontsize=9)
    axis.text(
        0.99, 0.97,
        "Validation-only checkpoint selection\n256 samples per split · EMA · Clean-FID 0.1.35 · clean mode\nNot final project KID",
        transform=axis.transAxes, va="top", ha="right", fontsize=8, color="#333333",
    )
    fig.savefig(CANONICAL / "validation_kid_by_milestone.png", dpi=180)
    plt.close(fig)


def preview_entries() -> list[dict[str, object]]:
    reference: list[dict[str, object]] | None = None
    for step in MILESTONES:
        manifest = json.loads((PREVIEWS / f"step_{step:06d}" / "manifest.json").read_text(encoding="utf-8"))
        if not (manifest["ema"] and manifest["clip_denoised"] and manifest["diagnostic_only"]):
            raise ValueError(f"preview manifest at {step} does not match the frozen protocol")
        entries = manifest["entries"]
        identity = [(entry["split"], entry["image_id"], entry["generation_seed"]) for entry in entries]
        if reference is None:
            reference = entries
            reference_identity = identity
        elif identity != reference_identity:
            raise ValueError("fixed preview cases differ across milestones")
    assert reference is not None
    return reference


def render_preview_panel() -> None:
    entries = preview_entries()
    fig, axes = plt.subplots(len(entries), len(MILESTONES), figsize=(8.7, 7.5))
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.075, top=0.90, wspace=0.08, hspace=0.06)
    for column, step in enumerate(MILESTONES):
        axes[0, column].set_title(f"{step // 1000}k", fontsize=11)
    for row, entry in enumerate(entries):
        label = f"{entry['split']}\nseed {entry['generation_seed']}"
        for column, step in enumerate(MILESTONES):
            folder = PREVIEWS / f"step_{step:06d}"
            tensor_path = folder / f"{row:02d}_{entry['split']}_{entry['image_id']}.pt"
            image = torch.load(tensor_path, map_location="cpu", weights_only=True)
            axes[row, column].imshow(image.permute(1, 2, 0).clamp(0, 1).numpy(), interpolation="nearest")
            axes[row, column].set_axis_off()
        position = axes[row, 0].get_position()
        fig.text(0.135, position.y0 + position.height / 2, label, ha="right", va="center", fontsize=8)
    fig.suptitle("DR03 Fixed EMA Preview Milestones", fontsize=13)
    fig.text(0.5, 0.018, "Fixed validation prompts and seeds · EMA · T = 1000 · clip_denoised = True · qualitative diagnostic only", ha="center", fontsize=8)
    fig.savefig(CANONICAL / "fixed_ema_preview_milestones.png", dpi=180)
    plt.close(fig)


def main() -> None:
    summary = json.loads((CANONICAL / "dr03_summary.json").read_text(encoding="utf-8"))
    render_training_curve(summary)
    render_kid_curve()
    render_preview_panel()


if __name__ == "__main__":
    main()
