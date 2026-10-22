# Model artifact provenance and distribution notice

The four checkpoint files in `artifacts/checkpoints/` are research artifacts
of this project, not Google Cartoon Set source files. They contain neither raw
Cartoon Set archives nor source images.

The distributed artifacts are `DDPM-U32`, `DDPM-C32`, `DDPM-U64`, and
`DDPM-C64`. Their paths, SHA-256 hashes, public roles, and provenance mapping
are authoritative in
[`public_model_manifest.json`](../../src/tiny_avatar_diffusion/public_model_manifest.json).
The manifest preserves technical identities in `internal_id`; frozen evidence
and embedded checkpoint metadata are not rewritten.

Google Cartoon Set attribution remains in
[`DATASET_ATTRIBUTION.md`](DATASET_ATTRIBUTION.md). The project code license
does not determine rights in data, checkpoint weights, generated images or
third-party components.
