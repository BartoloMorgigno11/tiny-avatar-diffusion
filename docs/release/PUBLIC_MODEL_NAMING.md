# Public model naming

The release exposes exactly four model identifiers: `DDPM-U32`, `DDPM-C32`,
`DDPM-U64`, and `DDPM-C64`. The public checkpoint files are respectively
`ddpm-u32-ema-40k.pt`, `ddpm-c32-ema-40k.pt`, `ddpm-u64-ema-60k.pt`, and
`ddpm-c64-ema-60k.pt`.

`src/tiny_avatar_diffusion/public_model_manifest.json` is the authoritative
public mapping. Its `internal_id` field preserves the technical identity needed
by existing adapters and checkpoint provenance; it is not a user-facing model
selection. Frozen records and embedded checkpoint metadata remain unchanged.
