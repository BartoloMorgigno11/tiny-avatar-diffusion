# Native64 recovery inventory

All SHA-256 values below are complete lowercase digests.  Each snapshot file was copied as bytes and re-hashed; every source/destination pair matched exactly.

| Original absolute path | Recovered path | Bytes | SHA-256 | Source worktree provenance |
|---|---|---:|---|---|
| `C:\Users\Bartolo\Desktop\PROGETTI\DL_Project\src\tiny_avatar_diffusion\v2_matched_budget_training.py` | `src/tiny_avatar_diffusion/v2_matched_budget_training.py` | 11111 | `7a076cb1f67acb5975b8789a1cb746d7a781a4b692697726c8f8ab2681ff8bfc` | `DL_Project`, `release/four-model-2026`, `8315c01ec4eb262d8933a9dcf73370d39c269dc1`; untracked, no committed history |
| `C:\Users\Bartolo\Desktop\PROGETTI\DL_Project\src\tiny_avatar_diffusion\v2_matched_budget_runner.py` | `src/tiny_avatar_diffusion/v2_matched_budget_runner.py` | 9501 | `6e2cd9556ce8c537ad987f396264f1917dcae5c1f2b07c9bff1bf534ce0bdea4` | same |
| `C:\Users\Bartolo\Desktop\PROGETTI\DL_Project\src\tiny_avatar_diffusion\v2a_continuation_runner.py` | `src/tiny_avatar_diffusion/v2a_continuation_runner.py` | 15970 | `6ce5531f0304808a325e963bcfa5365d06820873683ea9c8fdd2cb6e3e26805b` | same |
| `C:\Users\Bartolo\Desktop\PROGETTI\DL_Project\scripts\run_v2a_matched_budget.py` | `scripts/run_v2a_matched_budget.py` | 214 | `b2992537874b155f0f8a783b73c4cfbb5381de6669d6e79ed125fcb53b8db921` | same |
| `C:\Users\Bartolo\Desktop\PROGETTI\DL_Project\scripts\run_v2a_matched_budget_continuation_60k.py` | `scripts/run_v2a_matched_budget_continuation_60k.py` | 164 | `4ba23a79e05cdf8e08f43d51d9e84c086c0b1b402db228a182be99c6e26fe84b` | same |
| `C:\Users\Bartolo\Desktop\PROGETTI\DL_Project\scripts\run_v2a_unconditional_native64_60k.py` | `scripts/run_v2a_unconditional_native64_60k.py` | 17911 | `63a0d65c96fe587546ea4f0a4b602139db572dcff24061061e60a4ed4883208d` | same |
| `C:\Users\Bartolo\Desktop\PROGETTI\DL_Project\scripts\package_v2a_conditional_ema60.py` | `scripts/package_v2a_conditional_ema60.py` | 1978 | `8191f432990d0b5369bcba97a80e13f7d558a495862543163c6fecf51bb47675` | same |
| `C:\Users\Bartolo\Desktop\PROGETTI\DL_Project\scripts\package_v2a_unconditional_ema60.py` | `scripts/package_v2a_unconditional_ema60.py` | 2685 | `928e999ec063744c8efcd13e60e67b059ac6043df89dd7fdf348e197ecd69901` | same |
| `C:\Users\Bartolo\Desktop\PROGETTI\DL_Project_V2_NATIVE64\scripts\build_v2_authentic64_cache.py` | `scripts/build_v2_authentic64_cache.py` | 4572 | `f076f0da81ebe8dde9cdf09d7e3cb115dee32a56ebdb272f94df46aeedce320c` | detached `DL_Project_V2_NATIVE64` at `88b6fdd647bfb081f79b4a33b4f2dffd50a85d55`; untracked, no committed history |

## Authenticated artifact relationships

| Model | Relationship | Complete SHA-256 |
|---|---|---|
| C64 | Matched-budget 40k full state / continuation source | `15cb3cfc580e549a748aadac513ca90e2d588b281b46fc59b15e1b267ccc54f2` |
| C64 | 60k continuation full state | `f47132dc9e247861c6697e41b5d352ccfa347ba6a19e48dcee7f061206cb0f14` |
| C64 | Released conditional EMA inference package | `42666355b5de3dfe5091f1c48036acb6782c76c2bed013a129b885cc460762e4` |
| U64 | 60k EMA source package | `0404caf40f5317d04a9972fbeb8759086e40ff9497a0596772c82b8031673015` |
| U64 | 60k terminal full-state checkpoint | `244b5fc36d81ce3014bb49bb6a4f2382af830c8d50ef371d94b8a71120e58826` |
| U64 | Released unconditional EMA inference package | `36925385824a4915f1e13a0fe0c896d88c7c57c3296d31f24703fa18e08e14d0` |

The C64 packaging source verifies the 60k full-state SHA before extracting its EMA weights.  The U64 packaging source verifies the successful-60k telemetry and records both the U64 EMA source and terminal-state SHA before producing the released package.

## Referenced historical assets (not copied)

| Asset | Local availability | Identity / note |
|---|---|---|
| C64 40k and 60k full-state checkpoints | Available in `DL_Project/results/...` | hashes above |
| U64 terminal full-state and EMA-60k source | Available in `DL_Project/results/runs/v2a_unconditional_native64_60k_seed20261021/` | terminal and EMA hashes above |
| Native64 TRAIN cache | Available in `DL_Project_V2_NATIVE64/data/v2_authentic_native64_train_20261017/` | cache SHA `7776ca0bc4c14b3789f9f0a6491fefae1a30c8801637b65eaafeebdd9e4a25fd`; 8,388 × 3 × 64 × 64 uint8 |
| TRAIN manifest | Available in `DL_Project/data/processed/dr01_v1/text/10k/train.jsonl` | SHA `295f7c93766294243871ebb2bf742a9fdae2f5a716ac773b0ead16dc8e6eb014` |
| Cache integrity report | Available beside the native64 cache | direct CartoonSet RGB 500→64 preprocessing; report SHA `162b33720153973a8db469708a50aab39a691eac2fb12dc73bf2915bace218c8` |
| Original CartoonSet archive | Not found at the expected inspected local path | expected `data/raw/cartoonset/archives/cartoonset10k.tgz`; cache report records archive SHA `902145ad5096af10ca9ee200b648e2a60e8b6b83dd2f4dc5356817988be8ff43` |

The official academic repository contains only this source snapshot and its records; none of the listed large assets was copied.
