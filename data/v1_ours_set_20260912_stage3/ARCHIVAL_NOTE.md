# ARCHIVE ONLY — do not use for current results

**Arm:** ours (learned GRF + WBIC), v1 protocol
**Archived:** 2026-09-13   **Measured/trained:** 2026-09-12
**Stage:** 3 — 10M @ 1e-6, continued from stage 2 (40M cumulative)

## Why this is archived

**Trained under `push_semantics = "set"`.** The simulator push overwrote the
base velocity. quadcontrol `e443d08` and `80754b9` moved the study to
`"add"` (true delta-v) for **training as well as evaluation**, so this model
never saw the disturbance the study now uses. Re-measuring this same model
under `add` moves a single cell by up to 50 survival points
(`right_2p5mps`: 0% -> 50%), and reverses which direction is weakest.

**Trained at seed 0.** `quadcontrol/docs/OURS_MULTISEED_PROMPT.md` requires
seeds 1, 2, 3 with `--force-seed` set per seed, so the arms' tables line up.
Seed 0 is not one of them and has no companion seeds.

## Provenance

- Host: tarik-GE76-Dragon-Tiamat-11UH (laptop), **not** the reference Z8 Fury
- Simulator .so: `73f18e2d5990ddbee7f1313d7832edb44212b861415d90747391790bbcf8212c`
- Config: `/app/quadcontrol/config/robots/sim/envpool_train_v1.toml` (pre-`add`)
- Recipe: Gamma5 two-stage from scratch — 20M @ 1e-5, then 10M @ 1e-6
- 256 envs, `--adaptive-lr 0 --target-kl 0.01`

- Checkpoints every 1M (10 total), for checkpoint screening
- Added to test whether v1_ours was simply short of gamma5's 40M lineage.
  Measured n=10 under `add` semantics: **mean -0.2 survival points vs stage 2**,
  i.e. the extra 10M bought nothing detectable. Evidence against the
  undertraining explanation for the gamma5_stage3 gap.

## What replaced it

Nothing yet. The ours arm needs retraining under `add` semantics at seeds
1/2/3 before it can be compared against the kim and rudin arms. The models at
`data/v1_ours/` and `data/v1_ours_stage2/` are the reference machine's
earlier artifacts and are untouched by this archival.

Evaluation measured from this lineage is committed separately in quadcontrol at
`evaluations/analysis/baselines/ours_v1_set_20260912/`, carrying the same warning.
