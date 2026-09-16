# rudin + our curriculum, `add` semantics -- seed 1

The Rudin arm's height-corrected checkpoint (`rudin_add_height/v1_seed1/model_500.pt`,
base z = 0.304 m) continued 500 iterations under OUR cardinal velocity-kick
curriculum (six 3.0 m/s ceilings, add semantics, 20% no-force, matching
`envpool_train_v3_card_f.toml`'s terminal disturbance) instead of legged_gym's
own `push_robots`, which is disabled (`--no-push`).

The height term is kept ON throughout (`--base-height-scale -30
--base-height-target 0.30`) so ride height is held fixed while the disturbance
distribution is the only manipulated variable -- see the memory note below for
why an earlier attempt with the term off was aborted before any checkpoint.

| seed | trained | final mean reward | final mean ep length | base_height reward term |
|---|---|---|---|---|
| 1 | 2026-09-16 02:57-04:06 | 15.58 | 993.50 | -0.057 (steady) |

**DECLARED DEVIATION.** This is a curriculum-matched ablation, not Rudin et
al.'s recipe: the source's `max_push_vel_xy` is 1.0 m/s, x/y only. Report it
as such. The 50 Hz vs 100 Hz control-rate confound with the WBC arm is
unaffected by this change.

**Only `model_500.pt` is committed**, plus manifest, console log, tfevents and
source-provenance files. The ten intermediate 50-iteration checkpoints
(~44 MB) were left on disk, not committed, matching the convention in
`rudin_add/CONTENTS.md` and `rudin_add_height/CONTENTS.md`.

Simulator .so: `431313637a168865c3577692a8aeb365d744dfa7a8d3c734910df403a2e3a71f`
(pinned across training and both ladders below).

Evaluation: `20260916_rudin_cardf_seed1_n30` and `20260916_rudin_cardf_seed1_n100`
(survival-only, 6 directions, per-direction extension to a zero-survival
ceiling; n=100 confirms n=30 within sampling noise everywhere, largest move
+-15pp in the noisiest high-kick extension cells). Results summarised in
`RESULTS.md`. Full per-episode data:
`quadcontrol/evaluations/results/20260916_rudin_cardf_seed1_n30/` and
`.../20260916_rudin_cardf_seed1_n100/`.

See project memory `rudin-cardf-curriculum-arm.md`.
