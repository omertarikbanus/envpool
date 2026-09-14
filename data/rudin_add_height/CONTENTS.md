# rudin height-corrected arm, `add` semantics — seeds 1/2/3

**This is the rudin arm used in the three-arm comparison.** 500-iteration warm
starts off `data/rudin_add/v1_seed{1,2,3}`, `--base-height-scale -30
--base-height-target 0.30`. Each seed resumes from its OWN exact seed, so the
initial policy is identical and ride height is the only changed variable.

| seed | trained | base z (no-kick, n=100) | ladder |
|---|---|---|---|
| 1 | 2026-09-13 18:22-19:38 | 0.304 m | `20260913_rudinaddheight_seed1_n100` |
| 2 | 2026-09-14 12:10-13:21 | 0.298 m | `20260914_rudinaddheight_seed2_n100` |
| 3 | 2026-09-14 13:21-14:31 | 0.286 m | `20260914_rudinaddheight_seed3_n100` |

All three land within 1.4 cm of the 0.30 m target, up from the exact arm's
0.131 m crouch. The warm start exists because the from-scratch version of this
penalty never learns — see `data/rudin_height/ARCHIVAL_NOTE.md`.

**Caveat on the results.** The `fwd` row is non-monotonic on all three seeds
(3-seed mean dips to 34% at 3.5 m/s then rises to 53% at 4.0) and its seed
spread is +/-20.8 pp against <=7.4 pp everywhere else. Treat `fwd` as
unexplained; the seed SEM there is +/-12.0 points, not the +/-2.7 that pooling
300 episodes would imply.

**Only `model_500.pt` is committed** per seed, plus manifest, console log,
tfevents and source-provenance files. Intermediate checkpoints left on disk.

Simulator .so: `81f6ffcd15e0e9793bf0793078520d42e2774765b269ef79a43a073975018b67` (pinned across all three seeds and both ladders).
