# rudin exact arm, `add` semantics — seeds 1/2/3

The unitree_rl_gym GO2RoughCfg recipe reproduced exactly, trained against the
v1 plant with an additive (delta-v) push. 1500 iterations, 4096 envs, seed = S.

| seed | trained | final mean reward | final mean ep length |
|---|---|---|---|
| 1 | 2026-09-13 14:52-18:04 | 22.85 | 981.37 |
| 2 | 2026-09-14 06:22-09:14 | 23.40 | 1001.00 |
| 3 | 2026-09-14 09:14-12:09 | 23.16 | 1001.00 |

**These are warm-start sources, not a measured arm.** This arm walks at base
z = 0.131 m — a crouch nothing in the recipe prices, since `GO2RoughCfg` sets
`base_height` scale to 0 — so it was excluded from the three-arm comparison by
the user on 2026-09-14. It is committed because
`data/rudin_add_height/v1_seed{1,2,3}` is warm-started from it, each height
seed from its own matching exact seed.

**Only `model_1500.pt` is committed** per seed, plus the manifest, console log,
tfevents and source-provenance files. The 30 intermediate 50-iteration
checkpoints per seed (~420 MB) were left on disk.

Simulator .so: `81f6ffcd15e0e9793bf0793078520d42e2774765b269ef79a43a073975018b67`
Note seed 1 trained against `61a4547...`; the binary was rebuilt at 20:49 on
09-13 for the ours lineage. The only post-seed-1 change to this env is
`a753901`, whose `pd_command_vx_jitter` defaults to 0.0 and acts only under
`pd_fixed_command`, which `train_unitree.py` never sets — so training
behaviour is unchanged and the three seeds are comparable.
