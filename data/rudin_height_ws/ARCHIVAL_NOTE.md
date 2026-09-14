# ARCHIVE ONLY — do not use for current results

**Arm:** rudin height-corrected, warm-started
**Archived:** 2026-09-14   **Trained:** 2026-09-13
**Stage:** 500-iteration warm starts, scale -30, target 0.30, seeds 1/2/3

## Why this is archived

**Trained under `push_semantics = "set"`.** These are the warm starts that
proved the method works — they reached base z = 0.304, 0.300 and 0.288 m
against a 0.30 m target, from the exact arm's 0.131 m crouch — but they were
trained before the study moved to `"add"` (true delta-v) in training as well
as evaluation (quadcontrol `80754b9`, envpool `315cf4f`). They never saw the
disturbance the study now uses, exactly like the `set`-semantics ours and
rudin models archived alongside them.

They are kept because they are the evidence that the warm start fixes the
`only_positive_rewards` failure recorded in `data/rudin_height/` — a finding
that is semantics-independent.

## What replaced it

`data/rudin_add_height/v1_seed{1,2,3}`: the same recipe re-run under `add`
semantics, each seed resumed from its OWN exact seed so the initial policy is
identical and ride height is the only changed variable. Measured base z =
0.304 / 0.298 / 0.286 m.

## Provenance

- Simulator .so at archival: `81f6ffcd15e0e9793bf0793078520d42e2774765b269ef79a43a073975018b67`
- Only final checkpoints are committed; intermediate checkpoints left on disk.
