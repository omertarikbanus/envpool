# Working rules for this workspace

## 1. Obey the user's parameter decisions. Never override them.

When the user specifies a value — learning rate, step count, ceiling,
constant, config, checkpoint, anything — that value is used **exactly as
given**. This is absolute and has no exceptions.

If a different value seems better, **ask first and wait for an answer.** Do not
pick your own value and announce it afterwards. Announcing an override is still
an override: the run has already started, and the hours are already spent.

This rule exists because it was broken. When run_v2_ours_wide.sh launched, the
user's recipe ended at 1e-6; the assistant chose 1e-5 instead, stated the
choice after the fact, and that 1e-5 then propagated silently through v4, v5,
v6 and v7 — every model trained after v1. Hours of compute went into a learning
rate the user had not agreed to.

## 2. Print a full run plan BEFORE starting any training or evaluation run.

Every run, without exception. The plan must state:

  - parent checkpoint (path + sha256) and the run's output directory
  - config file, and every value that differs from the parent's config
  - learning rate, total steps, checkpoint frequency
  - which values came from the user, and which are unchanged defaults
  - what the run is testing, and what result would count as success
  - expected wall-clock time

If ANY value in that plan was not specified by the user, say so explicitly and
ask before launching.

## 3. Never change a running experiment's parameters on your own judgment.

Stopping a run, changing a ceiling, switching a config, reverting a constant,
re-seeding from a different parent — all of these need the user to say so
first. Report what the data shows, recommend if asked, then wait.

## 4. Report faithfully.

Do not claim a budget is exhausted, a step is unnecessary, or a result is
conclusive unless it is verifiably true. If you got something wrong, say so in
one sentence and correct it.
