#!/usr/bin/env python3
"""Run one evaluation case for a Rudin-arm (rsl_rl) checkpoint, optionally on video.

Policy side mirrors legged_gym's `play.py`: the actor's mean
(`ActorCritic.act_inference`), observation noise, pushes and friction
randomisation off. Evaluation side uses this project's shared protocol: a
fixed forward command (0.8 m/s by default, heading held at 0), one impulse
from `evaluations/core/forces.py` landed after three phase cycles, the
simulator's nominal friction, and survival scored exactly as
`evaluations/core/metrics.py` does -- body height inside [0.20, 0.75] m for the
whole 5.5 s post-onset window. The env's own early termination is disabled so
the trace and video always cover that window.

Run inside the container, one case per new output directory:

    python3 examples/eval_unitree.py \\
        --checkpoint data/rudin_unitree/trial1_seed1/model_1500.pt \\
        --out-dir /app/quadcontrol/evaluations/results/<new_dir> --video
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "third_party" / "rsl_rl"))
sys.path.insert(0, "/app/quadcontrol/evaluations")

from rsl_rl.modules import ActorCritic  # noqa: E402  (vendored v1.0.2)

import envpool  # noqa: E402
from core.forces import impulse, noforce  # noqa: E402
from core.metrics import T_EVAL, Z_MAX, Z_MIN  # noqa: E402

SIM_CONFIG = "/app/quadcontrol/config/robots/sim/envpool_train_rudin.toml"
POLICY_CFG = {"init_noise_std": 1.0, "actor_hidden_dims": [512, 256, 128],
              "critic_hidden_dims": [512, 256, 128], "activation": "elu"}
POLICY_DT = 0.02
# The joint-PD policy has no gait scheduler; the simulator's fallback phase
# clock (0.4 s, the nominal WBIC gait period) places the onset after three
# cycles, as for the other arms. It is invisible to the policy.
FALLBACK_PHASE_PERIOD = 0.4


def load_policy(path: Path):
    ac = ActorCritic(48, 48, 12, **POLICY_CFG)
    state = torch.load(path, map_location="cpu", weights_only=False)
    ac.load_state_dict(state["model_state_dict"])
    ac.eval()
    return ac.act_inference


def build_overlay(args, video_path: Path | None) -> str:
    spec = (impulse(args.impulse, args.direction, seed=args.seed)
            if args.impulse > 0 else noforce(seed=args.seed))
    lines = ["[simulation]",
             f"external_force_fallback_phase_period = {FALLBACK_PHASE_PERIOD}"]
    lines += spec._simulation_body()
    if video_path is not None:
        lines += ["headless = true", "video_record = true",
                  f'video_path = "{video_path}"', "video_width = 1280",
                  "video_height = 720", "video_fps = 50.0",
                  "camera_track = true"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--direction", default="fwd")
    ap.add_argument("--impulse", type=float, default=47.46, help="N·s; 0 = no push")
    ap.add_argument("--speed", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    case = f"{args.direction}_{args.impulse:g}Ns_v{args.speed:g}_seed{args.seed}"
    video = (out / f"rudin_{case}.mp4") if args.video else None
    overlay = build_overlay(args, video)
    (out / "sim_overlay.toml").write_text(overlay)

    policy = load_policy(args.checkpoint)
    env = envpool.make(
        "QuadrupedPD-v1", env_type="gymnasium", num_envs=1, seed=args.seed,
        num_threads=1, sim_config_path=SIM_CONFIG, sim_config_overlay=overlay,
        pd_fixed_command=True, pd_command_vx=args.speed, pd_command_vy=0.0,
        pd_command_heading=0.0, pd_obs_noise=False, pd_push_robots=False,
        pd_randomize_friction=False, pd_init_at_random_ep_len=False,
        terminate_when_unhealthy=False)

    obs, info = env.reset()
    rows, onset = [], -1.0
    for step in range(1001):
        with torch.no_grad():
            act = policy(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        obs, rew, term, trunc, info = env.step(act.astype(np.float64))
        g = obs[0, 6:9]
        rows.append({
            "t": float(info["sim_time"][0]),
            "z": float(info["body_height"][0]),
            "x": float(info["x_position"][0]),
            "y": float(info["y_position"][0]),
            "vx_world": float(info["x_velocity"][0]),
            "vy_world": float(info["y_velocity"][0]),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(-g[2], -1, 1)))),
            "fx": float(info["force_applied"][0][0]),
            "fy": float(info["force_applied"][0][1]),
            "fz": float(info["force_applied"][0][2]),
            "onset": float(info["force_onset"][0]),
            "reward": float(rew[0]),
        })
        onset = rows[-1]["onset"]
        if (onset >= 0 and rows[-1]["t"] >= onset + T_EVAL + 0.5) or term[0] or trunc[0]:
            break
    del env
    gc.collect()  # destroys the simulator, which closes the video encoder

    with (out / "trace.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    t = np.array([r["t"] for r in rows])
    z = np.array([r["z"] for r in rows])
    f_world = np.array([[r["fx"], r["fy"], r["fz"]] for r in rows])
    summary = {"checkpoint": str(args.checkpoint), "case": case,
               "command_mps": args.speed, "impulse_Ns_requested": args.impulse,
               "onset_time": onset, "steps": len(rows),
               "delivered_impulse_Ns": float(np.linalg.norm(f_world.sum(0)) * POLICY_DT)}
    if onset >= 0:
        win = (t >= onset) & (t <= onset + T_EVAL)
        inside = (z[win] >= Z_MIN) & (z[win] <= Z_MAX)
        first_out = t[win][~inside][0] - onset if (~inside).any() else None
        summary.update({
            "survived": bool(inside.all() and t[-1] >= onset + T_EVAL - 0.05),
            "survival_time_s": T_EVAL if first_out is None else float(first_out),
            "z_min_in_window": float(z[win].min()),
            "tilt_max_deg_in_window": float(max(r["tilt_deg"] for r, w_ in zip(rows, win) if w_)),
        })
    pre = t < (onset if onset >= 0 else t[-1])
    summary["z_mean_before_push"] = float(z[pre & (t > 1.0)].mean()) if (pre & (t > 1.0)).any() else None
    summary["vx_mean_before_push"] = (float(np.mean([r["vx_world"] for r, p in zip(rows, pre & (t > 1.0)) if p]))
                                      if (pre & (t > 1.0)).any() else None)
    if video is not None:
        summary["video"] = str(video)
        summary["video_bytes"] = video.stat().st_size if video.exists() else 0
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
