#!/usr/bin/env python3
"""Train the Rudin arm: unitree_rl_gym's Go2 recipe, exactly, on QuadrupedPD-v1.

Environment: `QuadrupedPD-v1` reproduces `LeggedRobot` + `GO2RoughCfg`
(unitree_rl_gym commit 276801e). Algorithm: rsl_rl v1.0.2, vendored unmodified
under `examples/third_party/rsl_rl`. The train config below is
`GO2RoughCfgPPO` over `LeggedRobotCfgPPO`, copied value for value.

Every run directory is self-describing: the manifest records both repositories'
HEAD, their uncommitted diffs (saved beside it), the envpool shared-object hash
and the vendored rsl_rl checksum, before the first step is taken. A run
directory is never overwritten.

Launch inside the container, console captured INSIDE the run directory (C++
output bypasses Python's sys.stdout, so it must be a shell redirect):

    RUN=data/rudin_unitree/trial1_seed1
    mkdir -p $RUN && python3 examples/train_unitree.py --run-dir $RUN \\
        > $RUN/console.log 2>&1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "third_party" / "rsl_rl"))
sys.path.insert(0, str(HERE))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402  (vendored v1.0.2)

from common.rsl_vec_env import TASK_ID, EnvPoolLeggedVecEnv  # noqa: E402

SIM_CONFIG = "/app/quadcontrol/config/robots/sim/envpool_train_rudin.toml"

# GO2RoughCfgPPO(LeggedRobotCfgPPO), unitree_rl_gym 276801e.
TRAIN_CFG = {
    "seed": 1,
    "runner_class_name": "OnPolicyRunner",
    "policy": {
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
    },
    "algorithm": {
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
    "runner": {
        "policy_class_name": "ActorCritic",
        "algorithm_class_name": "PPO",
        "num_steps_per_env": 24,
        "max_iterations": 1500,
        "save_interval": 50,
        "experiment_name": "rough_go2",
        "run_name": "",
        "resume": False,
        "load_run": -1,
        "checkpoint": -1,
        "resume_path": None,
    },
}
NUM_ENVS = 4096  # LeggedRobotCfg.env.num_envs
SOURCE_SUFFIXES = {".py", ".h", ".hh", ".hpp", ".cc", ".cpp", ".toml", ".md",
                   ".txt", ".xml", ".bzl", ".cfg", ""}


def set_seed(seed: int) -> None:
    """legged_gym.utils.helpers.set_seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _git(repo: Path, *args: str) -> str:
    # The container runs as root over host-owned checkouts, which git refuses
    # ("dubious ownership"). Git 2.34 ignores safe.directory passed with -c, so
    # a private global config is supplied instead of editing the container's.
    with tempfile.NamedTemporaryFile("w", suffix=".gitconfig") as cfg:
        cfg.write(f"[safe]\n\tdirectory = {repo}\n")
        cfg.flush()
        env = dict(os.environ, GIT_CONFIG_GLOBAL=cfg.name)
        try:
            return subprocess.run(["git", "-C", str(repo), *args], check=True,
                                  capture_output=True, text=True, env=env).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", "") or ""
            raise RuntimeError(f"git {' '.join(args)} failed in {repo}: "
                               f"{detail.strip() or exc}") from exc


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(run_dir: Path, args, train_cfg: dict, env_kwargs: dict) -> None:
    import envpool
    import envpool.mujoco as mj

    repos = {"envpool": HERE.parent, "quadcontrol": Path("/app/quadcontrol")}
    git = {}
    for name, repo in repos.items():
        diff = _git(repo, "diff", "HEAD", "--binary")
        untracked = _git(repo, "ls-files", "--others", "--exclude-standard")
        (run_dir / f"source_{name}.diff").write_text(diff)
        (run_dir / f"source_{name}_untracked.txt").write_text(untracked)
        # `git diff` omits untracked files, which is where new source lives
        # until it is committed. Archive their contents (run outputs excluded).
        with tarfile.open(run_dir / f"source_{name}_untracked.tar.gz", "w:gz") as tar:
            for rel in untracked.splitlines():
                path = repo / rel
                if (rel.startswith("data/") or not path.is_file()
                        or path.suffix not in SOURCE_SUFFIXES
                        or path.stat().st_size > 1 << 20):
                    continue
                tar.add(path, arcname=rel)
        git[name] = {"head": _git(repo, "rev-parse", "HEAD").strip(),
                     "branch": _git(repo, "branch", "--show-current").strip(),
                     "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}
    so_files = sorted(Path(mj.__file__).parent.glob("mujoco_gym_envpool*.so"))
    manifest = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_id": TASK_ID,
        "recipe": "unitree_rl_gym 276801e GO2RoughCfg / GO2RoughCfgPPO",
        "algorithm": "rsl_rl v1.0.2 (2ad79cf), vendored unmodified",
        "args": vars(args),
        "train_cfg": train_cfg,
        "env_kwargs": env_kwargs,
        "sim_config": {"path": args.sim_config_path,
                       "text": Path(args.sim_config_path).read_text()},
        "envpool_file": envpool.__file__,
        "envpool_so_sha256": {str(p): _sha256(p) for p in so_files},
        "rsl_rl_sha256sums": (HERE / "third_party/rsl_rl/SHA256SUMS").read_text(),
        "git": git,
        "torch": torch.__version__,
        "num_threads_torch": torch.get_num_threads(),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="new or empty directory; existing runs are never overwritten")
    p.add_argument("--num-envs", type=int, default=NUM_ENVS)
    p.add_argument("--seed", type=int, default=TRAIN_CFG["seed"])
    p.add_argument("--max-iterations", type=int,
                   default=TRAIN_CFG["runner"]["max_iterations"])
    p.add_argument("--num-threads", type=int, default=0,
                   help="EnvPool worker threads (0 = EnvPool default)")
    p.add_argument("--sim-config-path", default=SIM_CONFIG)
    p.add_argument("--no-push", action="store_true",
                   help="DEVIATION: disable legged_gym push_robots")
    args = p.parse_args()

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    clashes = [f.name for f in run_dir.iterdir() if f.name != "console.log"]
    if clashes:
        sys.exit(f"{run_dir} already holds a run ({clashes[:3]}...); refusing "
                 "to overwrite. Pick a new --run-dir.")

    train_cfg = json.loads(json.dumps(TRAIN_CFG))
    train_cfg["seed"] = args.seed
    train_cfg["runner"]["max_iterations"] = args.max_iterations
    env_kwargs = {"pd_push_robots": not args.no_push}

    set_seed(args.seed)
    write_provenance(run_dir, args, train_cfg, env_kwargs)
    print(f"[train_unitree] run_dir={run_dir} num_envs={args.num_envs} "
          f"seed={args.seed} iterations={args.max_iterations} "
          f"batch={args.num_envs * train_cfg['runner']['num_steps_per_env']}",
          flush=True)

    env = EnvPoolLeggedVecEnv(args.num_envs, args.seed, args.sim_config_path,
                              num_threads=args.num_threads, **env_kwargs)
    runner = OnPolicyRunner(env, train_cfg, log_dir=str(run_dir), device="cpu")
    # init_at_random_ep_len is realised inside the env (pd_init_at_random_ep_len),
    # so the runner's own randomisation of an unused buffer stays off.
    runner.learn(num_learning_iterations=args.max_iterations,
                 init_at_random_ep_len=False)
    env.close()
    print("[train_unitree] done", flush=True)


if __name__ == "__main__":
    main()
