"""Live cardinal-sampler audit. Run inside envpool-dev after make run."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import setup_environment, load_model_and_normalization

n = 128
# One thread per env, as training does: with fewer threads than envs, every env
# past the thread count keeps a frozen gait phase, never reaches the onset cycle
# and is never pushed. Without this the audit below fails on ~n-num_threads envs.
env = setup_environment("Humanoid-v4", n, 7, env_config={
    "sim_config_path": "/app/quadcontrol/config/robots/sim/envpool_train_Gamma6.toml",
    "num_threads": n})
path = "/app/envpool/data/gamma3/quadruped_ppo_model_ckpt_30001920"
model, env = load_model_and_normalization(path + ".zip", env, path + "_vecnormalize.pkl")
obs = env.reset()
finished = np.zeros(n, dtype=bool)
integrals = np.zeros(n)
records = [None] * n
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, infos = env.step(action)
    for i in np.flatnonzero(~finished):
        info = infos[i]
        integrals[i] += np.linalg.norm(info["force_applied"]) * .01
        onset, t = float(info["force_onset"]), float(info["sim_time"])
        if (onset >= 0 and t >= onset + .25) or done[i]:
            records[i] = {"direction": int(info["force_direction"]),
                          "requested": float(info["force_requested_impulse"]),
                          "delivered": float(integrals[i]), "onset": onset,
                          "completed_pulse": onset >= 0 and t >= onset + .2}
            finished[i] = True
    if finished.all():
        break
assert finished.all(), "some envs never reached onset"
counts = {i: sum(r["direction"] == i for r in records) for i in range(-1, 6)}
assert all(counts[i] > 0 for i in range(6)), counts
assert 10 <= counts[-1] <= 45, counts
for r in records:
    assert r["onset"] >= 0
    if r["completed_pulse"]:
        assert abs(r["delivered"] - r["requested"]) <= .06 * r["requested"] + 1e-6, r
report = {"counts": counts, "episodes": records}
Path("/app/envpool/data/gamma6_force_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print("CARDINAL SAMPLER PASSED", counts, flush=True)
env.close()
