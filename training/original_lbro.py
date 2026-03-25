# =============================================================
# training/original_lbro.py
# Baseline: Original LBRO (Nayyer et al.)
#
# Composite Resource Index (CRI) — equal weights:
#   CRI_i = (1 - cpu_util) * (1 - ram_util) * (1 - queue_util)
#   Broker picks cloudlet with highest CRI
# =============================================================

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.environment import LBROEnvironment
from simulator.config      import (
    NUM_CLOUDLETS, MAX_STEPS_PER_EP, RESULTS_DIR
)

os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_EPISODES = 100
RESULTS_CSV   = os.path.join(RESULTS_DIR, "original_lbro_results.csv")


def compute_cri(cloudlets):
    """
    Original LBRO composite resource index.
    Equal weights for CPU, RAM, queue — no ML.
    """
    scores = []
    for c in cloudlets:
        cri = (1 - c.cpu_util) * (1 - c.ram_util) * (1 - c.queue_util)
        scores.append(cri)
    return int(np.argmax(scores))


def run(num_episodes=EVAL_EPISODES, seed=100):
    env  = LBROEnvironment(seed=seed)
    rows = []

    for ep in range(num_episodes):
        state     = env.reset()
        ep_reward = 0.0

        for step in range(MAX_STEPS_PER_EP):
            action = compute_cri(env.cloudlets)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break

        summary = env.episode_summary()
        rows.append({
            "policy"   : "Original-LBRO",
            "episode"  : ep,
            "reward"   : round(ep_reward,                4),
            "drop_rate": round(summary["drop_rate"],     4),
            "avg_lat"  : round(summary["avg_latency_s"], 4),
            "avg_eng"  : round(summary["avg_energy_j"],  4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    avg = df.mean(numeric_only=True)
    print(f"  Original-LBRO  drop={avg['drop_rate']:.2%}  "
          f"lat={avg['avg_lat']:.4f}s  "
          f"eng={avg['avg_eng']:.4f}J  "
          f"reward={avg['reward']:.2f}")
    print(f"  Saved → {RESULTS_CSV}")
    return df


if __name__ == "__main__":
    run()
