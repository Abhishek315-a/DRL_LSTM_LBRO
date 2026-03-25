# =============================================================
# training/round_robin.py
# Baseline: Round-Robin scheduling
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
RESULTS_CSV   = os.path.join(RESULTS_DIR, "round_robin_results.csv")


def run(num_episodes=EVAL_EPISODES, seed=100):
    env  = LBROEnvironment(seed=seed)
    rows = []

    for ep in range(num_episodes):
        state     = env.reset()
        ep_reward = 0.0

        for step in range(MAX_STEPS_PER_EP):
            action = step % NUM_CLOUDLETS        # 0→1→2→0→...
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break

        summary = env.episode_summary()
        rows.append({
            "policy"   : "Round-Robin",
            "episode"  : ep,
            "reward"   : round(ep_reward,                4),
            "drop_rate": round(summary["drop_rate"],     4),
            "avg_lat"  : round(summary["avg_latency_s"], 4),
            "avg_eng"  : round(summary["avg_energy_j"],  4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    avg = df.mean(numeric_only=True)
    print(f"  Round-Robin  drop={avg['drop_rate']:.2%}  "
          f"lat={avg['avg_lat']:.4f}s  "
          f"eng={avg['avg_eng']:.4f}J  "
          f"reward={avg['reward']:.2f}")
    print(f"  Saved → {RESULTS_CSV}")
    return df


if __name__ == "__main__":
    run()
