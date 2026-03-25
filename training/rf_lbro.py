# =============================================================
# training/rf_lbro.py
# Baseline: Enhanced LBRO with RF classifier (prev. semester)
#
# RF classifies task as CPU-intensive / MEM-intensive / Balanced
# Weights in CRI change based on task type:
#   CPU-intensive → higher CPU weight
#   MEM-intensive → higher RAM weight
#   Balanced      → equal weights
# =============================================================

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.environment import LBROEnvironment
from models.rf_classifier  import RFTaskClassifier
from simulator.config      import (
    NUM_CLOUDLETS, MAX_STEPS_PER_EP, RESULTS_DIR,
    TASK_CPU, TASK_MEM, TASK_BAL
)

os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_EPISODES = 100
RESULTS_CSV   = os.path.join(RESULTS_DIR, "rf_lbro_results.csv")

# Weights per task type: [w_cpu, w_ram, w_queue]
WEIGHTS = {
    TASK_CPU: (0.6, 0.2, 0.2),   # CPU-intensive
    TASK_MEM: (0.2, 0.6, 0.2),   # MEM-intensive
    TASK_BAL: (0.33, 0.33, 0.34) # Balanced
}


def compute_rf_cri(cloudlets, task_type: int):
    w_cpu, w_ram, w_que = WEIGHTS.get(task_type, WEIGHTS[TASK_BAL])
    scores = []
    for c in cloudlets:
        cri = (w_cpu * (1 - c.cpu_util) +
               w_ram * (1 - c.ram_util) +
               w_que * (1 - c.queue_util))
        scores.append(cri)
    return int(np.argmax(scores))


def run(num_episodes=EVAL_EPISODES, seed=100):
    env    = LBROEnvironment(seed=seed)
    rf_clf = RFTaskClassifier.load()
    rows   = []

    for ep in range(num_episodes):
        state     = env.reset()
        ep_reward = 0.0

        for step in range(MAX_STEPS_PER_EP):
            task           = env.current_task
            task.task_type = rf_clf.predict(task)
            action         = compute_rf_cri(env.cloudlets, task.task_type)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break

        summary = env.episode_summary()
        rows.append({
            "policy"   : "RF-LBRO",
            "episode"  : ep,
            "reward"   : round(ep_reward,                4),
            "drop_rate": round(summary["drop_rate"],     4),
            "avg_lat"  : round(summary["avg_latency_s"], 4),
            "avg_eng"  : round(summary["avg_energy_j"],  4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    avg = df.mean(numeric_only=True)
    print(f"  RF-LBRO  drop={avg['drop_rate']:.2%}  "
          f"lat={avg['avg_lat']:.4f}s  "
          f"eng={avg['avg_eng']:.4f}J  "
          f"reward={avg['reward']:.2f}")
    print(f"  Saved → {RESULTS_CSV}")
    return df


if __name__ == "__main__":
    run()
