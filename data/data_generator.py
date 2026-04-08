# =============================================================
# data/data_generator.py
# DRL-LSTM-LBRO  —  Step 2: Training Data Generator
#
# Generates ONE dataset:
#
# 1. data/lstm_traces.csv       (50 episodes × 200 steps × 3 cloudlets)
#    → LSTM predictor training  (Step 4)
#    → Features: time-series cpu/ram/queue per cloudlet
#    → Task parameters sampled from Google Borg cluster trace (2019)
#
# RF classifier has been removed from the architecture (STATE_DIM=23).
#
# Run with:
#     cd /Users/abhishek.sk/Documents/college/DRL_LSTM_LBRO
#     python3 -m data.data_generator
# =============================================================

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.environment import LBROEnvironment
from simulator.task        import IoTTaskGenerator
from simulator.config      import (
    NUM_CLOUDLETS, NUM_ACTIONS, MAX_STEPS_PER_EP,
    DATA_DIR, LSTM_CSV_PATH,
)

# ── Output paths ──────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
LSTM_CSV = LSTM_CSV_PATH

# ── Generation settings ──────────────────────────────────────
NUM_LSTM_EPISODES = 50


# =============================================================
# PART 2 — LSTM Time-Series Dataset
# =============================================================

def generate_lstm_dataset(num_episodes: int = NUM_LSTM_EPISODES,
                          seed: int = 0) -> pd.DataFrame:
    """
    Run simulation episodes with round-robin action policy.
    Record per-slot cloudlet metrics for LSTM training.

    Each row = one (episode, slot, cloudlet_id) observation.

    The LSTM predictor (Step 4) trains on this time-series data
    to predict next-slot cpu_util and ram_util for each cloudlet.
    These predictions feed directly into the 23-dim state vector
    at indices [4],[5],[10],[11],[16],[17].

    Columns:
        episode      : episode index (0 to num_episodes-1)
        slot         : time slot within episode (0 to 199)
        cloudlet_id  : 0, 1, or 2
        cpu_util     : CPU utilisation [0,1]
        ram_util     : RAM utilisation [0,1]
        queue_util   : queue fill ratio [0,1]
        is_critical  : 1 if node is saturated
        action       : placement decision (0-3)
        latency_s    : task latency that slot
        energy_j     : task energy that slot
        reward       : reward signal that slot
    """
    print(f"\n[Step 2B] Generating LSTM dataset — "
          f"{num_episodes} episodes × {MAX_STEPS_PER_EP} steps...")

    env  = LBROEnvironment(seed=seed)
    rows = []

    for ep in tqdm(range(num_episodes), desc="  Running episodes", ncols=60):
        env.reset()

        for step in range(MAX_STEPS_PER_EP):
            # Round-robin policy — ensures all cloudlets get traffic
            action = step % NUM_ACTIONS
            _, reward, done, info = env.step(action)

            # Record state of all 3 cloudlets at this slot
            for cid, c in enumerate(env.cloudlets):
                rows.append({
                    "episode"     : ep,
                    "slot"        : step,
                    "cloudlet_id" : cid,
                    "cpu_util"    : round(float(c.cpu_util),   6),
                    "ram_util"    : round(float(c.ram_util),   6),
                    "queue_util"  : round(float(c.queue_util), 6),
                    "is_critical" : int(c.is_critical),
                    "action"      : info["action"],
                    "latency_s"   : round(info["latency_s"],   6),
                    "energy_j"    : round(info["energy_j"],    6),
                    "reward"      : round(info["reward"],      6),
                })

            if done:
                break

    return pd.DataFrame(rows)


# =============================================================
# MAIN
# =============================================================

def main():
    print("=" * 55)
    print("  DRL-LSTM-LBRO  —  Step 2: Data Generator")
    print("  (Task parameters from Google Borg cluster trace 2019)")
    print("=" * 55)

    lstm_df = generate_lstm_dataset()
    lstm_df.to_csv(LSTM_CSV, index=False)

    print(f"\n  ✅ LSTM dataset saved →  {LSTM_CSV}")
    print(f"     Rows      : {len(lstm_df):,}")
    print(f"     Columns   : {list(lstm_df.columns)}")
    print(f"\n     Per-cloudlet avg utilisation:")
    for cid in range(NUM_CLOUDLETS):
        sub = lstm_df[lstm_df["cloudlet_id"] == cid]
        print(f"       Cloudlet-{cid}  "
              f"cpu={sub['cpu_util'].mean():.4f}  "
              f"ram={sub['ram_util'].mean():.4f}  "
              f"queue={sub['queue_util'].mean():.4f}")

    print(f"\n{'=' * 55}")
    print(f"  Step 2 COMPLETE ✅")
    print(f"  Files generated:")
    print(f"    {LSTM_CSV}")
    print(f"  Next → Step 4: models/lstm_predictor.py")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
