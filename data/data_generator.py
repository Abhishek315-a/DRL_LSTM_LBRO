# =============================================================
# data/data_generator.py
# DRL-LSTM-LBRO  —  Step 2: Training Data Generator
#
# Generates TWO datasets:
#
# 1. data/workload_traces.csv   (10,000 rows)
#    → RF classifier training  (Step 3)
#    → Features: task characteristics
#    → Label   : task_type (0=CPU, 1=MEM, 2=BAL)
#
# 2. data/lstm_traces.csv       (50 episodes × 200 steps × 3 cloudlets)
#    → LSTM predictor training  (Step 4)
#    → Features: time-series cpu/ram/queue per cloudlet
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
    CPU_DEMAND_MIN, CPU_DEMAND_MAX,
    RAM_DEMAND_MIN, RAM_DEMAND_MAX,
    TASK_SIZE_MIN,  TASK_SIZE_MAX,
    DATA_DIR, WORKLOAD_CSV
)

# ── Output paths ──────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
RF_CSV   = WORKLOAD_CSV
LSTM_CSV = os.path.join(DATA_DIR, "lstm_traces.csv")

# ── Generation settings ───────────────────────────────────────
NUM_RF_TASKS      = 10_000
NUM_LSTM_EPISODES = 50

# ── Task type label map ───────────────────────────────────────
TYPE_NAMES = {0: "CPU_INTENSIVE", 1: "MEM_INTENSIVE", 2: "BALANCED"}


# =============================================================
# PART 1 — RF Classifier Dataset
# =============================================================

def generate_rf_dataset(num_tasks: int = NUM_RF_TASKS,
                        seed: int = 42) -> pd.DataFrame:
    """
    Generate task feature dataset for RF classifier training.

    Each row  = one task
    Features  = raw + normalised + engineered task characteristics
    Label     = task_type  (0=CPU_INTENSIVE, 1=MEM_INTENSIVE, 2=BALANCED)

    Feature engineering:
        cpu_ram_ratio   = cpu_demand / ram_demand
        compute_density = cpu_cycles / size_mbits
        load_score      = 0.6 × cpu_norm + 0.4 × ram_norm

    The RF classifier learns to predict task_type from these
    features so the LBRO broker can route tasks optimally:
        CPU tasks → highest MIPS cloudlet  (Cloudlet-0, 10k)
        MEM tasks → cloudlet with free RAM
        BAL tasks → least loaded cloudlet
    """
    print(f"\n[Step 2A] Generating RF dataset — {num_tasks:,} tasks...")

    gen  = IoTTaskGenerator(seed=seed)
    rows = []

    for i in tqdm(range(num_tasks), desc="  Building RF features", ncols=60):

        # Generate exactly one task per loop iteration
        task = gen._make_task(device_id=i % 50, current_slot=i)

        cpu  = task.cpu_demand_mips
        ram  = task.ram_demand_mb
        size = task.size_mbits
        pri  = task.static_priority
        cyc  = task.cpu_cycles       # C_i = S_i × β_i × 1e9  (AICDQN Eq. 2)

        # ── Normalised ────────────────────────────────────────
        cpu_n = (cpu  - CPU_DEMAND_MIN) / (CPU_DEMAND_MAX - CPU_DEMAND_MIN)
        ram_n = (ram  - RAM_DEMAND_MIN) / (RAM_DEMAND_MAX - RAM_DEMAND_MIN)
        siz_n = (size - TASK_SIZE_MIN)  / (TASK_SIZE_MAX  - TASK_SIZE_MIN)

        # ── Engineered ────────────────────────────────────────
        cpu_ram_ratio   = cpu / max(ram, 1.0)
        compute_density = cyc / max(size, 0.001)
        load_score      = 0.6 * cpu_n + 0.4 * ram_n

        rows.append({
            # ── Raw features ──────────────────────────────────
            "cpu_demand_mips"  : round(cpu,  2),
            "ram_demand_mb"    : round(ram,  2),
            "size_mbits"       : round(size, 4),
            "cpu_cycles"       : round(cyc,  2),
            "static_priority"  : pri,
            # ── Normalised features ───────────────────────────
            "cpu_norm"         : round(cpu_n, 6),
            "ram_norm"         : round(ram_n, 6),
            "size_norm"        : round(siz_n, 6),
            # ── Engineered features ───────────────────────────
            "cpu_ram_ratio"    : round(cpu_ram_ratio,   4),
            "compute_density"  : round(compute_density, 2),
            "load_score"       : round(load_score,      6),
            # ── Label ─────────────────────────────────────────
            "task_type"        : task.task_type,
            "task_type_name"   : TYPE_NAMES[task.task_type],
        })

    df = pd.DataFrame(rows)
    assert len(df) == num_tasks, f"Expected {num_tasks} rows, got {len(df)}"
    assert df["task_type"].nunique() == 3, "All 3 task types must be present"
    return df


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
    print("=" * 55)

    # ── Part 1: RF dataset ────────────────────────────────────
    rf_df = generate_rf_dataset()
    rf_df.to_csv(RF_CSV, index=False)

    print(f"\n  ✅ RF dataset saved  →  {RF_CSV}")
    print(f"     Rows      : {len(rf_df):,}")
    print(f"     Columns   : {list(rf_df.columns)}")
    print(f"\n     Task type distribution:")
    dist = rf_df["task_type_name"].value_counts()
    for name, count in dist.items():
        pct = 100 * count / len(rf_df)
        bar = "█" * int(pct / 2)
        print(f"       {name:15s}  {count:5,}  ({pct:.1f}%)  {bar}")

    # ── Part 2: LSTM dataset ──────────────────────────────────
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
    print(f"    {RF_CSV}")
    print(f"    {LSTM_CSV}")
    print(f"  Next → Step 3: models/rf_classifier.py")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
