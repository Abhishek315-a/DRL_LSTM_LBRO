# =============================================================
# simulator/task.py
# Task dataclass  (AICDQN Eq. 1)  +  Poisson IoT generator
# Dynamic priority score φ(i,t)    (AICDQN Eq. 4)
# =============================================================

import os
import ast
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from simulator.config import (GOOGLE_TRACE_PATH,
    NUM_DEVICES, TASK_ARRIVAL_PROB,
    TASK_SIZE_MIN, TASK_SIZE_MAX,
    PROCESSING_DENSITY,
    CPU_DEMAND_MIN, CPU_DEMAND_MAX,
    RAM_DEMAND_MIN, RAM_DEMAND_MAX,
    TASK_DEADLINE_SLOTS,
    TASK_CPU, TASK_MEM, TASK_BAL,
    TASK_PRIORITY_HIGH, TASK_PRIORITY_LOW,
    TASK_PRIORITY_HIGH_RATIO,
    THETA_1, THETA_2, THETA_3
)


# =============================================================
# Google Cluster Trace loader  (cached at class level)
# =============================================================

def _parse_google_trace(path: str):
    """
    Parse Google Borg cluster trace (2019) to extract CPU and RAM
    request distributions. Returns two numpy arrays scaled to the
    simulator's MIPS and MB ranges.

    Google fractions are normalised to machine capacity:
      - CPU:  1.0 ≈ 96,000 MIPS  (32 vCPUs × 3 GHz × 1000)
      - RAM:  1.0 = 131,072 MB   (128 GB)
    """
    if not os.path.exists(path):
        return None, None
    print(f"  Loading Google cluster trace → {path} ...")
    df = pd.read_csv(path, usecols=["resource_request"], nrows=300_000)
    cpu_vals, ram_vals = [], []
    for val in df["resource_request"].dropna():
        try:
            d = ast.literal_eval(val)
            c = d.get("cpus",   None)
            m = d.get("memory", None)
            if c and m and c > 0 and m > 0:
                cpu_vals.append(c)
                ram_vals.append(m)
        except Exception:
            continue
    if len(cpu_vals) < 1000:
        return None, None
    cpu_arr = np.clip(np.array(cpu_vals) * 96_000,  CPU_DEMAND_MIN, CPU_DEMAND_MAX)
    ram_arr = np.clip(np.array(ram_vals) * 131_072, RAM_DEMAND_MIN, RAM_DEMAND_MAX)
    print(f"  ✅ Trace parsed: {len(cpu_arr):,} tasks  "
          f"CPU [{cpu_arr.min():.0f}–{cpu_arr.max():.0f}] MIPS  "
          f"RAM [{ram_arr.min():.0f}–{ram_arr.max():.0f}] MB")
    return cpu_arr.astype(np.float32), ram_arr.astype(np.float32)


# =============================================================
# Task dataclass
# =============================================================

@dataclass
class Task:
    """
    Task tuple: τ_i = (S_i, β_i, D_it, p_i, T_arr, T_deadline)
    AICDQN paper, Eq. 1

    Fields:
        task_id         : unique integer ID
        device_id       : which IoT device generated this task
        size_mbits      : S_i  — task size in Mbits
        cpu_demand_mips : execution demand in MIPS
        ram_demand_mb   : RAM required in MB
        task_type       : 0=CPU_INTENSIVE, 1=MEM_INTENSIVE, 2=BALANCED
                          (heuristic here; RF classifier overrides in Step 3)
        static_priority : p_i ∈ {0.0, 1.0}
                          1.0 = real-time / urgent
                          0.0 = delay-tolerant
        arrival_slot    : T^arr_i  — simulation slot when task arrived
        deadline_slot   : T^deadline_i — must finish by this slot
    """

    # ── Core fields (Eq. 1) ───────────────────────────────────
    task_id         : int
    device_id       : int
    size_mbits      : float
    cpu_demand_mips : float
    ram_demand_mb   : float
    task_type       : int
    static_priority : float
    arrival_slot    : int
    deadline_slot   : int

    # ── Result fields (filled after execution) ────────────────
    assigned_node   : Optional[int]   = None  # 0/1/2=cloudlet, 3=cloud
    latency_s       : Optional[float] = None  # total end-to-end latency
    energy_j        : Optional[float] = None  # energy consumed
    dropped         : bool            = False  # True if queue was full
    deadline_missed : bool            = False  # True if finished after deadline

    # ══════════════════════════════════════════════════════════
    # Derived properties
    # ══════════════════════════════════════════════════════════

    @property
    def cpu_cycles(self) -> float:
        """
        C_i = S_i × β_i    (AICDQN Eq. 2)
        Total CPU cycles required to execute this task.
        S_i  = size in Mbits
        β_i  = PROCESSING_DENSITY in GCycles/Mbit
        Returns cycles (float).
        """
        return self.size_mbits * PROCESSING_DENSITY * 1e9

    @property
    def deadline_slack(self) -> int:
        """
        D_it = T_deadline - T_arr    (AICDQN Eq. 1)
        Number of slots available to complete this task.
        """
        return self.deadline_slot - self.arrival_slot

    @property
    def task_type_name(self) -> str:
        """Human-readable task type label."""
        return {
            TASK_CPU : "CPU_INTENSIVE",
            TASK_MEM : "MEM_INTENSIVE",
            TASK_BAL : "BALANCED"
        }.get(self.task_type, "UNKNOWN")

    # ══════════════════════════════════════════════════════════
    # Dynamic priority score  φ(i,t)  —  AICDQN Eq. 4
    # ══════════════════════════════════════════════════════════

    def dynamic_priority(self, current_slot: int,
                         queue_len: int,
                         queue_max: int) -> float:
        """
        φ(i,t) = θ1 × (1 / D_it)
               + θ2 × (Q_r(t) / Q_max_r)
               + θ3 × p_i

        AICDQN Eq. 4

        Args:
            current_slot : current simulation time slot
            queue_len    : current queue length at reference node
            queue_max    : max queue capacity at reference node

        Returns:
            phi : float — higher = more urgent, schedule first
        """
        # Remaining slots before deadline
        remaining = max(self.deadline_slot - current_slot, 1)

        # θ1 × deadline urgency  (increases as deadline approaches)
        urgency    = 1.0 / remaining

        # θ2 × queue congestion  (increases when queue is filling up)
        congestion = queue_len / max(queue_max, 1)

        # θ3 × static priority  (1.0 for real-time, 0.0 for tolerant)
        phi = (THETA_1 * urgency    +
               THETA_2 * congestion +
               THETA_3 * self.static_priority)

        return float(phi)

    # ══════════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════════

    def is_expired(self, current_slot: int) -> bool:
        """True if task has passed its hard deadline."""
        return current_slot > self.deadline_slot

    def to_feature_vector(self) -> list:
        """
        Returns raw feature list for RF classifier training (Step 3).
        Features: [cpu_demand, ram_demand, size_mbits, cpu_cycles]
        Label   : task_type
        """
        return [
            self.cpu_demand_mips,
            self.ram_demand_mb,
            self.size_mbits,
            self.cpu_cycles,
            self.static_priority,
        ]

    def __repr__(self):
        return (
            f"Task-{self.task_id} | "
            f"type={self.task_type_name:13s} | "
            f"cpu={self.cpu_demand_mips:6.0f} MIPS | "
            f"ram={self.ram_demand_mb:6.0f} MB | "
            f"size={self.size_mbits:.1f} Mb | "
            f"p={'HIGH' if self.static_priority else 'LOW '} | "
            f"deadline={self.deadline_slot}"
        )


# =============================================================
# IoT Task Generator
# =============================================================

class IoTTaskGenerator:
    """
    Poisson IoT task generator.

    Each of NUM_DEVICES devices independently generates a task
    with probability TASK_ARRIVAL_PROB per time slot.
    Task parameters (CPU demand, RAM demand) are sampled from the
    empirical distribution of the Google Borg cluster trace (2019).
    Falls back to uniform sampling if the trace file is not found.
    """

    _trace_cpu: np.ndarray = None   # class-level cache — loaded once
    _trace_ram: np.ndarray = None
    _trace_loaded: bool    = False

    @classmethod
    def _ensure_trace(cls):
        if cls._trace_loaded:
            return
        cls._trace_loaded = True
        cls._trace_cpu, cls._trace_ram = _parse_google_trace(GOOGLE_TRACE_PATH)

    def __init__(self, seed: int = 42):
        self.__class__._ensure_trace()
        self.rng           = np.random.default_rng(seed)
        self._task_counter = 0

    def generate(self, current_slot: int) -> list:
        """
        Generate tasks arriving at current_slot.

        Returns:
            list of Task objects (may be empty if no arrivals)
        """
        tasks = []
        for device_id in range(NUM_DEVICES):
            if self.rng.random() < TASK_ARRIVAL_PROB:
                tasks.append(
                    self._make_task(device_id, current_slot)
                )
        return tasks

    def _make_task(self, device_id: int, current_slot: int) -> Task:
        """
        Sample task parameters from Google Borg cluster trace distribution
        (or uniform fallback if trace unavailable).
        """
        self._task_counter += 1

        # CPU and RAM from Google trace (empirical) or uniform fallback
        if self.__class__._trace_cpu is not None:
            idx = int(self.rng.integers(0, len(self.__class__._trace_cpu)))
            cpu = float(self.__class__._trace_cpu[idx])
            ram = float(self.__class__._trace_ram[idx])
        else:
            cpu = float(self.rng.uniform(CPU_DEMAND_MIN, CPU_DEMAND_MAX))
            ram = float(self.rng.uniform(RAM_DEMAND_MIN, RAM_DEMAND_MAX))

        size  = float(self.rng.uniform(TASK_SIZE_MIN, TASK_SIZE_MAX))
        pri   = (TASK_PRIORITY_HIGH
                 if self.rng.random() < TASK_PRIORITY_HIGH_RATIO
                 else TASK_PRIORITY_LOW)

        # Heuristic type label (internal only — not used in DDQN state)
        cpu_norm = (cpu - CPU_DEMAND_MIN) / (CPU_DEMAND_MAX - CPU_DEMAND_MIN)
        ram_norm = (ram - RAM_DEMAND_MIN) / (RAM_DEMAND_MAX - RAM_DEMAND_MIN)

        if   cpu_norm > 0.65 : ttype = TASK_CPU
        elif ram_norm > 0.65 : ttype = TASK_MEM
        else                 : ttype = TASK_BAL

        return Task(
            task_id         = self._task_counter,
            device_id       = device_id,
            size_mbits      = size,
            cpu_demand_mips = cpu,
            ram_demand_mb   = ram,
            task_type       = ttype,
            static_priority = pri,
            arrival_slot    = current_slot,
            deadline_slot   = current_slot + TASK_DEADLINE_SLOTS,
        )

    def reset(self):
        """Reset counter — call at start of each episode."""
        self._task_counter = 0

    def generate_batch(self, num_tasks: int,
                       current_slot: int = 0) -> list:
        """
        Generate exactly num_tasks tasks regardless of arrival prob.
        Used by data_generator.py (Step 2) to build training dataset.
        """
        return [self._make_task(i % NUM_DEVICES, current_slot)
                for i in range(num_tasks)]
