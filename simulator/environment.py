# =============================================================
# simulator/environment.py
# DRL-LSTM-LBRO  —  LBRO Broker as MDP Environment
#
# MDP = (S, A, P, R, γ)  per AICDQN Section "Problem Formulation"
#
# State  S  (23-dim normalised vector):
# ┌──────────────────────────────────────────────────────────┐
# │ Per cloudlet [3 × 6 = 18 dims]:                         │
# │   cpu_util | ram_util | queue_util |                     │
# │   is_critical | lstm_pred_cpu | lstm_pred_ram            │
# │ Cloud [2 dims]:                                          │
# │   cloud_queue_util | cloud_wan_latency_norm              │
# │ Task [3 dims]:                                           │
# │   cpu_norm | ram_norm | priority_score_norm              │
# └──────────────────────────────────────────────────────────┘
# Total = 18 + 2 + 3 = 23 ✅
#
# Actions A:
#   0 → Cloudlet-0  (10k MIPS)
#   1 → Cloudlet-1  (8k  MIPS)
#   2 → Cloudlet-2  (6k  MIPS)
#   3 → Cloud       (fallback)
#
# Reward R (LBRO load balancing — updated):
#   R = −(w_lat·D̃ + w_eng·Ẽ + w_drop·P + w_imb·σ_CPU)
#   σ_CPU = std of CPU utilisation across cloudlets
#   ↑ This term is what makes it LOAD BALANCING not just offloading
# =============================================================

import numpy as np
from collections import deque

from simulator.config import (
    # Infrastructure
    NUM_CLOUDLETS, STATE_DIM, NUM_ACTIONS,
    MAX_STEPS_PER_EP, TIME_SLOT_S, LSTM_WINDOW,
    # Task
    CPU_DEMAND_MIN, CPU_DEMAND_MAX,
    RAM_DEMAND_MIN, RAM_DEMAND_MAX,
    TASK_DEADLINE_SLOTS,
    # Reward
    W_LATENCY, W_ENERGY, W_DROP, W_ADAPT_RATE,
    LATENCY_MAX_S, ENERGY_MAX_J, CLOUD_LATENCY_NORM,
)
from simulator.cloudlet import Cloudlet
from simulator.cloud    import CloudNode
from simulator.task     import IoTTaskGenerator, Task

# Load imbalance weight — core LBRO objective
# σ_CPU = std(cpu_utils) across cloudlets → 0 = perfect balance
W_IMBALANCE = 0.5


class LBROEnvironment:
    """
    LBRO Broker MDP Environment.

    This class IS the broker — it:
      1. Receives incoming tasks from IoT devices
      2. Observes all cloudlet states (global view)
      3. Executes DDQN placement decisions
      4. Returns reward signal to train DDQN

    Usage:
        env   = LBROEnvironment(seed=42)
        state = env.reset()
        while not done:
            action = ddqn_agent.act(state)
            next_state, reward, done, info = env.step(action)
    """

    def __init__(self, seed: int = 42):
        self.seed      = seed
        self.rng       = np.random.default_rng(seed)

        # ── Broker maintains global view of all nodes ─────────
        self.cloudlets = [Cloudlet(i) for i in range(NUM_CLOUDLETS)]
        self.cloud     = CloudNode(seed=seed)
        self.generator = IoTTaskGenerator(seed=seed)

        # ── LSTM history buffers ──────────────────────────────
        # Shape: (NUM_CLOUDLETS, LSTM_WINDOW)
        # Step 4 (lstm_predictor.py) writes real predictions into
        # lstm_cpu_pred and lstm_ram_pred after training.
        # Until then, placeholder = last observed value.
        self._cpu_hist     = np.zeros((NUM_CLOUDLETS, LSTM_WINDOW),
                                       dtype=np.float32)
        self._ram_hist     = np.zeros((NUM_CLOUDLETS, LSTM_WINDOW),
                                       dtype=np.float32)
        self.lstm_cpu_pred = np.zeros(NUM_CLOUDLETS, dtype=np.float32)
        self.lstm_ram_pred = np.zeros(NUM_CLOUDLETS, dtype=np.float32)

        # ── Adaptive reward weights (AICDQN Eq. 17–19) ───────
        self.w_lat  = float(W_LATENCY)
        self.w_eng  = float(W_ENERGY)
        self.w_drop = float(W_DROP)
        self.w_imb  = float(W_IMBALANCE)

        # ── Episode state ─────────────────────────────────────
        self.current_step = 0
        self.current_task = None
        self._task_buffer = deque()
        self._ep_stats    = self._blank_stats()

    # ══════════════════════════════════════════════════════════
    # reset#()
    # ══════════════════════════════════════════════════════════

    def reset(self) -> np.ndarray:
        """
        Reset broker + all nodes at start of each episode.
        Returns initial 23-dim state vector.
        """
        for c in self.cloudlets:
            c.reset()
        self.cloud.reset()
        self.generator.reset()

        self._cpu_hist     = np.zeros((NUM_CLOUDLETS, LSTM_WINDOW),
                                       dtype=np.float32)
        self._ram_hist     = np.zeros((NUM_CLOUDLETS, LSTM_WINDOW),
                                       dtype=np.float32)
        self.lstm_cpu_pred = np.zeros(NUM_CLOUDLETS, dtype=np.float32)
        self.lstm_ram_pred = np.zeros(NUM_CLOUDLETS, dtype=np.float32)

        self.w_lat  = float(W_LATENCY)
        self.w_eng  = float(W_ENERGY)
        self.w_drop = float(W_DROP)
        self.w_imb  = float(W_IMBALANCE)

        self.current_step = 0
        self._task_buffer = deque()
        self._ep_stats    = self._blank_stats()

        self.current_task = self._next_task()
        return self._build_state(self.current_task)

    # ══════════════════════════════════════════════════════════
    # step#()  —  broker executes one placement decision
    # ══════════════════════════════════════════════════════════

    def step(self, action: int):
        """
        Broker executes DDQN placement decision.

        Args:
            action : int in {0, 1, 2, 3}
                     0/1/2 = assign to cloudlet-0/1/2
                     3     = fallback to cloud

        Returns:
            next_state : np.ndarray  shape (23,)
            reward     : float       always ≤ 0
            done       : bool
            info       : dict
        """
        assert 0 <= action < NUM_ACTIONS, \
            f"Invalid action {action}. Must be 0–{NUM_ACTIONS - 1}"

        task = self.current_task

        # ── Broker dispatches task to chosen node ─────────────
        if action < NUM_CLOUDLETS:
            result = self.cloudlets[action].execute(task)
        else:
            result = self.cloud.execute(task)

        latency = result["latency"]
        energy  = result["energy"]
        dropped = result["dropped"]

        # ── Deadline check ────────────────────────────────────
        slots_needed      = latency / TIME_SLOT_S
        deadline_violated = (
            not dropped and
            (self.current_step + slots_needed > task.deadline_slot)
        )
        if deadline_violated:
            task.deadline_missed = True

        # ── Broker computes reward ────────────────────────────
        reward = self._compute_reward(
            latency, energy, dropped, deadline_violated, task
        )

        # ── Update LSTM history window ────────────────────────
        self._update_history()

        # ── Update episode stats ──────────────────────────────
        self._ep_stats["tasks"]   += 1
        self._ep_stats["latency"] += latency
        self._ep_stats["energy"]  += energy
        self._ep_stats["actions"][action] += 1
        if dropped or deadline_violated:
            self._ep_stats["dropped"] += 1

                # ── Decay cloudlet loads each slot ────────────────────
        for c in self.cloudlets:
            c.tick()                          # ← FIX: EMA load decay

        # ── Advance step ──────────────────────────────────────
        self.current_step += 1
        done = (self.current_step >= MAX_STEPS_PER_EP)

        # ── Broker observes next state ────────────────────────
        self.current_task = self._next_task()
        next_state        = self._build_state(self.current_task)


        info = {
            "step"             : self.current_step,
            "action"           : action,
            "latency_s"        : latency,
            "energy_j"         : energy,
            "dropped"          : dropped,
            "deadline_violated": deadline_violated,
            "reward"           : reward,
        }
        return next_state, reward, done, info

    # ══════════════════════════════════════════════════════════
    # _build_state()  —  broker observes all nodes
    # ══════════════════════════════════════════════════════════

    def _build_state(self, task: Task) -> np.ndarray:
        """
        Broker collects 23-dim normalised state vector.
        Adapted from AICDQN Eq. 13 for LBRO broker perspective.

        Index map:
          [0–5]   Cloudlet-0: cpu, ram, queue, critical, lstm_cpu, lstm_ram
          [6–11]  Cloudlet-1: same 6 features
          [12–17] Cloudlet-2: same 6 features
          [18]    Cloud queue utilisation
          [19]    Cloud WAN latency normalised (fixed 0.25)
          [20]    Task cpu demand normalised
          [21]    Task ram demand normalised
          [22]    Task dynamic priority score φ(i,t)  AICDQN Eq. 4
        """
        s = []

        # ── Per-cloudlet features (3 × 6 = 18) ───────────────
        for i, c in enumerate(self.cloudlets):
            s.append(float(c.cpu_util))                  # current CPU util
            s.append(float(c.ram_util))                  # current RAM util
            s.append(float(c.queue_util))                # queue fill ratio
            s.append(float(c.is_critical))               # 1 if saturated
            s.append(float(self.lstm_cpu_pred[i]))       # predicted CPU (Step 4)
            s.append(float(self.lstm_ram_pred[i]))       # predicted RAM (Step 4)

        # ── Cloud features (2) ────────────────────────────────
        cloud_q = min(self.cloud.queue_len / 100.0, 1.0)
        s.append(cloud_q)                                # cloud queue util
        s.append(CLOUD_LATENCY_NORM)                     # 0.25 = 250ms/1000ms

        # ── Task features (3) ─────────────────────────────────
        cpu_n = ((task.cpu_demand_mips - CPU_DEMAND_MIN) /
                 (CPU_DEMAND_MAX - CPU_DEMAND_MIN))

        ram_n = ((task.ram_demand_mb - RAM_DEMAND_MIN) /
                 (RAM_DEMAND_MAX - RAM_DEMAND_MIN))

        # φ(i,t) — dynamic priority score  (AICDQN Eq. 4)
        phi = task.dynamic_priority(
            current_slot = self.current_step,
            queue_len    = self.cloudlets[0].queue_len,
            queue_max    = self.cloudlets[0].q_max
        )

        s.append(float(np.clip(cpu_n, 0.0, 1.0)))       # [20]
        s.append(float(np.clip(ram_n, 0.0, 1.0)))       # [21]
        s.append(float(np.clip(phi,  0.0, 1.0)))        # [22]

        # ── Validate ──────────────────────────────────────────
        state = np.array(s, dtype=np.float32)
        assert len(state) == STATE_DIM, \
            f"State dim error: got {len(state)}, expected {STATE_DIM}"
        return state

    # ══════════════════════════════════════════════════════════
    # _compute_reward()  —  LBRO load balancing reward
    # ══════════════════════════════════════════════════════════

    def _compute_reward(self, latency: float, energy: float,
                        dropped: bool, deadline_violated: bool,
                        task: Task) -> float:
        """
        LBRO Load Balancing Reward:

        R(t) = −( w_lat · D̃
                + w_eng · Ẽ
                + w_drop · P
                + w_imb · σ_dist )    ← LOAD BALANCING TERM

        where:
            D̃     = latency / LATENCY_MAX          ∈ [0,1]
            Ẽ     = energy  / ENERGY_MAX           ∈ [0,1]
            P     = 1 if dropped or deadline missed
            σ_dist = std of task assignment proportions
                     across cloudlets  ∈ [0, ~0.47]
                     0   = perfectly balanced  ← broker wants this
                     0.47= all tasks on one cloudlet (worst)
        """
        # ── Normalised delay and energy ───────────────────────────
        d_norm = float(np.clip(latency / LATENCY_MAX_S, 0.0, 1.0))
        e_norm = float(np.clip(energy  / ENERGY_MAX_J,  0.0, 1.0))

        # ── Drop / deadline penalty ───────────────────────────────
        p_drop = 1.0 if (dropped or deadline_violated) else 0.0

        # Extra 1.5× penalty for missing HIGH priority deadline
        priority_mult = (1.5 if (task.static_priority == 1.0
                                  and deadline_violated) else 1.0)

        # ── Load imbalance penalty ────────────────────────────────
        # σ_dist = std of proportion of tasks sent to each cloudlet
        # Measures how UNevenly the broker has distributed work
        # DDQN learns to minimise this → even load distribution
        total_assigned = sum(c.stat_assigned for c in self.cloudlets)
        if total_assigned > 0:
            proportions = [c.stat_assigned / total_assigned
                           for c in self.cloudlets]
            imbalance = float(np.std(proportions))
        else:
            imbalance = 0.0

        # ── Update episode imbalance stat ─────────────────────────
        self._ep_stats["imbalance"] += imbalance

        # ── Final reward ──────────────────────────────────────────
        reward = -(self.w_lat  * d_norm                  +
                   self.w_eng  * e_norm                  +
                   self.w_drop * p_drop * priority_mult  +
                   self.w_imb  * imbalance)

        # ── Adaptive weight update (AICDQN Eq. 17–19) ────────────
        self.w_lat = max(0.1, self.w_lat + W_ADAPT_RATE * (d_norm - 0.5))
        self.w_eng = max(0.1, self.w_eng + W_ADAPT_RATE * (e_norm - 0.5))

        return float(reward)

    # ══════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════

    def _next_task(self) -> Task:
        """
        Pop task from buffer or generate new batch.
        Always returns exactly one Task — never blocks.
        """
        if self._task_buffer:
            return self._task_buffer.popleft()

        for _ in range(20):
            batch = self.generator.generate(self.current_step)
            if batch:
                t = batch[0]
                for extra in batch[1:]:
                    self._task_buffer.append(extra)
                return t

        # Fallback: synthetic balanced task (extremely rare)
        return Task(
            task_id         = 0,
            device_id       = 0,
            size_mbits      = 3.5,
            cpu_demand_mips = 2000.0,
            ram_demand_mb   = 768.0,
            task_type       = 2,
            static_priority = 0.0,
            arrival_slot    = self.current_step,
            deadline_slot   = self.current_step + TASK_DEADLINE_SLOTS,
        )

    def _update_history(self):
        """
        Slide CPU/RAM history window left by one slot.
        LSTM predictor (Step 4) reads this via get_lstm_input().
        """
        for i, c in enumerate(self.cloudlets):
            self._cpu_hist[i] = np.roll(self._cpu_hist[i], -1)
            self._ram_hist[i] = np.roll(self._ram_hist[i], -1)
            self._cpu_hist[i, -1] = c.cpu_util
            self._ram_hist[i, -1] = c.ram_util

        # Placeholder LSTM prediction = latest observation
        # Step 4 replaces with real LSTM predictions
        self.lstm_cpu_pred = self._cpu_hist[:, -1].copy()
        self.lstm_ram_pred = self._ram_hist[:, -1].copy()

    @staticmethod
    def _blank_stats() -> dict:
        return {
            "tasks"    : 0,
            "dropped"  : 0,
            "latency"  : 0.0,
            "energy"   : 0.0,
            "actions"  : [0] * NUM_ACTIONS,
            "imbalance": 0.0,
        }

    # ══════════════════════════════════════════════════════════
    # episode_summary()
    # ══════════════════════════════════════════════════════════

    def episode_summary(self) -> dict:
        s = self._ep_stats
        n = max(s["tasks"], 1)
        return {
            "total_tasks"    : s["tasks"],
            "drop_rate"      : s["dropped"]   / n,
            "avg_latency_s"  : s["latency"]   / n,
            "avg_energy_j"   : s["energy"]    / n,
            "avg_imbalance"  : s["imbalance"] / n,   # ← NEW
            "action_dist"    : [a / n for a in s["actions"]],
            "cloudlet_stats" : [c.summary() for c in self.cloudlets],
            "cloud_stats"    : self.cloud.summary(),
        }


    # ══════════════════════════════════════════════════════════
    # Gym-compatible accessors
    # ══════════════════════════════════════════════════════════

    @property
    def observation_space_dim(self) -> int:
        return STATE_DIM

    @property
    def action_space_dim(self) -> int:
        return NUM_ACTIONS

    def get_lstm_input(self, node_id: int) -> np.ndarray:
        """
        Returns (LSTM_WINDOW, 2) array for node_id.
        Column 0 = cpu_history, Column 1 = ram_history.
        Used by lstm_predictor.py in Step 4.
        """
        return np.stack([
            self._cpu_hist[node_id],
            self._ram_hist[node_id]
        ], axis=-1)

    def __repr__(self):
        nodes = "\n  ".join(str(c) for c in self.cloudlets)
        return (f"LBROBroker | step={self.current_step}\n"
                f"  {nodes}\n"
                f"  {self.cloud}")
