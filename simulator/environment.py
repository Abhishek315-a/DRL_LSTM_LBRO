# =============================================================
# simulator/environment.py
# DRL-LSTM-LBRO  —  LBRO Broker as MDP Environment
#
# Reward R (2-objective load balancing):
#   R = −(w_lat·D̃ + w_drop·P)
# =============================================================

import numpy as np
from collections import deque

from simulator.config import (
    NUM_CLOUDLETS, STATE_DIM, NUM_ACTIONS,
    MAX_STEPS_PER_EP, TIME_SLOT_S, LSTM_WINDOW,
    CPU_DEMAND_MIN, CPU_DEMAND_MAX,
    RAM_DEMAND_MIN, RAM_DEMAND_MAX,
    TASK_DEADLINE_SLOTS,
    W_LATENCY, W_DROP, W_IMBALANCE, W_OVERLOAD,
    LATENCY_MAX_S, CLOUD_LATENCY_NORM,
    TASK_CPU, TASK_MEM, TASK_BAL,
)
from simulator.cloudlet import Cloudlet
from simulator.cloud    import CloudNode
from simulator.task     import IoTTaskGenerator, Task


class LBROEnvironment:

    def __init__(self, seed: int = 42):
        self.seed      = seed
        self.rng       = np.random.default_rng(seed)

        self.cloudlets = [Cloudlet(i) for i in range(NUM_CLOUDLETS)]
        self.cloud     = CloudNode(seed=seed)
        self.generator = IoTTaskGenerator(seed=seed)

        self._cpu_hist     = np.zeros((NUM_CLOUDLETS, LSTM_WINDOW), dtype=np.float32)
        self._ram_hist     = np.zeros((NUM_CLOUDLETS, LSTM_WINDOW), dtype=np.float32)
        self.lstm_cpu_pred = np.zeros(NUM_CLOUDLETS, dtype=np.float32)
        self.lstm_ram_pred = np.zeros(NUM_CLOUDLETS, dtype=np.float32)
        self._lstm_predictors = None   # set via attach_lstm_predictors()

        self.w_lat     = float(W_LATENCY)
        self.w_drop    = float(W_DROP)
        self.w_imb     = float(W_IMBALANCE)
        self.w_overload = float(W_OVERLOAD)

        self.current_step = 0
        self.current_task = None
        self._task_buffer = deque()
        self._ep_stats    = self._blank_stats()


    def attach_lstm_predictors(self, predictors: list):
        """Call once after env creation to enable cold-start LSTM priming."""
        self._lstm_predictors = predictors

    def reset(self) -> np.ndarray:
        for c in self.cloudlets:
            c.reset()
        self.cloud.reset()
        self.generator.reset()

        self._cpu_hist     = np.zeros((NUM_CLOUDLETS, LSTM_WINDOW), dtype=np.float32)
        self._ram_hist     = np.zeros((NUM_CLOUDLETS, LSTM_WINDOW), dtype=np.float32)
        self.lstm_cpu_pred = np.zeros(NUM_CLOUDLETS, dtype=np.float32)
        self.lstm_ram_pred = np.zeros(NUM_CLOUDLETS, dtype=np.float32)

        self.w_lat      = float(W_LATENCY)
        self.w_drop     = float(W_DROP)
        self.w_imb      = float(W_IMBALANCE)
        self.w_overload = float(W_OVERLOAD)

        self.current_step = 0
        self._task_buffer = deque()
        self._ep_stats    = self._blank_stats()

        # Fix: prime LSTM predictions at episode start instead of using zeros
        if self._lstm_predictors is not None:
            for cid, predictor in enumerate(self._lstm_predictors):
                history = self.get_lstm_input(cid)   # all-zero window is fine here
                cpu_p, ram_p = predictor.predict(history)
                self.lstm_cpu_pred[cid] = cpu_p
                self.lstm_ram_pred[cid] = ram_p

        self.current_task = self._next_task()
        return self._build_state(self.current_task)


    def step(self, action: int):
        assert 0 <= action < NUM_ACTIONS, \
            f"Invalid action {action}. Must be 0–{NUM_ACTIONS - 1}"

        task = self.current_task

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

        reward = self._compute_reward(latency, dropped, deadline_violated, task)

        self._update_history()

        self._ep_stats["tasks"]   += 1
        self._ep_stats["energy"]  += energy
        self._ep_stats["actions"][action] += 1
        if dropped or deadline_violated:
            self._ep_stats["dropped"] += 1
        else:
            self._ep_stats["throughput"] += 1
            self._ep_stats["latency"]   += latency
            self._ep_stats["latency_n"] += 1
        for i, c in enumerate(self.cloudlets):
            self._ep_stats["cpu_util_sum"][i] += c.cpu_util

        for c in self.cloudlets:
            c.tick()

        self.current_step += 1
        done = (self.current_step >= MAX_STEPS_PER_EP)

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


    def _build_state(self, task: Task) -> np.ndarray:
        s = []

        for i, c in enumerate(self.cloudlets):
            s.append(float(c.cpu_util))
            s.append(float(c.ram_util))
            s.append(float(c.queue_util))
            s.append(float(c.is_critical))
            s.append(float(self.lstm_cpu_pred[i]))
            s.append(float(self.lstm_ram_pred[i]))

        cloud_q = min(self.cloud.queue_len / 100.0, 1.0)
        s.append(cloud_q)
        s.append(CLOUD_LATENCY_NORM)

        cpu_n = ((task.cpu_demand_mips - CPU_DEMAND_MIN) /
                 (CPU_DEMAND_MAX - CPU_DEMAND_MIN))
        ram_n = ((task.ram_demand_mb - RAM_DEMAND_MIN) /
                 (RAM_DEMAND_MAX - RAM_DEMAND_MIN))
        phi = task.dynamic_priority(
            current_slot = self.current_step,
            queue_len    = self.cloudlets[0].queue_len,
            queue_max    = self.cloudlets[0].q_max
        )

        s.append(float(np.clip(cpu_n, 0.0, 1.0)))
        s.append(float(np.clip(ram_n, 0.0, 1.0)))
        s.append(float(np.clip(phi,  0.0, 1.0)))

        state = np.array(s, dtype=np.float32)
        assert len(state) == STATE_DIM, \
            f"State dim error: got {len(state)}, expected {STATE_DIM}"
        return state


    def _compute_reward(self, latency, dropped, deadline_violated, task):
        d_norm = float(np.clip(latency / LATENCY_MAX_S, 0.0, 1.0))
        p_drop = 1.0 if (dropped or deadline_violated) else 0.0
        priority_mult = (1.5 if (task.static_priority == 1.0
                                and deadline_violated) else 1.0)

        # Imbalance: std of cpu_util across cloudlets (0 = perfectly balanced)
        cpu_utils = np.array([c.cpu_util for c in self.cloudlets], dtype=np.float32)
        imbalance = float(np.std(cpu_utils))

        # Queue saturation penalty: penalise any cloudlet with queue > 60% full
        # Grows quadratically above threshold so agent avoids hot-spotting
        OVERLOAD_THRESH = 0.6
        overload = 0.0
        for c in self.cloudlets:
            excess = float(c.queue_util) - OVERLOAD_THRESH
            if excess > 0.0:
                overload += excess ** 2

        reward = -(self.w_lat      * d_norm                 +
                   self.w_drop    * p_drop * priority_mult  +
                   self.w_imb     * imbalance               +
                   self.w_overload * overload)

        return float(reward)




    def _next_task(self) -> Task:
        if self._task_buffer:
            return self._task_buffer.popleft()

        for _ in range(20):
            batch = self.generator.generate(self.current_step)
            if batch:
                t = batch[0]
                for extra in batch[1:]:
                    self._task_buffer.append(extra)
                return t

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
        for i, c in enumerate(self.cloudlets):
            self._cpu_hist[i] = np.roll(self._cpu_hist[i], -1)
            self._ram_hist[i] = np.roll(self._ram_hist[i], -1)
            self._cpu_hist[i, -1] = c.cpu_util
            self._ram_hist[i, -1] = c.ram_util
        # lstm_cpu_pred and lstm_ram_pred are written by lstm_predictor
        # in train.py every 5 steps — do NOT overwrite here


    @staticmethod
    def _blank_stats() -> dict:
        return {
            "tasks"       : 0,
            "dropped"     : 0,
            "throughput"  : 0,
            "latency"     : 0.0,
            "latency_n"   : 0,
            "energy"      : 0.0,
            "actions"     : [0] * NUM_ACTIONS,
            "cpu_util_sum": [0.0] * NUM_CLOUDLETS,
        }


    def episode_summary(self) -> dict:
        s = self._ep_stats
        n = max(s["tasks"], 1)
        steps = max(self.current_step, 1)
        avg_cpu = [u / steps for u in s["cpu_util_sum"]]
        return {
            "total_tasks"     : s["tasks"],
            "throughput"      : s["throughput"],
            "drop_rate"       : s["dropped"]  / n,
            "success_rate"    : s["throughput"] / n,
            "avg_latency_s"   : s["latency"] / max(s["latency_n"], 1),
            "avg_energy_j"    : s["energy"]   / n,
            "action_dist"     : [a / n for a in s["actions"]],
            "avg_cpu_util"    : avg_cpu,
            "avg_resource_util": float(np.mean(avg_cpu)),
            "cloudlet_stats"  : [c.summary() for c in self.cloudlets],
            "cloud_stats"     : self.cloud.summary(),
        }

    @property
    def observation_space_dim(self) -> int:
        return STATE_DIM

    @property
    def action_space_dim(self) -> int:
        return NUM_ACTIONS

    def get_lstm_input(self, node_id: int) -> np.ndarray:
        return np.stack([
            self._cpu_hist[node_id],
            self._ram_hist[node_id]
        ], axis=-1)

    def __repr__(self):
        nodes = "\n  ".join(str(c) for c in self.cloudlets)
        return (f"LBROBroker | step={self.current_step}\n"
                f"  {nodes}\n"
                f"  {self.cloud}")
