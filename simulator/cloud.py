# =============================================================
# simulator/cloud.py
# Remote cloud — M/M/∞ queue (no queuing delay)
# WAN-dominated latency model  (AICDQN Eq. 10)
#
# D^CL = L^CL_trans + L^CL_net + 1/μ_CL
# =============================================================

import numpy as np

from simulator.config import (
    CLOUD_MIPS, CLOUD_RAM_MB, CLOUD_MAX_QUEUE,
    CLOUD_WAN_MBPS, CLOUD_PROP_DELAY_MS_MIN, CLOUD_PROP_DELAY_MS_MAX,
    P_TX_W, P_ACTIVE_W
)


class CloudNode:
    """
    Remote cloud modelled as M/M/∞ queue.
    Queuing delay is negligible (unlimited servers).
    Dominant cost = WAN transmission + propagation  (AICDQN Eq. 10).
    """

    def __init__(self, seed: int = 42):
        self.mips      = float(CLOUD_MIPS)
        self.ram_total = float(CLOUD_RAM_MB)
        self.q_max     = CLOUD_MAX_QUEUE
        self.rng       = np.random.default_rng(seed)

        # Live state
        self.queue_len  = 0
        self.cpu_util   = 0.0
        self.ram_util   = 0.0

        # Stats
        self.stat_assigned = 0
        self.stat_latency  = []
        self.stat_energy   = []

    def execute(self, task) -> dict:
        """
        Offload task to cloud.
        Latency = T_exec + T_tx_uplink + T_prop_WAN  (AICDQN Eq. 10)
        """
        self.stat_assigned += 1
        self.queue_len     += 1

        # T_exec = C_i / f_cloud  (fast — negligible vs WAN)
        t_exec = task.cpu_cycles / (self.mips * 1e6)

        # T_tx = S_i / B_WAN   (2 Mbps uplink — bottleneck)
        t_tx   = (task.size_mbits * 1e6) / (CLOUD_WAN_MBPS * 1e6)

        # T_prop = WAN round-trip latency  (stochastic)
        wan_ms  = self.rng.uniform(CLOUD_PROP_DELAY_MS_MIN,
                                   CLOUD_PROP_DELAY_MS_MAX)
        t_prop  = wan_ms / 1000.0           # → seconds

        total_latency = t_exec + t_tx + t_prop

        # Energy: uplink transmission (dominant) + minimal cloud compute
        e_tx      = P_TX_W  * t_tx
        e_compute = P_ACTIVE_W * t_exec * 0.01    # 1% share of cloud server
        energy    = e_tx + e_compute

        # Fill task
        task.assigned_node = 3              # cloud = action index 3
        task.latency_s     = total_latency
        task.energy_j      = energy

        self.queue_len = max(0, self.queue_len - 1)
        self.stat_latency.append(total_latency)
        self.stat_energy.append(energy)

        return {"latency": total_latency, "energy": energy, "dropped": False}

    @property
    def is_critical(self) -> bool:
        return False    # cloud never saturates (M/M/∞)

    def reset(self):
        self.queue_len     = 0
        self.cpu_util      = 0.0
        self.ram_util      = 0.0
        self.stat_assigned = 0
        self.stat_latency  = []
        self.stat_energy   = []

    def summary(self) -> dict:
        n = max(len(self.stat_latency), 1)
        return {
            "node"         : "cloud",
            "assigned"     : self.stat_assigned,
            "avg_latency_s": sum(self.stat_latency) / n,
            "avg_energy_j" : sum(self.stat_energy)  / n,
        }

    def __repr__(self):
        return f"CloudNode | Q={self.queue_len} | tasks={self.stat_assigned}"
