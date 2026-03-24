# =============================================================
# simulator/cloudlet.py
# Heterogeneous edge cloudlet — M/M/c queue + Erlang-C
# Energy: DVFS ACTIVE/IDLE/SLEEP  (AICDQN energy-aware scheduler)
#
# FIX v2:
#   - cpu_util / ram_util now use EMA load (not instant alloc)
#   - tick() added for per-slot load decay (call from env.step)
#   - reset() clears EMA load fields
# =============================================================

import math
import numpy as np

from simulator.config import (
    CLOUDLET_MIPS, CLOUDLET_RAM_MB, CLOUDLET_SERVERS,
    CLOUDLET_MAX_QUEUE, EDGE_LAN_MBPS, EDGE_PROP_DELAY_MS,
    CPU_CRITICAL_THRESH, RAM_CRITICAL_THRESH,
    P_ACTIVE_W, P_IDLE_W, P_SLEEP_W, P_TX_W,
    ENERGY_ACTIVE, ENERGY_IDLE, ENERGY_SLEEP,
    TIME_SLOT_S, CPU_DEMAND_MIN, CPU_DEMAND_MAX
)


class Cloudlet:
    """
    Edge cloudlet node modelled as M/M/c queue.

    Queuing model (AICDQN Eq. 7-9):
        rho   = lambda / (c * mu)
        P_W   = Erlang-C formula  (Eq. 8)
        W_q   = P_W / (c*mu*(1-rho))  seconds  (Eq. 9)

    Total latency for task i:
        T_total = T_exec + W_q + T_tx + T_prop

    Energy (DVFS):
        E = P_state * T_exec + P_TX * T_tx

    FIX: cpu_util/ram_util use EMA load so LSTM sees
         non-zero persistent signal across slots.
    """

    def __init__(self, node_id: int):
        self.node_id    = node_id
        self.mips       = float(CLOUDLET_MIPS[node_id])
        self.ram_total  = float(CLOUDLET_RAM_MB[node_id])
        self.c          = CLOUDLET_SERVERS[node_id]
        self.q_max      = CLOUDLET_MAX_QUEUE[node_id]
        self.prop_delay = EDGE_PROP_DELAY_MS[node_id] / 1000.0  # → seconds

        # ── Live state ────────────────────────────────────────
        self.queue_len    = 0
        self.cpu_used     = 0.0
        self.ram_used     = 0.0
        self.energy_state = ENERGY_IDLE

        # ── EMA load trackers (FIX: persist across slots) ─────
        # instant release in execute() means cpu_used=0 always.
        # _cpu_load / _ram_load are EMA-smoothed and decay via tick().
        self._cpu_load   = 0.0
        self._ram_load   = 0.0
        self._load_decay = 0.85   # load fades 15% per slot

        # ── Per-cloudlet arrival rate EMA ─────────────────────
        self._lam_ema   = 1.0
        self._ema_alpha = 0.2

        # ── Statistics ────────────────────────────────────────
        self.stat_assigned = 0
        self.stat_dropped  = 0
        self.stat_latency  = []
        self.stat_energy   = []


    # ── Utilisation properties ────────────────────────────────

    @property
    def cpu_util(self) -> float:
        """EMA-smoothed CPU utilisation [0, 1]."""
        return min(self._cpu_load, 1.0)

    @property
    def ram_util(self) -> float:
        """EMA-smoothed RAM utilisation [0, 1]."""
        return min(self._ram_load, 1.0)

    @property
    def queue_util(self) -> float:
        return self.queue_len / max(self.q_max, 1)

    @property
    def is_critical(self) -> bool:
        return (self.cpu_util >= CPU_CRITICAL_THRESH or
                self.ram_util >= RAM_CRITICAL_THRESH)


    # ── Per-slot load decay ───────────────────────────────────

    def tick(self):
        """
        Called once per time slot from environment.step().
        Decays EMA load to simulate tasks completing over time.
        Also decays queue length gradually.
        """
        self._cpu_load  *= self._load_decay
        self._ram_load  *= self._load_decay
        self.queue_len   = max(0, int(self.queue_len * self._load_decay))
        self._update_energy_state()


    # ── Erlang-C (AICDQN Eq. 8) ──────────────────────────────

    def _erlang_c(self, lam: float, mu: float) -> float:
        """
        P_W(c, a) — probability arriving task must wait.
        lam: arrival rate to this cloudlet (tasks/s)
        mu : service rate per server       (tasks/s)
        """
        if mu <= 0 or lam <= 0:
            return 0.0
        a   = lam / mu
        rho = a / self.c
        if rho >= 1.0:
            return 1.0
        try:
            num = (a ** self.c / math.factorial(self.c)) / (1.0 - rho)
        except OverflowError:
            return 1.0
        den = sum(a**k / math.factorial(k) for k in range(self.c)) + num
        return num / den if den > 0 else 0.0


    # ── Queuing delay W_q (AICDQN Eq. 9) ─────────────────────

    def _queuing_delay(self) -> float:
        """W^ES_q = P_W / (c * mu * (1 - rho))  [seconds]"""
        avg_cpu = (CPU_DEMAND_MIN + CPU_DEMAND_MAX) / 2.0
        mu      = self.mips / avg_cpu
        lam     = self._lam_ema
        rho     = lam / (self.c * mu)

        if rho >= 1.0:
            service_time = avg_cpu / self.mips
            return self.queue_len * service_time

        p_w   = self._erlang_c(lam, mu)
        denom = self.c * mu * (1.0 - rho)
        return p_w / denom if denom > 0 else 0.0


    # ── Capacity check ────────────────────────────────────────

    def can_accept(self, task) -> bool:
        return (self.queue_len < self.q_max and
                self.ram_used + task.ram_demand_mb <= self.ram_total)


    # ── Execute task ──────────────────────────────────────────

    def execute(self, task) -> dict:
        """
        Place task on this cloudlet.
        Returns metrics dict: {latency, energy, dropped}.
        """
        # ── Drop if full ──────────────────────────────────────
        if not self.can_accept(task):
            self.stat_dropped += 1
            task.dropped = True
            return {"latency": 999.0, "energy": 0.0, "dropped": True}

        # ── Admit task ────────────────────────────────────────
        self.stat_assigned += 1
        self.queue_len     += 1
        self.cpu_used      += task.cpu_demand_mips
        self.ram_used      += task.ram_demand_mb

        # ── FIX: update EMA load (persists across slots) ──────
        self._cpu_load = min(1.0, self._cpu_load +
                            task.cpu_demand_mips / (self.mips * self.c))
        self._ram_load = min(1.0, self._ram_load +
                            task.ram_demand_mb   / self.ram_total)

        self._update_energy_state()
        self._update_lam()

        # ── Latency breakdown ─────────────────────────────────
        t_exec  = task.cpu_cycles / (self.mips * 1e6)          # AICDQN Eq. 3
        t_queue = self._queuing_delay()                          # AICDQN Eq. 9
        t_tx    = (task.size_mbits * 1e6) / (EDGE_LAN_MBPS * 1e6)  # LAN
        t_prop  = self.prop_delay

        total_latency = t_exec + t_queue + t_tx + t_prop

        # ── Energy ────────────────────────────────────────────
        power_map = {ENERGY_ACTIVE: P_ACTIVE_W,
                     ENERGY_IDLE  : P_IDLE_W,
                     ENERGY_SLEEP : P_SLEEP_W}
        e_compute = power_map[self.energy_state] * t_exec
        e_tx      = P_TX_W * t_tx
        energy    = e_compute + e_tx

        # ── Release instantaneous alloc (EMA load remains) ────
        self.queue_len = max(0, self.queue_len - 1)
        self.cpu_used  = max(0.0, self.cpu_used - task.cpu_demand_mips)
        self.ram_used  = max(0.0, self.ram_used - task.ram_demand_mb)
        self._update_energy_state()

        # ── Fill task fields ──────────────────────────────────
        task.assigned_node = self.node_id
        task.latency_s     = total_latency
        task.energy_j      = energy

        self.stat_latency.append(total_latency)
        self.stat_energy.append(energy)

        return {"latency": total_latency, "energy": energy, "dropped": False}


    # ── Internal helpers ──────────────────────────────────────

    def _update_lam(self):
        """EMA update of per-cloudlet arrival rate (tasks/sec)."""
        instant       = 1.0 / TIME_SLOT_S
        self._lam_ema = ((1 - self._ema_alpha) * self._lam_ema +
                          self._ema_alpha       * instant)

    def _update_energy_state(self):
        if   self.cpu_util > 0.10: self.energy_state = ENERGY_ACTIVE
        elif self.cpu_util > 0.0 : self.energy_state = ENERGY_IDLE
        else                     : self.energy_state = ENERGY_SLEEP


    # ── Reset ─────────────────────────────────────────────────

    def reset(self):
        self.queue_len    = 0
        self.cpu_used     = 0.0
        self.ram_used     = 0.0
        self.energy_state = ENERGY_IDLE
        self._cpu_load    = 0.0     # FIX: clear EMA load
        self._ram_load    = 0.0     # FIX: clear EMA load
        self._lam_ema     = 1.0
        self.stat_assigned = 0
        self.stat_dropped  = 0
        self.stat_latency  = []
        self.stat_energy   = []


    # ── Summary ───────────────────────────────────────────────

    def summary(self) -> dict:
        n = max(len(self.stat_latency), 1)
        return {
            "node"         : self.node_id,
            "mips"         : self.mips,
            "assigned"     : self.stat_assigned,
            "dropped"      : self.stat_dropped,
            "avg_latency_s": sum(self.stat_latency) / n,
            "avg_energy_j" : sum(self.stat_energy)  / n,
            "cpu_util"     : self.cpu_util,
            "ram_util"     : self.ram_util,
        }

    def __repr__(self):
        st = "CRIT" if self.is_critical else "OK  "
        es = ["ACT", "IDLE", "SLP"][self.energy_state]
        return (f"Cloudlet-{self.node_id}({self.mips:.0f}MIPS) "
                f"CPU={self.cpu_util:.0%} RAM={self.ram_util:.0%} "
                f"Q={self.queue_len}/{self.q_max} [{st}] E={es}")
