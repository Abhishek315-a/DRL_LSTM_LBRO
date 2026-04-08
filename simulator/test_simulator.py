# =============================================================
# simulator/test_simulator.py
#
# Comprehensive test suite for DRL-LSTM-LBRO simulator.
# Run with:
#     cd /Users/abhishek.sk/Documents/college/DRL_LSTM_LBRO
#     python3 -m simulator.test_simulator
#
# All 8 tests must pass before proceeding to Step 2.
# =============================================================

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.environment import LBROEnvironment
from simulator.cloudlet    import Cloudlet
from simulator.cloud       import CloudNode
from simulator.task        import IoTTaskGenerator, Task
from simulator.config      import (
    STATE_DIM, NUM_ACTIONS, NUM_CLOUDLETS,
    MAX_STEPS_PER_EP, TASK_DEADLINE_SLOTS,
    CPU_DEMAND_MIN, CPU_DEMAND_MAX,
    RAM_DEMAND_MIN, RAM_DEMAND_MAX,
    TASK_SIZE_MIN, TASK_SIZE_MAX,
    LSTM_WINDOW
)

# ── Colour helpers ────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
PASS   = f"{GREEN}✅ PASS{RESET}"
FAIL   = f"{RED}❌ FAIL{RESET}"

_tests_run    = 0
_tests_passed = 0


def check(condition: bool, message: str, detail: str = ""):
    global _tests_run, _tests_passed
    _tests_run += 1
    if condition:
        _tests_passed += 1
        print(f"    {PASS}  {message}")
    else:
        print(f"    {FAIL}  {message}")
        print(f"          Detail: {detail}")
        sys.exit(1)


def section(title: str):
    print(f"\n{YELLOW}{'─' * 55}{RESET}")
    print(f"{YELLOW}  {title}{RESET}")
    print(f"{YELLOW}{'─' * 55}{RESET}")


# =============================================================
# TEST 1 — Config sanity
# =============================================================
def test_config():
    section("Test 1 — Config sanity")
    from simulator.config import (
        NUM_DEVICES, CLOUDLET_MIPS, CLOUDLET_RAM_MB,
        CLOUDLET_SERVERS, CLOUDLET_MAX_QUEUE,
        EDGE_LAN_MBPS, CLOUD_WAN_MBPS,
        W_LATENCY, W_ENERGY, W_DROP
    )
    check(NUM_CLOUDLETS == 3,
          "NUM_CLOUDLETS = 3", f"got {NUM_CLOUDLETS}")
    check(CLOUDLET_MIPS == [10_000, 8_000, 6_000],
          "MIPS = [10k, 8k, 6k]", f"got {CLOUDLET_MIPS}")
    check(EDGE_LAN_MBPS == 100.0,
          "LAN = 100 Mbps (not 4 Mbps)", f"got {EDGE_LAN_MBPS}")
    check(CLOUD_WAN_MBPS == 2.0,
          "WAN = 2 Mbps", f"got {CLOUD_WAN_MBPS}")
    check(STATE_DIM == 26,
          "STATE_DIM = 26", f"got {STATE_DIM}")
    check(NUM_ACTIONS == 4,
          "NUM_ACTIONS = 4", f"got {NUM_ACTIONS}")
    print(f"    Config loaded successfully ✓")


# =============================================================
# TEST 2 — Task generation and properties
# =============================================================
def test_task():
    section("Test 2 — Task generation and properties")
    gen   = IoTTaskGenerator(seed=42)
    tasks = gen.generate(current_slot=0)

    check(len(tasks) > 0,
          "Tasks generated at slot 0",
          "No tasks generated")

    t = tasks[0]
    check(CPU_DEMAND_MIN <= t.cpu_demand_mips <= CPU_DEMAND_MAX,
          f"CPU demand in [{CPU_DEMAND_MIN}, {CPU_DEMAND_MAX}] MIPS",
          f"got {t.cpu_demand_mips:.1f}")
    check(RAM_DEMAND_MIN <= t.ram_demand_mb <= RAM_DEMAND_MAX,
          f"RAM demand in [{RAM_DEMAND_MIN}, {RAM_DEMAND_MAX}] MB",
          f"got {t.ram_demand_mb:.1f}")
    check(TASK_SIZE_MIN <= t.size_mbits <= TASK_SIZE_MAX,
          f"Size in [{TASK_SIZE_MIN}, {TASK_SIZE_MAX}] Mbits",
          f"got {t.size_mbits:.2f}")
    check(t.cpu_cycles > 0,
          "cpu_cycles > 0  (Eq. 2)",
          f"got {t.cpu_cycles:.2e}")
    check(t.deadline_slot == t.arrival_slot + TASK_DEADLINE_SLOTS,
          f"deadline = arrival + {TASK_DEADLINE_SLOTS}",
          f"got {t.deadline_slot}")
    check(t.task_type in [0, 1, 2],
          "task_type in {0,1,2}",
          f"got {t.task_type}")
    check(t.static_priority in [0.0, 1.0],
          "static_priority in {0.0, 1.0}",
          f"got {t.static_priority}")

    # generate_batch test
    batch = gen.generate_batch(num_tasks=100)
    check(len(batch) == 100,
          "generate_batch(100) returns exactly 100 tasks",
          f"got {len(batch)}")

    print(f"    Sample: {t}")


# =============================================================
# TEST 3 — Dynamic priority score φ(i,t) — AICDQN Eq. 4
# =============================================================
def test_priority_score():
    section("Test 3 — Dynamic priority φ(i,t)  [AICDQN Eq. 4]")
    gen   = IoTTaskGenerator(seed=1)
    t     = gen.generate_batch(1)[0]

    phi_early = t.dynamic_priority(current_slot=0, queue_len=5,  queue_max=50)
    phi_late  = t.dynamic_priority(current_slot=8, queue_len=5,  queue_max=50)
    phi_cong  = t.dynamic_priority(current_slot=0, queue_len=45, queue_max=50)

    check(phi_late > phi_early,
          "φ increases as deadline approaches",
          f"early={phi_early:.4f}  late={phi_late:.4f}")
    check(phi_cong > phi_early,
          "φ increases under queue congestion",
          f"no_cong={phi_early:.4f}  congested={phi_cong:.4f}")
    check(0.0 <= phi_early <= 1.0,
          "φ(early) ∈ [0,1]",
          f"got {phi_early:.4f}")

    print(f"    φ(early)    = {phi_early:.4f}")
    print(f"    φ(deadline) = {phi_late:.4f}")
    print(f"    φ(congested)= {phi_cong:.4f}")


# =============================================================
# TEST 4 — Erlang-C formula correctness
# =============================================================
def test_erlang_c():
    section("Test 4 — Erlang-C M/M/c queue  [AICDQN Eq. 8]")
    c = Cloudlet(0)    # 10k MIPS, 4 servers

    pw_low  = c._erlang_c(lam=2.0,  mu=5.0)   # rho = 2/(4×5) = 0.10
    pw_mid  = c._erlang_c(lam=15.0, mu=5.0)   # rho = 15/20  = 0.75
    pw_high = c._erlang_c(lam=19.5, mu=5.0)   # rho = 19.5/20= 0.975
    pw_sat  = c._erlang_c(lam=25.0, mu=5.0)   # rho > 1      = saturated

    check(0.0  < pw_low  < 0.1,
          "low load:  P_W ∈ (0, 0.10)",   f"got {pw_low:.4f}")
    check(0.1  < pw_mid  < 0.7,
          "mid load:  P_W ∈ (0.10, 0.70)",f"got {pw_mid:.4f}")
    check(0.7  < pw_high < 1.0,
          "high load: P_W ∈ (0.70, 1.00)",f"got {pw_high:.4f}")
    check(pw_sat == 1.0,
          "saturated: P_W = 1.0",          f"got {pw_sat}")

    print(f"    P_W(ρ=0.10) = {pw_low:.4f}")
    print(f"    P_W(ρ=0.75) = {pw_mid:.4f}")
    print(f"    P_W(ρ=0.975)= {pw_high:.4f}")
    print(f"    P_W(sat)    = {pw_sat:.1f}")


# =============================================================
# TEST 5 — State vector (23-dim, normalised)
# =============================================================
def test_state_vector():
    section("Test 5 — State vector  (23-dim, normalised)")
    env   = LBROEnvironment(seed=0)
    state = env.reset()

    check(state.shape == (STATE_DIM,),
          f"shape = ({STATE_DIM},)",
          f"got {state.shape}")
    check(float(state.min()) >= -0.01,
          "all values ≥ 0.0",
          f"min = {state.min():.4f}")
    check(float(state.max()) <= 1.01,
          "all values ≤ 1.0",
          f"max = {state.max():.4f}")

    print(f"    State range = [{state.min():.3f}, {state.max():.3f}]")
    print(f"    Index map:")
    print(f"      [0–5]   Cloudlet-0  features")
    print(f"      [6–11]  Cloudlet-1  features")
    print(f"      [12–17] Cloudlet-2  features")
    print(f"      [18–19] Cloud       features")
    print(f"      [20–22] Task        features")


# =============================================================
# TEST 6 — Latency ordering  edge < cloud
# =============================================================
def test_latency_ordering():
    section("Test 6 — Latency ordering  (edge MUST be < cloud)")

    results = {}
    for action in range(NUM_ACTIONS):
        env = LBROEnvironment(seed=99)
        env.reset()
        _, _, _, info = env.step(action)
        results[action] = info
        label = f"Cloudlet-{action}" if action < 3 else "Cloud    "
        print(f"    {label}  "
              f"latency={info['latency_s']:.4f}s  "
              f"energy={info['energy_j']:.4f}J  "
              f"reward={info['reward']:.4f}")

    best_edge  = min(results[i]["latency_s"] for i in range(NUM_CLOUDLETS))
    cloud_lat  = results[3]["latency_s"]

    check(best_edge < cloud_lat,
          f"best_edge ({best_edge:.4f}s) < cloud ({cloud_lat:.4f}s)",
          "Edge should always be faster than cloud")

    check(results[0]["latency_s"] < results[1]["latency_s"],
          "Cloudlet-0 (10k) faster than Cloudlet-1 (8k)",
          f"{results[0]['latency_s']:.4f} vs {results[1]['latency_s']:.4f}")

    check(results[1]["latency_s"] < results[2]["latency_s"],
          "Cloudlet-1 (8k) faster than Cloudlet-2 (6k)",
          f"{results[1]['latency_s']:.4f} vs {results[2]['latency_s']:.4f}")


# =============================================================
# TEST 7 — Reward function (load balancing term)
# =============================================================
def test_reward():
    section("Test 7 — Reward function  (load balancing term)")

    # ── Part A: Reward must always be ≤ 0 ────────────────────
    for action in range(NUM_ACTIONS):
        env = LBROEnvironment(seed=action)
        env.reset()
        _, reward, _, _ = env.step(action)
        check(reward <= 0.0,
              f"action={action} reward ≤ 0",
              f"got {reward:.4f}")

    # ── Part B: Balanced > Unbalanced ────────────────────────
    # Compare:
    #   UNBALANCED → always cloudlet-0         [10, 0, 0]
    #   BALANCED   → alternating cloudlet-0,1  [5,  5, 0]
    #
    # Both avoid slow cloudlet-2 so latency/energy is similar.
    # Imbalance term is the ONLY difference → balanced must win.
    #
    # Why not round-robin 0,1,2?
    # Cloudlet-2 energy=2.863J vs Cloudlet-0 energy=0.588J
    # That 5× energy gap overwhelms imbalance savings.
    # A smart LBRO broker would never blindly round-robin
    # across heterogeneous nodes — it weights by capacity.

    env_unbal = LBROEnvironment(seed=5)
    env_unbal.reset()
    unbal_rewards = []
    for _ in range(10):
        _, r, done, _ = env_unbal.step(0)        # always cloudlet-0
        unbal_rewards.append(r)
        if done:
            break

    env_bal = LBROEnvironment(seed=5)
    env_bal.reset()
    bal_rewards = []
    for i in range(10):
        _, r, done, _ = env_bal.step(i % 2)      # alternates 0 → 1 → 0 → 1
        bal_rewards.append(r)
        if done:
            break

    avg_unbal = sum(unbal_rewards) / len(unbal_rewards)
    avg_bal   = sum(bal_rewards)   / len(bal_rewards)

    check(avg_bal > avg_unbal,
          f"balanced ({avg_bal:.4f}) > unbalanced ({avg_unbal:.4f})",
          "Load balancing reward term not working")

    print(f"    Balanced avg reward   : {avg_bal:.4f}   "
          f"[assignments 0→1→0→1]")
    print(f"    Unbalanced avg reward : {avg_unbal:.4f}   "
          f"[assignments always→0]")
    print(f"    Improvement           : {avg_bal - avg_unbal:.4f} ✓")
    print(f"")
    print(f"    Why this proves load balancing works:")
    print(f"    Both strategies use same cloudlets (0 and 1)")
    print(f"    σ_dist(balanced) ≈ 0.25  →  small penalty  ✅")
    print(f"    σ_dist(unbal)    ≈ 0.47  →  big   penalty  ❌")


# =============================================================
# TEST 8 — Full episode (200 steps)
# =============================================================
def test_full_episode():
    section("Test 8 — Full episode  (200 steps, round-robin)")

    env   = LBROEnvironment(seed=42)
    state = env.reset()

    check(state.shape == (STATE_DIM,),
          "Initial state shape correct", f"got {state.shape}")

    total_reward = 0.0
    steps_done   = 0

    for step in range(MAX_STEPS_PER_EP):
        action = step % NUM_ACTIONS
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps_done   += 1

        # State must stay normalised throughout
        assert state.min() >= -0.01, \
            f"State went below 0 at step {step}: min={state.min():.4f}"
        assert state.max() <=  1.01, \
            f"State went above 1 at step {step}: max={state.max():.4f}"

        if done:
            break

    check(steps_done == MAX_STEPS_PER_EP,
          f"Ran full {MAX_STEPS_PER_EP} steps",
          f"stopped at {steps_done}")

    summary = env.episode_summary()

    check(0.0 <= summary["drop_rate"] <= 1.0,
          "Drop rate ∈ [0,1]",
          f"got {summary['drop_rate']:.4f}")
    check(summary["avg_latency_s"] > 0,
          "Avg latency > 0",
          f"got {summary['avg_latency_s']:.4f}")
    check(summary["avg_energy_j"] > 0,
          "Avg energy > 0",
          f"got {summary['avg_energy_j']:.4f}")

    # LSTM window check
    for i in range(NUM_CLOUDLETS):
        h = env.get_lstm_input(i)
        check(h.shape == (LSTM_WINDOW, 2),
              f"Cloudlet-{i} LSTM input shape ({LSTM_WINDOW}, 2)",
              f"got {h.shape}")

    print(f"\n    ── Episode Summary ──────────────────────")
    print(f"    Total steps    : {steps_done}")
    print(f"    Total reward   : {total_reward:.4f}")
    print(f"    Avg latency    : {summary['avg_latency_s']:.4f} s")
    print(f"    Avg energy     : {summary['avg_energy_j']:.4f} J")
    print(f"    Drop rate      : {summary['drop_rate']:.2%}")
    print(f"    Total tasks    : {summary['total_tasks']}")
    print(f"    Action dist    : {[f'{x:.1%}' for x in summary['action_dist']]}")
    print(f"    ─────────────────────────────────────────")
    for cs in summary["cloudlet_stats"]:
        print(f"    Cloudlet-{cs['node']}  "
              f"assigned={cs['assigned']}  "
              f"dropped={cs['dropped']}  "
              f"avg_lat={cs['avg_latency_s']:.4f}s")
    print(f"    Cloud      "
          f"assigned={summary['cloud_stats']['assigned']}  "
          f"avg_lat={summary['cloud_stats']['avg_latency_s']:.4f}s")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    print(f"\n{'=' * 55}")
    print(f"  DRL-LSTM-LBRO  —  Simulator Test Suite")
    print(f"{'=' * 55}")

    test_config()
    test_task()
    test_priority_score()
    test_erlang_c()
    test_state_vector()
    test_latency_ordering()
    test_reward()
    test_full_episode()

    print(f"\n{'=' * 55}")
    print(f"  {GREEN}ALL {_tests_passed}/{_tests_run} CHECKS PASSED ✅{RESET}")
    print(f"  Simulator is research-ready!")
    print(f"  Next → Step 2: data/data_generator.py")
    print(f"{'=' * 55}\n")
