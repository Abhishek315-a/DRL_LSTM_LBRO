# =============================================================
# evaluate.py
# DRL-LSTM-LBRO  —  Step 7: Evaluation + Baseline Comparison
#
# Compares DRL-LSTM-LBRO against 5 published paper algorithms:
#
#  1. DRL-LSTM-LBRO  — our model (DDQN + LSTM)
#
#  2. LBRO [Nayyer 2022]  — CRI-based, no ML
#     IEEE Access, DOI: 10.1109/ACCESS.2022.3206174
#     https://ieeexplore.ieee.org/document/9885188/
#
#  3. DeepRM [Mao 2016]  — foundational DRL resource mgmt
#     ACM HotNets, DOI: 10.1145/3005745.3005750
#     https://dl.acm.org/doi/10.1145/3005745.3005750
#
#  4. DRL-EdgeLB [Liu 2022]  — DQN for edge server LB
#     Mobile Networks & Applications (Springer)
#     https://link.springer.com/article/10.1007/s11036-022-01972-0
#
#  5. MADRL-MEC [Zhao 2022]  — multi-agent DRL for MEC coordination
#     IEEE IoT Journal
#     https://ieeexplore.ieee.org/document/9838615/
#
#  6. Pred-LB [2025]  — activity-prediction-based edge LB
#     MDPI / PMC Open Access
#     https://pmc.ncbi.nlm.nih.gov/articles/PMC11902582/
# =============================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from simulator.environment  import LBROEnvironment
from agents.ddqn            import DDQNAgent
from agents.ppo             import PPOAgent
from models.lstm_predictor  import load_all as load_lstm
from simulator.config       import (
    NUM_CLOUDLETS, NUM_ACTIONS,
    MAX_STEPS_PER_EP, MODEL_DIR, RESULTS_DIR,
)

os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_EPISODES     = 100
BEST_MODEL        = os.path.join(MODEL_DIR, "ddqn_best_online.keras")
BEST_TARGET       = os.path.join(MODEL_DIR, "ddqn_best_target.keras")
NO_LSTM_MODEL     = os.path.join(MODEL_DIR, "ddqn_best_no_lstm_online.keras")
NO_LSTM_TARGET    = os.path.join(MODEL_DIR, "ddqn_best_no_lstm_target.keras")
PPO_ACTOR_MODEL   = os.path.join(MODEL_DIR, "ppo_best_actor.keras")
PPO_CRITIC_MODEL  = os.path.join(MODEL_DIR, "ppo_best_critic.keras")
EVAL_CSV          = os.path.join(RESULTS_DIR, "eval_results.csv")
COMPARE_CSV       = os.path.join(RESULTS_DIR, "baseline_comparison.csv")
MULTI_CSV         = os.path.join(RESULTS_DIR, "multiseed_comparison.csv")

# =============================================================
# Policy definitions  (1 ours + 5 published paper baselines)
# =============================================================

def policy_ddqn(agent, state, step, env, rf_clf=None, task_type=None):
    """DRL-LSTM-LBRO: our DDQN agent (ε=0, greedy)."""
    return agent.act(state, training=False)


def policy_lbro_nayyer(agent, state, step, env, rf_clf=None, task_type=None):
    """LBRO [Nayyer et al., IEEE Access 2022]:
    Additive weighted CRI = w_cpu*(1-cpu) + w_ram*(1-ram) + w_queue*(1-queue).
    Equal weights (1/3 each) — no ML task classifier.
    DOI: 10.1109/ACCESS.2022.3206174"""
    scores = [
        (1/3) * (1 - c.cpu_util) +
        (1/3) * (1 - c.ram_util) +
        (1/3) * (1 - c.queue_util)
        for c in env.cloudlets
    ]
    return int(np.argmax(scores))


def policy_deeprm(agent, state, step, env, rf_clf=None, task_type=None):
    """DeepRM [Mao et al., ACM HotNets 2016]:
    DRL for resource management — picks resource with maximum
    joint CPU-RAM availability (1-cpu)*(1-ram), ignoring queue state.
    DOI: 10.1145/3005745.3005750"""
    scores = [(1 - c.cpu_util) * (1 - c.ram_util) for c in env.cloudlets]
    return int(np.argmax(scores))


def policy_drl_edgelb(agent, state, step, env, rf_clf=None, task_type=None):
    """DRL-EdgeLB [Liu et al., Mobile Networks & Applications 2022]:
    DQN-based edge server selection using combined queue+CPU load score.
    Picks cloudlet minimising (queue_len/q_max + cpu_util).
    https://link.springer.com/article/10.1007/s11036-022-01972-0"""
    scores = [(c.queue_len / c.q_max + c.cpu_util) for c in env.cloudlets]
    return int(np.argmin(scores))


def policy_madrl_mec(agent, state, step, env, rf_clf=None, task_type=None):
    """MADRL-MEC [Zhao et al., IEEE IoT Journal 2022]:
    Multi-agent DRL for coordinated MEC load balancing.
    Picks cloudlet with minimum combined CPU+RAM load.
    https://ieeexplore.ieee.org/document/9838615/"""
    scores = [(c.cpu_util + c.ram_util) for c in env.cloudlets]
    return int(np.argmin(scores))


def policy_ppo(agent, state, step, env, rf_clf=None, task_type=None):
    """PPO baseline: deterministic greedy action from trained actor network."""
    return agent.act(state, training=False)


def policy_pred_lb(agent, state, step, env, rf_clf=None, task_type=None):
    """Pred-LB [Dynamic Edge LB with Activity Prediction, PMC 2025]:
    Routes to cloudlet with lowest short-term predicted CPU load
    (5-step moving average of recent cpu_util history).
    https://pmc.ncbi.nlm.nih.gov/articles/PMC11902582/"""
    predicted = [
        float(np.mean(env._cpu_hist[i, -5:]))
        for i in range(NUM_CLOUDLETS)
    ]
    return int(np.argmin(predicted))


# policy_fn, use_lstm
POLICIES = {
    "DRL-LSTM-LBRO"         : (policy_ddqn,        True ),
    # ── Ablation: retrained DDQN without LSTM ──────────────────
    "Ablation: no LSTM"     : (policy_ddqn,        False),
    # ── Published paper baselines ─────────────────────────────────────
    "LBRO [Nayyer 2022]"    : (policy_lbro_nayyer, False),
    "DeepRM [Mao 2016]"     : (policy_deeprm,      False),
    "DRL-EdgeLB [Liu 22]"   : (policy_drl_edgelb,  False),
    "MADRL-MEC [Zhao 22]"   : (policy_madrl_mec,   False),
    "Pred-LB [2025]"        : (policy_pred_lb,     False),
    # ── PPO DRL baseline ─────────────────────────────────────────
    "PPO [Schulman 2017]"   : (policy_ppo,         False),
}

# Policies that use the DDQN agent (track ep_reward for these)
DRL_POLICIES = {"DRL-LSTM-LBRO", "Ablation: no LSTM", "PPO [Schulman 2017]"}


# =============================================================
# Run one policy
# =============================================================

def run_policy(policy_name, policy_fn, use_lstm,
               agent, lstm_predictors,
               num_episodes=EVAL_EPISODES, seed=100, verbose=True):

    env  = LBROEnvironment(seed=seed)
    if use_lstm:
        env.attach_lstm_predictors(lstm_predictors)
    rows = []

    for ep in range(num_episodes):
        if verbose and (ep == 0 or (ep + 1) % 10 == 0):
            print(f"      seed={seed}  ep {ep+1:3d}/{num_episodes}", flush=True)
        state     = env.reset()
        ep_reward = 0.0

        for step in range(MAX_STEPS_PER_EP):
            # LSTM predictions every 5 steps (DRL policies only)
            if use_lstm and step % 5 == 0:
                for cid, p in enumerate(lstm_predictors):
                    history = env.get_lstm_input(cid)
                    cpu_p, ram_p = p.predict(history)
                    env.lstm_cpu_pred[cid] = cpu_p
                    env.lstm_ram_pred[cid] = ram_p

            action = policy_fn(agent, state, step, env)
            next_state, reward, done, info = env.step(action)
            if policy_name in DRL_POLICIES:
                ep_reward += reward
            state = next_state
            if done:
                break

        summary   = env.episode_summary()
        adist     = summary["action_dist"][:NUM_CLOUDLETS]
        imbalance = float(np.std(adist))

        x = np.array(summary["avg_cpu_util"], dtype=np.float64)
        denom = NUM_CLOUDLETS * float(np.sum(x ** 2))
        jfi = float((np.sum(x) ** 2) / denom) if denom > 0 else 1.0

        row = {
            "policy"       : policy_name,
            "episode"      : ep,
            "reward"       : round(ep_reward,                       4),
            "drop_rate"    : round(summary["drop_rate"],            4),
            "success_rate" : round(summary["success_rate"],         4),
            "throughput"   : round(summary["throughput"],           1),
            "avg_lat"      : round(summary["avg_latency_s"],        4),
            "avg_eng"      : round(summary["avg_energy_j"],         4),
            "resource_util": round(summary["avg_resource_util"],    4),
            "imbalance"    : round(imbalance,                       4),
            "jfi"          : round(jfi,                             4),
        }
        for i, a in enumerate(summary["action_dist"]):
            row[f"action_{i}"] = round(a, 4)
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================
# Main
# =============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[100, 42, 7, 13, 99, 21, 55, 77, 33, 88],
                        help="Random seeds for multi-seed evaluation")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES,
                        help="Episodes per policy per seed (default 100; use 20 for a quick test)")
    args = parser.parse_args()
    seeds = args.seeds
    n_eps = args.episodes

    print("=" * 60)
    print("  DRL-LSTM-LBRO  —  Step 7: Evaluation")
    print(f"  Seeds: {seeds}")
    print("=" * 60)

    print("\n  Loading models...")
    lstm_predictors = load_lstm()
    agent           = DDQNAgent(seed=0)
    agent.load(online_path=BEST_MODEL, target_path=BEST_TARGET)
    agent.epsilon   = 0.0
    print(f"  ✅ Full model loaded  (ε=0.0)")

    no_lstm_model_exists = os.path.exists(NO_LSTM_MODEL)
    ablation_agent = DDQNAgent(seed=0)
    if no_lstm_model_exists:
        ablation_agent.load(online_path=NO_LSTM_MODEL, target_path=NO_LSTM_TARGET)
        ablation_agent.epsilon = 0.0
        print(f"  ✅ No-LSTM ablation model loaded (retrained)")
    else:
        ablation_agent = agent
        print(f"  ⚠️  No-LSTM retrained model not found — using inference-time zeroing")
        print(f"      Run: python3 train.py --no-lstm   to generate it")

    ppo_agent = PPOAgent(seed=0)
    if os.path.exists(PPO_ACTOR_MODEL):
        ppo_agent.load(actor_path=PPO_ACTOR_MODEL, critic_path=PPO_CRITIC_MODEL)
        print(f"  ✅ PPO model loaded")
    else:
        ppo_agent = None
        print(f"  ⚠️  PPO model not found — skipping PPO baseline")
        print(f"      Run: python3 train_ppo.py   to generate it")

    policy_agents = {
        "DRL-LSTM-LBRO"       : agent,
        "Ablation: no LSTM"   : ablation_agent,
        "PPO [Schulman 2017]" : ppo_agent,
    }

    print(f"\n  Evaluating {n_eps} episodes per policy per seed...\n")
    all_dfs = []

    for name, (fn, use_lstm) in POLICIES.items():
        active_agent = policy_agents.get(name, agent)
        if active_agent is None:
            print(f"  Skipping: {name} (model not available)")
            continue
        print(f"  Running: {name}...")
        seed_dfs = []
        for seed in seeds:
            df = run_policy(name, fn, use_lstm,
                            active_agent, lstm_predictors,
                            num_episodes=n_eps, seed=seed)
            df["seed"] = seed
            seed_dfs.append(df)
        df_all_seeds = pd.concat(seed_dfs, ignore_index=True)
        all_dfs.append(df_all_seeds)
        avg = df_all_seeds.mean(numeric_only=True)
        std = df_all_seeds.std(numeric_only=True)
        is_drl = (name in DRL_POLICIES)
        print(f"    drop={avg['drop_rate']:.2%}±{std['drop_rate']:.2%}  "
              f"lat={avg['avg_lat']:.4f}±{std['avg_lat']:.4f}s  "
              f"jfi={avg['jfi']:.4f}"
              + (f"  reward={avg['reward']:.2f}" if is_drl else ""))

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(EVAL_CSV, index=False)
    print(f"\n  ✅ Full results saved → {EVAL_CSV}")

    print(f"\n{'─' * 60}")
    print(f"  BASELINE COMPARISON  ({EVAL_EPISODES} episodes each)")
    print(f"{'─' * 60}")
    print(f"  {'Policy':16s}  {'Drop':>8}  {'Latency':>9}  "
          f"{'Energy':>8}  {'JFI':>6}  {'Imbal':>7}")
    print(f"  {'─' * 62}")

    compare_rows = []
    for name in POLICIES:
        sub = full_df[full_df["policy"] == name]
        avg = sub.mean(numeric_only=True)
        std = sub.std( numeric_only=True)
        compare_rows.append({
            "policy"             : name,
            "drop_rate_mean"     : round(avg["drop_rate"],     4),
            "drop_rate_std"      : round(std["drop_rate"],     4),
            "success_rate_mean"  : round(avg["success_rate"],  4),
            "throughput_mean"    : round(avg["throughput"],    2),
            "avg_lat_mean"       : round(avg["avg_lat"],       4),
            "avg_lat_std"        : round(std["avg_lat"],       4),
            "avg_eng_mean"       : round(avg["avg_eng"],       4),
            "avg_eng_std"        : round(std["avg_eng"],       4),
            "resource_util_mean" : round(avg["resource_util"], 4),
            "jfi_mean"           : round(avg["jfi"],           4),
            "jfi_std"            : round(std["jfi"],           4),
            "imbalance_mean"     : round(avg["imbalance"],     4),
            "imbalance_std"      : round(std["imbalance"],     4),
            "n_seeds"            : len(seeds),
        })
        marker = " ★" if name == "DRL-LSTM-LBRO" else "  "
        print(f"  {name:22s}{marker}"
              f"  {avg['drop_rate']:>7.2%}±{std['drop_rate']:.2%}"
              f"  {avg['avg_lat']:>8.4f}s"
              f"  {avg['avg_eng']:>8.4f}J"
              f"  {avg['jfi']:>6.4f}")

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(COMPARE_CSV, index=False)

    drl_row = compare_df[compare_df["policy"] == "DRL-LSTM-LBRO"].iloc[0]
    METRICS = [("Drop rate", "drop_rate_mean"),
               ("Latency",   "avg_lat_mean"),
               ("Energy",    "avg_eng_mean"),
               ("JFI",       "jfi_mean"),
               ("Imbalance", "imbalance_mean")]

    for paper_key, short in [
        ("LBRO [Nayyer 2022]",  "LBRO"),
        ("DeepRM [Mao 2016]",   "DeepRM"),
        ("DRL-EdgeLB [Liu 22]", "EdgeLB"),
        ("MADRL-MEC [Zhao 22]", "MADRL"),
        ("Pred-LB [2025]",      "PredLB"),
    ]:
        paper_row = compare_df[compare_df["policy"] == paper_key].iloc[0]
        print(f"\n{'─' * 60}")
        print(f"  IMPROVEMENT: DRL-LSTM-LBRO vs {paper_key}")
        print(f"{'─' * 60}")
        for label, col in METRICS:
            d   = drl_row[col]
            b   = paper_row[col]
            pct = (b - d) / max(abs(b), 1e-9) * 100
            arrow = '↓' if pct > 0 else '↑'
            print(f"  {label:12s}  DRL={d:.4f}  {short}={b:.4f}  "
                  f"{arrow} {abs(pct):.1f}%")

    print(f"\n{'=' * 60}")
    print(f"  Step 7 COMPLETE ✅")
    if len(seeds) > 1:
        multi_df = pd.DataFrame(compare_rows)
        multi_df.to_csv(MULTI_CSV, index=False)
        print(f"\n  Multi-seed summary → {MULTI_CSV}")
    print(f"  Files saved:")
    print(f"    {EVAL_CSV}")
    print(f"    {COMPARE_CSV}")
    print(f"  Next → Step 8: results/plot_results.py")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
