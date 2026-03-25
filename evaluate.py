# =============================================================
# evaluate.py
# DRL-LSTM-LBRO  —  Step 7: Evaluation + Baseline Comparison
#
# Compares 6 policies over 100 evaluation episodes:
#   1. DRL-LSTM-LBRO  (trained DDQN — ε=0, greedy)
#   2. Round-Robin    (rotate 0→1→2→0)
#   3. Random         (random cloudlet each step)
#   4. Greedy-Best    (lowest CPU utilisation cloudlet)
#   5. Original-LBRO  (Nayyer et al. — CRI, no ML)
#   6. RF-LBRO        (prev. semester — RF + weighted CRI)
# =============================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from simulator.environment  import LBROEnvironment
from agents.ddqn            import DDQNAgent
from models.rf_classifier   import RFTaskClassifier
from models.lstm_predictor  import load_all as load_lstm
from simulator.config       import (
    NUM_CLOUDLETS, NUM_ACTIONS,
    MAX_STEPS_PER_EP, MODEL_DIR, RESULTS_DIR,
    TASK_CPU, TASK_MEM, TASK_BAL
)

os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_EPISODES = 100
BEST_MODEL    = os.path.join(MODEL_DIR, "ddqn_best_online.keras")
BEST_TARGET   = os.path.join(MODEL_DIR, "ddqn_best_target.keras")
EVAL_CSV      = os.path.join(RESULTS_DIR, "eval_results.csv")
COMPARE_CSV   = os.path.join(RESULTS_DIR, "baseline_comparison.csv")

# RF-LBRO weights per task type: [w_cpu, w_ram, w_queue]
RF_WEIGHTS = {
    TASK_CPU: (0.6, 0.2, 0.2),
    TASK_MEM: (0.2, 0.6, 0.2),
    TASK_BAL: (0.33, 0.33, 0.34),
}


# =============================================================
# Policy definitions
# =============================================================

def policy_ddqn(agent, state, step, env, rf_clf=None, task_type=None):
    return agent.act(state, training=False)


def policy_round_robin(agent, state, step, env, rf_clf=None, task_type=None):
    return step % NUM_CLOUDLETS


def policy_random(agent, state, step, env, rf_clf=None, task_type=None):
    return np.random.randint(0, NUM_CLOUDLETS)


def policy_greedy_best(agent, state, step, env, rf_clf=None, task_type=None):
    cpu_utils = [c.cpu_util for c in env.cloudlets]
    return int(np.argmin(cpu_utils))


def policy_original_lbro(agent, state, step, env, rf_clf=None, task_type=None):
    """Original LBRO: equal-weight CRI — no ML."""
    scores = [
        (1 - c.cpu_util) * (1 - c.ram_util) * (1 - c.queue_util)
        for c in env.cloudlets
    ]
    return int(np.argmax(scores))


def policy_rf_lbro(agent, state, step, env, rf_clf=None, task_type=None):
    """RF-LBRO: weighted CRI based on RF task type."""
    w_cpu, w_ram, w_que = RF_WEIGHTS.get(task_type, RF_WEIGHTS[TASK_BAL])
    scores = [
        w_cpu * (1 - c.cpu_util) +
        w_ram * (1 - c.ram_util) +
        w_que * (1 - c.queue_util)
        for c in env.cloudlets
    ]
    return int(np.argmax(scores))


# policy_fn, use_lstm, use_rf
POLICIES = {
    "DRL-LSTM-LBRO" : (policy_ddqn,          True,  True),
    "RF-LBRO"       : (policy_rf_lbro,        False, True),
    "Original-LBRO" : (policy_original_lbro,  False, False),
    "Round-Robin"   : (policy_round_robin,     False, False),
    "Greedy-Best"   : (policy_greedy_best,     False, False),
    "Random"        : (policy_random,          False, False),
}


# =============================================================
# Run one policy
# =============================================================

def run_policy(policy_name, policy_fn, use_lstm, use_rf,
               agent, rf_clf, lstm_predictors,
               num_episodes=EVAL_EPISODES, seed=100):

    env  = LBROEnvironment(seed=seed)
    rows = []

    for ep in range(num_episodes):
        state     = env.reset()
        ep_reward = 0.0

        for step in range(MAX_STEPS_PER_EP):
            task_type = TASK_BAL   # default

            # RF classification
            if use_rf:
                task           = env.current_task
                task.task_type = rf_clf.predict(task)
                task_type      = task.task_type

            # LSTM predictions (DRL only)
            if use_lstm and step % 5 == 0:
                for cid, p in enumerate(lstm_predictors):
                    history = env.get_lstm_input(cid)
                    cpu_p, ram_p = p.predict(history)
                    env.lstm_cpu_pred[cid] = cpu_p
                    env.lstm_ram_pred[cid] = ram_p

            action = policy_fn(agent, state, step, env,
                               rf_clf=rf_clf, task_type=task_type)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            state      = next_state
            if done:
                break

        summary   = env.episode_summary()
        adist     = summary["action_dist"][:NUM_CLOUDLETS]
        imbalance = float(np.std(adist))

        rows.append({
            "policy"   : policy_name,
            "episode"  : ep,
            "reward"   : round(ep_reward,                4),
            "drop_rate": round(summary["drop_rate"],     4),
            "avg_lat"  : round(summary["avg_latency_s"], 4),
            "avg_eng"  : round(summary["avg_energy_j"],  4),
            "imbalance": round(imbalance,                4),
            "action_0" : round(summary["action_dist"][0],4),
            "action_1" : round(summary["action_dist"][1],4),
            "action_2" : round(summary["action_dist"][2],4),
            "action_3" : round(summary["action_dist"][3],4),
        })

    return pd.DataFrame(rows)


# =============================================================
# Main
# =============================================================

def main():
    print("=" * 60)
    print("  DRL-LSTM-LBRO  —  Step 7: Evaluation")
    print("=" * 60)

    print("\n  Loading models...")
    rf_clf          = RFTaskClassifier.load()
    lstm_predictors = load_lstm()
    agent           = DDQNAgent(seed=0)
    agent.load(online_path=BEST_MODEL, target_path=BEST_TARGET)
    agent.epsilon   = 0.0
    print(f"  ✅ All models loaded  (ε=0.0 for evaluation)")

    print(f"\n  Evaluating {EVAL_EPISODES} episodes per policy...\n")
    all_dfs = []

    for name, (fn, use_lstm, use_rf) in POLICIES.items():
        print(f"  Running: {name}...")
        df = run_policy(name, fn, use_lstm, use_rf,
                        agent, rf_clf, lstm_predictors)
        all_dfs.append(df)
        avg = df.mean(numeric_only=True)
        print(f"    drop={avg['drop_rate']:.2%}  "
              f"lat={avg['avg_lat']:.4f}s  "
              f"eng={avg['avg_eng']:.4f}J  "
              f"reward={avg['reward']:.2f}")

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(EVAL_CSV, index=False)
    print(f"\n  ✅ Full results saved → {EVAL_CSV}")

    print(f"\n{'─' * 60}")
    print(f"  BASELINE COMPARISON  ({EVAL_EPISODES} episodes each)")
    print(f"{'─' * 60}")
    print(f"  {'Policy':16s}  {'Drop':>8}  {'Latency':>9}  "
          f"{'Energy':>8}  {'Imbal':>7}  {'Reward':>8}")
    print(f"  {'─' * 58}")

    compare_rows = []
    for name in POLICIES:
        sub = full_df[full_df["policy"] == name]
        avg = sub.mean(numeric_only=True)
        std = sub.std( numeric_only=True)
        compare_rows.append({
            "policy"        : name,
            "drop_rate_mean": round(avg["drop_rate"], 4),
            "drop_rate_std" : round(std["drop_rate"], 4),
            "avg_lat_mean"  : round(avg["avg_lat"],   4),
            "avg_lat_std"   : round(std["avg_lat"],   4),
            "avg_eng_mean"  : round(avg["avg_eng"],   4),
            "avg_eng_std"   : round(std["avg_eng"],   4),
            "imbalance_mean": round(avg["imbalance"], 4),
            "imbalance_std" : round(std["imbalance"], 4),
            "reward_mean"   : round(avg["reward"],    4),
            "reward_std"    : round(std["reward"],    4),
        })
        marker = " ★" if name == "DRL-LSTM-LBRO" else "  "
        print(f"  {name:16s}{marker}"
              f"  {avg['drop_rate']:>7.2%}"
              f"  {avg['avg_lat']:>9.4f}s"
              f"  {avg['avg_eng']:>8.4f}J"
              f"  {avg['imbalance']:>7.4f}"
              f"  {avg['reward']:>8.2f}")

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(COMPARE_CSV, index=False)

    # Improvement vs Original-LBRO (direct predecessor)
    print(f"\n{'─' * 60}")
    print(f"  IMPROVEMENT: DRL-LSTM-LBRO vs Original-LBRO")
    print(f"{'─' * 60}")
    drl_row  = compare_df[compare_df["policy"] == "DRL-LSTM-LBRO"].iloc[0]
    base_row = compare_df[compare_df["policy"] == "Original-LBRO"].iloc[0]
    for label, col in [("Drop rate", "drop_rate_mean"),
                       ("Latency",   "avg_lat_mean"),
                       ("Energy",    "avg_eng_mean"),
                       ("Imbalance", "imbalance_mean")]:
        d   = drl_row[col]
        b   = base_row[col]
        pct = (b - d) / max(b, 1e-9) * 100
        print(f"  {label:12s}  DRL={d:.4f}  LBRO={b:.4f}  "
              f"{'↓' if pct>0 else '↑'} {abs(pct):.1f}%")

    # Improvement vs RF-LBRO (direct predecessor)
    print(f"\n{'─' * 60}")
    print(f"  IMPROVEMENT: DRL-LSTM-LBRO vs RF-LBRO")
    print(f"{'─' * 60}")
    rf_row = compare_df[compare_df["policy"] == "RF-LBRO"].iloc[0]
    for label, col in [("Drop rate", "drop_rate_mean"),
                       ("Latency",   "avg_lat_mean"),
                       ("Energy",    "avg_eng_mean"),
                       ("Imbalance", "imbalance_mean")]:
        d   = drl_row[col]
        r   = rf_row[col]
        pct = (r - d) / max(r, 1e-9) * 100
        print(f"  {label:12s}  DRL={d:.4f}  RF={r:.4f}  "
              f"{'↓' if pct>0 else '↑'} {abs(pct):.1f}%")

    print(f"\n{'=' * 60}")
    print(f"  Step 7 COMPLETE ✅")
    print(f"  Files saved:")
    print(f"    {EVAL_CSV}")
    print(f"    {COMPARE_CSV}")
    print(f"  Next → Step 8: results/plot_results.py")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
