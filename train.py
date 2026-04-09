# =============================================================
# train.py
# DRL-LSTM-LBRO  —  Step 6: Main Training Loop
#
# Ties together ALL components:
#   ✅ LBROEnvironment    (simulator/environment.py)
#   ✅ DDQNAgent          (agents/ddqn.py)
#   ✅ LSTMPredictor ×3   (models/lstm_predictor.py)
#
# Training flow (per step):
#   1. Broker receives task from IoT device
#   2. RF classifier predicts task type
#   3. LSTM predictor updates cloudlet load forecasts
#   4. Broker builds 23-dim state vector
#   5. DDQN agent selects placement action (ε-greedy)
#   6. Broker executes placement → gets reward
#   7. DDQN learns from (s, a, r, s') experience
#   8. Repeat until episode ends (200 steps)
#
# Run with:
#     cd /Users/abhishek.sk/Documents/college/DRL_LSTM_LBRO
#     python3 train.py
# =============================================================

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from simulator.environment  import LBROEnvironment
from agents.ddqn            import DDQNAgent
from models.lstm_predictor  import LSTMPredictor, load_all as load_lstm
from simulator.config       import (
    NUM_CLOUDLETS, NUM_ACTIONS, STATE_DIM,
    MAX_STEPS_PER_EP, LSTM_WINDOW,
    DDQN_BATCH_SIZE, DDQN_TARGET_UPDATE,
    MODEL_DIR, RESULTS_DIR,
)

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Training hyperparameters ──────────────────────────────────
NUM_EPISODES    = 500
LOG_INTERVAL    = 10
SAVE_INTERVAL   = 50
WARMUP_EPISODES = 5
RESULTS_CSV     = os.path.join(RESULTS_DIR, "training_log.csv")


# =============================================================
# Training loop
# =============================================================

def train(use_lstm=True):
    tag = "full" if use_lstm else "no_lstm"

    print("=" * 60)
    print(f"  DRL-LSTM-LBRO  —  Step 6: Training  [{tag}]")
    print("=" * 60)

    # ── Load pretrained LSTM models ──────────────────────────
    print("\n  Loading pretrained models...")
    lstm_predictors = load_lstm() if use_lstm else None
    print(f"  {'✅' if use_lstm else '⬜'} LSTM predictors({'loaded' if use_lstm else 'DISABLED'})")

    # ── Initialise broker + DDQN agent ───────────────────────
    env   = LBROEnvironment(seed=42)
    if use_lstm:
        env.attach_lstm_predictors(lstm_predictors)
    agent = DDQNAgent(seed=42)
    print(f"  ✅ LBRO Broker initialised")
    print(f"  ✅ DDQN Agent  initialised  "
          f"(state={STATE_DIM}, actions={NUM_ACTIONS})")

    # ── Model save paths ─────────────────────────────────
    suffix      = "" if use_lstm else "_no_lstm"
    best_online = os.path.join(MODEL_DIR, f"ddqn_best{suffix}_online.keras")
    best_target = os.path.join(MODEL_DIR, f"ddqn_best{suffix}_target.keras")
    results_csv = os.path.join(RESULTS_DIR, f"training_log{suffix}.csv")

    # ── Training log ─────────────────────────────────────────
    log          = []
    best_reward  = -np.inf
    t_start      = time.time()

    print(f"\n  Starting training — {NUM_EPISODES} episodes...")
    print(f"  {'─' * 56}")
    print(f"  {'EP':>5}  {'Reward':>8}  {'DropRate':>9}  "
          f"{'AvgLat':>8}  {'AvgEng':>8}  "
          f"{'Loss':>8}  {'ε':>6}")
    print(f"  {'─' * 56}")

    for ep in range(1, NUM_EPISODES + 1):

        state      = env.reset()
        ep_reward  = 0.0
        ep_loss    = 0.0
        loss_count = 0

        for step in range(MAX_STEPS_PER_EP):

            # ── LSTM updates load predictions (every 5 slots) ─
            if use_lstm and step % 5 == 0:
                for cid, predictor in enumerate(lstm_predictors):
                    history = env.get_lstm_input(cid)
                    cpu_p, ram_p = predictor.predict(history)
                    env.lstm_cpu_pred[cid] = cpu_p
                    env.lstm_ram_pred[cid] = ram_p
            elif not use_lstm:
                for cid in range(NUM_CLOUDLETS):
                    env.lstm_cpu_pred[cid] = 0.0
                    env.lstm_ram_pred[cid] = 0.0

            # ── DDQN selects action ───────────────────────────
            training_mode = (ep > WARMUP_EPISODES)
            action = agent.act(state, training=training_mode)

            # ── Broker executes placement ─────────────────────
            next_state, reward, done, info = env.step(action)
            ep_reward += reward

            # ── Store + learn ─────────────────────────────────
            agent.remember(state, action, reward, next_state, done)
            if training_mode:
                loss = agent.learn()
                if loss > 0.0:
                    ep_loss    += loss
                    loss_count += 1

            state = next_state
            if done:
                break

        # ── Episode summary ───────────────────────────────────
        summary  = env.episode_summary()
        avg_loss = ep_loss / max(loss_count, 1)

        log.append({
            "episode"    : ep,
            "reward"     : round(ep_reward,        4),
            "drop_rate"  : round(summary["drop_rate"],       4),
            "avg_lat"    : round(summary["avg_latency_s"],   4),
            "avg_eng"    : round(summary["avg_energy_j"],    4),
            "avg_loss"   : round(avg_loss,                   6),
            "epsilon"    : round(agent.epsilon,              4),
            "buffer_size": len(agent.buffer),
            **{f"action_{i}": summary["action_dist"][i]
               for i in range(NUM_ACTIONS)},
        })

        # ── Print progress ────────────────────────────────────
        if ep % LOG_INTERVAL == 0 or ep == 1:
            elapsed = time.time() - t_start
            print(f"  {ep:>5}  "
                  f"{ep_reward:>8.2f}  "
                  f"{summary['drop_rate']:>8.1%}  "
                  f"{summary['avg_latency_s']:>8.4f}  "
                  f"{summary['avg_energy_j']:>8.4f}  "
                  f"{avg_loss:>8.5f}  "
                  f"{agent.epsilon:>6.4f}  "
                  f"[{elapsed:.0f}s]")

        # ── Save best model ───────────────────────────────────
        if ep_reward > best_reward and ep > WARMUP_EPISODES:
            best_reward = ep_reward
            agent.save(online_path=best_online, target_path=best_target)

        # ── Periodic checkpoint ───────────────────────────────
        if ep % SAVE_INTERVAL == 0:
            agent.save()
            pd.DataFrame(log).to_csv(results_csv, index=False)
            print(f"  {'─' * 56}")
            print(f"  Checkpoint saved at episode {ep}")
            print(f"  {'─' * 56}")

    # ── Training complete ─────────────────────────────────────
    elapsed = time.time() - t_start
    agent.save()
    df_log = pd.DataFrame(log)
    df_log.to_csv(results_csv, index=False)

    last50 = df_log.tail(50)
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE ✅  ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Episodes      : {NUM_EPISODES}")
    print(f"  Best reward   : {best_reward:.4f}")
    print(f"  Final ε       : {agent.epsilon:.4f}")
    print(f"\n  Last 50 episodes average:")
    print(f"    Reward      : {last50['reward'].mean():.4f}")
    print(f"    Drop rate   : {last50['drop_rate'].mean():.2%}")
    print(f"    Avg latency : {last50['avg_lat'].mean():.4f} s")
    print(f"    Avg energy  : {last50['avg_eng'].mean():.4f} J")
    print(f"\n  Action distribution (last 50 eps):")
    for i in range(NUM_ACTIONS):
        label = f"Cloudlet-{i}" if i < NUM_CLOUDLETS else "Cloud    "
        avg   = last50[f"action_{i}"].mean()
        bar   = "█" * int(avg * 40)
        print(f"    {label}: {avg:.1%}  {bar}")
    print(f"  Files saved:")
    print(f"    {results_csv}")
    print(f"    {os.path.join(MODEL_DIR, 'ddqn_online.keras')}")
    print(f"    {best_online}")
    print(f"  Next → Step 7: evaluate.py  (add --model {tag} flag if needed)")
    print(f"{'=' * 60}\n")

    return df_log


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DRL-LSTM-LBRO or an ablation variant."
    )
    parser.add_argument("--no-lstm", action="store_true",
                        help="Disable LSTM predictions during training (ablation)")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES,
                        help=f"Number of training episodes (default {NUM_EPISODES})")
    args = parser.parse_args()
    NUM_EPISODES = args.episodes
    train(use_lstm=not args.no_lstm)
