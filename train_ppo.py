# =============================================================
# train_ppo.py
# DRL-LSTM-LBRO  —  PPO Baseline Training
#
# Trains a PPO agent on the same LBROEnvironment used for DDQN.
# LSTM predictions are NOT used (PPO is a baseline without LSTM).
#
# Run with:
#     python3 train_ppo.py
#     python3 train_ppo.py --episodes 500
# =============================================================

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from simulator.environment import LBROEnvironment
from agents.ppo            import PPOAgent
from simulator.config      import (
    NUM_CLOUDLETS, NUM_ACTIONS, STATE_DIM,
    MAX_STEPS_PER_EP, MODEL_DIR, RESULTS_DIR,
)

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_EPISODES  = 500
LOG_INTERVAL  = 10
RESULTS_CSV   = os.path.join(RESULTS_DIR, "training_log_ppo.csv")
BEST_ACTOR    = os.path.join(MODEL_DIR,   "ppo_best_actor.keras")
BEST_CRITIC   = os.path.join(MODEL_DIR,   "ppo_best_critic.keras")


def train(num_episodes: int = NUM_EPISODES):
    print("=" * 60)
    print("  DRL-LSTM-LBRO  —  PPO Baseline Training")
    print(f"  Episodes: {num_episodes}  |  State: {STATE_DIM}  |  Actions: {NUM_ACTIONS}")
    print("=" * 60)

    env   = LBROEnvironment(seed=42)
    agent = PPOAgent(seed=42)

    log          = []
    best_reward  = -np.inf
    t_start      = time.time()

    print(f"\n  {'─' * 56}")
    print(f"  {'EP':>5}  {'Reward':>8}  {'DropRate':>9}  "
          f"{'AvgLat':>8}  {'Loss':>8}")
    print(f"  {'─' * 56}")

    for ep in range(1, num_episodes + 1):
        state     = env.reset()
        ep_reward = 0.0
        agent.clear_rollout()

        for step in range(MAX_STEPS_PER_EP):
            action = agent.step(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward_done(reward, done)
            ep_reward += reward
            state = next_state
            if done:
                break

        loss = agent.learn(last_value=0.0)
        summary = env.episode_summary()

        log.append({
            "episode"  : ep,
            "reward"   : round(ep_reward,              4),
            "drop_rate": round(summary["drop_rate"],   4),
            "avg_lat"  : round(summary["avg_latency_s"], 4),
            "avg_eng"  : round(summary["avg_energy_j"],  4),
            "loss"     : round(loss,                   6),
        })

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(actor_path=BEST_ACTOR, critic_path=BEST_CRITIC)

        if ep % LOG_INTERVAL == 0 or ep == 1:
            elapsed = time.time() - t_start
            print(f"  {ep:>5}  "
                  f"{ep_reward:>8.2f}  "
                  f"{summary['drop_rate']:>8.1%}  "
                  f"{summary['avg_latency_s']:>8.4f}  "
                  f"{loss:>8.5f}  "
                  f"[{elapsed:.0f}s]")

    elapsed = time.time() - t_start
    agent.save()
    pd.DataFrame(log).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(log)
    last50 = df.tail(50)
    print(f"\n{'=' * 60}")
    print(f"  PPO TRAINING COMPLETE ✅  ({elapsed:.1f}s)")
    print(f"  Best reward   : {best_reward:.4f}")
    print(f"  Last 50 eps:")
    print(f"    Reward      : {last50['reward'].mean():.4f}")
    print(f"    Drop rate   : {last50['drop_rate'].mean():.2%}")
    print(f"    Avg latency : {last50['avg_lat'].mean():.4f} s")
    print(f"  Models saved → {BEST_ACTOR}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    args = parser.parse_args()
    train(num_episodes=args.episodes)
