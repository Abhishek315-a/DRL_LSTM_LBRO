# =============================================================
# training/ddpg_baseline.py
# Baseline: DDPG agent (continuous action → discrete mapping)
#
# Actor  : outputs continuous value ∈ [0, NUM_ACTIONS)
# Critic : evaluates Q(s, a) for actor update
# Action : round(actor_output) → discrete cloudlet index
# =============================================================

import os, sys, random
import numpy as np
import pandas as pd
from collections import deque

import tensorflow as tf
from tensorflow.keras.models     import Model
from tensorflow.keras.layers     import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.environment import LBROEnvironment
from simulator.config      import (
    STATE_DIM, NUM_ACTIONS, NUM_CLOUDLETS,
    MAX_STEPS_PER_EP, RESULTS_DIR, MODEL_DIR
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

NUM_EPISODES  = 500
EVAL_EPISODES = 100
BATCH_SIZE    = 64
BUFFER_SIZE   = 50_000
ACTOR_LR      = 1e-4
CRITIC_LR     = 1e-3
GAMMA         = 0.99
TAU           = 0.005       # soft update rate
RESULTS_CSV   = os.path.join(RESULTS_DIR, "ddpg_results.csv")
ACTOR_PATH    = os.path.join(MODEL_DIR,   "ddpg_actor.keras")


# =============================================================
# Networks
# =============================================================

def build_actor(state_dim=STATE_DIM, num_actions=NUM_ACTIONS):
    inp = Input(shape=(state_dim,))
    x   = Dense(256, activation="relu")(inp)
    x   = Dense(128, activation="relu")(x)
    # sigmoid scaled to [0, num_actions) → round to discrete
    out = Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="Actor")


def build_critic(state_dim=STATE_DIM):
    s_inp = Input(shape=(state_dim,))
    a_inp = Input(shape=(1,))
    x     = Concatenate()([s_inp, a_inp])
    x     = Dense(256, activation="relu")(x)
    x     = Dense(128, activation="relu")(x)
    out   = Dense(1,   activation="linear")(x)
    return Model([s_inp, a_inp], out, name="Critic")


# =============================================================
# DDPG Agent
# =============================================================

class DDPGAgent:

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.actor        = build_actor()
        self.critic       = build_critic()
        self.target_actor  = build_actor()
        self.target_critic = build_critic()

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_opt  = Adam(ACTOR_LR)
        self.critic_opt = Adam(CRITIC_LR)

        self.buffer  = deque(maxlen=BUFFER_SIZE)
        self.noise   = 0.3    # exploration noise

    def act(self, state, training=True):
        s   = tf.constant(state.reshape(1, -1), dtype=tf.float32)
        raw = float(self.actor(s, training=False).numpy()[0][0])
        if training:
            raw += np.random.normal(0, self.noise)
        # scale to [0, NUM_CLOUDLETS) and discretise
        action = int(np.clip(round(raw * NUM_CLOUDLETS), 0,
                             NUM_CLOUDLETS - 1))
        return action

    def remember(self, s, a, r, ns, done):
        self.buffer.append((
            np.array(s,  dtype=np.float32),
            float(a) / NUM_CLOUDLETS,         # normalise to [0,1]
            float(r),
            np.array(ns, dtype=np.float32),
            float(done)
        ))

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        batch       = random.sample(self.buffer, BATCH_SIZE)
        states      = tf.constant(np.array([e[0] for e in batch], dtype=np.float32))
        actions     = tf.constant(np.array([e[1] for e in batch], dtype=np.float32).reshape(-1, 1))
        rewards     = tf.constant(np.array([e[2] for e in batch], dtype=np.float32).reshape(-1, 1))
        next_states = tf.constant(np.array([e[3] for e in batch], dtype=np.float32))
        dones       = tf.constant(np.array([e[4] for e in batch], dtype=np.float32).reshape(-1, 1))

        # Critic update
        next_actions = self.target_actor(next_states, training=False)
        target_q     = (rewards + GAMMA *
                        self.target_critic([next_states, next_actions],
                                        training=False) * (1 - dones))
        with tf.GradientTape() as tape:
            current_q = self.critic([states, actions], training=True)
            c_loss    = tf.reduce_mean(tf.square(target_q - current_q))
        grads = tape.gradient(c_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(
            zip(grads, self.critic.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            pred_actions = self.actor(states, training=True)
            a_loss       = -tf.reduce_mean(
                self.critic([states, pred_actions], training=False))
        grads = tape.gradient(a_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(
            zip(grads, self.actor.trainable_variables))

        # Soft update targets
        self._soft_update(self.target_actor,  self.actor)
        self._soft_update(self.target_critic, self.critic)

        # Decay noise
        self.noise = max(0.05, self.noise * 0.9999)


    def _soft_update(self, target, source):
        for t, s in zip(target.variables, source.variables):
            t.assign(TAU * s + (1 - TAU) * t)

    def save(self):
        self.actor.save(ACTOR_PATH)
        print(f"  ✅ DDPG actor saved → {ACTOR_PATH}")


# =============================================================
# Train
# =============================================================

def train():
    print("=" * 55)
    print("  DDPG Baseline — Training")
    print("=" * 55)

    env   = LBROEnvironment(seed=42)
    agent = DDPGAgent(seed=42)

    for ep in range(1, NUM_EPISODES + 1):
        state     = env.reset()
        ep_reward = 0.0

        for step in range(MAX_STEPS_PER_EP):
            action               = agent.act(state, training=True)
            next_state, r, done, _ = env.step(action)
            agent.remember(state, action, r, next_state, done)
            agent.learn()
            ep_reward += r
            state      = next_state
            if done:
                break

        if ep % 50 == 0:
            summary = env.episode_summary()
            print(f"  Ep {ep:4d}  reward={ep_reward:.2f}  "
                  f"drop={summary['drop_rate']:.2%}  "
                  f"noise={agent.noise:.4f}")

    agent.save()
    print("  Training complete ✅")


# =============================================================
# Evaluate
# =============================================================

def evaluate():
    print("\n  DDPG Baseline — Evaluation")

    env   = LBROEnvironment(seed=100)
    agent = DDPGAgent(seed=0)

    if os.path.exists(ACTOR_PATH):
        agent.actor = tf.keras.models.load_model(
            ACTOR_PATH, compile=False)
        print(f"  ✅ Loaded ← {ACTOR_PATH}")
    else:
        print("  ⚠️  No saved model found. Run train() first.")
        return None

    rows = []
    for ep in range(EVAL_EPISODES):
        state     = env.reset()
        ep_reward = 0.0
        for step in range(MAX_STEPS_PER_EP):
            action               = agent.act(state, training=False)
            state, r, done, _    = env.step(action)
            ep_reward           += r
            if done:
                break
        summary = env.episode_summary()
        rows.append({
            "policy"   : "DDPG",
            "episode"  : ep,
            "reward"   : round(ep_reward,                4),
            "drop_rate": round(summary["drop_rate"],     4),
            "avg_lat"  : round(summary["avg_latency_s"], 4),
            "avg_eng"  : round(summary["avg_energy_j"],  4),
        })

    df  = pd.DataFrame(rows)
    avg = df.mean(numeric_only=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"  DDPG  drop={avg['drop_rate']:.2%}  "
          f"lat={avg['avg_lat']:.4f}s  "
          f"eng={avg['avg_eng']:.4f}J  "
          f"reward={avg['reward']:.2f}")
    print(f"  Saved → {RESULTS_CSV}")
    return df


if __name__ == "__main__":
    train()
    evaluate()
