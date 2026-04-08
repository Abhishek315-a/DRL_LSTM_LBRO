# =============================================================
# agents/ppo.py
# DRL-LSTM-LBRO  —  PPO Baseline Agent (discrete action space)
#
# Actor-Critic with clipped surrogate objective (Schulman 2017).
# Used as an additional DRL baseline in the evaluation.
# =============================================================

import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.config import (
    STATE_DIM, NUM_ACTIONS, MODEL_DIR,
    DDQN_HIDDEN_UNITS,
)

os.makedirs(MODEL_DIR, exist_ok=True)

PPO_LR          = 3e-4
PPO_GAMMA       = 0.99
PPO_LAM         = 0.95      # GAE lambda
PPO_CLIP_EPS    = 0.2
PPO_EPOCHS      = 4
PPO_ENT_COEF    = 0.01      # entropy bonus coefficient
PPO_VF_COEF     = 0.5       # value loss coefficient

ACTOR_PATH  = os.path.join(MODEL_DIR, "ppo_actor.keras")
CRITIC_PATH = os.path.join(MODEL_DIR, "ppo_critic.keras")


# =============================================================
# Network builders
# =============================================================

def _build_actor(state_dim: int, action_dim: int) -> Model:
    inp = Input(shape=(state_dim,))
    x   = Dense(DDQN_HIDDEN_UNITS[0], activation="relu")(inp)
    x   = Dense(DDQN_HIDDEN_UNITS[1], activation="relu")(x)
    out = Dense(action_dim, activation=None)(x)   # raw logits
    return Model(inp, out, name="ppo_actor")


def _build_critic(state_dim: int) -> Model:
    inp = Input(shape=(state_dim,))
    x   = Dense(DDQN_HIDDEN_UNITS[0], activation="relu")(inp)
    x   = Dense(DDQN_HIDDEN_UNITS[1], activation="relu")(x)
    out = Dense(1, activation=None)(x)
    return Model(inp, out, name="ppo_critic")


# =============================================================
# PPO Agent
# =============================================================

class PPOAgent:

    def __init__(self, state_dim: int = STATE_DIM,
                 action_dim: int = NUM_ACTIONS, seed: int = 42):
        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.actor  = _build_actor(state_dim, action_dim)
        self.critic = _build_critic(state_dim)

        self.actor_opt  = Adam(learning_rate=PPO_LR)
        self.critic_opt = Adam(learning_rate=PPO_LR)

        self.rollout_states      = []
        self.rollout_actions     = []
        self.rollout_log_probs   = []
        self.rollout_rewards     = []
        self.rollout_values      = []
        self.rollout_dones       = []

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------

    def _policy(self, state: np.ndarray):
        """Returns (action, log_prob, value) for a single state."""
        s     = tf.constant(state[None], dtype=tf.float32)
        logits = self.actor(s, training=False)[0]
        dist   = tf.nn.softmax(logits)
        action = tf.squeeze(tf.random.categorical(logits[None], 1), axis=1)[0]
        log_p  = tf.math.log(dist[action] + 1e-8)
        value  = self.critic(s, training=False)[0, 0]
        return int(action.numpy()), float(log_p.numpy()), float(value.numpy())

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Greedy (deterministic) at evaluation; sampled during training."""
        s      = tf.constant(state[None], dtype=tf.float32)
        logits = self.actor(s, training=False)[0]
        if training:
            action = tf.squeeze(tf.random.categorical(logits[None], 1))[0]
        else:
            action = tf.argmax(logits)
        return int(action.numpy())

    # ----------------------------------------------------------
    # Rollout buffer
    # ----------------------------------------------------------

    def step(self, state, action=None):
        """Called at each env step during training to record transition."""
        a, log_p, v = self._policy(state)
        self.rollout_states.append(state.copy())
        self.rollout_actions.append(a)
        self.rollout_log_probs.append(log_p)
        self.rollout_values.append(v)
        return a

    def store_reward_done(self, reward: float, done: bool):
        self.rollout_rewards.append(reward)
        self.rollout_dones.append(float(done))

    def clear_rollout(self):
        self.rollout_states      = []
        self.rollout_actions     = []
        self.rollout_log_probs   = []
        self.rollout_rewards     = []
        self.rollout_values      = []
        self.rollout_dones       = []

    # ----------------------------------------------------------
    # GAE + PPO update
    # ----------------------------------------------------------

    def _compute_gae(self, last_value: float = 0.0):
        rewards  = self.rollout_rewards
        values   = self.rollout_values + [last_value]
        dones    = self.rollout_dones
        T        = len(rewards)
        adv      = np.zeros(T, dtype=np.float32)
        gae      = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + PPO_GAMMA * values[t+1] * (1 - dones[t]) - values[t]
            gae   = delta + PPO_GAMMA * PPO_LAM * (1 - dones[t]) * gae
            adv[t] = gae
        returns = adv + np.array(self.rollout_values, dtype=np.float32)
        return adv, returns

    def learn(self, last_value: float = 0.0) -> float:
        """Run PPO_EPOCHS of updates on the current rollout. Returns mean loss."""
        advantages, returns = self._compute_gae(last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states    = tf.constant(np.array(self.rollout_states),    dtype=tf.float32)
        actions   = tf.constant(np.array(self.rollout_actions),   dtype=tf.int32)
        old_lp    = tf.constant(np.array(self.rollout_log_probs), dtype=tf.float32)
        adv_t     = tf.constant(advantages,                        dtype=tf.float32)
        ret_t     = tf.constant(returns,                           dtype=tf.float32)

        total_loss = 0.0
        for _ in range(PPO_EPOCHS):
            with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
                logits  = self.actor(states,  training=True)
                dist    = tf.nn.softmax(logits)
                log_p   = tf.math.log(
                    tf.gather(dist, actions, batch_dims=1) + 1e-8)
                ratio   = tf.exp(log_p - old_lp)
                clip_r  = tf.clip_by_value(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS)
                policy_loss = -tf.reduce_mean(
                    tf.minimum(ratio * adv_t, clip_r * adv_t))
                entropy = -tf.reduce_mean(tf.reduce_sum(dist * tf.math.log(dist + 1e-8), axis=1))
                actor_loss = policy_loss - PPO_ENT_COEF * entropy

                values  = tf.squeeze(self.critic(states, training=True), axis=1)
                vf_loss = tf.reduce_mean(tf.square(values - ret_t))

            grads_a = tape_a.gradient(actor_loss,  self.actor.trainable_variables)
            grads_c = tape_c.gradient(vf_loss,     self.critic.trainable_variables)
            self.actor_opt.apply_gradients(
                zip(grads_a, self.actor.trainable_variables))
            self.critic_opt.apply_gradients(
                zip(grads_c, self.critic.trainable_variables))
            total_loss += float(actor_loss + PPO_VF_COEF * vf_loss)

        self.clear_rollout()
        return total_loss / PPO_EPOCHS

    # ----------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------

    def save(self, actor_path: str = ACTOR_PATH,
             critic_path: str = CRITIC_PATH):
        self.actor.save(actor_path)
        self.critic.save(critic_path)

    def load(self, actor_path: str = ACTOR_PATH,
             critic_path: str = CRITIC_PATH):
        self.actor  = tf.keras.models.load_model(actor_path)
        self.critic = tf.keras.models.load_model(critic_path)
        print(f"  ✅ PPO loaded ← {actor_path}")
