# =============================================================
# agents/ddqn.py
# DRL-LSTM-LBRO  —  Step 5: Double DQN Agent
# =============================================================

import os
import sys
import random
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import Dense, Input
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.config import (
    STATE_DIM, NUM_ACTIONS,
    DDQN_LR, DDQN_GAMMA,
    DDQN_EPSILON, DDQN_EPSILON_MIN, DDQN_EPSILON_DECAY,
    DDQN_BATCH_SIZE, DDQN_MEMORY_SIZE, DDQN_TARGET_UPDATE,
    DDQN_HIDDEN_UNITS, MODEL_DIR
)

os.makedirs(MODEL_DIR, exist_ok=True)

DDQN_MODEL_PATH  = os.path.join(MODEL_DIR, "ddqn_online.keras")
DDQN_TARGET_PATH = os.path.join(MODEL_DIR, "ddqn_target.keras")


# =============================================================
# Replay Buffer
# =============================================================

class ReplayBuffer:

    def __init__(self, maxlen: int = DDQN_MEMORY_SIZE):
        self.buffer = deque(maxlen=maxlen)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size: int = DDQN_BATCH_SIZE):
        batch       = random.sample(self.buffer, batch_size)
        states      = np.array([e[0] for e in batch], dtype=np.float32)
        actions     = np.array([e[1] for e in batch], dtype=np.int32)
        rewards     = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones       = np.array([e[4] for e in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    @property
    def ready(self) -> bool:
        return len(self.buffer) >= DDQN_BATCH_SIZE


# =============================================================
# Q-Network
# =============================================================

def build_q_network(state_dim:    int  = STATE_DIM,
                    num_actions:  int  = NUM_ACTIONS,
                    hidden_units: list = None) -> tf.keras.Model:
    if hidden_units is None:
        hidden_units = DDQN_HIDDEN_UNITS    # [256, 128]

    model = Sequential(name="Q_Network")
    model.add(Input(shape=(state_dim,), name="state_input"))
    for i, units in enumerate(hidden_units):
        model.add(Dense(units, activation="relu", name=f"hidden_{i+1}"))
    model.add(Dense(num_actions, activation="linear", name="q_values"))

    model.compile(optimizer=Adam(learning_rate=DDQN_LR), loss="mse")
    return model


# =============================================================
# DDQNAgent
# =============================================================

class DDQNAgent:

    def __init__(self,
                 state_dim:   int   = STATE_DIM,
                 num_actions: int   = NUM_ACTIONS,
                 lr:          float = DDQN_LR,
                 gamma:       float = DDQN_GAMMA,
                 epsilon:     float = DDQN_EPSILON,
                 seed:        int   = 42):

        self.state_dim   = state_dim
        self.num_actions = num_actions
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.lr          = lr
        self.step_count  = 0
        self.learn_count = 0

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.online_net = build_q_network(state_dim, num_actions)
        self.target_net = build_q_network(state_dim, num_actions)
        self._sync_target()

        self.buffer   = ReplayBuffer(DDQN_MEMORY_SIZE)
        self.losses   = []
        self.epsilons = []


    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_t = tf.constant(state.reshape(1, -1), dtype=tf.float32)
        q_vals  = self.online_net(state_t, training=False).numpy()[0]
        return int(np.argmax(q_vals))


    def q_values(self, state: np.ndarray) -> np.ndarray:
        state_t = tf.constant(state.reshape(1, -1), dtype=tf.float32)
        return self.online_net(state_t, training=False).numpy()[0]


    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1


    def learn(self) -> float:
        """
        DDQN update rule:
            a*  = argmax_a  Q_online(s', a)    ← online selects
            y   = r + γ · Q_target(s', a*)     ← target evaluates
            L   = MSE(Q_online(s, a),  y)

        Uses train_on_batch instead of model.fit()
        for significantly faster per-step training.
        """
        if not self.buffer.ready:
            return 0.0

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(DDQN_BATCH_SIZE)

        # Online net selects best next action
        q_next_online = self.online_net(next_states, training=False).numpy()
        best_actions  = np.argmax(q_next_online, axis=1)

        # Target net evaluates that action
        q_next_target = self.target_net(next_states, training=False).numpy()
        best_q_next   = q_next_target[
            np.arange(DDQN_BATCH_SIZE), best_actions]

        # Bellman targets
        targets_full = self.online_net(states, training=False).numpy()
        for i in range(DDQN_BATCH_SIZE):
            if dones[i]:
                targets_full[i, actions[i]] = rewards[i]
            else:
                targets_full[i, actions[i]] = (
                    rewards[i] + self.gamma * best_q_next[i]
                )

        # ── Single batch update — much faster than model.fit() ──
        loss = float(self.online_net.train_on_batch(states, targets_full))

        # Epsilon decay
        if self.epsilon > DDQN_EPSILON_MIN:
            self.epsilon *= DDQN_EPSILON_DECAY

        self.losses.append(loss)
        self.epsilons.append(self.epsilon)
        self.learn_count += 1

        # Sync target every N steps
        if self.learn_count % DDQN_TARGET_UPDATE == 0:
            self._sync_target()

        return loss


    def _sync_target(self):
        self.target_net.set_weights(self.online_net.get_weights())


    def save(self, online_path: str = DDQN_MODEL_PATH,
             target_path:  str = DDQN_TARGET_PATH):
        self.online_net.save(online_path)
        self.target_net.save(target_path)
        print(f"  ✅ DDQN saved → {online_path}")
        print(f"                  {target_path}")


    def load(self, online_path: str = DDQN_MODEL_PATH,
             target_path:  str = DDQN_TARGET_PATH):
        assert os.path.exists(online_path), \
            f"Model not found: {online_path}"
        self.online_net = tf.keras.models.load_model(
            online_path, compile=False)
        self.online_net.compile(
            optimizer=Adam(learning_rate=self.lr), loss="mse")
        self.target_net = tf.keras.models.load_model(
            target_path, compile=False)
        self.target_net.compile(
            optimizer=Adam(learning_rate=self.lr), loss="mse")
        print(f"  ✅ DDQN loaded ← {online_path}")


    def summary(self) -> dict:
        return {
            "step_count" : self.step_count,
            "learn_count": self.learn_count,
            "epsilon"    : round(self.epsilon, 4),
            "buffer_size": len(self.buffer),
            "avg_loss"   : (float(np.mean(self.losses[-100:]))
                            if self.losses else 0.0),
        }

    def print_architecture(self):
        print("\n  Online Network:")
        self.online_net.summary()


# =============================================================
# MAIN  —  smoke test
# =============================================================

def main():
    print("=" * 55)
    print("  DRL-LSTM-LBRO  —  Step 5: DDQN Agent")
    print("=" * 55)

    agent = DDQNAgent(seed=42)
    agent.print_architecture()

    print("\n  Smoke tests:")
    state  = np.random.rand(STATE_DIM).astype(np.float32)
    next_s = np.random.rand(STATE_DIM).astype(np.float32)

    action = agent.act(state, training=True)
    assert 0 <= action < NUM_ACTIONS
    print(f"    ✅ act()      → action={action}  (ε={agent.epsilon:.2f})")

    qv = agent.q_values(state)
    assert qv.shape == (NUM_ACTIONS,)
    print(f"    ✅ q_values() → {np.round(qv, 4)}")

    for _ in range(DDQN_BATCH_SIZE + 5):
        agent.remember(state, action, -0.3, next_s, False)
    assert len(agent.buffer) == DDQN_BATCH_SIZE + 5
    print(f"    ✅ remember() → buffer={len(agent.buffer)}")

    loss = agent.learn()
    assert loss > 0.0
    print(f"    ✅ learn()    → loss={loss:.6f}")

    assert agent.epsilon < DDQN_EPSILON
    print(f"    ✅ epsilon    → {agent.epsilon:.6f}")

    agent.save()
    agent2 = DDQNAgent(seed=0)
    agent2.load()
    qv_now = agent.q_values(state)
    qv2    = agent2.q_values(state)
    assert np.allclose(qv_now, qv2, atol=1e-4)
    print(f"    ✅ save/load  → Q-values match after reload")

    print(f"\n{'=' * 55}")
    print(f"  Step 5 COMPLETE ✅")
    print(f"  State dim  : {STATE_DIM}")
    print(f"  Actions    : {NUM_ACTIONS}  (C0, C1, C2, Cloud)")
    print(f"  Hidden     : {DDQN_HIDDEN_UNITS}")
    print(f"  Buffer     : {DDQN_MEMORY_SIZE:,}")
    print(f"  γ (gamma)  : {DDQN_GAMMA}")
    print(f"  ε decay    : {DDQN_EPSILON} → {DDQN_EPSILON_MIN}")
    print(f"  Next → Step 6: train.py")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
