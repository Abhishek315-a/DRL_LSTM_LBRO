# =============================================================
# models/lstm_predictor.py
# DRL-LSTM-LBRO  —  Step 4: GRU-LSTM Workload Predictor
#
# PURPOSE:
#   Predict next-slot CPU and RAM utilisation for each
#   cloudlet node using a hybrid GRU-LSTM architecture.
#   (AICDQN Section "Load Forecaster GRU-LSTM Module")
#
# These predictions feed directly into the 23-dim state
# vector at indices:
#   [4]  Cloudlet-0 predicted CPU
#   [5]  Cloudlet-0 predicted RAM
#   [10] Cloudlet-1 predicted CPU
#   [11] Cloudlet-1 predicted RAM
#   [16] Cloudlet-2 predicted CPU
#   [17] Cloudlet-2 predicted RAM
#
# Architecture (AICDQN Eq. 11):
#   h_t = LSTM(GRU(Q_t, h_{t-1}))
#   λ̂_{t+1} = W_o · h_t + b_o
#
# INPUT :  data/lstm_traces.csv
# OUTPUT:  models/lstm_c0.h5
#          models/lstm_c1.h5
#          models/lstm_c2.h5
#
# Run with:
#     cd /Users/abhishek.sk/Documents/college/DRL_LSTM_LBRO
#     python3 -m models.lstm_predictor
# =============================================================

import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models     import Sequential, load_model
from tensorflow.keras.layers     import GRU, LSTM, Dense, Dropout
from tensorflow.keras.callbacks  import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.config import (
    NUM_CLOUDLETS, LSTM_WINDOW, LSTM_UNITS,
    LSTM_EPOCHS, LSTM_BATCH,
    DATA_DIR, MODEL_DIR, LSTM_MODEL_PATH
)

os.makedirs(MODEL_DIR, exist_ok=True)

LSTM_CSV = os.path.join(DATA_DIR, "lstm_traces.csv")


# =============================================================
# Data preparation
# =============================================================

def build_sequences(series: np.ndarray,
                    window: int = LSTM_WINDOW):
    """
    Convert time-series array into (X, y) sliding windows.

    X shape : (N, window, 2)
    y shape : (N, 2)  → [next_cpu, next_ram]
    """
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window, :2])
    return np.array(X, dtype=np.float32), \
           np.array(y, dtype=np.float32)


def prepare_data(csv_path: str = LSTM_CSV,
                 cloudlet_id: int = 0,
                 window: int = LSTM_WINDOW,
                 test_ratio: float = 0.2):
    """
    Load lstm_traces.csv and build train/test sequences
    for a single cloudlet node.

    Features per timestep : [cpu_util, ram_util]
    Target                : [next_cpu, next_ram]
    """
    df  = pd.read_csv(csv_path)
    sub = df[df["cloudlet_id"] == cloudlet_id].copy()
    sub = sub.sort_values(["episode", "slot"]).reset_index(drop=True)

    series = sub[["cpu_util", "ram_util"]].values.astype(np.float32)
    X, y   = build_sequences(series, window)

    split           = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test


# =============================================================
# Model architecture
# =============================================================

def build_gru_lstm_model(window: int     = LSTM_WINDOW,
                         n_features: int = 2,
                         units: int      = LSTM_UNITS) -> tf.keras.Model:
    """
    Hybrid GRU-LSTM architecture (AICDQN Eq. 11):
        h_t = LSTM(GRU(Q_t, h_{t-1}))
        λ̂_{t+1} = W_o · h_t + b_o

    GRU  → captures short-term load fluctuations
    LSTM → captures long-term workload dependencies
    """
    model = Sequential([
        GRU(units,
            input_shape      = (window, n_features),
            return_sequences = True,
            name             = "gru_layer"),
        Dropout(0.1, name="dropout_1"),

        LSTM(units,
             return_sequences = False,
             name             = "lstm_layer"),
        Dropout(0.1, name="dropout_2"),

        Dense(32, activation="relu",    name="dense_hidden"),
        Dense(2,  activation="sigmoid", name="output"),
        # sigmoid → output ∈ [0,1] matching normalised cpu/ram
    ], name="GRU_LSTM_Predictor")

    model.compile(
        optimizer = Adam(learning_rate=1e-3),
        loss      = "mse",      # AICDQN Eq. 12: L_MSE
        metrics   = ["mae"]
    )
    return model


# =============================================================
# LSTMPredictor  —  wrapper class
# =============================================================

class LSTMPredictor:
    """
    GRU-LSTM workload predictor for one cloudlet node.

    After training, call predict(history) from environment.py
    to get the next-slot CPU and RAM prediction which feeds
    into the 23-dim state vector (indices [4],[5] per node).

    Methods:
        train(csv_path)       : train on lstm_traces.csv
        predict(history)      : predict next cpu/ram
        save(path)            : save .h5 model
        load(cloudlet_id)     : load saved .h5 model
    """

    def __init__(self, cloudlet_id: int,
                 window: int = LSTM_WINDOW,
                 units:  int = LSTM_UNITS):
        self.cloudlet_id = cloudlet_id
        self.window      = window
        self.units       = units
        self.model       = None
        self.is_trained  = False
        self.history     = None
        self.model_path  = LSTM_MODEL_PATH.format(cloudlet_id)

    # ── Train ─────────────────────────────────────────────────
    def train(self, csv_path: str = LSTM_CSV) -> dict:
        """
        Load data → build model → train → evaluate → save.
        """
        print(f"\n  [Cloudlet-{self.cloudlet_id}] Training GRU-LSTM...")

        X_train, X_test, y_train, y_test = prepare_data(
            csv_path, self.cloudlet_id, self.window
        )
        print(f"    Train sequences : {len(X_train):,}")
        print(f"    Test  sequences : {len(X_test):,}")
        print(f"    Input shape     : {X_train.shape}")

        self.model = build_gru_lstm_model(self.window, 2, self.units)

        if self.cloudlet_id == 0:
            self.model.summary()

        callbacks = [
            EarlyStopping(
                monitor              = "val_loss",
                patience             = 8,
                restore_best_weights = True,
                verbose              = 0
            ),
            ModelCheckpoint(
                filepath       = self.model_path,
                monitor        = "val_loss",
                save_best_only = True,
                verbose        = 0
            )
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data = (X_test, y_test),
            epochs          = LSTM_EPOCHS,
            batch_size      = LSTM_BATCH,
            callbacks       = callbacks,
            verbose         = 1
        )
        self.is_trained = True

        y_pred   = self.model.predict(X_test, verbose=0)
        mse      = float(np.mean((y_pred - y_test) ** 2))
        mae      = float(np.mean(np.abs(y_pred - y_test)))
        best_ep  = int(np.argmin(
                       self.history.history["val_loss"])) + 1

        print(f"    Best epoch : {best_ep}/{LSTM_EPOCHS}")
        print(f"    Test  MSE  : {mse:.6f}")
        print(f"    Test  MAE  : {mae:.6f}")

        return {"mse": mse, "mae": mae, "best_epoch": best_ep}

    # ── Predict ───────────────────────────────────────────────
    def predict(self, history: np.ndarray) -> tuple:
        """
        Predict next-slot CPU and RAM utilisation.

        Args:
            history : np.ndarray shape (LSTM_WINDOW, 2)
                      from env.get_lstm_input(cloudlet_id)
                      Column 0 = cpu_util history
                      Column 1 = ram_util history

        Returns:
            (cpu_pred, ram_pred) : floats ∈ [0, 1]
        """
        assert self.is_trained, \
            "Model not trained. Call train() or load() first."
        assert history.shape == (self.window, 2), \
            f"Expected ({self.window}, 2), got {history.shape}"

        x     = history.reshape(1, self.window, 2).astype(np.float32)
        pred  = self.model.predict(x, verbose=0)[0]
        cpu_p = float(np.clip(pred[0], 0.0, 1.0))
        ram_p = float(np.clip(pred[1], 0.0, 1.0))
        return cpu_p, ram_p

    # ── Save / Load ───────────────────────────────────────────
    def save(self, path: str = None):
        path = path or self.model_path
        self.model.save(path)          # saves as .keras automatically
        size_kb = os.path.getsize(path) / 1024
        print(f"    ✅ Saved → {path}  ({size_kb:.1f} KB)")

    @classmethod
    def load(cls, cloudlet_id: int, path: str = None):
        path = path or LSTM_MODEL_PATH.format(cloudlet_id)
        assert os.path.exists(path), \
            f"Model not found: {path}. Run train() first."
        obj            = cls(cloudlet_id)
        obj.model      = load_model(path, compile=False)   # ← compile=False
        obj.model.compile(
            optimizer = Adam(learning_rate=1e-3),
            loss      = "mse",
            metrics   = ["mae"]
        )
        obj.is_trained = True
        print(f"    ✅ Loaded ← {path}")
        return obj



# =============================================================
# Train / load all 3 cloudlets
# =============================================================

def train_all(csv_path: str = LSTM_CSV) -> dict:
    """Train one GRU-LSTM model per cloudlet."""
    print("\n  Training GRU-LSTM for all 3 cloudlets...")
    all_metrics = {}
    for cid in range(NUM_CLOUDLETS):
        predictor                        = LSTMPredictor(cloudlet_id=cid)
        metrics                          = predictor.train(csv_path)
        all_metrics[f"cloudlet_{cid}"]   = metrics
    return all_metrics


def load_all() -> list:
    """Load all 3 saved LSTM models. Returns list of LSTMPredictor."""
    predictors = []
    for cid in range(NUM_CLOUDLETS):
        predictors.append(LSTMPredictor.load(cid))
    return predictors


# =============================================================
# MAIN
# =============================================================

def main():
    print("=" * 55)
    print("  DRL-LSTM-LBRO  —  Step 4: GRU-LSTM Predictor")
    print("=" * 55)

    # ── Train ─────────────────────────────────────────────────
    all_metrics = train_all()

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"  TRAINING SUMMARY")
    print(f"{'─' * 50}")
    for cid in range(NUM_CLOUDLETS):
        m = all_metrics[f"cloudlet_{cid}"]
        print(f"  Cloudlet-{cid}  "
              f"MSE={m['mse']:.6f}  "
              f"MAE={m['mae']:.6f}  "
              f"best_epoch={m['best_epoch']}")

    # ── Verify load + inference ───────────────────────────────
    print(f"\n  Verifying load + inference...")
    predictors = load_all()

    from simulator.environment import LBROEnvironment
    env = LBROEnvironment(seed=7)
    env.reset()
    for _ in range(LSTM_WINDOW + 5):
        env.step(0)

    print(f"\n  Inference test (after {LSTM_WINDOW + 5} steps):")
    for cid, p in enumerate(predictors):
        history      = env.get_lstm_input(cid)
        cpu_p, ram_p = p.predict(history)
        actual_cpu   = env.cloudlets[cid].cpu_util
        actual_ram   = env.cloudlets[cid].ram_util
        print(f"    Cloudlet-{cid}  "
              f"pred_cpu={cpu_p:.4f}  actual={actual_cpu:.4f}  |  "
              f"pred_ram={ram_p:.4f}  actual={actual_ram:.4f}")

    print(f"\n{'=' * 55}")
    print(f"  Step 4 COMPLETE ✅")
    print(f"  Models saved:")
    for cid in range(NUM_CLOUDLETS):
        path = LSTM_MODEL_PATH.format(cid)
        size = os.path.getsize(path) / 1024
        print(f"    {path}  ({size:.1f} KB)")
    print(f"  Next → Step 5: agents/ddqn.py")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
