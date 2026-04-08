# =============================================================
# models/lstm_predictor.py
# DRL-LSTM-LBRO  —  Step 4: GRU-LSTM Workload Predictor
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
    DATA_DIR, MODEL_DIR, LSTM_MODEL_PATH, LSTM_CSV_PATH
)

os.makedirs(MODEL_DIR, exist_ok=True)

LSTM_CSV      = LSTM_CSV_PATH
LSTM_HORIZON  = 5   # predict 5 slots ahead instead of 1


# =============================================================
# Data preparation
# =============================================================

def build_sequences(series: np.ndarray,
                    window:  int = LSTM_WINDOW,
                    horizon: int = LSTM_HORIZON):
    """
    Convert time-series into (X, y) sliding windows.
    X shape : (N, window, 2)
    y shape : (N, 2)  → [cpu @ t+horizon, ram @ t+horizon]
    """
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i : i + window])
        y.append(series[i + window + horizon - 1, :2])  # t+5 ahead
    return np.array(X, dtype=np.float32), \
           np.array(y, dtype=np.float32)


def prepare_data(csv_path:    str = LSTM_CSV,
                 cloudlet_id: int = 0,
                 window:      int = LSTM_WINDOW,
                 test_ratio:  float = 0.2):
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

def build_gru_lstm_model(window:     int = LSTM_WINDOW,
                         n_features: int = 2,
                         units:      int = LSTM_UNITS) -> tf.keras.Model:
    """
    Hybrid GRU-LSTM architecture (AICDQN Eq. 11):
        h_t = LSTM(GRU(Q_t, h_{t-1}))
        λ̂_{t+horizon} = W_o · h_t + b_o
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
    ], name="GRU_LSTM_Predictor")

    model.compile(
        optimizer = Adam(learning_rate=1e-3),
        loss      = "mse",
        metrics   = ["mae"]
    )
    return model


# =============================================================
# LSTMPredictor  —  wrapper class
# =============================================================

class LSTMPredictor:

    def __init__(self, cloudlet_id: int,
                 window:  int = LSTM_WINDOW,
                 units:   int = LSTM_UNITS):
        self.cloudlet_id = cloudlet_id
        self.window      = window
        self.units       = units
        self.model       = None
        self.is_trained  = False
        self.history     = None
        self.model_path  = LSTM_MODEL_PATH.format(cloudlet_id)

    def train(self, csv_path: str = LSTM_CSV) -> dict:
        print(f"\n  [Cloudlet-{self.cloudlet_id}] Training GRU-LSTM "
              f"(horizon={LSTM_HORIZON} slots ahead)...")

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

        y_pred  = self.model(X_test, training=False).numpy()
        mse     = float(np.mean((y_pred - y_test) ** 2))
        mae     = float(np.mean(np.abs(y_pred - y_test)))
        best_ep = int(np.argmin(self.history.history["val_loss"])) + 1

        print(f"    Best epoch : {best_ep}/{LSTM_EPOCHS}")
        print(f"    Test  MSE  : {mse:.6f}")
        print(f"    Test  MAE  : {mae:.6f}")

        return {"mse": mse, "mae": mae, "best_epoch": best_ep}

    def predict(self, history: np.ndarray) -> tuple:
        """
        Predict CPU and RAM utilisation 5 slots ahead.

        Args:
            history : np.ndarray shape (LSTM_WINDOW, 2)
                      from env.get_lstm_input(cloudlet_id)
        Returns:
            (cpu_pred, ram_pred) : floats ∈ [0, 1]
        """
        assert self.is_trained, \
            "Model not trained. Call train() or load() first."
        assert history.shape == (self.window, 2), \
            f"Expected ({self.window}, 2), got {history.shape}"

        x     = history.reshape(1, self.window, 2).astype(np.float32)
        pred  = self.model(x, training=False)[0].numpy()
        cpu_p = float(np.clip(pred[0], 0.0, 1.0))
        ram_p = float(np.clip(pred[1], 0.0, 1.0))
        return cpu_p, ram_p

    def save(self, path: str = None):
        path = path or self.model_path
        self.model.save(path)
        size_kb = os.path.getsize(path) / 1024
        print(f"    ✅ Saved → {path}  ({size_kb:.1f} KB)")

    @classmethod
    def load(cls, cloudlet_id: int, path: str = None):
        path = path or LSTM_MODEL_PATH.format(cloudlet_id)
        assert os.path.exists(path), \
            f"Model not found: {path}. Run train() first."
        obj            = cls(cloudlet_id)
        obj.model      = load_model(path, compile=False)
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
    print(f"\n  Training GRU-LSTM for all {NUM_CLOUDLETS} cloudlets...")
    all_metrics = {}
    for cid in range(NUM_CLOUDLETS):
        predictor                      = LSTMPredictor(cloudlet_id=cid)
        metrics                        = predictor.train(csv_path)
        all_metrics[f"cloudlet_{cid}"] = metrics
    return all_metrics


def load_all() -> list:
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

    all_metrics = train_all()

    print(f"\n{'─' * 50}")
    print(f"  TRAINING SUMMARY  (horizon = {LSTM_HORIZON} slots ahead)")
    print(f"{'─' * 50}")
    for cid in range(NUM_CLOUDLETS):
        m = all_metrics[f"cloudlet_{cid}"]
        print(f"  Cloudlet-{cid}  "
              f"MSE={m['mse']:.6f}  "
              f"MAE={m['mae']:.6f}  "
              f"best_epoch={m['best_epoch']}")

    print(f"\n  Verifying load + inference...")
    predictors = load_all()

    from simulator.environment import LBROEnvironment
    env = LBROEnvironment(seed=7)
    env.reset()
    for _ in range(LSTM_WINDOW + 5):
        env.step(0)

    print(f"\n  Inference test (predicting {LSTM_HORIZON} slots ahead):")
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
    for cid in range(NUM_CLOUDLETS):
        path = LSTM_MODEL_PATH.format(cid)
        size = os.path.getsize(path) / 1024
        print(f"    {path}  ({size:.1f} KB)")
    print(f"  Next → Step 5: agents/ddqn.py")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
