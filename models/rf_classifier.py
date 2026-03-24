# =============================================================
# models/rf_classifier.py
# DRL-LSTM-LBRO  —  Step 3: Random Forest Task Classifier
#
# PURPOSE:
#   Classify each incoming task as:
#       0 = CPU_INTENSIVE  → route to highest MIPS cloudlet
#       1 = MEM_INTENSIVE  → route to cloudlet with free RAM
#       2 = BALANCED       → route to least loaded cloudlet
#
# INPUT :  data/workload_traces.csv  (generated in Step 2)
# OUTPUT:  models/rf_model.pkl       (saved trained model)
#
# Features used (11 total):
#   Raw       : cpu_demand_mips, ram_demand_mb, size_mbits,
#               cpu_cycles, static_priority
#   Normalised: cpu_norm, ram_norm, size_norm
#   Engineered: cpu_ram_ratio, compute_density, load_score
#
# Run with:
#     cd /Users/abhishek.sk/Documents/college/DRL_LSTM_LBRO
#     python3 -m models.rf_classifier
# =============================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (accuracy_score,
                                      classification_report,
                                      confusion_matrix)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.config import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_TEST_SPLIT,
    DATA_DIR, MODEL_DIR, WORKLOAD_CSV, RF_MODEL_PATH
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature columns ───────────────────────────────────────────
FEATURE_COLS = [
    "cpu_demand_mips",    # raw CPU demand
    "ram_demand_mb",      # raw RAM demand
    "size_mbits",         # raw task size
    "cpu_cycles",         # C_i = S_i × β_i  (AICDQN Eq. 2)
    "static_priority",    # 0=low, 1=high
    "cpu_norm",           # normalised CPU
    "ram_norm",           # normalised RAM
    "size_norm",          # normalised size
    "cpu_ram_ratio",      # cpu / ram  (engineered)
    "compute_density",    # cycles / size  (engineered)
    "load_score",         # 0.6×cpu_n + 0.4×ram_n  (engineered)
]
LABEL_COL   = "task_type"
CLASS_NAMES = ["CPU_INTENSIVE", "MEM_INTENSIVE", "BALANCED"]


# =============================================================
# RFTaskClassifier
# =============================================================

class RFTaskClassifier:
    """
    Random Forest classifier for IoT task type prediction.

    After training, call predict(task) from the LBRO broker
    to get the task type label for routing decisions.

    Methods:
        train(csv_path)      : load CSV → train → evaluate → save
        predict(task)        : predict single task type (int 0/1/2)
        predict_proba(task)  : probability over 3 classes
        save(path)           : save model to .pkl
        load(path)           : load model from .pkl
        feature_importance() : sorted feature importance DataFrame
    """

    def __init__(self, n_estimators: int = RF_N_ESTIMATORS,
                 max_depth: int = RF_MAX_DEPTH,
                 seed: int = 42):
        self.model = RandomForestClassifier(
            n_estimators = n_estimators,
            max_depth    = max_depth,
            random_state = seed,
            n_jobs       = -1,            # use all CPU cores
            class_weight = "balanced",    # handle class imbalance
        )
        self.is_trained   = False
        self.feature_cols = FEATURE_COLS
        self.class_names  = CLASS_NAMES
        self.train_acc    = None
        self.test_acc     = None
        self.cv_scores    = None

    # ══════════════════════════════════════════════════════════
    # train()
    # ══════════════════════════════════════════════════════════

    def train(self, csv_path: str = WORKLOAD_CSV) -> dict:
        """
        Load CSV → split → train → evaluate → save model.
        Returns dict with all performance metrics.
        """
        print(f"\n[Step 3] Training RF Classifier...")
        print(f"  Loading  : {csv_path}")

        # ── Load dataset ──────────────────────────────────────
        df = pd.read_csv(csv_path)
        print(f"  Rows     : {len(df):,}")
        print(f"  Features : {len(self.feature_cols)}")

        X = df[self.feature_cols].values.astype(np.float32)
        y = df[LABEL_COL].values.astype(int)

        # ── Train / test split ────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = RF_TEST_SPLIT,
            random_state = 42,
            stratify     = y        # preserve class balance
        )
        print(f"\n  Train set : {len(X_train):,} samples")
        print(f"  Test  set : {len(X_test):,}  samples")

        # ── Train ─────────────────────────────────────────────
        print(f"\n  Training Random Forest...")
        print(f"    n_estimators = {RF_N_ESTIMATORS}")
        print(f"    max_depth    = {RF_MAX_DEPTH}")
        print(f"    n_features   = {len(self.feature_cols)}")

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # ── Evaluate ──────────────────────────────────────────
        y_pred_train = self.model.predict(X_train)
        y_pred_test  = self.model.predict(X_test)

        self.train_acc = accuracy_score(y_train, y_pred_train)
        self.test_acc  = accuracy_score(y_test,  y_pred_test)

        # 5-fold cross validation on full dataset
        self.cv_scores = cross_val_score(
            self.model, X, y, cv=5, scoring="accuracy"
        )

        # ── Print results ─────────────────────────────────────
        self._print_results(y_test, y_pred_test)

        # ── Save model ────────────────────────────────────────
        self.save(RF_MODEL_PATH)

        return {
            "train_acc": self.train_acc,
            "test_acc" : self.test_acc,
            "cv_mean"  : self.cv_scores.mean(),
            "cv_std"   : self.cv_scores.std(),
        }

    # ══════════════════════════════════════════════════════════
    # predict()
    # ══════════════════════════════════════════════════════════

    def predict(self, task) -> int:
        """
        Predict task type from a Task object.
        Called by LBRO broker in environment.py (Step 6).

        Returns:
            0 = CPU_INTENSIVE
            1 = MEM_INTENSIVE
            2 = BALANCED
        """
        assert self.is_trained, \
            "Model not trained. Call train() or load() first."

        features = self._extract_features(task)
        return int(self.model.predict(features)[0])

    def predict_proba(self, task) -> np.ndarray:
        """
        Returns probability distribution over 3 classes.
        Shape: (3,)  → [P(CPU), P(MEM), P(BAL)]
        """
        assert self.is_trained, \
            "Model not trained. Call train() or load() first."

        features = self._extract_features(task)
        return self.model.predict_proba(features)[0]

    # ══════════════════════════════════════════════════════════
    # save() / load()
    # ══════════════════════════════════════════════════════════

    def save(self, path: str = RF_MODEL_PATH):
        """Save trained model to .pkl file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"\n  ✅ Model saved → {path}  ({size_kb:.1f} KB)")

    @classmethod
    def load(cls, path: str = RF_MODEL_PATH):
        """Load saved model from .pkl file."""
        assert os.path.exists(path), \
            f"Model not found at {path}. Run train() first."
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"  ✅ Model loaded ← {path}")
        return obj

    # ══════════════════════════════════════════════════════════
    # feature_importance()
    # ══════════════════════════════════════════════════════════

    def feature_importance(self) -> pd.DataFrame:
        """Returns sorted feature importance DataFrame."""
        assert self.is_trained
        imp = self.model.feature_importances_
        df  = pd.DataFrame({
            "feature"    : self.feature_cols,
            "importance" : imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df

    # ══════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════

    def _extract_features(self, task) -> np.ndarray:
        """
        Build (1, 11) feature array from a Task object.
        Matches exactly the columns used during training.
        """
        from simulator.config import (
            CPU_DEMAND_MIN, CPU_DEMAND_MAX,
            RAM_DEMAND_MIN, RAM_DEMAND_MAX,
            TASK_SIZE_MIN,  TASK_SIZE_MAX,
        )

        cpu  = task.cpu_demand_mips
        ram  = task.ram_demand_mb
        size = task.size_mbits
        cyc  = task.cpu_cycles
        pri  = task.static_priority

        cpu_n = (cpu  - CPU_DEMAND_MIN) / (CPU_DEMAND_MAX - CPU_DEMAND_MIN)
        ram_n = (ram  - RAM_DEMAND_MIN) / (RAM_DEMAND_MAX - RAM_DEMAND_MIN)
        siz_n = (size - TASK_SIZE_MIN)  / (TASK_SIZE_MAX  - TASK_SIZE_MIN)

        return np.array([[
            cpu,                        # cpu_demand_mips
            ram,                        # ram_demand_mb
            size,                       # size_mbits
            cyc,                        # cpu_cycles
            pri,                        # static_priority
            cpu_n,                      # cpu_norm
            ram_n,                      # ram_norm
            siz_n,                      # size_norm
            cpu / max(ram,  1.0),       # cpu_ram_ratio
            cyc / max(size, 0.001),     # compute_density
            0.6 * cpu_n + 0.4 * ram_n, # load_score
        ]], dtype=np.float32)

    def _print_results(self, y_test, y_pred_test):
        """Print formatted evaluation results."""
        print(f"\n{'─' * 50}")
        print(f"  RESULTS")
        print(f"{'─' * 50}")
        print(f"  Train accuracy : {self.train_acc:.4f}  "
              f"({self.train_acc * 100:.2f}%)")
        print(f"  Test  accuracy : {self.test_acc:.4f}  "
              f"({self.test_acc  * 100:.2f}%)")
        print(f"  CV    accuracy : {self.cv_scores.mean():.4f} "
              f"± {self.cv_scores.std():.4f}")

        print(f"\n  Classification Report:")
        print(classification_report(
            y_test, y_pred_test,
            target_names = self.class_names,
            digits       = 4
        ))

        print(f"  Confusion Matrix:")
        cm     = confusion_matrix(y_test, y_pred_test)
        header = f"{'':15s}" + "".join(
            f"{n:>15s}" for n in self.class_names)
        print(f"  {header}")
        for i, row in enumerate(cm):
            row_str = "".join(f"{v:>15d}" for v in row)
            print(f"  {self.class_names[i]:15s}{row_str}")

        print(f"\n  Feature Importance (top 5):")
        fi = self.feature_importance().head(5)
        for _, r in fi.iterrows():
            bar = "█" * int(r["importance"] * 40)
            print(f"    {int(r['rank']):2d}. {r['feature']:20s} "
                  f"{r['importance']:.4f}  {bar}")


# =============================================================
# MAIN
# =============================================================

def main():
    print("=" * 55)
    print("  DRL-LSTM-LBRO  —  Step 3: RF Classifier")
    print("=" * 55)

    # ── Train ─────────────────────────────────────────────────
    clf     = RFTaskClassifier()
    metrics = clf.train(WORKLOAD_CSV)

    # ── Verify saved model loads correctly ────────────────────
    print(f"\n  Verifying saved model loads correctly...")
    clf2 = RFTaskClassifier.load(RF_MODEL_PATH)
    assert clf2.is_trained, "Loaded model not trained"
    print(f"  ✅ Load verified")

    # ── Quick inference test ──────────────────────────────────
    print(f"\n  Quick inference test (5 tasks):")
    from simulator.task import IoTTaskGenerator
    gen   = IoTTaskGenerator(seed=99)
    tasks = [gen._make_task(i, 0) for i in range(5)]

    correct = 0
    for t in tasks:
        pred  = clf2.predict(t)
        proba = clf2.predict_proba(t)
        true  = t.task_type
        match = "✅" if pred == true else "⚠️ "
        if pred == true:
            correct += 1
        print(f"    {match} cpu={t.cpu_demand_mips:6.0f}  "
              f"ram={t.ram_demand_mb:6.0f}  "
              f"true={CLASS_NAMES[true]:15s}  "
              f"pred={CLASS_NAMES[pred]:15s}  "
              f"conf={max(proba):.3f}")

    print(f"\n  Inference: {correct}/5 correct")
    print(f"\n{'=' * 55}")
    print(f"  Step 3 COMPLETE ✅")
    print(f"  Train acc : {metrics['train_acc'] * 100:.2f}%")
    print(f"  Test  acc : {metrics['test_acc']  * 100:.2f}%")
    print(f"  CV    acc : {metrics['cv_mean']   * 100:.2f}% "
          f"± {metrics['cv_std'] * 100:.2f}%")
    print(f"  Saved  →  {RF_MODEL_PATH}")
    print(f"  Next   →  Step 4: models/lstm_predictor.py")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
