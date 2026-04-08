# =============================================================
# scale_experiment.py
# DRL-LSTM-LBRO  —  Scalability Experiment (6 cloudlets)
#
# Trains LSTM predictors + DDQN and evaluates DRL-LSTM-LBRO
# and all heuristic baselines at 6 cloudlets (vs 3 default).
#
# Models/results saved to models_6c/ and results_6c/ to avoid
# overwriting the 3-cloudlet production models.
#
# Run with:
#     python3 scale_experiment.py
#     python3 scale_experiment.py --skip-train   # eval only
# =============================================================

import os
import sys
import subprocess
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

ENV6 = {**os.environ, "NUM_CLOUDLETS_OVERRIDE": "6"}


def run(cmd, **kwargs):
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, env=ENV6, **kwargs)
    if result.returncode != 0:
        print(f"  ❌ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; run evaluation only")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Eval episodes per seed (default 50)")
    args = parser.parse_args()

    print("=" * 60)
    print("  DRL-LSTM-LBRO  —  Scalability Experiment (6 Cloudlets)")
    print("=" * 60)

    os.makedirs(os.path.join(ROOT, "models_6c"),  exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results_6c"), exist_ok=True)

    if not args.skip_train:
        print("\n[1/4] Generating LSTM trace data for 6 cloudlets...")
        run([sys.executable, "-m", "data.data_generator"])

        print("\n[2/4] Training LSTM predictors for 6 cloudlets...")
        run([sys.executable, "-m", "models.lstm_predictor"])

        print("\n[3/4] Training DDQN for 6 cloudlets (500 episodes)...")
        run([sys.executable, "train.py", "--episodes", "500"])
    else:
        print("\n  ⏭  Skipping training (--skip-train)")

    print(f"\n[4/4] Evaluating all policies at 6 cloudlets "
          f"({args.episodes} eps × 3 seeds)...")
    run([sys.executable, "evaluate.py",
         "--episodes", str(args.episodes),
         "--seeds", "100", "42", "7"])

    csv = os.path.join(ROOT, "results_6c", "baseline_comparison.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        print("\n" + "─" * 60)
        print("  SCALABILITY RESULTS  (6 cloudlets)")
        print("─" * 60)
        cols = ["policy", "drop_rate_mean", "drop_rate_std",
                "avg_lat_mean", "throughput_mean", "jfi_mean"]
        print(df[cols].to_string(index=False))
        print("─" * 60)
    else:
        print("  ⚠️  Results CSV not found — check for errors above")


if __name__ == "__main__":
    main()
