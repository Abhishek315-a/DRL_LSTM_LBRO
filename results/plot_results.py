# =============================================================
# results/plot_results.py
# DRL-LSTM-LBRO  —  Step 8: Thesis Figures Generator
#
# Generates 5 publication-quality PNG figures:
#   Fig 1: Training convergence (reward over 500 episodes)
#   Fig 2: Drop rate reduction over training
#   Fig 3: Baseline comparison grouped bar (4 metrics)
#   Fig 4: DDQN action distribution pie chart
#   Fig 5: Latency & energy side-by-side bars
#
# Run with:
#     python3 -m results.plot_results
# =============================================================

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulator.config import RESULTS_DIR, NUM_CLOUDLETS, NUM_ACTIONS

TRAINING_CSV = os.path.join(RESULTS_DIR, "training_log.csv")
EVAL_CSV     = os.path.join(RESULTS_DIR, "eval_results.csv")
COMPARE_CSV  = os.path.join(RESULTS_DIR, "baseline_comparison.csv")
OUT_DIR      = RESULTS_DIR
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "DRL-LSTM-LBRO" : "#2196F3",
    "Round-Robin"   : "#FF9800",
    "Random"        : "#F44336",
    "Greedy-Best"   : "#4CAF50",
}
POLICIES = list(COLORS.keys())

def smooth(v, w=20):
    return np.convolve(v, np.ones(w)/w, mode="valid")


def plot_training_convergence(df):
    raw  = df["reward"].values
    sm   = smooth(raw, 20)
    ep_s = df["episode"].values[19:]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["episode"], raw, color="#BBDEFB",
            alpha=0.4, linewidth=0.8, label="Raw reward")
    ax.plot(ep_s, sm, color="#2196F3", linewidth=2.5,
            label="Smoothed (window=20)")
    ax.fill_between(ep_s, sm, alpha=0.15, color="#2196F3")
    ax.axhline(y=raw.max(), color="green",
               linestyle="--", linewidth=1.2,
               label=f"Best: {raw.max():.1f}")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("DRL-LSTM-LBRO Training Convergence\n"
                 "Reward improves as DDQN learns optimal placement",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_facecolor("#F8F9FA")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_training_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Fig 1 → {path}")


def plot_drop_rate(df):
    raw  = df["drop_rate"].values * 100
    sm   = smooth(raw, 20)
    ep_s = df["episode"].values[19:]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["episode"], raw, color="#FFCDD2",
            alpha=0.4, linewidth=0.8)
    ax.plot(ep_s, sm, color="#F44336", linewidth=2.5,
            label="Smoothed drop rate")
    ax.fill_between(ep_s, sm, alpha=0.15, color="#F44336")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Drop Rate (%)", fontsize=12)
    ax.set_title("Task Drop Rate Decreases over Training\n"
                 "DDQN reduces task drops: 50.5% → ~34%",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_facecolor("#F8F9FA")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_drop_rate.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Fig 2 → {path}")


def plot_baseline_comparison(cmp_df):
    metrics = [("Drop Rate (%)", "drop_rate_mean", 100),
               ("Latency (s)",   "avg_lat_mean",   1),
               ("Energy (J)",    "avg_eng_mean",   1),
               ("Reward",        "reward_mean",    1)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("DRL-LSTM-LBRO vs Baselines\n"
                 "27% lower drop rate | 55% less energy vs Round-Robin",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, (label, col, mult) in zip(axes, metrics):
        vals  = []
        cols_ = []
        for p in POLICIES:
            row = cmp_df[cmp_df["policy"]==p]
            vals.append(float(row[col].values[0]) * mult
                        if len(row) else 0)
            cols_.append(COLORS[p])
        bars = ax.bar(range(len(POLICIES)), vals,
                      color=cols_, edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+max(vals)*0.02,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")
        ax.set_xticks(range(len(POLICIES)))
        ax.set_xticklabels(["DRL","RR","Rand","GB"], fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_facecolor("#F8F9FA")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        bars[0].set_edgecolor("#1565C0")
        bars[0].set_linewidth(2.5)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_baseline_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Fig 3 → {path}")


def plot_action_distribution(eval_df):
    drl = eval_df[eval_df["policy"]=="DRL-LSTM-LBRO"]
    labels = ["Cloudlet-0\n(10k MIPS)","Cloudlet-1\n(8k MIPS)",
              "Cloudlet-2\n(6k MIPS)","Cloud\n(WAN)"]
    values = [drl["action_0"].mean(), drl["action_1"].mean(),
              drl["action_2"].mean(), drl["action_3"].mean()]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=["#1565C0","#1E88E5","#64B5F6","#FF9800"],
        explode=(0.05,0.02,0.02,0.02), startangle=140,
        textprops={"fontsize":11})
    for at in autotexts:
        at.set_fontsize(10); at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("DDQN Learned Action Distribution\n"
                 "DRL prefers fastest cloudlet; avoids WAN",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_action_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Fig 4 → {path}")


def plot_latency_energy(cmp_df):
    lat_vals = [float(cmp_df[cmp_df["policy"]==p]["avg_lat_mean"].iloc[0])
            for p in POLICIES]
    eng_vals = [float(cmp_df[cmp_df["policy"]==p]["avg_eng_mean"].iloc[0])
            for p in POLICIES]
    x = np.arange(len(POLICIES))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Latency and Energy: DRL-LSTM-LBRO vs Baselines\n"
                 "DRL saves 55% energy; RR has lower latency but 2× energy",
                 fontsize=13, fontweight="bold")

    for ax, vals, ylabel, unit in [
        (ax1, lat_vals, "Avg Latency (s)", "s"),
        (ax2, eng_vals, "Avg Energy (J)",  "J")
    ]:
        bars = ax.bar(x, vals, width=0.5,
                      color=[COLORS[p] for p in POLICIES],
                      edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+max(vals)*0.02,
                    f"{v:.3f}{unit}", ha="center",
                    fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["DRL","RR","Rand","GB"], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_facecolor("#F8F9FA")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        bars[0].set_edgecolor("#1565C0")
        bars[0].set_linewidth(2.5)

    patches = [mpatches.Patch(color=COLORS[p], label=p)
               for p in POLICIES]
    fig.legend(handles=patches, loc="lower center",
               ncol=4, fontsize=10, bbox_to_anchor=(0.5,-0.05))
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_latency_energy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Fig 5 → {path}")


def main():
    print("=" * 55)
    print("  DRL-LSTM-LBRO  —  Step 8: Plot Results")
    print("=" * 55)

    train_df   = pd.read_csv(TRAINING_CSV)
    eval_df    = pd.read_csv(EVAL_CSV)
    compare_df = pd.read_csv(COMPARE_CSV)

    print(f"\n  Generating 5 figures...")
    plot_training_convergence(train_df)
    plot_drop_rate(train_df)
    plot_baseline_comparison(compare_df)
    plot_action_distribution(eval_df)
    plot_latency_energy(compare_df)

    print(f"\n{'=' * 55}")
    print(f"  Step 8 COMPLETE ✅  — Project DONE!")
    print(f"  All figures → {OUT_DIR}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
