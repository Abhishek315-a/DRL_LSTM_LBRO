"""
Generate fig_architecture1.png for DRL-LSTM-LBRO paper.
Publication-quality B&W block diagram.
Updated: RF classifier removed, state dim R^23, arrival prob p=0.2,
         Google Borg trace workload, two-stage pipeline (LSTM + DDQN).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── helpers ───────────────────────────────────────────────────────
def R(x, y, w, h, lw=1.4, ls="-", fc="white"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc,
                            edgecolor="black", linewidth=lw,
                            linestyle=ls, zorder=2))

def Rborder(x, y, w, h, lw=1.2, ls="--"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none",
                            edgecolor="black", linewidth=lw,
                            linestyle=ls, zorder=2))

def T(x, y, text, fs=9.5, bold=False, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs,
            fontweight="bold" if bold else "normal",
            color="black", zorder=3)

def box(x, y, w, h, line1, line2=None, fs=9.0, lw=1.2, fc="white"):
    R(x, y, w, h, lw=lw, fc=fc)
    cx, cy = x + w / 2, y + h / 2
    if line2:
        T(cx, cy + 0.16, line1, fs=fs, bold=True)
        T(cx, cy - 0.18, line2, fs=fs - 1.5)
    else:
        T(cx, cy, line1, fs=fs, bold=True)

def arrow(x1, y1, x2, y2, lbl="", lx=0, ly=0.13, ha="center"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=1.0, mutation_scale=13), zorder=4)
    if lbl:
        T((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, lbl, fs=8.0, ha=ha)

# ══════════════════════════════════════════════════════════════════
#  TOP ROW — group boxes y: 5.6 → 9.2  |  titles at y: 9.50
# ══════════════════════════════════════════════════════════════════

# ── IoT Devices Layer ─────────────────────────────────────────────
T(1.70, 9.50, "IoT Devices Layer", fs=11.0, bold=True)
R(0.20, 5.50, 3.00, 3.60, lw=1.6)

# Task Generator: top at y=8.00 so arrow at y=8.20 clears it
box(0.40, 6.90, 2.60, 1.10,
    "Task Generator",
    "50 devices  |  p = 0.2 / slot")

box(0.40, 5.70, 2.60, 0.95,
    "Workload Parameters",
    "Google Borg Cluster Trace (2019)")

arrow(1.70, 6.90, 1.70, 6.65, "", lx=0, ly=0)

ax.annotate("", xy=(3.60, 8.20), xytext=(3.20, 8.20),
            arrowprops=dict(arrowstyle="-|>", color="black",
                            lw=1.0, mutation_scale=13), zorder=4)
T(2.80, 8.48, "Past CPU / Queue History", fs=8.0, ha="center")

# ── Stage 1 — Prediction Module ──────────────────────────────────
T(6.35, 9.50, "Stage 1 — Prediction Module", fs=11.0, bold=True)
R(3.60, 5.50, 5.50, 3.60, lw=1.6)

box(3.80, 7.75, 5.10, 1.00,
    "GRU-LSTM Load Predictor",
    r"Window W=10  $\to$  5-step forecast per cloudlet")

box(3.80, 5.75, 5.10, 1.30,
    "DDQN Agent  (Double Q-Learning)",
    r"Online + Target MLP  [256$\to$128$\to$4]  |  Replay Buffer 50k")

# GRU-LSTM → DDQN arrow + label
ax.annotate("", xy=(6.35, 7.75), xytext=(6.35, 7.05),
            arrowprops=dict(arrowstyle="-|>", color="black",
                            lw=1.0, mutation_scale=13), zorder=4)
T(6.95, 7.40, r"pred $\hat{u}$", fs=8.5, ha="left")

# ── Load Balancer ─────────────────────────────────────────────────
T(11.05, 9.50, "Load Balancer", fs=11.0, bold=True)
Rborder(9.60, 5.50, 2.90, 3.60, lw=1.4)

for yi, lbl in zip([8.75, 8.00, 7.25, 6.50],
                   [r"$a=0$ :  Cloudlet $C_0$",
                    r"$a=1$ :  Cloudlet $C_1$",
                    r"$a=2$ :  Cloudlet $C_2$",
                    r"$a=3$ :  Cloud  (WAN)"]):
    box(9.80, yi - 0.28, 2.50, 0.52, lbl, fs=8.5)

# DDQN → Load Balancer (centered at mid-height of both group boxes)
arrow(9.10, 7.30, 9.60, 7.30, "")

# ══════════════════════════════════════════════════════════════════
#  BOTTOM ROW — group boxes y: 0.40 → 5.40  |  titles at y: 5.52
# ══════════════════════════════════════════════════════════════════

# ── Stage 2 — System State Encoder ───────────────────────────────
T(1.70, 5.20, "Stage 2 — System State Encoder", fs=11.0, bold=True,
  va="bottom")
R(0.20, 0.20, 3.00, 4.90, lw=1.6)

for yi, lbl in zip([4.45, 3.75, 3.05],
                   [r"$u_{cpu}$", r"$u_{ram}$", r"$q_{util}$"]):
    T(0.58, yi, lbl, fs=9.5, ha="center")
    ax.annotate("", xy=(1.25, yi), xytext=(0.85, yi),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=0.9, mutation_scale=10), zorder=4)

box(1.30, 3.45, 1.70, 1.35, "State\nBuffer", fs=9.0)
arrow(2.15, 3.45, 2.15, 2.70)

# State Vector box (thick border, prominent)
box(0.38, 2.00, 2.65, 0.62,
    r"State Vector   $\mathbf{s} \in \mathbb{R}^{23}$",
    lw=2.0, fs=9.5)
T(1.70, 1.72, r"(18 cloudlet + 2 cloud + 3 task features)", fs=7.5)

box(0.38, 0.60, 1.20, 0.95, "Task\nFeatures", fs=9.0)
box(1.72, 0.60, 1.30, 0.95, "LSTM\nPredictions", fs=9.0)
arrow(2.15, 2.00, 2.15, 1.80)

# State Vector → DDQN (diagonal through gap; label rotated along the arrow)
ax.annotate("", xy=(3.60, 5.90), xytext=(3.20, 2.31),
            arrowprops=dict(arrowstyle="-|>", color="black",
                            lw=1.2, mutation_scale=13), zorder=4)
ax.text(3.22, 4.10, r"$\mathbf{s} \in \mathbb{R}^{23}$",
        ha="left", va="center", fontsize=9.0, fontweight="bold",
        rotation=84, color="black", zorder=5)

# ── Execution Layer ───────────────────────────────────────────────
T(6.85, 5.20, "Execution Layer", fs=11.0, bold=True, va="bottom")
Rborder(3.60, 0.20, 5.80, 4.90, lw=1.4)

# Cloud box — top-right, clearly separate from cloudlets
box(7.50, 4.00, 1.60, 0.80, "Cloud", fs=9.0)

# Cloudlet boxes — spread wide, tall for clarity
box(3.82, 1.20, 1.55, 2.20,
    "Cloudlet $C_0$", r"M/M/4 | Q$\leq$30", fs=9.0)
box(5.62, 1.20, 1.55, 2.20,
    "Cloudlet $C_1$", r"M/M/3 | Q$\leq$25", fs=9.0)
box(7.42, 1.20, 1.55, 2.20,
    "Cloudlet $C_2$", r"M/M/2 | Q$\leq$20", fs=9.0)

# Load Balancer → Execution Layer
arrow(11.05, 5.60, 9.30, 4.35,
      "Assign", lx=0.18, ly=0.15, ha="left")

# ── Execution Feedback — closes the RL loop ───────────────────────
# Step 1: horizontal from Execution Layer right → far-right rail
ax.plot([9.40, 13.80], [0.80, 0.80], color="black", lw=1.0, zorder=4)
# Step 2: vertical UP along right rail — goes ABOVE Load Balancer (top = 9.10)
ax.plot([13.80, 13.80], [0.80, 9.25], color="black", lw=1.0, zorder=4)
T(14.28, 5.00, "Execution\nFeedback\n" r"($r_t$)", fs=8.0, ha="left")
# Step 3: horizontal left ABOVE Load Balancer top
ax.plot([9.35, 13.80], [9.25, 9.25], color="black", lw=1.0, zorder=4)
T(13.00, 9.40, r"$(s,a,r_t,s')\ {\to}\ $Replay Buffer",
  fs=8.0, ha="center")
# Step 4: vertical DOWN at x=9.05 (just right of GRU-LSTM right edge 8.90)
ax.plot([9.35, 9.35], [6.40, 9.25], color="black", lw=1.0, zorder=4)
# Step 5: short arrow LEFT into DDQN right side
ax.annotate("", xy=(8.90, 6.40), xytext=(9.35, 6.40),
            arrowprops=dict(arrowstyle="-|>", color="black",
                            lw=1.2, mutation_scale=13), zorder=5)

# ── Reward formula (bottom right) ────────────────────────────────
T(11.50, 0.42,
  r"$r_t = -W_l\,\bar{\ell} - W_d\,d - W_i\,\sigma(u)"
  r" - W_o\,\max(0,\,u_q{-}0.6)$",
  fs=8.5, ha="center")

# ─────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.4)
plt.savefig("paper/fig_architecture1.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("Saved: paper/fig_architecture1.png")
