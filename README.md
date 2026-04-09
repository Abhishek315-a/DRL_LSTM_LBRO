# DRL-LSTM-LBRO

**Dynamic Load Balancing for Resource-Constrained Mobile Edge Computing**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

DRL-LSTM-LBRO is a two-stage intelligent task scheduling framework for heterogeneous IoT edge-cloud environments. A per-cloudlet **LSTM load forecaster** enriches the state observed by a **Double DQN (DDQN) agent**, enabling proactive, queue-aware task placement that reduces task drops and latency versus five published baselines and a PPO baseline — without architectural changes across 3-cloudlet and 6-cloudlet topologies.

---

## Table of Contents

- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [State, Action and Reward](#state-action-and-reward)
- [Baselines Compared](#baselines-compared)
- [Technology Stack](#technology-stack)
- [Quick Start](#quick-start)
- [Step-by-Step Usage](#step-by-step-usage)
- [Scaling Experiment (6 Cloudlets)](#scaling-experiment-6-cloudlets)
- [PPO Baseline](#ppo-baseline)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Research Context](#research-context)
- [Author](#author)

---

## Key Results

Evaluated over **500 episodes (10 seeds × 50 episodes)** on a simulated 3-cloudlet MEC testbed against five published load balancing algorithms and a PPO baseline. Latency is reported over **completed tasks only** (dropped tasks excluded).

| Algorithm | Drop Rate (%) ↓ | Latency (s) ↓ | Throughput ↑ | JFI ↑ |
|---|---|---|---|---|
| **DRL-LSTM-LBRO (ours)** | **0.58 ± 0.21** | **0.260** | **198.8** | 0.876 |
| Ablation: no LSTM (retrained) | 0.61 ± 0.27 | 0.285 | 198.8 | 0.953 |
| PPO [Schulman 2017] | 0.79 ± 0.50 | 0.369 | 198.4 | 0.903 |
| LBRO [Nayyer 2022] | 2.76 ± 1.23 | 0.340 | 194.5 | 0.997 |
| DeepRM [Mao 2016] | 2.00 ± 0.94 | 0.335 | 196.0 | 0.990 |
| DRL-EdgeLB [Liu 2022] | 4.35 ± 1.69 | 0.349 | 191.3 | 0.999 |
| MADRL-MEC [Zhao 2022] | 2.07 ± 1.02 | 0.335 | 195.9 | 0.991 |
| Pred-LB [2025] | 5.24 ± 1.77 | 0.348 | 189.5 | 0.999 |

**6-cloudlet scalability** (3 seeds × 30 episodes): DRL-LSTM-LBRO achieves **0.77% drop rate** and **0.255 s latency** — a 37.4% drop reduction over LBRO — without any architectural changes.

---

## System Architecture

### Decision Pipeline

```
IoT Task Arrival
      │
      ▼
Resource Monitoring  ──────────────────────────────────┐
(CPU, RAM, queue, critical-state per cloudlet)         │
      │                                                │
      ▼                                                │
LSTM Predictors (one per cloudlet)                     │
(forecast CPU & RAM utilisation 5 steps ahead)         │
      │                                                │
      └──────────────► State Vector (23-dim) ◄─────────┘
                              │
                              ▼
                       DDQN Agent (ε-greedy)
                              │
               ┌──────────────┼──────────────┬──────────┐
               ▼              ▼              ▼          ▼
          Cloudlet-0     Cloudlet-1     Cloudlet-2    Cloud
         (4 servers,    (3 servers,    (2 servers,  (fallback)
          16 GB RAM)     8 GB RAM)      4 GB RAM)
               │              │              │          │
               └──────────────┴──────────────┴──────────┘
                              │
                   Latency + Energy + Drop + Imbalance
                              │
                        Multi-objective Reward
                              │
                         Agent Update (DDQN)
```

### Two-Stage Framework

1. **Stage 1 — LSTM Load Forecasting**: A GRU-LSTM model per cloudlet predicts next-step CPU and RAM utilisation from a sliding window of the last 10 time slots. Predictions are refreshed every 5 simulation steps.
2. **Stage 2 — DDQN Scheduling**: A Double DQN with prioritised experience replay observes a 23-dimensional state (current metrics + LSTM forecasts + task features) and selects the optimal placement action. Reward penalises latency, task drops, deadline violations, and load imbalance jointly.

---

## Project Structure

```
DRL_LSTM_LBRO/
│
├── simulator/
│   ├── config.py           # All hyperparameters, paths, and dynamic scaling
│   ├── task.py             # IoT task definition and Google Borg-based generation
│   ├── cloudlet.py         # M/M/c edge node: Erlang-C queuing, EMA load tracking
│   ├── cloud.py            # Cloud fallback with WAN propagation delay
│   └── environment.py      # MDP: state construction, reward, episode metrics
│
├── data/
│   ├── data_generator.py   # Generates lstm_traces.csv from environment simulation
│   └── lstm_traces.csv     # LSTM training traces (auto-generated)
│
├── models/
│   ├── lstm_predictor.py   # GRU-LSTM per-cloudlet load forecaster (train + infer)
│   └── lstm_c*.keras       # Saved LSTM model files (auto-generated)
│
├── agents/
│   ├── ddqn.py             # Double DQN agent: replay buffer, target network, learn()
│   └── ppo.py              # PPO baseline: actor-critic, GAE, clipped surrogate
│
├── results/
│   ├── plot_results.py         # Publication-quality figures from eval outputs
│   ├── training_log.csv        # Per-episode training metrics
│   ├── eval_results.csv        # Full per-episode evaluation results
│   ├── baseline_comparison.csv # Summary comparison table (seed-aggregated)
│   └── fig*.png                # Generated figures
│
├── train.py            # DDQN training loop (supports --no-lstm ablation)
├── train_ppo.py        # PPO training loop
├── evaluate.py         # Evaluation + baseline comparison (10 seeds)
├── scale_experiment.py # End-to-end 6-cloudlet scalability experiment
└── README.md
```

> **Scale experiment artefacts**: models and results for 6 cloudlets are saved in `models_6c/` and `results_6c/` respectively, keeping 3-cloudlet results untouched.

---

## State, Action and Reward

### State Vector (23-dimensional for 3 cloudlets)

| Group | Features | Count |
|---|---|---|
| Per cloudlet (×3) | CPU util, RAM util, queue util, critical flag, LSTM CPU pred, LSTM RAM pred | 18 |
| Cloud | Queue util, normalised WAN latency | 2 |
| Current task | Normalised CPU demand, RAM demand, priority | 3 |
| **Total** | | **23** |

For N cloudlets the state scales to `6·N + 5` dimensions (e.g. 41-dim for 6 cloudlets).

### Action Space

| Action | Destination |
|---|---|
| 0 | Cloudlet-0 (strongest: 4 servers, 16 GB RAM) |
| 1 | Cloudlet-1 (3 servers, 8 GB RAM) |
| 2 | Cloudlet-2 (2 servers, 4 GB RAM) |
| 3 | Cloud (WAN fallback, unlimited capacity) |

For 6 cloudlets: actions 0–5 map to cloudlets, action 6 maps to cloud.

### Multi-Objective Reward

```
R(t) = −( W_lat·D̃ + W_eng·Ẽ + W_drop·P_drop + W_dead·P_dead
          + W_imb·σ_imb + W_ovld·P_ovld )
```

| Term | Symbol | Weight | Description |
|---|---|---|---|
| Latency | D̃ | 1.0 | Normalised end-to-end delay |
| Energy | Ẽ | 0.5 | Normalised compute energy |
| Drop penalty | P_drop | 5.0 | Binary — task was rejected |
| Deadline penalty | P_dead | 3.0 | Task missed deadline |
| Load imbalance | σ_imb | 2.0 | Std-dev of CPU util across cloudlets |
| Overload penalty | P_ovld | 3.0 | Any cloudlet above saturation threshold |

The imbalance and overload penalties prevent the known policy-collapse failure mode (routing >95% to the strongest node).

---

## Baselines Compared

| Algorithm | Type | Key Characteristic |
|---|---|---|
| **LBRO** [Nayyer 2022] | Heuristic | Additive weighted CRI; no ML |
| **DeepRM** [Mao 2016] | DRL | Policy-gradient resource manager; no LSTM |
| **DRL-EdgeLB** [Liu 2022] | DRL | DQN for edge server load balancing |
| **MADRL-MEC** [Zhao 2022] | Multi-agent DRL | Cooperative edge coordination |
| **Pred-LB** [2025] | Predictive heuristic | Activity-prediction-based scheduling |
| **PPO** [Schulman 2017] | DRL | Actor-critic with clipped surrogate (our addition) |

All baselines are reimplemented in the identical M/M/c simulator; no external code is reused.

---

## Technology Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.13 | Core language |
| TensorFlow / Keras | 2.x (Keras 3) | LSTM, DDQN, PPO networks |
| NumPy | latest | Numerical arrays |
| Pandas | latest | Data logging and CSV I/O |
| Matplotlib | latest | Result visualisation |
| tqdm | latest | Progress bars |

---

## Quick Start

```bash
# 1. Install dependencies
pip install tensorflow numpy pandas matplotlib tqdm

# 2. Generate LSTM training data
python3 -m data.data_generator

# 3. Train LSTM load predictors
python3 -m models.lstm_predictor

# 4. Train DDQN agent (1 000 episodes)
python3 train.py

# 5. Evaluate against all baselines (10 seeds × 50 episodes)
python3 evaluate.py

# 6. Plot results
python3 -m results.plot_results
```

---

## Step-by-Step Usage

### Step 1 — Generate LSTM Trace Data

Simulates the environment under random actions to produce load traces for LSTM training.

```bash
python3 -m data.data_generator
```

**Output:** `data/lstm_traces.csv` (~50 k rows, columns: `cloudlet_id`, `cpu_util`, `ram_util`)

---

### Step 2 — Train LSTM Predictors

Trains one GRU-LSTM model per cloudlet on the generated traces.

```bash
python3 -m models.lstm_predictor
```

**Output:** `models/lstm_c0.keras`, `models/lstm_c1.keras`, `models/lstm_c2.keras`

Architecture: `GRU(64) → LSTM(32) → Dense(2)` — predicts [CPU, RAM] 5 steps ahead.

---

### Step 3 — Train DDQN Agent

```bash
python3 train.py                  # default 1 000 episodes
python3 train.py --episodes 500   # custom episode count
python3 train.py --no-lstm        # ablation: train without LSTM input
```

**Outputs:**
- `models/ddqn_best_online.keras` / `ddqn_best_target.keras` — best checkpoint
- `results/training_log.csv` — per-episode reward, drop rate, latency, action distribution

---

### Step 4 — Evaluate

```bash
python3 evaluate.py --episodes 50 --seeds 100 42 7 13 99 21 55 77 33 88
```

Runs DRL-LSTM-LBRO and all baselines for the given seeds and episode count, then writes aggregated results.

**Outputs:**
- `results/eval_results.csv` — full per-episode data
- `results/baseline_comparison.csv` — seed-aggregated summary table

---

### Step 5 — Plot Results

```bash
python3 -m results.plot_results
```

**Outputs in `results/`:**

| File | Content |
|---|---|
| `fig1_training_convergence.png` | Episode reward + 20-ep smoothed curve |
| `fig2_drop_rate.png` | Drop rate comparison bar chart |
| `fig3_baseline_comparison.png` | Multi-metric comparison grid |
| `fig4_action_distribution.png` | Per-cloudlet action distribution |
| `fig5_latency_energy.png` | Latency and energy scatter |
| `fig6_jfi_comparison.png` | Jain's Fairness Index comparison |

---

## Scaling Experiment (6 Cloudlets)

Runs the full pipeline — data generation, LSTM training, DDQN training, evaluation — at 6 cloudlets using the `NUM_CLOUDLETS_OVERRIDE` environment variable. All artefacts are stored in `models_6c/` and `results_6c/` to avoid overwriting 3-cloudlet results.

```bash
python3 scale_experiment.py --episodes 30
```

To skip retraining and run only evaluation on existing 6-cloudlet models:

```bash
NUM_CLOUDLETS_OVERRIDE=6 python3 evaluate.py --episodes 30 --seeds 100 42 7
```

**6-cloudlet results (3 seeds × 30 episodes):**

| Algorithm | Drop (%) | Latency (s) | Throughput |
|---|---|---|---|
| **DRL-LSTM-LBRO** | **0.77 ± 0.53** | **0.255** | **198.5** |
| LBRO | 1.23 ± 0.38 | 0.327 | 197.5 |
| DeepRM | 1.27 ± 0.38 | 0.324 | 197.5 |
| MADRL-MEC | 1.30 ± 0.40 | 0.324 | 197.4 |
| DRL-EdgeLB | 1.38 ± 0.49 | 0.333 | 197.2 |
| Pred-LB | 1.52 ± 0.57 | 0.331 | 197.0 |

---

## PPO Baseline

A Proximal Policy Optimisation (PPO) actor-critic agent is included as a modern DRL baseline. It uses the same environment and state/action space as DDQN.

```bash
# Train PPO (500 episodes)
python3 train_ppo.py

# PPO is included automatically when running evaluate.py
python3 evaluate.py
```

**Result (10 seeds × 50 episodes):** PPO achieves 0.79% drop rate and 0.369 s latency — 36.2% and 41.9% worse than DRL-LSTM-LBRO on those metrics respectively. The DDQN + LSTM formulation benefits from off-policy learning (replay buffer) and temporally-enriched state.

---

## Configuration

All hyperparameters live in `simulator/config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `NUM_CLOUDLETS` | 3 | Number of edge cloudlets (override with `NUM_CLOUDLETS_OVERRIDE` env var) |
| `NUM_ACTIONS` | `NUM_CLOUDLETS + 1` | Placement actions (cloudlets + cloud) |
| `STATE_DIM` | `6·N + 5` | State vector dimension |
| `MAX_STEPS_PER_EP` | 200 | Simulation steps per episode |
| `LSTM_WINDOW` | 10 | LSTM history window (time slots) |
| `LSTM_UNITS` | 64 / 32 | GRU and LSTM hidden units |
| `DDQN_LR` | 1e-3 | DDQN learning rate |
| `GAMMA` | 0.95 | Discount factor |
| `EPSILON_DECAY` | 0.9998 | Exploration decay (slow) |
| `BATCH_SIZE` | 64 | Replay buffer batch size |
| `TARGET_UPDATE_FREQ` | 10 | Target network sync frequency (episodes) |
| `W_IMBALANCE` | 2.0 | Load imbalance penalty weight |
| `W_OVERLOAD` | 3.0 | Overload penalty weight |

**Dynamic scaling**: set `NUM_CLOUDLETS_OVERRIDE=6` (or any value) before running any script to switch to an N-cloudlet topology. Cloudlet capacities, state dimension, and file paths all adjust automatically.

---

## Outputs

### Training (`train.py`)

| File | Description |
|---|---|
| `models/ddqn_best_online.keras` | Online network at best reward checkpoint |
| `models/ddqn_best_target.keras` | Target network at best reward checkpoint |
| `results/training_log.csv` | Episode-level: reward, drop rate, latency, energy, ε, action distribution |

### Evaluation (`evaluate.py`)

| File | Description |
|---|---|
| `results/eval_results.csv` | Full per-episode results for all policies and seeds |
| `results/baseline_comparison.csv` | Seed-aggregated means and std-devs for all metrics |

### Ablation (`train.py --no-lstm`)

| File | Description |
|---|---|
| `models/ddqn_best_no_lstm_online.keras` | Best no-LSTM model |
| `results/training_log_no_lstm.csv` | No-LSTM training log |

---

## Research Context

This project accompanies the paper:

> **"DRL-LSTM-LBRO: Predictive Load-Aware Task Scheduling for IoT Mobile Edge Computing"**  
> Submitted to IEEE Transactions on Vehicular Technology / IEEE Internet of Things Journal.

**Key contributions:**
1. **LSTM-augmented state enrichment** — per-cloudlet CPU/RAM forecasting 5 steps ahead improves training-time policy quality (retrained ablation: 0.61% vs 0.58% drop rate; structural proxy DeepRM: 2.00%).
2. **Multi-objective reward with collapse prevention** — joint penalty for latency, drop, deadline, imbalance, and overload eliminates the known failure mode of routing all tasks to the strongest node.
3. **Scalability** — same architecture scales from 3 to 6 cloudlets without modification.
4. **Google Borg trace grounding** — task CPU/RAM demands sampled from the 2019 Borg cluster trace for realistic workload distributions.

---

## Author

**Abhishek Kumar Singh**  
M.Tech, Computer Science and Engineering  
National Institute of Technology Rourkela  
Supervisor: Prof. Bibhudatta Sahoo