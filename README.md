# DRL-LSTM-LBRO  
## Dynamic Load Balancing in Resource-Constrained Edge Computing Infrastructure

DRL-LSTM-LBRO is a broker-based intelligent task scheduling framework for heterogeneous edge-cloud environments. It extends the classical LBRO idea with three learning components:

- **Random Forest (RF)** for task-type classification.
- **LSTM** for short-term resource/load prediction.
- **DDQN** for adaptive task placement and load balancing.

The goal is to improve task placement decisions across edge cloudlets and the cloud while reducing latency, energy consumption, task drops, and load imbalance.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Core Idea](#core-idea)
- [System Flow](#system-flow)
- [Project Structure](#project-structure)
- [State Action and Reward](#state-action-and-reward)
- [Technology Stack](#technology-stack)
- [How to Run](#how-to-run)
- [Outputs Generated](#outputs-generated)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Research Context](#research-context)
- [Author](#author)

---

## Overview

In edge computing, incoming IoT tasks must be assigned to limited edge resources quickly and intelligently. Static methods such as Round-Robin or Random assignment ignore node heterogeneity, current congestion, and future load trends.

This project builds a **learning-enabled broker** that observes the system state and decides where each task should execute:

- **Cloudlet-0**
- **Cloudlet-1**
- **Cloudlet-2**
- **Cloud** (fallback option)

The broker combines:
1. **Task awareness** using RF classification.
2. **Future load awareness** using LSTM prediction.
3. **Decision learning** using DDQN reinforcement learning.

---

## Problem Statement

Edge nodes have limited CPU, RAM, queue capacity, and energy. If tasks are placed poorly, the system suffers from:

- Higher end-to-end latency
- More dropped tasks
- Node saturation and instability
- Poor energy efficiency
- Uneven load across cloudlets

The project models the placement problem as a **Markov Decision Process (MDP)** and trains an agent to learn better scheduling policies than heuristic baselines.

---

## Core Idea

The central idea is simple:

- A new task arrives from an IoT device.
- The broker first understands the task type.
- It then estimates near-future load on each cloudlet.
- Using current state + predicted state, the DDQN agent chooses the best destination.
- The environment returns latency, energy, drop status, and reward.
- The agent improves over time.

This turns a reactive load balancer into a **predictive and adaptive scheduler**.

---

## System Flow

### End-to-End Decision Pipeline

```text
IoT Task Arrival
      ↓
Random Forest Classifier
      ↓
Task type = CPU / MEM / BALANCED
      ↓
Resource Monitoring
(current cpu, ram, queue, critical state)
      ↓
LSTM Predictor
(predicted cpu and ram for each cloudlet)
      ↓
State Vector Construction
      ↓
DDQN Agent selects action
      ↓
Action ∈ {Cloudlet-0, Cloudlet-1, Cloudlet-2, Cloud}
      ↓
Task Execution
      ↓
Latency + Energy + Drop + Imbalance measured
      ↓
Reward computed
      ↓
Agent learns and updates policy

** Conceptual Architecture

            +----------------------+
            |     IoT Devices      |
            |  generate tasks      |
            +----------+-----------+
                       |
                       v
        +-------------------------------+
        |      Enhanced LBRO Broker     |
        |-------------------------------|
        | 1. RF task classifier         |
        | 2. LSTM load predictor        |
        | 3. DDQN decision agent        |
        +------+-----------+------------+
                |           | 
                |           +------------------+
                |                              |
                v                              v
+--------------+  +--------------+  +--------------+
| Cloudlet-0   |  | Cloudlet-1   |  | Cloudlet-2   |
| Edge node    |  | Edge node    |  | Edge node    |
+--------------+  +--------------+  +--------------+
           \             |             /
            \            |            /
             \           |           /
              v          v          v
                  +-------------+
                  |    Cloud    |
                  |  Fallback   |
                  +-------------+

** Project Structure

DRL_LSTM_LBRO/
│
├── simulator/
│   ├── __init__.py
│   ├── config.py
│   ├── task.py
│   ├── cloudlet.py
│   ├── cloud.py
│   └── environment.py
│
├── data/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── workload_traces.csv
│   └── lstm_traces.csv
│
├── models/
│   ├── __init__.py
│   ├── rf_classifier.py
│   ├── lstm_predictor.py
│   ├── rf_model.pkl
│   └── lstm_c*.keras
│
├── agents/
│   ├── __init__.py
│   └── ddqn.py
│
├── results/
│   ├── __init__.py
│   ├── plot_results.py
│   ├── training_log.csv
│   ├── eval_results.csv
│   ├── baseline_comparison.csv
│   └── fig*.png
│
├── train.py
├── evaluate.py
└── README.md
```

# Module Description

## simulator/config.py
Contains all global constants and hyperparameters:
- cloudlet capacities
- queue limits
- task ranges
- reward weights
- RL hyperparameters
- file paths

---

## simulator/task.py
Defines task objects and task generation logic.

A task includes:
- task size
- CPU demand
- RAM demand
- task type
- priority
- arrival slot
- deadline slot

---

## simulator/cloudlet.py
Implements edge cloudlet behavior:
- heterogeneous node resources
- M/M/c queueing model
- Erlang-C waiting delay
- energy-aware state model
- EMA-based load tracking for stable LSTM traces

---

## simulator/cloud.py
Implements remote cloud execution:
- very high compute capacity
- WAN transmission delay
- fallback destination under edge congestion

---

## simulator/environment.py
The broker MDP environment:
- builds the 23-dimensional state
- executes selected action
- computes reward
- tracks episode-level metrics
- exposes Gym-like `reset()` and `step()` workflow

---

## data/data_generator.py
Generates:
- `workload_traces.csv` for RF training
- `lstm_traces.csv` for LSTM training

---

## models/rf_classifier.py
Trains a Random Forest classifier to identify task type:
- CPU-intensive
- MEM-intensive
- Balanced

---

## models/lstm_predictor.py
Trains per-cloudlet LSTM models to predict next-step:
- CPU utilization
- RAM utilization

---

## agents/ddqn.py
Implements the Double DQN agent:
- replay memory
- action selection
- target network update
- Q-learning loss

---

## train.py
Runs training episodes and learns the task placement policy.

---

## evaluate.py
Compares the trained DRL policy against baselines such as:
- Round-Robin
- Random
- Greedy-Best

---

## results/plot_results.py
Generates publication/thesis figures from training and evaluation outputs.

# State, Action and Reward

## State
The broker observes a **23-dimensional state vector**.

### Per Cloudlet Features
For each of the 3 cloudlets:
- CPU utilization
- RAM utilization
- Queue utilization
- Critical-state flag
- Predicted CPU utilization from LSTM
- Predicted RAM utilization from LSTM

**Total:**  
3 cloudlets × 6 features = **18**

---

### Cloud Features
- Cloud queue utilization
- Normalized WAN latency  

**Total:** 2 features

---

### Current Task Features
- Normalized CPU demand
- Normalized RAM demand
- Normalized priority score  

**Total:** 3 features

---

### Final State Size
**Total = 18 + 2 + 3 = 23**

---

## Action
The DDQN agent chooses one of 4 actions:
- 0 → Cloudlet-0  
- 1 → Cloudlet-1  
- 2 → Cloudlet-2  
- 3 → Cloud  

---

## Reward
The reward is designed to penalize poor scheduling decisions.

```text
R(t) = −( w_lat·D̃ + w_eng·Ẽ + w_drop·P + w_imb·σ )
```
Where:
- D̃ = normalized latency
- Ẽ = normalized energy
- P = drop/deadline penalty
- σ = imbalance penalty across cloudlets

This encourages both efficient offloading and load balancing.

# Technology Stack:
- Python
- NumPy
- Pandas
- scikit-learn
- TensorFlow / Keras
- Matplotlib
- tqdm

# How to Run
## 1. Create Package Init Files:
```text
- touch data/__init__.py
- touch simulator/__init__.py
- touch models/__init__.py
- touch agents/__init__.py
- touch results/__init__.py
```
## 2. Generate Data
```
- python3 -m data.data_generator
```

## Outputs:

- data/workload_traces.csv
- data/lstm_traces.csv
## 3. Train Random Forest
```
python3 -m models.rf_classifier
```
## Output:

- RF model saved in models/
## 4. Train LSTM
```
python3 -m models.lstm_predictor
```
## Outputs:

- LSTM models saved in models/
## 5. Train DDQN Agent
```
python3 train.py
```
## Outputs:

- DDQN model
- results/training_log.csv
## 6. Evaluate
```
python3 evaluate.py
```
## Outputs:

- results/eval_results.csv
- results/baseline_comparison.csv
## 7. Plot Results
```
python3 -m results.plot_results
```
## Outputs:

- fig1_training_convergence.png
- fig2_drop_rate.png
- fig3_baseline_comparison.png
- fig4_action_distribution.png
- fig5_latency_energy.png

## Full Run Sequence
```
touch data/__init__.py
touch simulator/__init__.py
touch models/__init__.py
touch agents/__init__.py
touch results/__init__.py

python3 -m data.data_generator
python3 -m models.rf_classifier
python3 -m models.lstm_predictor
python3 train.py
python3 evaluate.py
python3 -m results.plot_results
```

## Author
```
Abhishek Kumar Singh
M.Tech, Computer Science and Engineering
NIT Rourkela
Supervisor: Prof. Bibhudatta Sahoo
```