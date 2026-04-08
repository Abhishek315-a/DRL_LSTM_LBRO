# =============================================================
# simulator/config.py
# DRL-LSTM-LBRO: All simulation constants
# Aligned with: AICDQN (Scientific Reports 2026) + your thesis
# =============================================================


# ── IoT Layer ─────────────────────────────────────────────────
NUM_DEVICES          = 50
TASK_ARRIVAL_PROB    = 0.2     # 50 devices × 0.2 = ~10 tasks/slot, ~2000 tasks/episode
TASK_SIZE_MIN        = 2.0          # Mbits
TASK_SIZE_MAX        = 5.0          # Mbits
PROCESSING_DENSITY   = 0.297        # GCycles/Mbit  (AICDQN Eq. 2)
CPU_DEMAND_MIN       = 1000.0       # MIPS
CPU_DEMAND_MAX       = 3000.0       # MIPS
RAM_DEMAND_MIN       = 256.0        # MB
RAM_DEMAND_MAX       = 1280.0       # MB
TASK_DEADLINE_SLOTS  = 20           # slots before drop

TASK_PRIORITY_HIGH       = 1.0
TASK_PRIORITY_LOW        = 0.0
TASK_PRIORITY_HIGH_RATIO = 0.3

TASK_CPU = 0
TASK_MEM = 1
TASK_BAL = 2


# ── Edge Cloudlets ────────────────────────────────────────────
NUM_CLOUDLETS      = 3
CLOUDLET_MIPS      = [8_000, 8_000, 8_000]  # equal MIPS — forces smart routing
CLOUDLET_RAM_MB    = [16384, 8192,  4096]   # different RAM — RF type routing needed
CLOUDLET_SERVERS   = [4,     3,     2]      # C1 gets extra server — useful for spreading
CLOUDLET_MAX_QUEUE = [30,    25,    20]     # tighter caps — C0 overflows at realistic load
EDGE_LAN_MBPS      = 100.0
EDGE_PROP_DELAY_MS = [2.0,   3.0,   4.0]

CPU_CRITICAL_THRESH = 0.85
RAM_CRITICAL_THRESH = 0.85


# ── Cloud Node ────────────────────────────────────────────────
CLOUD_MIPS              = 100_000
CLOUD_RAM_MB            = 1_000_000
CLOUD_MAX_QUEUE         = 10_000
CLOUD_WAN_MBPS          = 2.0
CLOUD_PROP_DELAY_MS_MIN = 200.0
CLOUD_PROP_DELAY_MS_MAX = 300.0


# ── Energy Model ──────────────────────────────────────────────
P_ACTIVE_W = 15.0
P_IDLE_W   = 5.0
P_SLEEP_W  = 0.5
P_TX_W     = 0.5
P_RX_W     = 0.1
KAPPA      = 1e-28

ENERGY_ACTIVE = 0
ENERGY_IDLE   = 1
ENERGY_SLEEP  = 2


# ── MDP Settings ──────────────────────────────────────────────
NUM_ACTIONS      = 4
STATE_DIM        = 23   # 18 cloudlet + 2 cloud + 3 task_demand (RF one-hot removed)
TIME_SLOT_S      = 0.1
MAX_STEPS_PER_EP = 200
NUM_EPISODES     = 1000   # increased from 500


# ── Reward Weights ────────────────────────────────────────────
W_LATENCY    = 2.0    # latency penalty
W_DROP       = 5.0    # drop / deadline miss penalty
W_IMBALANCE  = 2.0    # ← raised from 0.3: forces agent to spread load
W_OVERLOAD   = 3.0    # ← new: per-cloudlet queue saturation penalty (>60% full)
LATENCY_MAX_S      = 3.0
ENERGY_MAX_J       = 5.0
CLOUD_LATENCY_NORM = 0.25


# ── LSTM Settings ─────────────────────────────────────────────
LSTM_WINDOW = 10
LSTM_UNITS  = 64
LSTM_EPOCHS = 50
LSTM_BATCH  = 32


# ── RF Classifier Settings (legacy — RF removed from architecture) ────
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = 10
RF_TEST_SPLIT   = 0.2


# ── Priority Score Weights (AICDQN Eq. 4) ────────────────────
THETA_1 = 0.5
THETA_2 = 0.3
THETA_3 = 0.2


# ── DDQN Hyperparameters ──────────────────────────────────────
DDQN_LR            = 1e-3
DDQN_GAMMA         = 0.99
DDQN_EPSILON       = 1.0
DDQN_EPSILON_MIN   = 0.01
DDQN_EPSILON_DECAY = 0.9998   # ← FIXED: was 0.995 (hit min in 5 eps)
DDQN_BATCH_SIZE    = 64
DDQN_MEMORY_SIZE   = 50_000
DDQN_TARGET_UPDATE = 10
DDQN_HIDDEN_UNITS  = [256, 128]


# ── File Paths ────────────────────────────────────────────────
DATA_DIR          = "data/"
MODEL_DIR         = "models/"
RESULTS_DIR       = "results/"
WORKLOAD_CSV      = "data/workload_traces.csv"
RF_MODEL_PATH     = "models/rf_model.pkl"
LSTM_MODEL_PATH   = "models/lstm_c{}.keras"
GOOGLE_TRACE_PATH = "data/google_cluster_2019/borg_traces_data.csv"
