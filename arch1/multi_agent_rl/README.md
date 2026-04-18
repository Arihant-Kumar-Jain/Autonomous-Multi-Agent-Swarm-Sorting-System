# 🤖 Multi-Agent Collaborative Reinforcement Learning

A production-quality Pygame simulation where **3 agents** learn (via tabular Q-learning) to collaboratively collect **10 balls** on a 2D grid world — as efficiently as possible.

---

## 📁 Project Structure

```
multi_agent_rl/
├── main.py              # Entry point — simulation loop + Pygame rendering
├── config.py            # ALL tunable parameters (grid, rewards, RL, display)
├── env/
│   └── environment.py   # Grid, ball placement, collision resolution, rewards
├── agents/
│   └── agent.py         # Agent logic: RL + rule-based, stuck detection
├── rl/
│   └── rl_model.py      # Tabular Q-learning (QLearner class)
├── utils/
│   └── utils.py         # BFS, state encoding, distance helpers, CSV logger
├── models/              # Auto-saved Q-tables (agent_0.pkl, etc.)
├── logs/
│   └── training_log.csv # Episode stats
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
python main.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `T` | Toggle RL ↔ Rule-based mode |
| `S` | Save Q-tables to `models/` |
| `L` | Load Q-tables from `models/` |
| `Q` | Quit (auto-saves) |

---

## 🧠 How It Works

### Environment
- 20×20 grid (configurable in `config.py`)
- 10 gold balls scattered randomly each episode
- 10 static obstacles
- 3 agents start at random non-overlapping positions

### Agents
Each agent:
1. Observes its local window + global ball positions
2. Encodes a compact state tuple
3. Picks an action via **epsilon-greedy Q-learning**
4. Receives a shaped reward
5. Updates its own Q-table

### Reward Function
| Signal | Value | Why |
|--------|-------|-----|
| Ball collected | +20 | Core task objective |
| All balls cleared | +50 | Episode completion bonus |
| Per step | −0.5 | Encourages speed |
| Collision | −5 | Avoid blocking each other |
| Anti-clustering | −3 | Spread out, don't chase same ball |
| Distance shaping | ±2×Δdist | Dense signal for early learning |
| Revisit penalty | −1 | Discourage oscillation |

### Collision Resolution
- No two agents may share a cell
- Lower-index agent has priority
- Blocked agent falls back to current position

### Stuck Detection
- If an agent hasn't moved more than 1 cell in the last 15 steps → forced random action

---

## 🔀 Modes

| Mode | Description |
|------|-------------|
| **RL** | Tabular Q-learning with epsilon-greedy exploration |
| **Rule** | Greedy BFS to nearest ball (no learning) |

Press **T** to switch modes mid-run and compare performance.

---

## ⚙️ Key Config Options (`config.py`)

```python
GRID_ROWS / GRID_COLS = 20     # World size
NUM_BALLS             = 10     # Balls per episode
NUM_OBSTACLES         = 10     # Static obstacles
MAX_STEPS             = 500    # Steps before forced reset
FPS                   = 15     # Rendering speed
ALPHA                 = 0.15   # Q-learning rate
GAMMA                 = 0.95   # Discount factor
EPSILON_DECAY         = 0.997  # Exploration decay
SAVE_EVERY            = 50     # Auto-save interval
```

---

## 📊 Training Progress

After ~200 episodes in RL mode, you should observe:
- Agents spreading out instead of clustering
- Faster completion times
- Fewer collisions
- Epsilon approaching minimum (0.05)

Check `logs/training_log.csv` for episode-by-episode stats.

---

## 🏗️ Architecture Highlights

- **Centralised environment, decentralised learning** — agents share world state but each has its own Q-table
- **Persistent Q-tables** — learning never resets across episodes
- **Compact state encoding** — coarse position bins + directional signs = manageable Q-table size
- **Modular codebase** — each concern is isolated; swap Q-learning for DQN by replacing `rl_model.py` only
