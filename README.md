# CartPole RL Agent
### Reinforcement Learning — Deep Q-Network (DQN) | PyTorch + Gymnasium

A reinforcement learning agent that learns to balance a pole on a moving cart 
using Deep Q-Learning (DQN). The agent starts with zero knowledge and learns 
purely through trial and error.

---

## Demo

https://github.com/user-attachments/assets/3fbe59bd-14f3-4478-a830-b1bef8679753

---

## The Problem

A pole is attached to a cart. The agent can push the cart left or right.
The goal is to keep the pole balanced upright for as long as possible.

- **State:** 4 values — cart position, cart velocity, pole angle, pole angular velocity
- **Actions:** 2 choices — push left or push right
- **Reward:** +1 for every step the pole stays upright
- **Solved:** Average score ≥ 475 over 100 consecutive episodes

---

## How the Agent Learns (DQN)

Deep Q-Network combines Q-Learning with a neural network:

1. **Explore** — take random actions (epsilon-greedy)
2. **Remember** — store experiences in replay memory
3. **Replay** — sample random batches and learn from them
4. **Exploit** — as epsilon decays, use learned policy more

Key components:
- **Policy network** — trained every step via Bellman equation
- **Target network** — updated every 10 episodes to stabilise training
- **Experience replay** — breaks correlation between consecutive samples
- **Reward shaping** — -10 penalty on failure to discourage early termination

---

## Training Progress

| Episode | Avg Score | Status |
|---|---|---|
| 10 | 27.6 | Pure random exploration |
| 100 | 34.4 | Starting to learn |
| 340 | 188.7 | First perfect score (500)! |
| 500 | 263.1 | Consistently improving |
| 780 | 273.0 | Regularly hitting 500 |
| 1000 | 332.6 | Strong stable agent |

---

## Evaluation Results (10 episodes)

| Episode | Score | Result |
|---|---|---|
| 1 | 40 | — |
| 2 | 500 | ✅ Perfect |
| 3 | 500 | ✅ Perfect |
| 4 | 500 | ✅ Perfect |
| 5 | 36 | — |
| 6 | 97 | — |
| 7 | 500 | ✅ Perfect |
| 8 | 500 | ✅ Perfect |
| 9 | 38 | — |
| 10 | 500 | ✅ Perfect |
| **Mean** | **321.1** | **6/10 perfect** |

---

## Model Architecture
Input (4) → Linear(128) → ReLU → Linear(128) → ReLU → Output (2)
- **Algorithm:** Deep Q-Network (DQN)
- **Optimizer:** Adam (lr=0.001)
- **Loss:** MSE (Bellman target)
- **Replay buffer:** 10,000 experiences
- **Batch size:** 64
- **Epsilon:** 1.0 → 0.01 (decay: 0.995)
- **Gradient clipping:** max norm 1.0
- **Target network update:** every 10 episodes

---

## Best Practices Applied
- ✅ Reproducible seeds
- ✅ Separate policy and target networks
- ✅ Experience replay buffer
- ✅ Epsilon-greedy exploration with decay
- ✅ Reward shaping
- ✅ Gradient clipping
- ✅ Early stopping on solve

---

## Project Structure
cartpole-rl-agent/
├── notebooks/
│   ├── 01_train.py       # DQN training loop
│   ├── 02_evaluate.py    # Evaluation + behaviour plots
│   └── 03_render.py      # Live visual rendering
├── models/
│   └── dqn_best.pth      # Best saved model
├── outputs/
│   ├── training_progress.png
│   ├── epsilon_decay.png
│   ├── evaluation_scores.png
│   ├── agent_behaviour.png
│   └── cartpole_agent_demo.mp4
├── requirements.txt
└── README.md

---

## How to Run
```bash
git clone https://github.com/Ruonaobos/cartpole-rl-agent.git
cd cartpole-rl-agent
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

# Train the agent
python notebooks/01_train.py

# Evaluate performance
python notebooks/02_evaluate.py

# Watch the agent live
python notebooks/03_render.py
```

---

## Tools
Python 3.11 | PyTorch | Gymnasium | NumPy | Matplotlib | Pygame

---

*Built as part of the Microsoft ML Learning Pathway — Reinforcement Learning*
