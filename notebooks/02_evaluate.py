import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Reproducibility ───────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────
ENV_NAME    = "CartPole-v1"
STATE_SIZE  = 4
ACTION_SIZE = 2
DEVICE      = torch.device("cpu")

# ── Load model ────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x):
        return self.net(x)

model = DQN(STATE_SIZE, ACTION_SIZE).to(DEVICE)
model.load_state_dict(torch.load("models/dqn_best.pth", weights_only=True))
model.eval()
print("Model loaded.")

# ── Run 10 evaluation episodes ────────────────────────
env    = gym.make(ENV_NAME)
scores = []

print("\nRunning 10 evaluation episodes...")
for ep in range(10):
    state, _ = env.reset(seed=SEED + ep)
    score    = 0
    for _ in range(500):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action  = model(state_t).argmax().item()
        state, _, terminated, truncated, _ = env.step(action)
        score += 1
        if terminated or truncated:
            break
    scores.append(score)
    print(f"  Episode {ep+1}: {score} steps")

env.close()
print(f"\nMean score: {np.mean(scores):.1f}")
print(f"Min score:  {min(scores)}")
print(f"Max score:  {max(scores)}")

# ── Plot evaluation scores ────────────────────────────
plt.figure(figsize=(10, 5))
colors = ["#1D9E75" if s == 500 else "#378ADD" for s in scores]
bars   = plt.bar(range(1, 11), scores, color=colors)
plt.axhline(500, color="#D85A30", linestyle="--", linewidth=1, label="Max score (500)")
plt.axhline(np.mean(scores), color="#7F77DD", linestyle="-.", linewidth=1.5,
            label=f"Mean ({np.mean(scores):.1f})")
plt.title("DQN Agent — Evaluation Performance (10 Episodes)")
plt.xlabel("Episode")
plt.ylabel("Steps Balanced")
plt.ylim(0, 550)
plt.legend()
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(score), ha="center", fontsize=10, fontweight="500")
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("outputs/evaluation_scores.png", dpi=150)
plt.show()
print("Saved: outputs/evaluation_scores.png")

# ── Visualise one episode frame by frame ──────────────
print("\nRecording one episode for visualisation...")
env   = gym.make(ENV_NAME)
state, _ = env.reset(seed=SEED)

cart_positions = []
pole_angles    = []
actions_taken  = []
step_scores    = []

for step in range(500):
    cart_pos, cart_vel, pole_angle, pole_vel = state
    cart_positions.append(cart_pos)
    pole_angles.append(np.degrees(pole_angle))

    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action  = model(state_t).argmax().item()

    actions_taken.append(action)
    step_scores.append(step + 1)
    state, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()
print(f"Episode lasted {len(cart_positions)} steps")

# ── Plot agent behaviour ──────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(cart_positions, color="#378ADD", linewidth=1.5)
axes[0].axhline(0,  color="gray", linestyle="--", linewidth=0.8, label="Center")
axes[0].axhline(2.4,  color="#D85A30", linestyle=":", linewidth=0.8, label="Boundary")
axes[0].axhline(-2.4, color="#D85A30", linestyle=":", linewidth=0.8)
axes[0].set_title("Cart Position Over Time")
axes[0].set_ylabel("Position")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(pole_angles, color="#7F77DD", linewidth=1.5)
axes[1].axhline(0,    color="gray",   linestyle="--", linewidth=0.8, label="Upright")
axes[1].axhline(12,   color="#D85A30", linestyle=":", linewidth=0.8, label="Fail zone")
axes[1].axhline(-12,  color="#D85A30", linestyle=":", linewidth=0.8)
axes[1].set_title("Pole Angle Over Time (degrees)")
axes[1].set_ylabel("Angle (°)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].fill_between(range(len(actions_taken)),
                      actions_taken, alpha=0.6,
                      color="#1D9E75", step="pre")
axes[2].set_title("Actions Taken (0=Left, 1=Right)")
axes[2].set_ylabel("Action")
axes[2].set_xlabel("Step")
axes[2].set_yticks([0, 1])
axes[2].set_yticklabels(["Left", "Right"])
axes[2].grid(True, alpha=0.3)

plt.suptitle(f"DQN Agent Behaviour — {len(cart_positions)} Steps Balanced", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/agent_behaviour.png", dpi=150)
plt.show()
print("Saved: outputs/agent_behaviour.png")

print("\nEvaluation complete!")