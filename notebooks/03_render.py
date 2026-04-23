import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

# ── Config ────────────────────────────────────────────
SEED        = 42
STATE_SIZE  = 4
ACTION_SIZE = 2
EPISODES    = 10  # record 10 episodes
DEVICE      = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
print("Model loaded. Starting live render...")
print("Close the window after each episode to proceed to the next.\n")

# ── Render loop ───────────────────────────────────────
env = gym.make("CartPole-v1", render_mode="human")

for ep in range(EPISODES):
    state, _ = env.reset(seed=SEED + ep)
    score    = 0
    print(f"Episode {ep+1}/{EPISODES} — watch the window!")

    for _ in range(500):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action  = model(state_t).argmax().item()
        state, _, terminated, truncated, _ = env.step(action)
        score += 1
        if terminated or truncated:
            break

    print(f"  Score: {score} steps\n")

env.close()
print("Render complete!")