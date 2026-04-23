import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# ── Reproducibility ───────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────
ENV_NAME     = "CartPole-v1"
EPISODES     = 1000
GAMMA        = 0.99      # discount factor
LR           = 0.001
BATCH_SIZE   = 64
MEMORY_SIZE  = 10000
EPSILON_START= 1.0
EPSILON_END  = 0.01
EPSILON_DECAY= 0.995
TARGET_UPDATE= 10        # update target network every N episodes
SOLVE_SCORE  = 475       # considered solved at this average score
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Environment ───────────────────────────────────────
env = gym.make(ENV_NAME)
env.action_space.seed(SEED)

STATE_SIZE  = env.observation_space.shape[0]  # 4
ACTION_SIZE = env.action_space.n              # 2
print(f"State size: {STATE_SIZE} | Action size: {ACTION_SIZE}")

# ── Deep Q-Network ────────────────────────────────────
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

# Online network (trained every step)
# Target network (updated every N episodes — stabilises training)
policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(DEVICE)
target_net = DQN(STATE_SIZE, ACTION_SIZE).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
criterion = nn.MSELoss()

# ── Replay Memory ─────────────────────────────────────
memory = deque(maxlen=MEMORY_SIZE)

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def replay():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states      = torch.FloatTensor(np.array(states)).to(DEVICE)
    actions     = torch.LongTensor(actions).to(DEVICE)
    rewards     = torch.FloatTensor(rewards).to(DEVICE)
    next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
    dones       = torch.FloatTensor(dones).to(DEVICE)

    # Current Q values
    current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q values (Bellman equation)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0]
        target_q   = rewards + GAMMA * max_next_q * (1 - dones)

    loss = criterion(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

# ── Epsilon-greedy action selection ───────────────────
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # explore
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        return policy_net(state_t).argmax().item()  # exploit

# ── Training loop ─────────────────────────────────────
scores        = []
avg_scores    = []
epsilon       = EPSILON_START
best_avg      = 0
solved        = False

print(f"\nTraining DQN on {ENV_NAME} for {EPISODES} episodes...")
print(f"{'Episode':>8} | {'Score':>6} | {'Avg(100)':>9} | {'Epsilon':>8}")
print("-" * 45)

for episode in range(1, EPISODES + 1):
    state, _ = env.reset()
    score    = 0

    for step in range(500):
        action               = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done                 = terminated or truncated

        # Shape reward — penalise failure
        reward = reward if not terminated else -10

        remember(state, action, reward, next_state, done)
        replay()

        state  = next_state
        score += 1
        if done:
            break

    scores.append(score)
    avg_score = np.mean(scores[-100:])
    avg_scores.append(avg_score)

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Update target network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Log every 10 episodes
    if episode % 10 == 0:
        print(f"{episode:>8} | {score:>6} | {avg_score:>9.1f} | {epsilon:>8.4f}")

    # Save best model
    if avg_score > best_avg:
        best_avg = avg_score
        torch.save(policy_net.state_dict(), "models/dqn_best.pth")

    # Check if solved
    if avg_score >= SOLVE_SCORE and not solved:
        print(f"\n✓ Solved at episode {episode} with avg score {avg_score:.1f}!")
        solved = True
        break

env.close()
print(f"\nTraining complete. Best avg score: {best_avg:.1f}")
print(f"Model saved to models/dqn_best.pth")

# ── Plot 1: Score per episode ─────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(scores,     alpha=0.4, color="#378ADD", label="Episode Score")
plt.plot(avg_scores, color="#D85A30", linewidth=2, label="100-episode Average")
plt.axhline(SOLVE_SCORE, color="#1D9E75", linestyle="--", linewidth=1, label=f"Solved ({SOLVE_SCORE})")
plt.title("DQN Agent — CartPole Training Progress")
plt.xlabel("Episode")
plt.ylabel("Score (steps balanced)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/training_progress.png", dpi=150)
plt.show()
print("Saved: outputs/training_progress.png")

# ── Plot 2: Epsilon decay ─────────────────────────────
epsilons = [max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** i)) for i in range(len(scores))]
plt.figure(figsize=(10, 4))
plt.plot(epsilons, color="#7F77DD", linewidth=2)
plt.title("Exploration Rate (Epsilon) Decay")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/epsilon_decay.png", dpi=150)
plt.show()
print("Saved: outputs/epsilon_decay.png")

print("\nAll done!")