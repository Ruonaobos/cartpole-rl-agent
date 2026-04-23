import gymnasium as gym
import torch
import numpy as np

env = gym.make("CartPole-v1")
obs, info = env.reset()
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Initial observation: {obs}")
print("All good!")
env.close()