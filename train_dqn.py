
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.recommender_env import RecommenderEnv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# Hyperparameters
EPISODES = 500
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON_START = 1.0 
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
TARGET_UPDATE = 10

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Train the DQN Agent
def train_dqn():
    # Load environment data
    interaction_matrix = np.load('data/interaction_matrix.npy', allow_pickle=True).item()
    user_features = np.load('data/user_features.npy')
    item_features = np.load('data/item_features.npy')

    # Initialize environment and DQN
    env = RecommenderEnv(interaction_matrix, user_features, item_features)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim).float()
    target_net = DQN(input_dim, output_dim).float()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = deque(maxlen=10000)
    epsilon = EPSILON_START

    def select_action(state):
        if random.random() < epsilon:
            return random.randint(0, output_dim - 1)  # Random action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return policy_net(state_tensor).argmax(dim=1).item()  # Greedy action

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        batch = random.sample(memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = policy_net(states).gather(1, actions).squeeze()
        next_q_values = target_net(next_states).max(1)[0]
        expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Training loop
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            optimize_model()

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    # Save the trained model
    os.makedirs('models', exist_ok=True)
    torch.save(policy_net.state_dict(), 'models/dqn_policy.pth')
    print("Model saved as models/dqn_policy.pth")

if __name__ == "__main__":
    train_dqn()
