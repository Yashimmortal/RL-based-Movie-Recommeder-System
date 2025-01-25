import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from env.recommender_env import RecommenderEnv
from models.dqn_policy import DQNAgent
from models.ppo_policy import PPOAgent
from models.average_tracker import AverageStateTracker
from utils.evaluation import calculate_cumulative_reward, calculate_diversity, calculate_coverage


interaction_matrix = np.load('./data/interaction_matrix.npy', allow_pickle=True)
item_features = np.load('./data/item_features.npy')
user_features = np.load('./data/user_features.npy')


env = RecommenderEnv(interaction_matrix, user_features, item_features)
tracker = AverageStateTracker(interaction_matrix, item_features)


state_dim = item_features.shape[1]
action_dim = interaction_matrix.shape[1]


dqn_agent = DQNAgent(state_dim, action_dim)
ppo_agent = PPOAgent(state_dim, action_dim)


episodes = 10
agents = {"DQN": dqn_agent, "PPO": ppo_agent}
results = {}


for name, agent in agents.items():
    total_rewards = []
    diversity_scores = []
    coverage_scores = []

    for episode in range(episodes):
        state = env.reset()
        state = tracker.get_state(env.current_user)
        rewards = []
        actions = []

        for step in range(env.max_steps):
            action = agent.select_action(state) if name == "DQN" else agent.select_action(state)[0]
            next_state, reward, done, _ = env.step(action)
            next_state = tracker.get_state(env.current_user)

            rewards.append(reward)
            actions.append(action)
            state = next_state

            if done:
                break

        
        total_rewards.append(calculate_cumulative_reward(rewards))
        diversity_scores.append(calculate_diversity(actions, item_features))
        coverage_scores.append(calculate_coverage(actions, item_features.shape[0]))

    results[name] = {
        "Average Reward": np.mean(total_rewards),
        "Diversity": np.mean(diversity_scores),
        "Coverage": np.mean(coverage_scores),
    }


for name, metrics in results.items():
    print(f"Agent: {name}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}")
