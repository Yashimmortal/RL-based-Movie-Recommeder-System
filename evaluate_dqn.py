import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from env.recommender_env import RecommenderEnv
from train_dqn import DQN  # Import the DQN model architecture

def evaluate_dqn(model_path, num_episodes=100, max_steps=100):
    """
    Evaluates a trained DQN model on the Recommender Environment.
    
    Args:
        model_path (str): Path to the saved DQN model.
        num_episodes (int): Number of episodes to run for evaluation.
        max_steps (int): Maximum number of steps per episode to avoid infinite loops.
    
    Returns:
        None
    """
    # Load preprocessed data
    interaction_matrix = np.load('data/interaction_matrix.npy', allow_pickle=True).item()
    user_features = np.load('data/user_features.npy')
    item_features = np.load('data/item_features.npy')
    
    # Initialize environment
    env = RecommenderEnv(interaction_matrix, user_features, item_features)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Rebuild the DQN model architecture
    model = DQN(input_dim, output_dim).float()
    
    # Load the saved model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    total_rewards = []

    # Evaluate the model
    for episode in range(num_episodes):
        state = env.reset()  # Reset environment for a random user
        episode_reward = 0
        done = False
        step_count = 0  # Track the number of steps in the episode

        while not done and step_count < max_steps:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Ensure correct shape
                action = model(state_tensor).argmax(dim=1).item()  # Choose action using the trained model
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            step_count += 1  # Increment the step count

        if step_count >= max_steps:
            print(f"Warning: Episode {episode + 1} reached the maximum step limit ({max_steps}).")

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")

    # Calculate average reward
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    # Path to the trained DQN model
    model_path = 'models/dqn_policy.pth'
    
    # Run evaluation
    evaluate_dqn(model_path=model_path, num_episodes=100)
