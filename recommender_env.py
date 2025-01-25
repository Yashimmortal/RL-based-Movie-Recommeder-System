import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gym
from gym import spaces
import numpy as np
from scipy.sparse import csr_matrix

class RecommenderEnv(gym.Env):
    """
    Custom Reinforcement Learning Environment for a Movie Recommender System.
    Users are treated as states, and movie recommendations are actions.
    Rewards are derived from the interaction matrix.
    """
    def __init__(self, interaction_matrix, user_features, item_features):
        super(RecommenderEnv, self).__init__()
        
        # Initialize environment components
        self.interaction_matrix = interaction_matrix  # Sparse user-item interaction matrix
        self.user_features = user_features           # Array of user IDs
        self.item_features = item_features           # Array of movie IDs
        
        # Define action and observation spaces
        self.num_users, self.num_items = interaction_matrix.shape
        self.action_space = spaces.Discrete(self.num_items)  # Actions = recommending a movie (movie IDs)
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(self.num_items,), dtype=np.float32
        )  # Observation = user's interaction history
        
        # Placeholder for the current user
        self.current_user = None

    def reset(self):
        """
        Resets the environment by selecting a random user.
        Returns the user's interaction history as the initial state.
        """
        self.current_user = np.random.randint(self.num_users)  # Select a random user
        state = self.interaction_matrix[self.current_user].toarray().flatten()  # Sparse row to dense array
        return state

    def step(self, action):
        """
        Simulates recommending a movie and retrieves the reward from the interaction matrix.
        
        Args:
            action (int): The movie ID to recommend (column index in the interaction matrix).
        
        Returns:
            state (array): Updated interaction history of the current user (unchanged in this simple example).
            reward (float): Rating the user gave for the recommended movie.
            done (bool): Whether the episode is finished (reward is 0).
            info (dict): Additional information (empty in this case).
        """
        # Get the reward (rating) for the recommended movie
        reward = self.interaction_matrix[self.current_user, action]
        
        # If reward is 0, it means the user didn't interact with the movie, and the episode ends
        done = reward == 0
        
        # The state remains unchanged in this example
        state = self.interaction_matrix[self.current_user].toarray().flatten()
        
        return state, reward, done, {}

if __name__ == "__main__":
    # Load preprocessed data
    interaction_matrix = np.load('data/interaction_matrix.npy', allow_pickle=True).item()
    user_features = np.load('data/user_features.npy')
    item_features = np.load('data/item_features.npy')
    
    # Initialize the environment
    env = RecommenderEnv(interaction_matrix, user_features, item_features)
    
    # Test the environment
    print("Testing the Recommender Environment...")
    state = env.reset()
    print(f"Initial state (first 10 movie ratings): {state[:10]}")  # Show the first 10 movie ratings
    action = np.random.choice(env.num_items)  # Randomly recommend a movie
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
