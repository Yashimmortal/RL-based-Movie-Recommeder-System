import streamlit as st
import numpy as np
import torch
from env.recommender_env import RecommenderEnv
from train_dqn import DQN

# Function to load the trained model
@st.cache_resource
def load_model(model_path, input_dim, output_dim):
    model = DQN(input_dim, output_dim).float()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to recommend movies
def recommend_movies(model, user_id, env, top_k=5):
    env.reset(user_id=user_id)
    state = env.get_user_state(user_id)
    recommendations = []

    for _ in range(top_k):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = model(state_tensor).argmax(dim=1).item()
        _, reward, done, _ = env.step(action)
        recommendations.append((env.item_index_to_movie[action], reward))
        if done:
            break

    return recommendations

# Load preprocessed data
@st.cache_resource
def load_environment():
    interaction_matrix = np.load('data/interaction_matrix.npy', allow_pickle=True).item()
    user_features = np.load('data/user_features.npy')
    item_features = np.load('data/item_features.npy')
    return RecommenderEnv(interaction_matrix, user_features, item_features)

# Main Streamlit app
def main():
    st.title("Movie Recommender System")
    st.write("Interact with the movie recommendation system powered by Reinforcement Learning.")

    # Load environment and model
    env = load_environment()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    model = load_model('models/dqn_policy.pth', input_dim, output_dim)

    # Sidebar for user selection
    user_id = st.sidebar.number_input("Enter User ID", min_value=0, max_value=env.num_users - 1, step=1, value=0)
    top_k = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

    # Recommend movies
    if st.button("Recommend Movies"):
        recommendations = recommend_movies(model, user_id, env, top_k)
        st.subheader(f"Top {top_k} Movie Recommendations for User {user_id}:")
        for i, (movie, reward) in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie} (Predicted Reward: {reward:.2f})")

if __name__ == "__main__":
    main()
