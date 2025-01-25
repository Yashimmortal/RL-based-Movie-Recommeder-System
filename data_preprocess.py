import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def preprocess_movielens(data_path, output_path):
    # Load raw data
    print("Loading data...")
    ratings = pd.read_csv(data_path + 'ratings.csv')  # Example columns: userId, movieId, rating, timestamp
    movies = pd.read_csv(data_path + 'movies.csv')    # Example columns: movieId, title, genres

    # Merge datasets
    print("Merging datasets...")
    data = pd.merge(ratings, movies, on='movieId')
    
    # Filter top users and movies to reduce size
    print("Filtering top users and movies...")
    top_users = data['userId'].value_counts().head(10000).index  # Keep top 10,000 users
    top_movies = data['movieId'].value_counts().head(5000).index  # Keep top 5,000 movies
    data = data[data['userId'].isin(top_users) & data['movieId'].isin(top_movies)]

    # Split into train and test
    print("Splitting into train and test...")
    train = data.sample(frac=0.8, random_state=42)  # 80% training data
    test = data.drop(train.index)                   # Remaining 20% as test data

    # Save train and test datasets
    train.to_csv(output_path + 'train.csv', index=False)
    test.to_csv(output_path + 'test.csv', index=False)
    print(f"Train and test data saved to {output_path}")

    # Create sparse interaction matrix
    print("Creating sparse interaction matrix...")
    user_map = {user_id: idx for idx, user_id in enumerate(train['userId'].unique())}
    movie_map = {movie_id: idx for idx, movie_id in enumerate(train['movieId'].unique())}
    train['user_idx'] = train['userId'].map(user_map)
    train['movie_idx'] = train['movieId'].map(movie_map)

    interaction_matrix = csr_matrix(
        (train['rating'], (train['user_idx'], train['movie_idx'])),
        shape=(len(user_map), len(movie_map))
    )

    # Save sparse matrix and features
    print("Saving interaction matrix and features...")
    np.save(output_path + 'interaction_matrix.npy', interaction_matrix)
    np.save(output_path + 'user_features.npy', list(user_map.keys()))
    np.save(output_path + 'item_features.npy', list(movie_map.keys()))

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_movielens(data_path='data/', output_path='data/')
