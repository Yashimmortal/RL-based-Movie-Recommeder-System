import numpy as np

def calculate_cumulative_reward(rewards):
    """Calculate the total reward over an episode."""
    return np.sum(rewards)

def calculate_diversity(actions, item_features):
    """
    Measure diversity as the average pairwise cosine distance
    between the features of the recommended items.
    """
    selected_features = item_features[actions]
    if len(actions) < 2:
        return 0  
    distances = []
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            dist = 1 - np.dot(selected_features[i], selected_features[j]) / (
                np.linalg.norm(selected_features[i]) * np.linalg.norm(selected_features[j])
            )
            distances.append(dist)
    return np.mean(distances)

def calculate_coverage(actions, total_items):
    """Measure the proportion of items recommended."""
    return len(set(actions)) / total_items
