import numpy as np

class AverageStateTracker:
    def __init__(self, interaction_matrix, item_features):
        """
        Initialize the tracker with user-item interactions and item features.
        """
        self.interaction_matrix = interaction_matrix
        self.item_features = item_features

    def get_state(self, user_id):
        """
        Compute the state for a user by averaging the features of interacted items.
        """
        user_interactions = self.interaction_matrix[user_id]
        interacted_items = np.where(user_interactions > 0)[0]
        
        if len(interacted_items) == 0:
            # No interactions yet; return zero vector
            return np.zeros(self.item_features.shape[1])
        
        # Average the features of the interacted items
        state = np.mean(self.item_features[interacted_items], axis=0)
        return state
