import numpy as np


# Define the class for individual nodes in the decision tree
class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        quality=None,
        value=None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.quality = quality
        self.value = value


class DecisionTree:
    # Initialize the decision tree with specified index and maximum depth
    def __init__(self, index, max_depth=1):
        self.index = index  # index for splitting
        self.max_depth = max_depth  # Maximum depth of the tree

    # Fit the decision tree model to the training data
    def fit(self, X, y):
        # Combine features and target into one dataset
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)

        # Grow the decision tree recursively
        self.tree = self._grow_tree(dataset)

    # Calculate feature importance based on the trained tree
    def feature_importance(self, X):
        # Initialize an array to store feature importances
        feature_importance = np.zeros(X.shape[1])

        # Recursive function to traverse the tree and accumulate feature importance
        def traverse_tree(node, importance):
            if node.left is None and node.right is None:
                # Leaf node, return importance
                return importance
            else:
                # Non-leaf node, accumulate importance
                importance[node.feature_index] += node.quality
                importance = traverse_tree(node.left, importance)
                importance = traverse_tree(node.right, importance)
                return importance

        # Call the recursive function to accumulate feature importance
        feature_importance = traverse_tree(self.tree, feature_importance)

        # Normalize feature importances to sum up to 1
        feature_importance /= np.sum(feature_importance)

        return feature_importance

    # Recursively grow the decision tree
    def _grow_tree(self, dataset, depth=1):
        X, y = dataset[:, :-1], dataset[:, -1]  # Split dataset into features and target
        num_samples = np.shape(X)[0]  # Number of samples

        # Set max_depth to number of samples if not specified
        if self.max_depth is None:
            self.max_depth = num_samples

        # If current depth is within maximum depth limit
        if depth <= self.max_depth:
            # Find the best split for the current dataset
            best_split = find_best_split(dataset, self.index)

            # If a valid split is found
            if len(best_split) != 0:
                # If the quality of the split is greater than 0 (indicating impurity reduction)
                if best_split["quality"] > 0:
                    # Recursively grow the left and right subtrees
                    left_subtree = self._grow_tree(best_split["left"], depth + 1)
                    right_subtree = self._grow_tree(best_split["right"], depth + 1)

                    # Create a node representing the current split
                    return Node(
                        best_split["feature_index"],
                        best_split["threshold"],
                        left_subtree,
                        right_subtree,
                        best_split["quality"],
                    )

        # Leaf node: assign the majority class as the value
        leaf_value = np.argmax(np.bincount(y))
        return Node(value=leaf_value)

    # Make predictions using the trained decision tree
    def predict(self, X):
        predictions = [self._predict_tree(v, self.tree) for v in X]
        return predictions

    # Recursively predict the value for a given input using the decision tree
    def _predict_tree(self, x, tree_node):
        # If the current node is a leaf node, return its assigned value
        if tree_node.value is not None:
            return tree_node.value

        # Extract the feature value for the current node from the input data point
        feature_val = x[tree_node.feature_index]

        # If the feature value is less than or equal to the threshold, traverse the left subtree
        if feature_val <= tree_node.threshold:
            return self._predict_tree(x, tree_node.left)
        # Otherwise, traverse the right subtree
        else:
            return self._predict_tree(x, tree_node.right)


# Split the dataset based on a specified feature and threshold
def split_dataset(X, y, feature_index, threshold):
    dataset = np.concatenate(
        (X, y.reshape(-1, 1)), axis=1
    )  # Combine features and target
    # Split the dataset into left and right subsets based on the specified feature and threshold
    left = np.array([row for row in dataset if row[feature_index] <= threshold])
    right = np.array([row for row in dataset if row[feature_index] > threshold])
    return left, right


# Find the best split for the current dataset
def find_best_split(dataset, index):
    X, y = dataset[:, :-1], dataset[:, -1]  # Separate features and target
    best_split = {}  # Store information about the best split
    max_quality = -float("inf")  # Initialize maximum quality

    # List of all feature indices
    feature_idx_list = [i for i in range(X.shape[1])]

    # For each feature index
    for feature_index in feature_idx_list:
        feature_values = dataset[:, feature_index]  # Extract feature values
        possible_thresholds = np.unique(feature_values)  # Get unique feature values
        thresholds = [
            (possible_thresholds[i] + possible_thresholds[i + 1]) / 2
            for i in range(len(possible_thresholds) - 1)
        ]

        # For each potential threshold
        for threshold in thresholds:
            # Split the dataset based on the current feature and threshold
            left, right = split_dataset(X, y, feature_index, threshold)

            # Ensure both resulting subsets have at least one data point
            if len(left) > 0 and len(right) > 0:
                left_y, right_y = left[:, -1], right[:, -1]  # Separate target values
                # Calculate the quality of this split
                current_quality = gain(y, left_y, right_y, index)

                # Update the best split if the current split has higher quality
                if current_quality > max_quality:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["left"] = left
                    best_split["right"] = right
                    best_split["quality"] = current_quality
                    max_quality = current_quality

    return best_split


# Calculate the gain based on the provided splitting index
def gain(parent, left_child, right_child, mode):
    weight_left = len(left_child) / len(parent)
    weight_right = len(right_child) / len(parent)

    # Gain calculation
    gain = mode(parent) - (
        weight_left * mode(left_child) + weight_right * mode(right_child)
    )

    return gain


# Calculate the entropy of a given set of labels
def entropy(y):
    n = len(y)
    _, count = np.unique(y, return_counts=True)
    prob = count / n

    entropy = -1 * np.sum(prob * np.log2(prob))

    return entropy


# Calculate the Gini index of a given set of labels
def gini(y):
    n = len(y)
    _, count = np.unique(y, return_counts=True)
    prob = count / n

    gini = 1 - np.sum(prob**2)

    return gini