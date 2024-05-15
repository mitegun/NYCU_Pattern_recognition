import torch
import numpy as np
from src.utils import entropy_loss
from src.utils import WeakClassifier


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10):
        # Initialize the AdaBoostClassifier with specified input dimension and number of weak learners
        self.sample_weights = None  # Initialize sample weights
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]  # Create a list of weak learners
        self.alphas = []  # List to store alpha values for each weak learner
        self.feature_means = None  # Initialize feature means
        self.feature_stds = None  # Initialize feature standard deviations
        self.feature_importance = None  # Initialize feature importance

    def normalize_data(self, X, train=False):
        # Normalize input data
        if train:
            # Compute and store feature means and standard deviations during training
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
        X_normalized = (X - self.feature_means) / (
            self.feature_stds + 1e-8
        )  # Normalize data using computed means and standard deviations
        return X_normalized

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        # Train the AdaBoost classifier
        X_train_normalized = self.normalize_data(
            X_train, train=True
        )  # Normalize training data
        losses_of_models = []  # List to store losses of each weak learner
        num_samples = len(X_train_normalized)  # Number of samples in training data
        self.sample_weights = (
            torch.ones(num_samples) / num_samples
        )  # Initialize sample weights equally
        for model in self.learners:  # Iterate over weak learners
            for epoch in range(num_epochs):  # Iterate over epochs
                losses = []  # List to store losses for each epoch

                model.zero_grad()  # Clear gradients
                outputs = model(
                    torch.tensor(X_train_normalized, dtype=torch.float32)
                )  # Get predictions from the model
                targets = torch.tensor(y_train, dtype=torch.float32).view(
                    -1, 1
                )  # Convert targets to tensor

                # Compute entropy loss
                loss = entropy_loss(outputs, targets)
                losses.append(loss.item())  # Append loss value to losses list

                losses = torch.tensor(losses)  # Convert losses list to tensor
                error = (
                    losses.unsqueeze(1) * self.sample_weights
                ).sum()  # Compute weighted error
                alpha = 0.5 * torch.log((1 - error) / error)  # Compute alpha
                self.alphas.append(alpha.item())  # Append alpha value to alphas list

                predicted = (
                    outputs.squeeze() > 0
                ).float()  # Convert predictions to binary
                incorrect = predicted != targets.squeeze()  # Find incorrect predictions
                exponents = torch.exp(
                    -learning_rate * alpha * incorrect.float()
                )  # Compute exponents for sample weights update
                self.sample_weights *= exponents  # Update sample weights
                self.sample_weights /= (
                    self.sample_weights.sum()
                )  # Normalize sample weights

                losses_of_models.append(
                    losses
                )  # Append losses of current model to losses_of_models list

        return losses_of_models  # Return losses of all models

    def predict_learners(self, X):
        # Make weighted predictions using trained weak learners
        X_normalized = self.normalize_data(X)  # Normalize input data
        predictions_classes = []  # List to store predicted classes
        predictions_probs = []  # List to store predicted probabilities
        for model, alpha in zip(
            self.learners, self.alphas
        ):  # Iterate over weak learners and their corresponding alphas
            outputs = model(
                torch.tensor(X_normalized, dtype=torch.float32)
            )  # Get predictions from the model
            weighted_outputs = (
                alpha * outputs
            )  # Apply alpha weights to the model predictions
            predictions_classes.append(
                (weighted_outputs.squeeze() > 0).int().tolist()
            )  # Convert predictions to classes and append to list
            predictions_probs.append(
                weighted_outputs.squeeze().tolist()
            )  # Append weighted probabilities to list
        return (
            predictions_classes,
            predictions_probs,
        )  # Return weighted predicted classes and probabilities

    def compute_feature_importance(self):
        feature_importance = np.zeros(self.learners[0].fc.weight.shape[1])
        for alpha, learner in zip(self.alphas, self.learners):
            # Get weights from the linear layer of the neural network
            weights = np.abs(learner.fc.weight.data.numpy().flatten())
            # Add weights weighted by alpha to get feature importance
            feature_importance += weights * alpha

        # Normalize feature importance
        feature_importance /= np.sum(feature_importance)

        self.feature_importance = feature_importance.tolist()
        return self.feature_importance
