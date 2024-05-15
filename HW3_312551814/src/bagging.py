import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from src.utils import WeakClassifier


class BaggingClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        # Create 10 weak learners
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        # Normalize the training data
        X_train_normalized = self.scaler.fit_transform(X_train)

        losses_of_models = []
        for model in self.learners:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            criterion = nn.BCEWithLogitsLoss()

            # Select a random subset of data for training
            indices = np.random.choice(
                len(X_train_normalized), len(X_train_normalized), replace=True
            )
            X_subset = X_train_normalized[indices]
            y_subset = y_train[indices]

            # Convert data to torch tensors
            X_subset_torch = torch.tensor(X_subset, dtype=torch.float32)
            y_subset_torch = torch.tensor(y_subset, dtype=torch.float32).view(-1, 1)

            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_subset_torch)
                loss = criterion(outputs, y_subset_torch)
                loss.backward()
                optimizer.step()

            # Store the loss
            losses_of_models.append(loss.item())

        return losses_of_models

    def predict_learners(self, X) -> t.Tuple[t.Sequence[int], t.Sequence[float]]:
        # Normalize test data
        X_test_normalized = self.scaler.transform(X)

        predictions_probs = []
        for model in self.learners:
            model.eval()
            with torch.no_grad():
                outputs = model(torch.tensor(X_test_normalized, dtype=torch.float32))
                probabilities = torch.sigmoid(outputs).numpy().flatten()
                predictions_probs.append(probabilities)

        # Average probabilities from all models
        avg_probs = np.mean(predictions_probs, axis=0)
        # Use 0.5 threshold for binary class prediction
        predictions_classes = (avg_probs > 0.5).astype(int)

        return predictions_classes, predictions_probs

    def compute_feature_importance(self):
        feature_names = ["age", "sex", "cp", "fbs", "thalach", "thal"]
        feature_importance = np.zeros(len(feature_names))

        for model in self.learners:
            # Get feature weights from the model
            weights = np.abs(model.fc.weight.detach().numpy().squeeze())

            # Add feature weights to overall importance
            feature_importance += weights

        # Normalize feature importance
        feature_importance /= np.sum(feature_importance)

        return feature_importance.tolist()
