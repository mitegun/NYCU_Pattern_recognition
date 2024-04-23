import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


# Class implementing linear regression using closed-form solution
class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        # Add a column of ones for the intercept term
        ones_column = np.ones((X.shape[0], 1))
        X = np.concatenate((ones_column, X), axis=1)
        # Calculate weights using closed-form solution
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
        # Make predictions using the calculated weights and intercept
        return X @ self.weights + self.intercept


# Class implementing linear regression using gradient descent
class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate, epochs):
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)
        # Initialize parameters
        num_samples = X.shape[0]
        num_features = X.shape[1]
        self.weights = np.zeros(num_features)
        self.intercept = 0
        losses = []

        # Gradient descent loop
        for _ in range(epochs):
            y_pred = np.dot(X, self.weights) + self.intercept
            y_pred = np.squeeze(y_pred)
            y_reshaped = np.squeeze(y)
            d_weights = -(2 / num_samples) * np.dot(X.T, (y_reshaped - y_pred))
            d_intercept = -(2 / num_samples) * np.sum(y_reshaped - y_pred)
            self.weights -= learning_rate * d_weights
            self.intercept -= learning_rate * d_intercept
            mse = np.mean((y_reshaped - y_pred) ** 2)
            losses.append(mse)
        return losses

    def predict(self, X):
        # Make predictions
        return np.dot(X, self.weights) + self.intercept

    def plot_learning_curve(self, losses):
        # Plot learning curve
        plt.plot(losses)
        plt.title("Learning Curve for Gradient Descent")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.show()


# Class for linear regression gradient descent with normalization
class GradientdescentNormalized(LinearRegressionBase):
    def fit(self, X, y, learning_rate, epochs):
        # Initialize parameters
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.intercept = 0
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_normalized = (X - self.mean) / self.std
        losses = []

        # Gradient descent loop
        for _ in range(epochs):
            y_pred = np.dot(X_normalized, self.weights) + self.intercept
            d_weights = -(2 / num_samples) * np.dot(X_normalized.T, (y - y_pred))
            d_intercept = -(2 / num_samples) * np.sum(y - y_pred)
            self.weights -= learning_rate * d_weights
            self.intercept -= learning_rate * d_intercept
            mse = np.mean((y - y_pred) ** 2)
            losses.append(mse)
        return losses

    def predict(self, X):
        X_normalized = (X - self.mean) / self.std
        return np.dot(X_normalized, self.weights) + self.intercept

    def plot_learning_curve(self, losses):
        # Plot learning curve
        plt.plot(losses)
        plt.title("Learning Curve for Gradient descent Normalized")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.show()


# Class implementing linear regression with L1 regularization
class LinearRegressionL1Regularization(LinearRegressionGradientdescent):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def fit(self, X, y, learning_rate, epochs):
        # Initialize parameters
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.intercept = 0
        losses = []

        # Normalize features
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # Gradient descent loop with L1 regularization
        for _ in range(epochs):
            # Compute predictions
            y_pred = np.dot(X_normalized, self.weights) + self.intercept

            # Compute gradients
            d_weights = -(2 / num_samples) * np.dot(X_normalized.T, (y - y_pred))
            d_intercept = -(2 / num_samples) * np.sum(y - y_pred)

            # L1 Regularization
            d_weights += self.alpha * np.sign(self.weights)

            # Update parameters
            self.weights -= learning_rate * d_weights
            self.intercept -= learning_rate * d_intercept

            # Compute and store mean squared error with L1 regularization
            loss = np.mean((y_pred - y) ** 2) + self.alpha * np.sum(
                np.abs(self.weights)
            )
            losses.append(loss)

        return losses

    def predict(self, X):
        # Make predictions using L1 regularization
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        return np.dot(X_normalized, self.weights) + self.intercept

    def plot_learning_curve(self, losses):
        # Plot learning curve for L1 regularization
        plt.plot(losses)
        plt.title("Learning Curve for L1 regularization")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.show()


# Function to compute Mean Squared Error
def compute_mse(prediction, ground_truth):
    return np.mean(np.square(prediction - ground_truth))


# Main function
def main():
    # Load training data
    train_df = pd.read_csv("./train.csv")
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    # Initialize and fit models
    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f"{LR_CF.weights=}, {LR_CF.intercept=:.4f}")

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=1e-4, epochs=1000000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f"{LR_GD.weights=}, {LR_GD.intercept=:.4f}")

    LR_GDN = GradientdescentNormalized()
    losses = LR_GDN.fit(train_x, train_y, learning_rate=1e-3, epochs=10000)
    LR_GDN.plot_learning_curve(losses)
    logger.info(f"{LR_GDN.weights=}, {LR_GDN.intercept=:.4f}")

    LR_L1 = LinearRegressionL1Regularization(alpha=0.01)
    losses_l1 = LR_L1.fit(train_x, train_y, learning_rate=1e-3, epochs=10000)
    LR_L1.plot_learning_curve(losses_l1)
    logger.info(f"{LR_L1.weights=}, {LR_L1.intercept=:.4f}")

    # Load test data
    test_df = pd.read_csv("./test.csv")
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    # Make predictions
    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_gdn = LR_GDN.predict(test_x)
    y_preds_l1 = LR_L1.predict(test_x)

    # Calculate prediction differences
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).sum()
    y_preds_diff_gn = np.abs(y_preds_gdn - y_preds_cf).sum()
    y_preds_diff_L1 = np.abs(y_preds_l1 - y_preds_cf).sum()
    a = "Prediction difference between Gradient descent and Closed form:"
    logger.info(f"{a} {y_preds_diff:.4f}")
    b = "Prediction difference between"
    g = "Gradient descent Normalized and Closed form:"
    logger.info(f"{b} {g} {y_preds_diff_gn:.4f}")
    c = "Prediction difference between L1Regularization and Closed form:"
    logger.info(f"{c} {y_preds_diff_L1:.4f}")

    # Calculate Mean Squared Error
    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    mse_gdn = compute_mse(y_preds_gdn, test_y)
    mse_l1 = compute_mse(y_preds_l1, test_y)
    # Calculate percentage differences in MSE
    diff = ((mse_gd - mse_cf) / mse_cf) * 100
    diff_gdn = ((mse_gdn - mse_cf) / mse_cf) * 100
    diff_l1 = ((mse_l1 - mse_cf) / mse_cf) * 100
    d = "Difference between Gradient descent and Closed form: "
    logger.info(f"{mse_cf=:.4f}, {mse_gd=:.4f}. {d} {diff:.3f}%")
    e = "Difference between Gradient descent Normalized and Closed form:"
    logger.info(f"{mse_cf=:.4f}, {mse_gdn=:.4f}. {e} {diff_gdn:.3f}%")
    f = "Difference between L1Regularization and Closed form:"
    logger.info(f"{mse_cf=:.4f}, {mse_l1=:.4f}. {f} {diff_l1:.3f}%")


# Entry point of the program
if __name__ == "__main__":
    main()
