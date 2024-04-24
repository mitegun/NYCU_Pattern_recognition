import typing as t
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-3, num_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:

        # Normalize input data
        self.mean = np.mean(inputs, axis=0)
        self.std = np.std(inputs, axis=0)
        normalized_inputs = (inputs - self.mean) / self.std

        # Initialize weights and intercept
        self.weights = np.zeros(normalized_inputs.shape[1])
        self.intercept = 0

        # Training the model
        for _ in range(self.num_iterations):
            logits = np.dot(normalized_inputs, self.weights) + self.intercept
            predicted_probs = self.sigmoid(logits)
            gradient = np.dot(normalized_inputs.T, (predicted_probs - targets))
            # Update weights and intercept using gradient descent
            self.weights -= self.learning_rate * gradient
            self.intercept -= self.learning_rate * np.mean(predicted_probs - targets)

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:

        # Normalize input data
        normalized_inputs = (inputs - self.mean) / self.std
        # Calculate logits and predicted probabilities
        logits = np.dot(normalized_inputs, self.weights) + self.intercept
        predicted_probs = self.sigmoid(logits)
        # Convert probabilities to binary classes
        predicted_classes = np.where(predicted_probs >= 0.5, 1, 0)
        return predicted_probs, predicted_classes

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class FLD:
    def __init__(self):
        # Initialize variables for Fisher's Linear Discriminant
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        # Fit the FLD model to the training data
        self.train_inputs = inputs
        self.train_targets = targets
        self.mean = np.mean(inputs, axis=0)
        self.std = np.std(inputs, axis=0)
        class_0 = inputs[targets == 0]
        class_1 = inputs[targets == 1]
        self.m0 = np.mean(class_0, axis=0)
        self.m1 = np.mean(class_1, axis=0)
        self.sw = np.dot((class_0 - self.m0).T, class_0 - self.m0) + np.dot(
            (class_1 - self.m1).T, class_1 - self.m1
        )
        self.sb = np.dot(
            (self.m1 - self.m0).reshape(-1, 1), (self.m1 - self.m0).reshape(1, -1)
        )
        self.w = np.linalg.inv(self.sw).dot(self.m1 - self.m0)
        self.w /= np.linalg.norm(self.w)
        self.slope = -self.w[0] / self.w[1]
        self.intercept = self.m1 @ self.w

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Sequence[t.Union[int, bool]]:
        # Predict class labels for input data
        projected_inputs = inputs @ self.w
        m0_reshaped = np.tile(self.m0, (len(inputs), 1)).T
        m1_reshaped = np.tile(self.m1, (len(inputs), 1)).T
        dist_to_m0 = np.linalg.norm(projected_inputs - m0_reshaped, axis=0)
        dist_to_m1 = np.linalg.norm(projected_inputs - m1_reshaped, axis=0)
        predictions = dist_to_m0 > dist_to_m1
        return predictions.astype(int)

    def plot_projection(self, x_test: np.ndarray[float]) -> None:
        # Plot the projection line
        
        plt.title(
            f"Projection Line: Slope={self.slope:.2f}, Intercept={self.intercept:.2f}"
        )
        # Plot the decision boundary
        x_boundary = np.array(
            [min(self.train_inputs[:, 0]), max(self.train_inputs[:, 0])]
        )
        y_boundary = self.slope * x_boundary + self.intercept
        plt.plot(x_boundary, y_boundary, color="red", label="Projection Line")

        # Obtain the prediction of the testing set
        predictions = self.predict(x_test)

        # Colorize points based on prediction
        unique_classes = np.unique(predictions)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_classes)))
        for i, c in enumerate(unique_classes):
            class_points = x_test[predictions == c]
            plt.scatter(
                class_points[:, 0],
                class_points[:, 1],
                color=colors[i],
                label=f"Class {c}",
            )
        plt.legend()
        plt.show()


def compute_auc(y_trues, y_preds) -> float:
    # Compute Area Under the Curve
    return roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds) -> float:
    # Compute accuracy score
    return np.mean(y_trues == y_preds)


def main():
    # Read data
    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")

    # Logistic Regression
    x_train = train_df.drop(["target"], axis=1).to_numpy()
    y_train = train_df["target"].to_numpy()
    x_test = test_df.drop(["target"], axis=1).to_numpy()
    y_test = test_df["target"].to_numpy()
    LR = LogisticRegression(learning_rate=1e-3, num_iterations=1000)
    LR.fit(x_train, y_train)
    y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes[1])
    auc = compute_auc(y_test, y_pred_classes[1])
    logger.info(f"LR: Weights: {LR.weights[:5]}, Intercept: {LR.intercept}")
    logger.info(f"LR: AUC={auc:.4f}")
    logger.info(f"LR: Accuracy={accuracy:.4f}")

    # FLD
    cols = ["27", "30"]
    x_train = train_df[cols].to_numpy()
    y_train = train_df["target"].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df["target"].to_numpy()
    fld_model = FLD()
    fld_model.fit(x_train, y_train)
    y_preds = fld_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_preds)
    logger.info(f"FLD: m0={fld_model.m0}, m1={fld_model.m1}")
    logger.info(f"FLD: Sw=\n{fld_model.sw}")
    logger.info(f"FLD: Sb=\n{fld_model.sb}")
    logger.info(f"FLD: w=\n{fld_model.w}")
    logger.info(f"FLD: Accuracy={accuracy:.4f}")

    # Plot the projection line and testing data
    fld_model.plot_projection(x_test)


if __name__ == "__main__":
    main()
