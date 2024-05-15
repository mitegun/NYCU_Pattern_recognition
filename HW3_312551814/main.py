import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from src.adaboost import AdaBoostClassifier
from src.bagging import BaggingClassifier
from src.decision_tree import DecisionTree, gini, entropy
from src.utils import plot_learners_roc


def main():
    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")

    X_train = train_df.drop(["target"], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df["target"].to_numpy()  # (n_samples, )

    X_test = test_df.drop(["target"], axis=1).to_numpy()
    y_test = test_df["target"].to_numpy()

    feature_names = list(train_df.drop(["target"], axis=1).columns)

    """
    Feel free to modify the following section if you need.
    Remember to print out logs with loguru.
    """

    # AdaBoost
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_adaboost.fit(
        X_train,
        y_train,
        num_epochs=500,
        learning_rate=0.1,
    )
    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    y_pred_classes, _ = clf_adaboost.predict_learners(X_test)

    # Convert y_pred_classes to numpy array
    y_pred_classes = np.array(y_pred_classes)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred_classes.mean(axis=0).round())

    logger.info(f"AdaBoost - Accuracy: {accuracy:.4f}")

    # Plot ROC curves
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="roc_curves.png",  # Specify the file path
    )

    # Compute feature importance
    feature_importance = clf_adaboost.compute_feature_importance()

    # Plotting
    feature_names = ["age", "sex", "cp", "fbs", "thalach", "thal"]

    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, feature_importance, align="center")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.show()

    # Bagging
    clf_bagging = BaggingClassifier(input_dim=X_train.shape[1])
    _ = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=5000,  # 15000
        learning_rate=0.001,
    )

    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy = accuracy_score(y_test, y_pred_classes)

    logger.info(f"Bagging - Accuracy: {accuracy:.4f}")

    plot_learners_roc(
        y_preds=y_pred_probs,  # Utilisez les probabilités prédites
        y_trues=y_test,
        fpath="roc_curves.png",
    )
    feature_names = ["age", "sex", "cp", "fbs", "thalach", "thal"]
    feature_importance = clf_bagging.compute_feature_importance()

    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, feature_importance, align="center")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.show()

    # Decision Tree

    example = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    logger.info(f"Gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1] is : {gini(example):.4f}")
    logger.info(
        f"Entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1] is : {entropy(example):.4f}"
    )

    clf_tree = DecisionTree(index=entropy, max_depth=7)
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_decision_tree = accuracy_score(y_test, y_pred_classes)
    logger.info(f"DecisionTree - Accuracy: {accuracy_decision_tree:.4f}")

    # Compute feature importance
    feature_importance = clf_tree.feature_importance(X_train)

    # Plotting feature importance
    feature_names = ["age", "sex", "cp", "fbs", "thalach", "thal"]
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, feature_importance, align="center")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.show()
    # plt.show()


if __name__ == "__main__":
    main()
