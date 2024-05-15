import typing as t
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


class WeakClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super(WeakClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # No activation for the hidden layer
        hidden = self.fc(x)
        # Sigmoid activation for the output
        output = torch.tanh(hidden)
        return output


def entropy_loss(outputs, targets):
    # Apply sigmoid to convert logits to probabilities
    outputs = torch.sigmoid(outputs)

    # Compute binary cross entropy
    bce_loss = -targets * torch.log(outputs) - (1 - targets) * torch.log(1 - outputs)

    # Sum along the classes dimension
    bce_loss = torch.sum(bce_loss, dim=1)

    # Compute entropy loss
    entropy_loss = torch.mean(bce_loss)

    return entropy_loss


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath="./tmp.png",
):
    plt.figure(figsize=(8, 6))
    for y_pred in y_preds:
        fpr, tpr, _ = roc_curve(y_trues, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label="ROC curve (area = %0.2f)" % roc_auc)

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig(fpath)
    plt.close()
