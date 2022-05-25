from sklearn.metrics import f1_score, precision_score
import numpy as np


def model_performance(y_true, y_pred, losses, pos_weight):
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    loss = sum(losses) / len(y_true)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    out = {
        "loss": loss,
        "f1": f1,
        "precision": precision
    }
    return out
