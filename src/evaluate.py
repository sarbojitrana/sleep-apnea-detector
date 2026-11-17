"""Evaluation uses the same fixed zero-margin decision as inference."""
import math

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, fbeta_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from src.data import profile_groups
from src.model import train_tuned_model


def wilson_interval(successes, total):
    """Descriptive 95% binomial interval; repeated profiles are not independent."""
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def classification_metrics(y_true, predictions, scores):
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()]
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": tn / (tn + fp) if tn + fp else None,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "false_negatives": fn, "false_positives": fp,
        "sensitivity_interval_95": wilson_interval(tp, tp + fn),
        "report": classification_report(y_true, predictions, labels=[0, 1],
            target_names=["Other dataset label", "Apnea dataset label"], zero_division=0),
    }


def evaluate_model(model, X_test, y_test):
    scores = model.decision_function(X_test)
    return classification_metrics(y_test, (scores >= 0).astype(int), scores)


def cross_validate_model(X, y, cv=4):
    """Nested CV: retune from scratch inside every outer development fold."""
    scores = np.empty(len(X))
    fold_metrics = []
    for train, test in StratifiedGroupKFold(cv, shuffle=True, random_state=73).split(
            X, y, groups=profile_groups(X)):
        model, _ = train_tuned_model(X.iloc[train], y.iloc[train])
        scores[test] = model.decision_function(X.iloc[test])
        fold_metrics.append(evaluate_model(model, X.iloc[test], y.iloc[test]))
    return {
        "folds": cv,
        "pooled": classification_metrics(y, (scores >= 0).astype(int), scores),
        "f2_mean": float(np.mean([m["f2"] for m in fold_metrics])),
        "f2_std": float(np.std([m["f2"] for m in fold_metrics])),
        "fold_metrics": fold_metrics,
    }


def format_metrics(metrics):
    return "\n".join([
        f"Sensitivity/recall : {metrics['recall']:.3f}",
        f"Specificity        : {metrics['specificity']:.3f}",
        f"Precision          : {metrics['precision']:.3f}",
        f"F2 / F1            : {metrics['f2']:.3f} / {metrics['f1']:.3f}",
        f"ROC AUC / avg prec : {metrics['roc_auc']:.3f} / {metrics['average_precision']:.3f}",
        f"Missed / flagged other rows: {metrics['false_negatives']} / {metrics['false_positives']}",
        "Confusion matrix (actual rows / predicted columns; other, apnea):",
        str(metrics["confusion_matrix"]),
    ])
