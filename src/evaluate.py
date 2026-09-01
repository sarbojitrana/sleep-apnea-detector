import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "report": classification_report(y_test, y_pred, target_names=["No Apnea", "Apnea"], zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def cross_validate_model(model, X, y, cv=5, scoring="f1"):
    scores = cross_val_score(
        model, X, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        n_jobs=-1,
    )
    return float(np.mean(scores)), float(np.std(scores))


def format_metrics(metrics):
    lines = [
        f"Accuracy   : {metrics['accuracy']:.3f}",
        f"Precision  : {metrics['precision']:.3f}",
        f"Recall     : {metrics['recall']:.3f}",
        f"F1         : {metrics['f1']:.3f}",
        f"ROC AUC    : {metrics['roc_auc']:.3f}",
        "",
        metrics["report"],
        "Confusion matrix (rows = actual, cols = predicted):",
        str(metrics["confusion_matrix"]),
    ]
    return "\n".join(lines)
