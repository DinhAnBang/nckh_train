"""Shared binary metrics and per-attack detection analysis."""

from collections import Counter
from time import perf_counter

import numpy as np


def scores(model, matrix):
    probabilities = model.predict_proba(matrix)
    if probabilities.shape[1] != 2:
        raise ValueError("Expected probabilities for two classes")
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Attack class 1 is absent from model classes")
    return np.asarray(probabilities[:, classes.index(1)], dtype=np.float64)


def summarize(y_true, probability, attack_types, threshold=0.5):
    """Return metrics and attack-type rows for one fixed threshold."""
    from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                                 f1_score, precision_recall_curve, precision_score, recall_score, auc)
    truth = np.asarray(y_true, dtype=np.uint8)
    chance = np.asarray(probability, dtype=np.float64)
    if len(truth) != len(chance) or len(truth) != len(attack_types):
        raise ValueError("Prediction, label and metadata lengths differ")
    if not np.isfinite(chance).all() or ((chance < 0) | (chance > 1)).any():
        raise ValueError("Invalid attack probabilities")
    predicted = (chance >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    precision_curve, recall_curve, _ = precision_recall_curve(truth, chance)
    metrics = {
        "rows": len(truth), "normal": int((truth == 0).sum()), "attack": int((truth == 1).sum()),
        "threshold": threshold, "accuracy": accuracy_score(truth, predicted),
        "precision": precision_score(truth, predicted, zero_division=0),
        "recall": recall_score(truth, predicted, zero_division=0),
        "f1": f1_score(truth, predicted, zero_division=0),
        "macro_f1": f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0),
        "pr_auc": auc(recall_curve, precision_curve),
        "average_precision": average_precision_score(truth, chance),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }
    counts = Counter()
    for kind, actual, guess in zip(attack_types, truth, predicted):
        if actual == 1:
            counts[(str(kind), "rows")] += 1
            counts[(str(kind), "detected")] += int(guess == 1)
    by_type = [{"type": kind, "rows": counts[(kind, "rows")],
                "detected": counts[(kind, "detected")],
                "missed": counts[(kind, "rows")] - counts[(kind, "detected")],
                "recall": counts[(kind, "detected")] / counts[(kind, "rows")]}
               for kind in sorted({kind for kind, _ in counts})]
    return metrics, by_type


def evaluate_batches(model, batches, threshold=0.5):
    labels, probabilities, types = [], [], []
    prediction_seconds = 0.0
    for x, y, kinds in batches:
        started = perf_counter()
        p = scores(model, x)
        prediction_seconds += perf_counter() - started
        labels.append(y)
        probabilities.append(p)
        types.extend(kinds)
    metrics, by_type = summarize(np.concatenate(labels), np.concatenate(probabilities), types, threshold)
    metrics["prediction_seconds"] = round(prediction_seconds, 3)
    return metrics, by_type
