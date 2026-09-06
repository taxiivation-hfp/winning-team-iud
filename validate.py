"""Batch-aware validation - the only kind that predicts the leaderboard here.

A random split mixes measurements from the same batch into train and validation,
so a model that has merely memorised batch-specific offsets scores well and then
collapses on batch 10. Both schemes below hold out *whole batches*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from data import CLASS_NAMES, CLASSES


def macro_f1(y_true, y_pred) -> float:
    """The competition metric: unweighted mean of the six per-class F1 scores."""
    return float(f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0))


def leave_one_batch_out(batch: np.ndarray):
    """Hold out each batch in turn, train on all the others."""
    for b in np.unique(batch):
        val = batch == b
        yield b, np.flatnonzero(~val), np.flatnonzero(val)


def forward_chaining(batch: np.ndarray, min_train_batches: int = 3):
    """Train on batches <= k, validate on batch k+1.

    This is the honest rehearsal of the real task: every fold predicts a *later*
    batch from earlier ones, exactly as batch 10 is predicted from 1-9.
    """
    batches = np.unique(batch)
    for i in range(min_train_batches, len(batches)):
        yield batches[i], np.flatnonzero(np.isin(batch, batches[:i])), np.flatnonzero(batch == batches[i])


SCHEMES = {"forward": forward_chaining, "lobo": leave_one_batch_out}


def evaluate(model, X: pd.DataFrame, y: np.ndarray, batch: np.ndarray, scheme: str = "forward") -> pd.DataFrame:
    """Run a batch-aware CV and return per-fold macro F1.

    The mean is unweighted across folds, so a tiny batch counts as much as a big
    one. Watch the last folds especially - they are the closest analogue to
    batch 10. Note that some batches are missing classes entirely (batches 3-5
    contain no Toluene), so a fold can only be as good as the classes it holds.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {sorted(SCHEMES)}, got {scheme!r}")

    rows = []
    for held_out, tr, va in SCHEMES[scheme](batch):
        fold = clone(model)
        fold.fit(X.iloc[tr], y[tr])
        pred = fold.predict(X.iloc[va])
        rows.append(
            {
                "held_out_batch": held_out,
                "n_train": len(tr),
                "n_val": len(va),
                "classes_present": len(np.unique(y[va])),
                "macro_f1": macro_f1(y[va], pred),
            }
        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> str:
    mean = results["macro_f1"].mean()
    last = results["macro_f1"].iloc[-1]
    table = results.to_string(index=False, float_format="%.4f")
    return f"{table}\n\nmean macro-F1 {mean:.4f} | last fold {last:.4f}"


def per_class_report(y_true, y_pred) -> str:
    """Where the macro average is being dragged down - usually a rare class."""
    return classification_report(
        y_true,
        y_pred,
        labels=CLASSES,
        target_names=[CLASS_NAMES[c] for c in CLASSES],
        zero_division=0,
    )


def confusion(y_true, y_pred) -> pd.DataFrame:
    names = [CLASS_NAMES[c] for c in CLASSES]
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=CLASSES), index=names, columns=names)
