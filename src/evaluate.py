"""Evaluation: metrics, threshold selection, confusion matrix and importance plots."""

from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, proba, threshold: float) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
    }


def select_threshold(y_val, proba_val, strategy: str = "max_f1", min_recall: float = 0.0) -> float:
    """Pick the decision threshold using VALIDATION data only."""
    if strategy != "max_f1":
        raise ValueError(f"Unknown threshold strategy: {strategy}")
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    ok = rec >= min_recall
    if not ok.any():
        ok = np.ones_like(ok, dtype=bool)
    idx = int(np.argmax(np.where(ok, f1, -1.0)))
    return float(thr[idx])


def permutation_feature_importance(pipeline, X, y, seed: int, n_repeats: int = 5) -> dict:
    """Model-agnostic importance on raw input columns (drop in PR-AUC when shuffled)."""
    res = permutation_importance(
        pipeline, X, y, scoring="average_precision", n_repeats=n_repeats, random_state=seed
    )
    pairs = sorted(
        zip(X.columns, res.importances_mean, res.importances_std, strict=False), key=lambda t: -t[1]
    )
    return {c: {"mean": float(m), "std": float(s)} for c, m, s in pairs}


def plot_confusion_matrix(cm: dict, path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1], ["No failure", "Failure"])
    ax.set_yticks([0, 1], ["No failure", "Failure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                int(arr[i, j]),
                ha="center",
                va="center",
                color="white" if arr[i, j] > arr.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_importance(importance: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(importance)[::-1]
    vals = [importance[n]["mean"] for n in names]
    errs = [importance[n]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(names, vals, xerr=errs, color="#4c78a8")
    ax.set_xlabel("Drop in PR-AUC when feature is shuffled (validation set)")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
