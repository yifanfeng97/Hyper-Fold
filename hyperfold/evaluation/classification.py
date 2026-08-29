"""Classification metrics for Hyper-Fold: EC and fold evaluation.

EC-number prediction is evaluated with the GearNet-protocol metrics:
protein-centric Fmax (the maximum F1 over decision thresholds) and
micro-averaged AUPR. Fold classification is evaluated with top-1 accuracy.
All metrics are pure NumPy.
"""
import numpy as np

__all__ = [
    "compute_fmax",
    "compute_micro_aupr",
    "compute_accuracy",
    # GearNet-protocol naming aliases.
    "compute_f1_max",
    "compute_auprc_micro",
]


def compute_micro_aupr(y_true, y_score):
    """Micro-averaged area under the precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels. Any shape is accepted; it is flattened
        internally.
    y_score : np.ndarray
        Predicted scores of the same shape as ``y_true``.

    Returns
    -------
    float
        Micro-averaged AUPR. Returns ``0.0`` when there are no positive
        labels, avoiding division-by-zero NaNs.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    positives = y_true == 1
    n_pos = positives.sum()
    if n_pos == 0:
        return 0.0

    order = np.argsort(y_score, kind="mergesort")[::-1]
    target = y_true[order]
    precision = target.cumsum() / np.arange(1, len(target) + 1)
    auprc = precision[target == 1].sum() / n_pos
    return float(auprc)


def compute_fmax(y_true, y_score):
    """Protein-centric Fmax (GearNet/TorchDrug ``f1_max`` protocol).

    For each sample with at least one positive label, computes precision and
    recall over the ranked class scores, averages those precision/recall
    values across samples at each ranking threshold, then returns the maximum
    F1 obtained over thresholds. Implemented with NumPy and guarded so that
    rows without positive labels (which contribute no signal) do not produce
    NaNs.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples, n_classes)
        Ground-truth binary labels.
    y_score : np.ndarray, shape (n_samples, n_classes)
        Predicted scores.

    Returns
    -------
    float
        Maximum F1 score. Returns ``0.0`` when no sample has a positive label.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.ndim != 2 or y_score.ndim != 2:
        raise ValueError("y_true and y_score must be 2-dimensional")
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    row_sum = y_true.sum(axis=1)
    valid = row_sum > 0
    if valid.sum() == 0:
        return 0.0

    pred = y_score[valid]
    target = y_true[valid]

    # Descending sort order within each row.
    order = np.argsort(-pred, axis=1, kind="mergesort")
    target_sorted = np.take_along_axis(target, order, axis=1)

    precision = target_sorted.cumsum(axis=1) / np.arange(1, target.shape[1] + 1)
    recall = target_sorted.cumsum(axis=1) / (target.sum(axis=1, keepdims=True) + 1e-10)

    # Mark the top-ranked element of each row as a "start" in the global sort.
    is_start = np.zeros_like(target, dtype=bool)
    is_start[np.arange(is_start.shape[0]), order[:, 0]] = True

    # Deterministic tie-breaker: within each row give the top-ranked entry the
    # largest auxiliary score, the second-ranked a slightly smaller one, etc.
    # This guarantees that all per-row starts appear before any later entry in
    # the global sort, preventing NaNs from ``is_start.cumsum(0)`` being zero
    # at the first global rank.
    eps = 1e-5
    n = pred.shape[1]
    bias = np.zeros_like(pred)
    bias[np.arange(pred.shape[0])[:, None], order] = (n - np.arange(n)) * eps
    pred_for_sort = pred + bias

    all_order = np.argsort(-pred_for_sort.ravel(), kind="mergesort")

    # Map local (row, col) indices to a flat global index.
    order_flat = order + np.arange(order.shape[0])[:, None] * order.shape[1]
    order_flat = order_flat.ravel()
    inv_order = np.zeros_like(order_flat)
    inv_order[order_flat] = np.arange(order_flat.shape[0])

    is_start_flat = is_start.ravel()[all_order]
    all_order = inv_order[all_order]
    precision_flat = precision.ravel()
    recall_flat = recall.ravel()

    prev_precision = np.where(is_start_flat, 0.0, precision_flat[all_order - 1])
    all_precision = precision_flat[all_order] - prev_precision
    all_precision = all_precision.cumsum() / is_start_flat.cumsum()

    prev_recall = np.where(is_start_flat, 0.0, recall_flat[all_order - 1])
    all_recall = recall_flat[all_order] - prev_recall
    all_recall = all_recall.cumsum() / pred.shape[0]

    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return float(all_f1.max())


def compute_accuracy(y_true, y_pred):
    """Top-1 accuracy for single-label (fold) classification.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground-truth class indices.
    y_pred : np.ndarray, shape (n_samples,) or (n_samples, n_classes)
        Predicted class indices, or per-class logits/scores (argmaxed).

    Returns
    -------
    float
        Top-1 accuracy. Returns ``0.0`` for empty inputs.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(axis=1)
    y_pred = y_pred.ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have compatible shapes")
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


# GearNet-protocol naming aliases.
compute_f1_max = compute_fmax
compute_auprc_micro = compute_micro_aupr
