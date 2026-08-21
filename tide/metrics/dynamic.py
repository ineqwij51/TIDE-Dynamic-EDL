"""Static-aware trajectory metrics for 20-second emotion curves."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import rankdata


STATIC_EPSILON = 1e-8
LOG_FLOOR = 1e-6
DYNAMIC_METRICS = (
    "StaticAware_SRCC",
    "StaticAware_PCC",
    "StaticAware_KendallTauB",
    "TimeAugmented_Discrete_Frechet",
    "TWED_lambda1_nu1",
    "DTW",
    "ZMAE",
    "ZRMSE",
    "ZLinf",
    "DeltaCorr_v1",
    "ChangeMagnitudeCorr",
    "LogRatioChangeCorr",
    "Directional_Accuracy",
    "Balanced_Nonflat_Directional_Accuracy",
    "Flat_Transition_Predicted_Change",
)
LOWER_IS_BETTER = frozenset(
    {
        "TimeAugmented_Discrete_Frechet",
        "TWED_lambda1_nu1",
        "DTW",
        "ZMAE",
        "ZRMSE",
        "ZLinf",
        "Flat_Transition_Predicted_Change",
        "Chebyshev",
        "Clark",
        "Canberra",
        "KL",
    }
)


def _batch_curves(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or pred.shape != truth.shape or pred.shape[1] < 2:
        raise ValueError("dynamic metrics require matching finite [windows, seconds] arrays")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("dynamic metric inputs must be finite")
    return pred, truth


def _batch_zscore(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = value.std(axis=1, ddof=0)
    static = scale <= STATIC_EPSILON
    safe_scale = np.where(static, 1.0, scale)
    result = (value - value.mean(axis=1, keepdims=True)) / safe_scale[:, None]
    result[static] = 0.0
    return result, static


def _batch_correlation(
    left: np.ndarray,
    right: np.ndarray,
    left_static: np.ndarray | None = None,
    right_static: np.ndarray | None = None,
) -> np.ndarray:
    left_z, computed_left_static = _batch_zscore(left)
    right_z, computed_right_static = _batch_zscore(right)
    left_static = computed_left_static if left_static is None else left_static
    right_static = computed_right_static if right_static is None else right_static
    result = np.clip(np.mean(left_z * right_z, axis=1), -1.0, 1.0)
    result[left_static & right_static] = 1.0
    result[left_static ^ right_static] = 0.0
    return result


def _batch_kendall_tau_b(
    prediction: np.ndarray,
    target: np.ndarray,
    prediction_static: np.ndarray,
    target_static: np.ndarray,
) -> np.ndarray:
    rows, seconds = prediction.shape
    concordant = np.zeros(rows, dtype=np.float64)
    discordant = np.zeros(rows, dtype=np.float64)
    ties_prediction = np.zeros(rows, dtype=np.float64)
    ties_target = np.zeros(rows, dtype=np.float64)
    for left in range(seconds - 1):
        prediction_sign = np.sign(
            prediction[:, left + 1 :] - prediction[:, [left]]
        )
        target_sign = np.sign(target[:, left + 1 :] - target[:, [left]])
        product = prediction_sign * target_sign
        concordant += np.sum(product > 0, axis=1)
        discordant += np.sum(product < 0, axis=1)
        ties_prediction += np.sum(
            (prediction_sign == 0) & (target_sign != 0), axis=1
        )
        ties_target += np.sum(
            (target_sign == 0) & (prediction_sign != 0), axis=1
        )
    comparable = concordant + discordant
    denominator = np.sqrt(
        (comparable + ties_prediction) * (comparable + ties_target)
    )
    result = np.divide(
        concordant - discordant,
        denominator,
        out=np.zeros(rows, dtype=np.float64),
        where=denominator > 0,
    )
    result[prediction_static & target_static] = 1.0
    result[prediction_static ^ target_static] = 0.0
    return result


def _batch_frechet(prediction_z: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    rows, seconds = prediction_z.shape
    time = np.linspace(0.0, 1.0, seconds, dtype=np.float64)
    local = np.sqrt(
        np.square(prediction_z[:, :, None] - target_z[:, None, :])
        + np.square(time[None, :, None] - time[None, None, :])
    )
    dynamic = np.full((rows, seconds, seconds), np.inf, dtype=np.float64)
    for left in range(seconds):
        for right in range(seconds):
            if left == 0 and right == 0:
                previous = np.zeros(rows, dtype=np.float64)
            elif left == 0:
                previous = dynamic[:, left, right - 1]
            elif right == 0:
                previous = dynamic[:, left - 1, right]
            else:
                previous = np.minimum.reduce(
                    (
                        dynamic[:, left - 1, right],
                        dynamic[:, left - 1, right - 1],
                        dynamic[:, left, right - 1],
                    )
                )
            dynamic[:, left, right] = np.maximum(previous, local[:, left, right])
    return dynamic[:, -1, -1]


def _batch_dtw(prediction_z: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    rows, seconds = prediction_z.shape
    dynamic = np.full(
        (rows, seconds + 1, seconds + 1), np.inf, dtype=np.float64
    )
    dynamic[:, 0, 0] = 0.0
    for left in range(1, seconds + 1):
        for right in range(1, seconds + 1):
            dynamic[:, left, right] = np.abs(
                prediction_z[:, left - 1] - target_z[:, right - 1]
            ) + np.minimum.reduce(
                (
                    dynamic[:, left - 1, right],
                    dynamic[:, left, right - 1],
                    dynamic[:, left - 1, right - 1],
                )
            )
    return dynamic[:, -1, -1]


def _batch_twed(prediction_z: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    rows, seconds = prediction_z.shape
    time = np.linspace(0.0, 1.0, seconds, dtype=np.float64)
    dynamic = np.full(
        (rows, seconds + 1, seconds + 1), np.inf, dtype=np.float64
    )
    dynamic[:, 0, 0] = 0.0
    for left in range(1, seconds + 1):
        previous = prediction_z[:, left - 2] if left > 1 else 0.0
        previous_time = time[left - 2] if left > 1 else 0.0
        dynamic[:, left, 0] = (
            dynamic[:, left - 1, 0]
            + np.abs(prediction_z[:, left - 1] - previous)
            + abs(time[left - 1] - previous_time)
            + 1.0
        )
    for right in range(1, seconds + 1):
        previous = target_z[:, right - 2] if right > 1 else 0.0
        previous_time = time[right - 2] if right > 1 else 0.0
        dynamic[:, 0, right] = (
            dynamic[:, 0, right - 1]
            + np.abs(target_z[:, right - 1] - previous)
            + abs(time[right - 1] - previous_time)
            + 1.0
        )
    for left in range(1, seconds + 1):
        prediction_previous = prediction_z[:, left - 2] if left > 1 else 0.0
        prediction_previous_time = time[left - 2] if left > 1 else 0.0
        for right in range(1, seconds + 1):
            target_previous = target_z[:, right - 2] if right > 1 else 0.0
            target_previous_time = time[right - 2] if right > 1 else 0.0
            delete_prediction = (
                dynamic[:, left - 1, right]
                + np.abs(prediction_z[:, left - 1] - prediction_previous)
                + abs(time[left - 1] - prediction_previous_time)
                + 1.0
            )
            delete_target = (
                dynamic[:, left, right - 1]
                + np.abs(target_z[:, right - 1] - target_previous)
                + abs(time[right - 1] - target_previous_time)
                + 1.0
            )
            match = (
                dynamic[:, left - 1, right - 1]
                + np.abs(prediction_z[:, left - 1] - target_z[:, right - 1])
                + np.abs(prediction_previous - target_previous)
                + abs(time[left - 1] - time[right - 1])
                + abs(prediction_previous_time - target_previous_time)
            )
            dynamic[:, left, right] = np.minimum.reduce(
                (delete_prediction, delete_target, match)
            )
    return dynamic[:, -1, -1]


def _balanced_accuracy(
    target_sign: np.ndarray,
    prediction_sign: np.ndarray,
) -> np.ndarray:
    rows = target_sign.shape[0]
    total = np.zeros(rows, dtype=np.float64)
    classes = np.zeros(rows, dtype=np.float64)
    for direction in (-1, 1):
        mask = target_sign == direction
        count = mask.sum(axis=1)
        recall = np.divide(
            ((prediction_sign == target_sign) & mask).sum(axis=1),
            count,
            out=np.zeros(rows, dtype=np.float64),
            where=count > 0,
        )
        total += recall
        classes += count > 0
    return np.divide(
        total,
        classes,
        out=np.ones(rows, dtype=np.float64),
        where=classes > 0,
    )


def batch_dynamic_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the frozen dynamic metrics for matching ``[N,T]`` curves."""

    pred, truth = _batch_curves(prediction, target)
    if (pred < 0).any() or (truth < 0).any():
        raise ValueError("dynamic log-ratio metrics require nonnegative trajectories")
    pred_z, pred_static = _batch_zscore(pred)
    truth_z, truth_static = _batch_zscore(truth)
    difference = np.abs(pred_z - truth_z)
    delta_pred = np.diff(pred_z, axis=1)
    delta_truth = np.diff(truth_z, axis=1)
    raw_delta_truth = np.diff(truth, axis=1)
    pred_delta_static = delta_pred.std(axis=1, ddof=0) <= STATIC_EPSILON
    truth_delta_static = delta_truth.std(axis=1, ddof=0) <= STATIC_EPSILON
    nonflat = raw_delta_truth != 0.0
    pred_sign = np.sign(delta_pred).astype(np.int8)
    truth_sign = np.sign(delta_truth).astype(np.int8)
    nonflat_count = nonflat.sum(axis=1)
    direction_accuracy = np.divide(
        ((pred_sign == truth_sign) & nonflat).sum(axis=1),
        nonflat_count,
        out=np.ones(pred.shape[0], dtype=np.float64),
        where=nonflat_count > 0,
    )
    flat = ~nonflat
    flat_count = flat.sum(axis=1)
    flat_change = np.divide(
        (np.abs(delta_pred) * flat).sum(axis=1),
        flat_count,
        out=np.zeros(pred.shape[0], dtype=np.float64),
        where=flat_count > 0,
    )
    return {
        "StaticAware_SRCC": _batch_correlation(
            rankdata(pred, axis=1, method="average"),
            rankdata(truth, axis=1, method="average"),
            pred_static,
            truth_static,
        ),
        "StaticAware_PCC": _batch_correlation(
            pred, truth, pred_static, truth_static
        ),
        "StaticAware_KendallTauB": _batch_kendall_tau_b(
            pred, truth, pred_static, truth_static
        ),
        "TimeAugmented_Discrete_Frechet": _batch_frechet(pred_z, truth_z),
        "TWED_lambda1_nu1": _batch_twed(pred_z, truth_z),
        "DTW": _batch_dtw(pred_z, truth_z),
        "ZMAE": difference.mean(axis=1),
        "ZRMSE": np.sqrt(np.mean(np.square(difference), axis=1)),
        "ZLinf": difference.max(axis=1),
        "DeltaCorr_v1": _batch_correlation(
            delta_pred,
            delta_truth,
            pred_delta_static,
            truth_delta_static,
        ),
        "ChangeMagnitudeCorr": _batch_correlation(
            np.abs(delta_pred), np.abs(delta_truth)
        ),
        "LogRatioChangeCorr": _batch_correlation(
            np.diff(np.log(pred + LOG_FLOOR), axis=1),
            np.diff(np.log(truth + LOG_FLOOR), axis=1),
        ),
        "Directional_Accuracy": direction_accuracy,
        "Balanced_Nonflat_Directional_Accuracy": _balanced_accuracy(
            np.where(nonflat, truth_sign, 0),
            np.where(nonflat, pred_sign, 0),
        ),
        "Flat_Transition_Predicted_Change": flat_change,
    }


def static_aware_pcc(
    prediction: Iterable[float] | np.ndarray,
    target: Iterable[float] | np.ndarray,
) -> float:
    pred, truth = _batch_curves(
        np.asarray(prediction, dtype=np.float64).reshape(1, -1),
        np.asarray(target, dtype=np.float64).reshape(1, -1),
    )
    _, pred_static = _batch_zscore(pred)
    _, truth_static = _batch_zscore(truth)
    return float(
        _batch_correlation(pred, truth, pred_static, truth_static)[0]
    )


def static_aware_srcc(
    prediction: Iterable[float] | np.ndarray,
    target: Iterable[float] | np.ndarray,
) -> float:
    pred, truth = _batch_curves(
        np.asarray(prediction, dtype=np.float64).reshape(1, -1),
        np.asarray(target, dtype=np.float64).reshape(1, -1),
    )
    _, pred_static = _batch_zscore(pred)
    _, truth_static = _batch_zscore(truth)
    return float(
        _batch_correlation(
            rankdata(pred, axis=1, method="average"),
            rankdata(truth, axis=1, method="average"),
            pred_static,
            truth_static,
        )[0]
    )


__all__ = [
    "DYNAMIC_METRICS",
    "LOG_FLOOR",
    "LOWER_IS_BETTER",
    "STATIC_EPSILON",
    "batch_dynamic_metrics",
    "static_aware_pcc",
    "static_aware_srcc",
]
