"""Summarize untouched holdout batches and rank pending destructive checks."""

from collections import defaultdict

import numpy as np

from .dropout import ComboModel
from .pls import PLSPath
from .records import missing_bands, predictors, strengths
from .settings import FEATURES, TARGET


def trace_prediction(records: list[dict], path: PLSPath, count: int) -> list[dict]:
    x = predictors(records)
    predictions = path.predict(x, count)
    scores = path.scores(x, count)
    outside = (x < path.low) | (x > path.high)
    result = []
    for record, estimate, latent, flags in zip(records, predictions, scores, outside):
        result.append({
            "id": record["id"], "batch_id": record["batch_id"],
            "predicted_mpa": float(estimate),
            "outside_features": ",".join(field for field, flag in zip(FEATURES, flags) if flag),
            **{f"score_{number}": round(float(score), 8) for number, score in enumerate(latent, 1)},
        })
    return result


def trace_combo(records: list[dict], combo: ComboModel) -> list[dict]:
    x = predictors(records, combo.features)
    predictions = combo.path.predict(x, combo.count)
    scores = combo.path.scores(x, combo.count)
    outside = (x < combo.path.low) | (x > combo.path.high)
    result = []
    for record, estimate, latent, flags in zip(records, predictions, scores, outside):
        result.append({
            "id": record["id"], "batch_id": record["batch_id"],
            "predicted_mpa": float(estimate),
            "outside_features": ",".join(field for field, flag in zip(combo.features, flags) if flag),
            **{f"score_{number}": round(float(score), 8) for number, score in enumerate(latent, 1)},
        })
    return result


def outside_readings(record: dict, path: PLSPath) -> str:
    flags = []
    for index, field in enumerate(FEATURES):
        value = record[field]
        if value is not None and (value < path.low[index] or value > path.high[index]):
            flags.append(field)
    return ",".join(flags)


def pending_estimates(records: list[dict], full_path: PLSPath, full_count: int,
                      combos: list[ComboModel], score_count: int) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[missing_bands(record)].append(record)
    passed = {combo.bands: combo for combo in combos}
    result = []
    for bands, group in groups.items():
        if not bands:
            rows = trace_prediction(group, full_path, full_count)
            for row in rows:
                row["model"] = "全波段"
        elif bands in passed:
            rows = trace_combo(group, passed[bands])
            for row in rows:
                row["model"] = f"缺 {'+'.join(bands)}"
        else:
            rows = [{
                "id": record["id"], "batch_id": record["batch_id"], "predicted_mpa": None,
                "outside_features": outside_readings(record, full_path), "model": "无可用模型",
            } for record in group]
        result.extend(rows)
    for row in result:
        for number in range(1, score_count + 1):
            row.setdefault(f"score_{number}", None)
    return result


def holdout_diagnostics(records: list[dict], path: PLSPath, count: int) -> tuple[list[dict], list[dict]]:
    trace = trace_prediction(records, path, count)
    groups = defaultdict(list)
    for index, (row, record) in enumerate(zip(trace, records)):
        actual = record[TARGET]
        row.update(actual_mpa=actual, error_mpa=row["predicted_mpa"] - actual)
        groups[row["batch_id"]].append(index)
    predictions = np.array([row["predicted_mpa"] for row in trace])
    actual = strengths(records)
    residual = predictions - actual
    baseline_residual = path.target_mean - actual
    results = []
    for scope, batch, index in [
        ("overall", "", np.arange(len(trace))),
        *(("batch", batch, selected) for batch, selected in sorted(groups.items())),
    ]:
        errors = residual[index]
        results.append({
            "scope": scope, "batch_id": batch, "strips": len(index),
            "rmse_mpa": round(float(np.sqrt(np.mean(errors ** 2))), 8),
            "mae_mpa": round(float(np.mean(np.abs(errors))), 8),
            "bias_mpa": round(float(np.mean(errors)), 8),
            "baseline_rmse_mpa": round(float(np.sqrt(np.mean(baseline_residual[index] ** 2))), 8),
            "outside_strips": sum(bool(trace[i]["outside_features"]) for i in index),
        })
    return results, trace


def pending_queue(trace: list[dict], threshold: float) -> tuple[list[dict], int]:
    for row in trace:
        if row["predicted_mpa"] is None:
            row.update(action="无完全匹配的已通过模型，直接实测", priority=1)
        elif row["outside_features"]:
            row.update(action="核对读数并优先实测", priority=1)
        elif row["predicted_mpa"] < threshold:
            row.update(action="低于筛查线，优先实测", priority=2)
        else:
            row.update(action="按班次常规实测", priority=3)
    trace.sort(key=lambda row: (row["priority"], row["predicted_mpa"] is not None,
                                row["predicted_mpa"] or 0.0, row["id"]))
    for index, row in enumerate(trace, 1):
        row["position"] = index
    return trace, sum(row["priority"] < 3 for row in trace)
