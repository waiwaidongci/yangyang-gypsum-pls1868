"""Domain entry points used by the catalog, seed script and background worker."""

from .dropout import fit_combo, verify_combos
from .evaluation import holdout_diagnostics, pending_estimates, pending_queue
from .fixtures import demo_rows
from .pls import fit_path
from .records import normalize_rows, predictors, strengths
from .reports import calibration_tables, dropout_tables, handoff_tables, selection_tables
from .selection import select_components
from .settings import SPEC, parse_settings

__all__ = ["SPEC", "demo_rows", "normalize_rows", "analyze"]


def analyze(rows: list[dict], settings: dict) -> dict:
    maximum, threshold, margin, registered = parse_settings(settings)
    records = sorted(normalize_rows(rows), key=lambda row: (row["batch_id"], row["id"]))
    train = [row for row in records if row["split"] == "train"]
    holdout = [row for row in records if row["split"] == "holdout"]
    pending = [row for row in records if row["split"] == "pending"]
    selected, candidates, folds = select_components(train, maximum)
    model = fit_path(predictors(train), strengths(train), selected.count)
    if model.components != selected.count:
        raise ValueError("全量训练数据无法重拟合已选分量，请核对各批次的测量与实测分布。")
    summary, holdout_trace = holdout_diagnostics(holdout, model, selected.count)
    combos = [fit_combo(train, bands, maximum) for bands in registered]
    checks, outcomes = verify_combos(holdout, model, selected.count, combos, margin)
    passed = [outcome.model for outcome in outcomes if not outcome.disabled]
    score_count = max([selected.count, *(combo.count for combo in passed)])
    queue, priority = pending_queue(
        pending_estimates(pending, model, selected.count, passed, score_count), threshold)
    unestimated = sum(row["predicted_mpa"] is None for row in queue)
    overall = summary[0]
    warnings = []
    if any(candidate.rmse is None for candidate in candidates):
        warnings.append("部分候选分量在至少一个训练折中不可提取，请查看逐折误差。")
    if overall["outside_strips"]:
        warnings.append(f'独立留出有 {overall["outside_strips"]} 条试条的读数超出训练范围，已保留在总体误差中。')
    if overall["rmse_mpa"] >= overall["baseline_rmse_mpa"]:
        warnings.append("独立留出 RMSE 未优于训练均值基线，请复核校准批覆盖范围。")
    disabled = [outcome.model.key for outcome in outcomes if outcome.disabled]
    if disabled:
        warnings.append(f'缺测组合 {"、".join(disabled)} 未通过逐批核验，已列入禁用清单。')
    if unestimated:
        warnings.append(f"{unestimated} 条待测试条没有完全匹配的已通过模型，未给出估计，直接安排实测。")
    if not pending:
        warnings.append("本次台账无待测试条。")
    combo_note = (f"登记缺测组合 {len(outcomes)} 种，核验通过 {len(passed)} 种、禁用 {len(disabled)} 种。"
                  if outcomes else "")
    return {
        "summary": (f"用 {selected.folds} 个校准批的折外误差选出 {selected.count} 个潜变量。"
                    f"冻结模型后，在 {len(summary) - 1} 个独立批的 {len(holdout)} 条试条上核验。"
                    f"待测 {len(pending)} 条，其中 {priority} 条优先实测。{combo_note}"),
        "metrics": {
            "选定潜变量数": selected.count, "训练批折外 RMSE（MPa）": round(selected.rmse, 8),
            "独立留出 RMSE（MPa）": overall["rmse_mpa"],
            "独立留出均值基线 RMSE（MPa）": overall["baseline_rmse_mpa"],
            "训练抗折均值（MPa）": model.target_mean, "优先实测试条数": priority,
            "登记缺测组合数": len(outcomes), "通过缺测组合数": len(passed),
            "禁用缺测组合数": len(disabled), "未估计待测条数": unestimated,
        },
        "tables": [*handoff_tables(summary, holdout_trace, queue, selected.count, score_count),
                   *selection_tables(selected, candidates, folds), *calibration_tables(model, selected.count),
                   *dropout_tables(outcomes, checks, margin)],
        "warnings": warnings,
    }
