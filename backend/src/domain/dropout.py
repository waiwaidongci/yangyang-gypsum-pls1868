"""Fit one frozen model per registered missing-band combo and verify it batch by batch."""

from dataclasses import dataclass

import numpy as np

from .pls import PLSPath, fit_path
from .records import predictors, strengths
from .selection import select_components_xy
from .settings import FEATURES

TOLERANCE = 1e-10


@dataclass(frozen=True)
class ComboModel:
    bands: tuple
    features: tuple
    count: int
    cv_rmse: float | None
    path: PLSPath | None

    @property
    def key(self) -> str:
        return "+".join(self.bands)


@dataclass(frozen=True)
class ComboCheck:
    key: str
    batch: str
    strips: int
    combo_rmse: float | None
    full_rmse: float
    excess: float | None
    passed: bool


@dataclass(frozen=True)
class ComboOutcome:
    model: ComboModel
    holdout_rmse: float | None
    disabled: bool
    reason: str


def fit_combo(train: list[dict], bands: tuple, maximum: int) -> ComboModel:
    features = tuple(field for field in FEATURES if field not in bands)
    x = predictors(train, features)
    y = strengths(train)
    groups = np.array([row["batch_id"] for row in train])
    try:
        selected, _, _ = select_components_xy(x, y, groups, maximum)
        path = fit_path(x, y, selected.count)
        if path.components != selected.count:
            raise ValueError("重拟合分量不足")
    except (ValueError, np.linalg.LinAlgError):
        return ComboModel(bands, features, 0, None, None)
    return ComboModel(bands, features, selected.count, selected.rmse, path)


def verify_combos(holdout: list[dict], full_path: PLSPath, full_count: int,
                  models: list[ComboModel], margin: float) -> tuple[list[ComboCheck], list[ComboOutcome]]:
    y = strengths(holdout)
    groups = np.array([row["batch_id"] for row in holdout])
    full_residual = full_path.predict(predictors(holdout), full_count) - y
    checks, outcomes = [], []
    for model in models:
        disabled = model.path is None
        reason = "完整校准批无法拟合该组合" if disabled else ""
        combo_residual = None
        if not disabled:
            combo_residual = model.path.predict(predictors(holdout, model.features), model.count) - y
        for batch in sorted(np.unique(groups)):
            mask = groups == batch
            full_rmse = float(np.sqrt(np.mean(full_residual[mask] ** 2)))
            if combo_residual is None:
                checks.append(ComboCheck(model.key, str(batch), int(mask.sum()), None, full_rmse, None, False))
                continue
            combo_rmse = float(np.sqrt(np.mean(combo_residual[mask] ** 2)))
            excess = combo_rmse - full_rmse
            passed = excess <= margin + TOLERANCE
            if not passed:
                disabled = True
                if not reason:
                    reason = f"留出批 {batch} 误差 {combo_rmse:.4f} MPa 超出全波段 {full_rmse:.4f} MPa 加约定值"
            checks.append(ComboCheck(model.key, str(batch), int(mask.sum()), combo_rmse, full_rmse, excess, passed))
        holdout_rmse = None
        if combo_residual is not None:
            holdout_rmse = float(np.sqrt(np.mean(combo_residual ** 2)))
        outcomes.append(ComboOutcome(model, holdout_rmse, disabled, reason))
    return checks, outcomes
