"""Columns, measurement units and run options shared by the application and domain."""

FEATURES = (
    "nir_1200", "nir_1450", "nir_1550", "nir_1720", "nir_1940", "nir_2100",
    "density_kg_m3", "thickness_mm",
)
BANDS = tuple(field for field in FEATURES if field.startswith("nir_"))
TARGET = "flexural_mpa"
COLUMNS = ("id", "batch_id", "split", *FEATURES, TARGET)
FIELDS = {
    "id": "试条编号",
    "batch_id": "制板批号",
    "split": "样本用途",
    "nir_1200": "1200 nm 吸光度",
    "nir_1450": "1450 nm 吸光度",
    "nir_1550": "1550 nm 吸光度",
    "nir_1720": "1720 nm 吸光度",
    "nir_1940": "1940 nm 吸光度",
    "nir_2100": "2100 nm 吸光度",
    "density_kg_m3": "体积密度（kg/m³）",
    "thickness_mm": "实测试条厚度（mm）",
    TARGET: "实测抗折强度（MPa）",
}
MAX_COMBOS = 16
SETTINGS = [
    {"key": "max_components", "label": "候选潜变量上限", "default": 6,
     "min": 1, "max": len(FEATURES), "step": 1},
    {"key": "screening_mpa", "label": "优先实测筛查线（MPa）", "default": 4.2,
     "min": 0.1, "max": 30, "step": 0.1},
    {"key": "dropout_margin_mpa", "label": "缺测组合逐批误差约定值（MPa）", "default": 0.1,
     "min": 0.0, "max": 5.0, "step": 0.01},
    {"key": "missing_combos", "label": "登记的缺测波段组合", "default": [],
     "type": "combos", "bands": list(BANDS)},
]
SPEC = {
    "columns": list(COLUMNS),
    "fields": FIELDS,
    "settings": SETTINGS,
    "modules": {
        "datasets": "试条测量台账",
        "runs": "潜变量校准",
        "results": "批次留出结果",
        "reviews": "工坊实测交接",
    },
    "limitation": "抗折估计用于安排破坏性实测。新批超出校准范围时优先实测，放行仍依据实测。",
    "dataset_name": "20261125-再生石膏板试条.csv",
    "recommendation": "先复核独立批误差，再领取待测试条清单，优先实测越界或低于筛查线的试条。",
}


def normalize_combos(value) -> list[list[str]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("登记的缺测波段组合须为数组，每项为一组波段编号。")
    seen = set()
    combos = []
    for item in value:
        if not isinstance(item, (list, tuple)) or not item:
            raise ValueError("每种缺测组合须为非空波段数组。")
        unknown = [band for band in item if band not in BANDS]
        if unknown:
            raise ValueError(f"缺测组合含未知波段：{', '.join(map(str, unknown))}。")
        canonical = tuple(field for field in BANDS if field in set(item))
        if canonical in seen:
            raise ValueError(f"缺测组合 {'+'.join(canonical)} 重复登记。")
        seen.add(canonical)
        combos.append(list(canonical))
    if len(combos) > MAX_COMBOS:
        raise ValueError(f"缺测组合最多登记 {MAX_COMBOS} 种。")
    return combos


def parse_settings(settings: dict) -> tuple[int, float, float, list[tuple[str, ...]]]:
    if settings.keys() - {option["key"] for option in SETTINGS}:
        raise ValueError("运行参数仅接受 max_components、screening_mpa、dropout_margin_mpa 和 missing_combos。")
    parsed = {}
    for option in SETTINGS:
        value = settings.get(option["key"], option["default"])
        if option.get("type") == "combos":
            parsed[option["key"]] = [tuple(combo) for combo in normalize_combos(value)]
            continue
        low, high = option["min"], option["max"]
        if type(value) not in (int, float) or not low <= value <= high:
            raise ValueError(f'{option["label"]}须为 {low} 至 {high} 的数值。')
        parsed[option["key"]] = value
    maximum = parsed["max_components"]
    if maximum != int(maximum):
        raise ValueError("候选潜变量上限须为整数。")
    return (int(maximum), float(parsed["screening_mpa"]),
            float(parsed["dropout_margin_mpa"]), parsed["missing_combos"])
