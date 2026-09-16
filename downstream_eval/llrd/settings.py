from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from downstream_eval.llrd.model_config import normalize_model_entry
from downstream_eval.llrd.runtime import SCRIPT_DIR

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"


class DatasetMode(Enum):
    RGZ = "rgz"
    MIRABEST = "mirabest"
    LOTSS_HORTON = "lotss_horton"
    ALL = "all"


class EvalMode(Enum):
    FINETUNE = "finetune"
    LINEAR_PROBE = "linear_probe"
    ALL = "all"


def load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    uncommented = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml

            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError:
                data = json.loads(uncommented)
        except ModuleNotFoundError:
            data = json.loads(uncommented)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping/object at the top level.")
    return data


def apply_matplotlib_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.serif": ["STIXTwoText", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.linewidth": 1.2,
        "axes.edgecolor": "black",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "xtick.minor.size": 5,
        "ytick.minor.size": 5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "figure.dpi": 300,
    })


_RAW_CONFIG = load_mapping(DEFAULT_CONFIG_PATH)
_RUN = _RAW_CONFIG["run"]
_PLOT = _RAW_CONFIG.get("plot", {})
_DATASETS = {item["key"]: item for item in _RAW_CONFIG["datasets"]}
_ANALYSIS = _RAW_CONFIG["analysis"]

EVAL_MODE = EvalMode(_RUN["eval_mode"])
DATASET_MODE = DatasetMode(_RUN["dataset_mode"])
USE_CLASS_WEIGHTED_LOSS = bool(_RUN["use_class_weighted_loss"])
GLOBAL_SEED = int(_RUN["global_seed"])
BATCH_SIZE = int(_RUN["batch_size"])
SUBSAMPLE_FRACTION = _RUN["subsample_fraction"]
NUM_FOLDS = int(_RUN["num_folds"])
WORK_DIR = _RUN.get("work_dir") or str(Path(_RUN["output_dir"]).parent / "work")
CACHE_JSON_PATH = _RUN.get("cache_json_path") or str(Path(_RUN["output_dir"]) / "finetune_results_cache_llrd.json")

FINETUNE_EPOCHS = int(_RUN["finetune_epochs"])
FINETUNE_WEIGHT_DECAY = float(_RUN["finetune_weight_decay"])
WARMUP_RATIO_FT = float(_RUN["warmup_ratio_ft"])
WARMUP_RATIO_LP = float(_RUN["warmup_ratio_lp"])
MAX_GRAD_NORM = float(_RUN["max_grad_norm"])
DEBUG_LAYER_DECAY = int(_RUN["debug_layer_decay"])
LR_CONFIGS = [dict(item) for item in _RUN["lr_configs"]]

MIN_INCHES_PER_MODEL = float(_PLOT.get("min_inches_per_model", 6.0))
INCHES_PER_ROW = float(_PLOT.get("inches_per_row", 4.0))
FIG_DPI = int(_PLOT.get("fig_dpi", 300))
MAX_MODELS_PER_ROW = int(_PLOT.get("max_models_per_row", 6))
TOP_N_MODELS = int(_PLOT.get("top_n_models", 60))

RGZ_DATA_DIR = _DATASETS["rgz"]["root"]
MIRABEST_DATA_DIR = _DATASETS["mirabest"]["root"]
LOTSS_HORTON_DATA_DIR = _DATASETS["lotss_horton"]["root"]
DATA_INDEX_CACHE_DIR = _DATASETS["rgz"].get("index_cache_dir", "./dataset_index_cache")
DATA_INDEX_WORKERS = int(_DATASETS["rgz"].get("index_workers", 4))

_MODEL_SPECS = [normalize_model_entry(item) for item in _RAW_CONFIG["models"]]
_ENABLED_MODELS = [item for item in _MODEL_SPECS if item.get("enabled", True)]
MODEL_ENTRIES = [(item["alias"], item["encoder"], item["model_id"]) for item in _ENABLED_MODELS]

ANALYSIS_MODELS = [
    {
        "lookup_alias": item["alias"],
    }
    for item in _ENABLED_MODELS
    if item.get("analysis", False)
]
ANALYSIS_DATASET_ORDER = [
    (item["label"], item.get("analysis_label") or item["label"])
    for item in sorted(_DATASETS.values(), key=lambda item: int(item["analysis_order"]))
    if item.get("enabled", True)
]
ANALYSIS_COMPACT_TABLE_LAYOUT = bool(_ANALYSIS["compact_table_layout"])
ANALYSIS_SAVE_COMBINED_LP_FT = bool(_ANALYSIS["save_combined_lp_ft"])
ANALYSIS_SAVE_INDIVIDUAL_MODE_PLOTS = bool(_ANALYSIS["save_individual_mode_plots"])
ANALYSIS_CM_CMAP_NAME = _ANALYSIS["cm_cmap_name"]
ANALYSIS_CM_NORMALIZATION = _ANALYSIS["cm_normalization"]
ANALYSIS_HEADER_FONT_SIZE = int(_ANALYSIS["header_font_size"])
ANALYSIS_CLASS_ORDER = {
    item["label"]: list(item["class_order"])
    for item in _DATASETS.values()
    if item.get("enabled", True)
}


def _fmt_lr(val: float) -> str:
    return f"{val:.0e}"


def _cfg_tag(cfg: dict) -> str:
    return (
        f"bb={_fmt_lr(cfg['bb_lr'])}"
        f"_headx{int(cfg['head_mult'])}"
        f"_decay{cfg['layer_decay']:.2f}"
    )


def _cfg_label(cfg: dict) -> str:
    return (
        f"{_fmt_lr(cfg['bb_lr'])}/x{int(cfg['head_mult'])}"
        f"/{cfg['layer_decay']:.2f}"
    )
