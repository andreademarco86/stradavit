from __future__ import annotations

from pathlib import Path

from downstream_eval.llrd.run_types import (
    AnalysisConfig,
    DatasetMode,
    DatasetSpec,
    EvalMode,
    EvalRunConfig,
    LLRDConfig,
    ModelSpec,
)
from downstream_eval.llrd.model_config import normalize_model_entry
from downstream_eval.llrd.settings import DEFAULT_CONFIG_PATH, load_mapping


def _config_from_mapping(data: dict) -> LLRDConfig:
    missing = [key for key in ("run", "datasets", "models", "analysis") if key not in data]
    if missing:
        raise ValueError(f"LLRD config is missing required sections: {missing}")
    run = dict(data["run"])
    run.setdefault("deterministic_eval", True)
    run.setdefault("work_dir", str(Path(run.get("output_dir", "downstream_eval/llrd/outputs")).parent / "work"))
    run.pop("num_workers", None)
    run.pop("paper_parity", None)

    cfg = LLRDConfig(
        run=EvalRunConfig(**run),
        datasets=[DatasetSpec(**item) for item in data["datasets"]],
        models=[ModelSpec(**normalize_model_entry(item)) for item in data["models"]],
        analysis=AnalysisConfig(**data["analysis"]),
    )
    validate_config(cfg)
    return cfg


def load_config(path: str | None = None) -> LLRDConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    return _config_from_mapping(load_mapping(config_path))


def validate_config(config: LLRDConfig) -> None:
    valid_eval_modes = {item.value for item in EvalMode}
    valid_dataset_modes = {item.value for item in DatasetMode}
    if config.run.eval_mode not in valid_eval_modes:
        raise ValueError(f"Invalid eval_mode={config.run.eval_mode!r}; expected one of {sorted(valid_eval_modes)}.")
    if config.run.dataset_mode not in valid_dataset_modes:
        raise ValueError(f"Invalid dataset_mode={config.run.dataset_mode!r}; expected one of {sorted(valid_dataset_modes)}.")
    if config.run.num_folds < 2:
        raise ValueError("num_folds must be >= 2.")
    if not config.selected_datasets():
        raise ValueError("No datasets selected.")
    if not config.selected_models():
        raise ValueError("No models selected.")
    aliases = [model.alias for model in config.models]
    if len(aliases) != len(set(aliases)):
        raise ValueError("Model aliases must be unique.")
