from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import downstream_eval.llrd.settings as settings


DatasetMode = settings.DatasetMode
EvalMode = settings.EvalMode


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    root: str
    enabled: bool
    index_cache_dir: str
    index_workers: int
    analysis_order: int
    analysis_label: str
    class_order: list[str]


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    encoder: str
    model_id: str
    enabled: bool
    analysis: bool


@dataclass(frozen=True)
class EvalRunConfig:
    eval_mode: str
    dataset_mode: str
    work_dir: str
    output_dir: str
    cache_json_path: str | None
    use_class_weighted_loss: bool
    global_seed: int
    deterministic_eval: bool
    batch_size: int
    subsample_fraction: float | None
    num_folds: int
    finetune_epochs: int
    finetune_weight_decay: float
    warmup_ratio_ft: float
    warmup_ratio_lp: float
    max_grad_norm: float
    debug_layer_decay: int
    lr_configs: list[dict[str, float]]
    hf_cache_path: str


@dataclass(frozen=True)
class AnalysisConfig:
    compact_table_layout: bool
    save_combined_lp_ft: bool
    save_individual_mode_plots: bool
    cm_cmap_name: str
    cm_normalization: str
    header_font_size: int


@dataclass(frozen=True)
class LLRDConfig:
    run: EvalRunConfig
    datasets: list[DatasetSpec]
    models: list[ModelSpec]
    analysis: AnalysisConfig

    def selected_eval_modes(self) -> list[str]:
        if self.run.eval_mode == EvalMode.ALL.value:
            return [EvalMode.LINEAR_PROBE.value, EvalMode.FINETUNE.value]
        return [self.run.eval_mode]

    def selected_datasets(self) -> list[DatasetSpec]:
        return [
            dataset
            for dataset in self.datasets
            if dataset.enabled and (self.run.dataset_mode == DatasetMode.ALL.value or dataset.key == self.run.dataset_mode)
        ]

    def selected_models(self) -> list[ModelSpec]:
        return [model for model in self.models if model.enabled]

    def cache_path(self) -> str:
        if self.run.cache_json_path:
            return self.run.cache_json_path
        return str(Path(self.run.output_dir) / "finetune_results_cache_llrd.json")
