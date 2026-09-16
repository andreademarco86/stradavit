from __future__ import annotations

import math
import os
import types
from dataclasses import dataclass

import torch.nn as nn
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments

import downstream_eval.llrd.models as llrd_models
from downstream_eval.llrd.models import (
    DINOv2PoolerClassifier,
    DifferentialLRTrainer,
    apply_strict_linear_probe_head,
    compute_metrics,
)
from downstream_eval.llrd.run_types import EvalMode
from downstream_eval.llrd.runtime import guarded_print
from pretraining.stradavit_model import StradaViTForImageClassification


@dataclass(frozen=True)
class BatchPlan:
    per_device_batch_size: int
    gradient_accumulation_steps: int
    effective_global_batch_size: int
    world_size: int


def build_classifier_model(
    *,
    encoder_family: str,
    checkpoint_path: str,
    num_classes: int,
    class_weights,
    n_registers: int,
):
    if encoder_family == "mae":
        model = StradaViTForImageClassification(
            checkpoint_path=checkpoint_path,
            num_labels=num_classes,
            class_weights=class_weights,
            n_registers=n_registers,
        )
        if n_registers > 0:
            guarded_print(f"   → Using register-aware MAE model (n_registers={n_registers})")
        else:
            guarded_print("   → Using global average pooling for MAE model")
        return model

    if encoder_family == "dinov2":
        return DINOv2PoolerClassifier(
            checkpoint_path=checkpoint_path,
            num_labels=num_classes,
            class_weights=class_weights,
        )

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint_path,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    guarded_print("   → Using HF AutoModelForImageClassification (CLS pooling)")
    return model


def apply_linear_probe_runtime(model, num_classes: int):
    model = apply_strict_linear_probe_head(model, num_classes)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    guarded_print(
        f"   → Linear probe (strict): {trainable:,} / {total:,} params trainable "
        f"(head=classifier)"
    )
    if os.getenv("DEBUG_LINEAR_PROBE", "0") == "1":
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        guarded_print(f"   → Linear probe trainable params: {trainable_names}")

    def _linear_probe_train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            for module_name, module in self.named_modules():
                if not module_name:
                    continue
                if module_name != "classifier" and not module_name.startswith("classifier."):
                    module.eval()
        return self

    model.train = types.MethodType(_linear_probe_train, model)
    return model


def build_batch_plan(target_global_batch_size: int, encoder_family: str) -> BatchPlan:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    base_per_device = max(1, target_global_batch_size // max(1, world_size))
    per_device_bs = base_per_device
    eff_per_step = per_device_bs * max(1, world_size)
    grad_acc_steps = max(1, math.ceil(target_global_batch_size / eff_per_step)) if eff_per_step > 0 else 1
    eff_global_bs = per_device_bs * max(1, world_size) * grad_acc_steps
    return BatchPlan(
        per_device_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_acc_steps,
        effective_global_batch_size=eff_global_bs,
        world_size=world_size,
    )


def log_batch_plan(plan: BatchPlan, target_global_batch_size: int) -> None:
    guarded_print(
        f"   → Batch config: per-device={plan.per_device_batch_size}, "
        f"grad_acc_steps={plan.gradient_accumulation_steps}, "
        f"effective_global_batch≈{plan.effective_global_batch_size} "
        f"(target={target_global_batch_size})"
    )


def should_find_unused_parameters(encoder_family: str, n_registers: int) -> bool:
    return bool(encoder_family == "dinov2" or (encoder_family == "mae" and n_registers > 0))


def _deterministic_dataloader_workers() -> int:
    return 8


def _dataloader_prefetch_factor(workers: int) -> int | None:
    if workers <= 0:
        return None
    return 2


def build_training_args(
    *,
    output_dir: str,
    batch_plan: BatchPlan,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    max_grad_norm: float,
    ddp_find_unused_parameters: bool,
    deterministic_eval: bool,
    full_determinism: bool = False,
) -> TrainingArguments:
    workers = _deterministic_dataloader_workers()
    kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_plan.per_device_batch_size,
        "per_device_eval_batch_size": batch_plan.per_device_batch_size,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": warmup_ratio,
        "logging_strategy": "epoch",
        "eval_strategy": "epoch",
        "save_strategy": "no",
        "report_to": [],
        "bf16": True,
        "dataloader_num_workers": workers,
        "dataloader_prefetch_factor": _dataloader_prefetch_factor(workers),
        "dataloader_persistent_workers": False,
        "load_best_model_at_end": False,
        "seed": seed,
        "disable_tqdm": True,
        "max_grad_norm": max_grad_norm,
        "gradient_accumulation_steps": batch_plan.gradient_accumulation_steps,
        "ddp_find_unused_parameters": ddp_find_unused_parameters,
        "log_level": "error",
        "log_level_replica": "error",
    }
    if deterministic_eval:
        kwargs["data_seed"] = seed
        kwargs["full_determinism"] = full_determinism
    guarded_print(
        f"   → Trainer workers: {workers} "
        f"(prefetch={_dataloader_prefetch_factor(workers)}, deterministic_eval={deterministic_eval})"
    )
    return TrainingArguments(**kwargs)


def build_trainer(
    *,
    eval_mode: EvalMode,
    model,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    lr_config: dict,
    debug_layer_decay: int,
):
    trainer_cls = Trainer if eval_mode == EvalMode.LINEAR_PROBE else DifferentialLRTrainer
    if trainer_cls is DifferentialLRTrainer:
        llrd_models.HEAD_LR_MULT = lr_config["head_mult"]
        llrd_models.LAYER_DECAY = lr_config["layer_decay"]
        llrd_models.DEBUG_LAYER_DECAY = debug_layer_decay
    return trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
