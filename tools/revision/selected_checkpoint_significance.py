#!/usr/bin/env python3
"""Run the post-selection STRADAViT significance analysis.

The selected STRADAViT checkpoint and its ViT-MAE initialization are evaluated
under linear probing and full fine-tuning on fixed MiraBest, LoTSS DR2, and RGZ
DR1 splits. A secondary comparison evaluates the HCL-adapted DINOv2-Base(R)
checkpoint against its off-the-shelf DINOv2-Base(R) initialization. Each
model/protocol pair uses the same 15 training seeds. Dataset
processing, D4 augmentation, class weighting, model construction, batching,
optimizer settings, and Trainer configuration are imported from or matched to
the standard downstream evaluator used by OPS.

Edit the configuration block and launch with the same four-GPU execution shape
used by OPS, for example:

    torchrun --nproc_per_node=4 tools/revision/selected_checkpoint_significance.py

Completed runs are stored individually and reused on restart. Per-run outputs
retain metrics, compressed logits and probabilities, labels, predictions,
sample indices, and training histories. The final output also includes aggregate
summaries, paired tests, Holm-adjusted p-values, split manifests, and LaTeX rows
for the main paper and MiraBest appendix.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
from copy import copy
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BASELINE_MODEL_PATH = "facebook/vit-mae-base"
SELECTED_MODEL_PATH = os.getenv(
    "STRADA_SELECTED_MODEL_PATH",
    str(
        PROJECT_ROOT
        / "pretraining"
        / "Models"
        / "vitmae-b-mode=mae_then_contrastive--stage=phase2--init=mae-phase1--loss=soft_hcl--lr=5e-4--proj=mlp-L2-H2048-O128--prephase=l2_l1--phase1_loss=l2_l1--phase1_mask=0.75--regs=4--nodinoenc--views=2--patch=16--mask=0.0"
        / "checkpoints"
    ),
)
DINOV2_BASELINE_MODEL_PATH = "facebook/dinov2-with-registers-base"
DINOV2_ADAPTED_MODEL_PATH = os.getenv(
    "STRADA_DINOV2_ADAPTED_MODEL_PATH",
    str(
        PROJECT_ROOT
        / "pretraining"
        / "Models"
        / "dinov2withregisters-b-mode=contrastive_only--stage=main--init=dinov2-hf--loss=hcl--lr=1e-4--proj=mlp-L2-H2048-O128--prephase=none--regs=4--views=2"
        / "checkpoints"
    ),
)

MIRABEST_ROOT = PROJECT_ROOT / "data" / "mirabestv2"
LOTSS_ROOT = PROJECT_ROOT / "data" / "lotss_dr2_horton_hires_cutouts"
RGZ_ROOT = PROJECT_ROOT / "data" / "rgz_dr1_data" / "cutouts_pad1.50_minpix8"
OUTPUT_DIR = PROJECT_ROOT / "revision_outputs" / "selected_checkpoint_significance"
HF_CACHE_PATH = OUTPUT_DIR / "hf_cache"
RGZ_INDEX_CACHE = OUTPUT_DIR / "dataset_index_cache" / "rgz_index.pkl"

DATASET_KEYS = ("mirabest", "lotss", "rgz")
PROTOCOLS = ("linear_probe", "fine_tuning")
RUN_SEEDS = tuple(range(15))
SPLIT_SEED = 0
NUM_SPLIT_FOLDS = 3
FIXED_FOLD_INDEX = 0

EPOCHS = 20
TARGET_GLOBAL_BATCH_SIZE = 256
BACKBONE_LR = 5e-4
LINEAR_PROBE_LR = 5e-3
HEAD_LR_MULTIPLIER = 10.0
LAYERWISE_LR_DECAY = 0.65
WEIGHT_DECAY = 0.05
LINEAR_PROBE_WARMUP_RATIO = 0.05
FINE_TUNING_WARMUP_RATIO = 0.20
MAX_GRAD_NORM = 1.0
EXPECTED_WORLD_SIZE = 4

MIRABEST_CLASS_NAMES = ("FRI", "FRII")
MIRABEST_EXPECTED_TRAIN_COUNT = 729
MIRABEST_EXPECTED_TEST_COUNT = 104
MIRABEST_STANDARD_CODES = {100: 0, 102: 0, 104: 0, 200: 1, 201: 1}
MIRABEST_COMPACT_CODES = {0: 0, 1: 0, 2: 0, 5: 1, 6: 1}

MODEL_SPECS = (
    ("baseline", "ViT-MAE baseline", BASELINE_MODEL_PATH, "mae"),
    ("selected", "Selected STRADAViT", SELECTED_MODEL_PATH, "mae"),
    ("dinov2_baseline", "DINOv2-Base(R) baseline", DINOV2_BASELINE_MODEL_PATH, "dinov2"),
    ("dinov2_adapted", "Adapted DINOv2-Base(R)", DINOV2_ADAPTED_MODEL_PATH, "dinov2"),
)


@dataclass
class PreparedDataset:
    key: str
    label: str
    class_names: list[str]
    all_targets: list[int]
    train_indices: Any
    test_indices: Any
    split_description: str
    order_digest: str
    dataset_factory: Callable[[str, Any, int], Any]


@dataclass
class RuntimeDatasetBundle:
    train_dataset: Any
    test_dataset: Any
    class_weights: Any


_RUNTIME_DATASET_CACHE: dict[tuple[Any, ...], RuntimeDatasetBundle] = {}


@dataclass
class MiraBestDatasetF:
    data: Any
    targets: list[int]
    image_size: int
    transform: Any

    def __post_init__(self) -> None:
        from utils.radioastroprocessor import RadioFoundationProcessor

        self.processor = RadioFoundationProcessor(image_size=self.image_size)
        self.classes = list(MIRABEST_CLASS_NAMES)
        self.class_to_idx = {name: index for index, name in enumerate(MIRABEST_CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        tensor = self.processor.process_numpy(self.data[index], resize=True)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return {"pixel_values": tensor, "labels": int(self.targets[index])}


def is_main_process() -> bool:
    return os.getenv("RANK", "0") in ("0", "") and os.getenv("LOCAL_RANK", "0") in ("0", "")


def distributed_barrier() -> None:
    import torch.distributed as distributed

    if distributed.is_available() and distributed.is_initialized():
        distributed.barrier()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=json_fallback) + "\n")
    os.replace(temporary, path)


def json_fallback(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def validate_configuration() -> None:
    for setting_name, environment_name, configured_path in (
        ("SELECTED_MODEL_PATH", "STRADA_SELECTED_MODEL_PATH", SELECTED_MODEL_PATH),
        ("DINOV2_ADAPTED_MODEL_PATH", "STRADA_DINOV2_ADAPTED_MODEL_PATH", DINOV2_ADAPTED_MODEL_PATH),
    ):
        local_path = Path(configured_path).expanduser()
        if not local_path.is_dir():
            raise FileNotFoundError(
                f"Set {environment_name} to the corresponding local checkpoint directory before running "
                f"({setting_name} currently resolves to {local_path})."
            )
    for label, path in (
        ("MiraBest", MIRABEST_ROOT),
        ("LoTSS DR2", LOTSS_ROOT),
        ("RGZ DR1", RGZ_ROOT),
    ):
        if not Path(path).is_dir():
            raise FileNotFoundError(f"{label} dataset directory not found: {path}")
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"This comparison must use the four-GPU OPS execution shape; WORLD_SIZE={world_size}. "
            "Launch with torchrun --nproc_per_node=4."
        )


def reset_seed(seed: int) -> None:
    import numpy as np
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mirabest_batch_paths(split: str) -> list[Path]:
    directory = MIRABEST_ROOT / "F_batches"
    if not directory.is_dir():
        raise FileNotFoundError(f"MiraBest Dataset-F directory not found: {directory}")
    if split == "train":
        paths = sorted(directory.glob("data_batch_*"))
    elif split == "test":
        paths = [directory / "test_batch"]
    else:
        raise ValueError(f"Unsupported MiraBest split: {split}")
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Missing MiraBest Dataset-F {split} batches in {directory}")
    return paths


def mirabest_label_field(entry: dict[str, Any]) -> list[int]:
    candidates = [entry.get("fine_labels"), entry.get("labels")]
    for mapping in (MIRABEST_STANDARD_CODES, MIRABEST_COMPACT_CODES):
        for values in candidates:
            if values is None:
                continue
            numeric = [int(value) for value in values]
            if set(numeric).intersection(mapping):
                return numeric
    raise ValueError("Could not identify MiraBest Dataset-F labels.")


def load_mirabest_split(split: str) -> tuple[Any, list[int]]:
    import numpy as np

    arrays: list[Any] = []
    targets: list[int] = []
    for path in mirabest_batch_paths(split):
        with path.open("rb") as handle:
            entry = pickle.load(handle, encoding="latin1")
        if "data" not in entry:
            raise ValueError(f"{path} does not contain image data.")
        labels = mirabest_label_field(entry)
        data = entry["data"]
        if len(data) != len(labels):
            raise ValueError(f"Data/label length mismatch in {path}")
        for image, raw_label in zip(data, labels):
            mapped = MIRABEST_STANDARD_CODES.get(raw_label)
            if mapped is None:
                mapped = MIRABEST_COMPACT_CODES.get(raw_label)
            if mapped is not None:
                arrays.append(image[None, ...])
                targets.append(mapped)
    if not targets:
        raise RuntimeError(f"No confident FR I/FR II samples were retained from MiraBest {split}.")
    return np.vstack(arrays), targets


def validate_mirabest_split(train_targets: list[int], test_targets: list[int]) -> None:
    observed_train = dict(Counter(train_targets))
    observed_test = dict(Counter(test_targets))
    expected_train = {0: 348, 1: 381}
    expected_test = {0: 49, 1: 55}
    if len(train_targets) != MIRABEST_EXPECTED_TRAIN_COUNT or len(test_targets) != MIRABEST_EXPECTED_TEST_COUNT:
        raise RuntimeError(
            "MiraBest Dataset-F requires the fixed 729/104 partition; "
            f"found {len(train_targets)}/{len(test_targets)}."
        )
    if observed_train != expected_train or observed_test != expected_test:
        raise RuntimeError(
            "MiraBest Dataset-F class counts must be 348/381 for training and 49/55 for testing; "
            f"found {observed_train}/{observed_test}."
        )


def labels_from_dataset(dataset: Any) -> list[int]:
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]
    if hasattr(dataset, "targets"):
        return [int(label) for label in dataset.targets]
    raise TypeError(f"Cannot extract labels from {dataset.__class__.__name__}")


def sequence_digest(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def sample_order_digest(dataset: Any, labels: list[int]) -> str:
    if hasattr(dataset, "samples"):
        return sequence_digest(f"{path}|{label}" for path, label in dataset.samples)
    return sequence_digest(f"{index}|{label}" for index, label in enumerate(labels))


def fixed_fold_indices(labels: list[int]) -> tuple[Any, Any]:
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    indices = np.arange(len(labels), dtype=np.int64)
    splitter = StratifiedKFold(n_splits=NUM_SPLIT_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    folds = list(splitter.split(indices, np.asarray(labels, dtype=np.int64)))
    train_relative, test_relative = folds[FIXED_FOLD_INDEX]
    return indices[train_relative], indices[test_relative]


def prepare_mirabest() -> PreparedDataset:
    import numpy as np

    train_data, train_targets = load_mirabest_split("train")
    test_data, test_targets = load_mirabest_split("test")
    validate_mirabest_split(train_targets, test_targets)
    class_names = list(MIRABEST_CLASS_NAMES)

    def factory(split: str, transform: Any, image_size: int):
        if split == "train":
            return MiraBestDatasetF(train_data, train_targets, image_size, transform)
        if split == "test":
            return MiraBestDatasetF(test_data, test_targets, image_size, transform)
        raise ValueError(split)

    all_targets = [*train_targets, *test_targets]
    digest_values = [f"train|{index}|{label}" for index, label in enumerate(train_targets)]
    digest_values.extend(f"test|{index}|{label}" for index, label in enumerate(test_targets))
    return PreparedDataset(
        key="mirabest",
        label="MiraBest",
        class_names=class_names,
        all_targets=all_targets,
        train_indices=np.arange(len(train_targets), dtype=np.int64),
        test_indices=np.arange(len(test_targets), dtype=np.int64),
        split_description="MiraBest Confident Dataset-F fixed 729/104 train/test partition",
        order_digest=sequence_digest(digest_values),
        dataset_factory=factory,
    )


def prepare_lotss() -> PreparedDataset:
    from torch.utils.data import Subset

    from utils.strada_datasets import LoTSSDR2HortonDataset
    from utils.radioastroprocessor import RadioFoundationProcessor

    base_dataset = LoTSSDR2HortonDataset(root=str(LOTSS_ROOT), transform=None, image_size=224)
    labels = labels_from_dataset(base_dataset)
    train_indices, test_indices = fixed_fold_indices(labels)
    class_names = list(base_dataset.classes)

    def factory(split: str, transform: Any, image_size: int):
        full_dataset = copy(base_dataset)
        full_dataset.transform = transform
        full_dataset.processor = RadioFoundationProcessor(image_size=image_size)
        indices = train_indices if split == "train" else test_indices
        return Subset(full_dataset, indices)

    return PreparedDataset(
        key="lotss",
        label="LoTSS DR2",
        class_names=class_names,
        all_targets=labels,
        train_indices=train_indices,
        test_indices=test_indices,
        split_description=f"fold {FIXED_FOLD_INDEX + 1} of the seed-{SPLIT_SEED} stratified {NUM_SPLIT_FOLDS}-fold partition",
        order_digest=sample_order_digest(base_dataset, labels),
        dataset_factory=factory,
    )


def prepare_rgz() -> PreparedDataset:
    from torch.utils.data import Subset

    from utils.strada_datasets import RadioGalaxyDataset
    from utils.radioastroprocessor import RadioFoundationProcessor

    base_dataset = RadioGalaxyDataset(
        root=str(RGZ_ROOT),
        transform=None,
        image_size=224,
        index_cache_path=str(RGZ_INDEX_CACHE),
        index_num_workers=4,
    )
    labels = labels_from_dataset(base_dataset)
    train_indices, test_indices = fixed_fold_indices(labels)
    class_names = list(base_dataset.classes)

    def factory(split: str, transform: Any, image_size: int):
        full_dataset = copy(base_dataset)
        full_dataset.transform = transform
        full_dataset.processor = RadioFoundationProcessor(image_size=image_size)
        indices = train_indices if split == "train" else test_indices
        return Subset(full_dataset, indices)

    return PreparedDataset(
        key="rgz",
        label="RGZ DR1",
        class_names=class_names,
        all_targets=labels,
        train_indices=train_indices,
        test_indices=test_indices,
        split_description=f"fold {FIXED_FOLD_INDEX + 1} of the seed-{SPLIT_SEED} stratified {NUM_SPLIT_FOLDS}-fold partition",
        order_digest=sample_order_digest(base_dataset, labels),
        dataset_factory=factory,
    )


def prepare_datasets() -> dict[str, PreparedDataset]:
    builders = {
        "mirabest": prepare_mirabest,
        "lotss": prepare_lotss,
        "rgz": prepare_rgz,
    }
    unsupported = set(DATASET_KEYS).difference(builders)
    if unsupported:
        raise ValueError(f"Unsupported datasets: {sorted(unsupported)}")
    return {key: builders[key]() for key in DATASET_KEYS}


@lru_cache(maxsize=None)
def processor_config(model_path: str, encoder_family: str) -> tuple[str, tuple[float, ...], tuple[float, ...], int]:
    from transformers import AutoConfig, AutoImageProcessor

    from downstream_eval.llrd.evaluation import _resolve_checkpoint_dir
    from downstream_eval.llrd.models import _extract_image_size_from_processor, _pretrained_candidate_dirs

    resolved = _resolve_checkpoint_dir(model_path) or model_path
    local_model = os.path.isdir(resolved)
    candidates = [resolved]
    if local_model:
        candidates.extend(_pretrained_candidate_dirs(resolved))
    processor_fallback = (
        "facebook/dinov2-with-registers-base" if encoder_family == "dinov2" else "facebook/vit-mae-base"
    )
    candidates.append(processor_fallback)
    processor = None
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            try:
                processor = AutoImageProcessor.from_pretrained(candidate, use_fast=True)
            except Exception:
                processor = AutoImageProcessor.from_pretrained(candidate, use_fast=False)
            break
        except Exception:
            continue
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_size = None
    if processor is not None:
        mean = list(processor.image_mean)
        std = list(processor.image_std)
        image_size = _extract_image_size_from_processor(processor)
    if image_size is None:
        try:
            image_size = int(getattr(AutoConfig.from_pretrained(resolved), "image_size", 224) or 224)
        except Exception:
            image_size = 224
    return resolved, tuple(float(value) for value in mean), tuple(float(value) for value in std), int(image_size)


def transforms_for_model(mean: tuple[float, ...], std: tuple[float, ...]) -> tuple[Any, Any]:
    from downstream_eval.llrd.evaluation import build_image_transforms

    return build_image_transforms(mean, std)


def balanced_class_weights(targets: list[int], num_classes: int) -> Any:
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(num_classes, dtype=np.int64)
    return compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(targets, dtype=np.int64)).astype(
        np.float32
    )


def runtime_dataset_bundle(
    dataset: PreparedDataset,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    image_size: int,
) -> RuntimeDatasetBundle:
    cache_key = (dataset.key, image_size, mean, std)
    cached = _RUNTIME_DATASET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    train_transform, eval_transform = transforms_for_model(mean, std)
    bundle = RuntimeDatasetBundle(
        train_dataset=dataset.dataset_factory("train", train_transform, image_size),
        test_dataset=dataset.dataset_factory("test", eval_transform, image_size),
        class_weights=balanced_class_weights(dataset.all_targets, len(dataset.class_names)),
    )
    _RUNTIME_DATASET_CACHE[cache_key] = bundle
    if is_main_process():
        print(
            f"[{dataset.label}] prepared datasets cached for image_size={image_size}; "
            "all subsequent model/seed runs reuse them.",
            flush=True,
        )
    return bundle


def run_record_path(dataset_key: str, protocol: str, model_key: str, seed: int) -> Path:
    return OUTPUT_DIR / "runs" / dataset_key / protocol / model_key / f"seed_{seed:02d}.json"


def run_predictions_path(dataset_key: str, protocol: str, model_key: str, seed: int) -> Path:
    return OUTPUT_DIR / "runs" / dataset_key / protocol / model_key / f"seed_{seed:02d}_predictions.npz"


def run_history_path(dataset_key: str, protocol: str, model_key: str, seed: int) -> Path:
    return OUTPUT_DIR / "runs" / dataset_key / protocol / model_key / f"seed_{seed:02d}_history.json"


def evaluate_once(
    dataset: PreparedDataset,
    protocol: str,
    model_key: str,
    model_label: str,
    model_path: str,
    encoder_family: str,
    seed: int,
) -> dict[str, Any] | None:
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    from downstream_eval.llrd.models import infer_n_registers
    from downstream_eval.llrd.run_types import EvalMode
    from downstream_eval.llrd.trainer_runtime import (
        apply_linear_probe_runtime,
        build_batch_plan,
        build_classifier_model,
        build_trainer,
        build_training_args,
        should_find_unused_parameters,
    )

    record_path = run_record_path(dataset.key, protocol, model_key, seed)
    predictions_path = run_predictions_path(dataset.key, protocol, model_key, seed)
    history_path = run_history_path(dataset.key, protocol, model_key, seed)
    if record_path.is_file() and predictions_path.is_file() and history_path.is_file():
        distributed_barrier()
        return json.loads(record_path.read_text()) if is_main_process() else None
    if record_path.is_file() and is_main_process():
        print(
            f"[{dataset.label}] incomplete retained artifacts for {protocol} seed={seed:02d} "
            f"model={model_label}; rerunning once to preserve predictions and history.",
            flush=True,
        )

    reset_seed(seed)
    checkpoint_path, mean, std, image_size = processor_config(model_path, encoder_family)
    dataset_bundle = runtime_dataset_bundle(dataset, mean, std, image_size)
    train_dataset = dataset_bundle.train_dataset
    test_dataset = dataset_bundle.test_dataset
    class_weights = dataset_bundle.class_weights
    register_count = infer_n_registers(checkpoint_path, encoder_family)
    model = build_classifier_model(
        encoder_family=encoder_family,
        checkpoint_path=checkpoint_path,
        num_classes=len(dataset.class_names),
        class_weights=class_weights,
        n_registers=register_count,
    )
    eval_mode = EvalMode.LINEAR_PROBE if protocol == "linear_probe" else EvalMode.FINETUNE
    if eval_mode == EvalMode.LINEAR_PROBE:
        model = apply_linear_probe_runtime(model, len(dataset.class_names))
    learning_rate = LINEAR_PROBE_LR if eval_mode == EvalMode.LINEAR_PROBE else BACKBONE_LR
    warmup_ratio = LINEAR_PROBE_WARMUP_RATIO if eval_mode == EvalMode.LINEAR_PROBE else FINE_TUNING_WARMUP_RATIO
    lr_config = {
        "bb_lr": BACKBONE_LR,
        "head_mult": HEAD_LR_MULTIPLIER,
        "layer_decay": LAYERWISE_LR_DECAY,
    }
    batch_plan = build_batch_plan(TARGET_GLOBAL_BATCH_SIZE, encoder_family)
    run_directory = OUTPUT_DIR / "trainer_runs" / dataset.key / protocol / model_key / f"seed_{seed:02d}"
    training_args = build_training_args(
        output_dir=str(run_directory),
        batch_plan=batch_plan,
        num_epochs=EPOCHS,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=warmup_ratio,
        seed=seed,
        max_grad_norm=MAX_GRAD_NORM,
        ddp_find_unused_parameters=should_find_unused_parameters(encoder_family, register_count),
        deterministic_eval=True,
        full_determinism=False,
    )
    trainer = build_trainer(
        eval_mode=eval_mode,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        lr_config=lr_config,
        debug_layer_decay=0,
    )
    trainer.train()
    prediction_output = trainer.predict(test_dataset)
    trainer.accelerator.wait_for_everyone()
    record = None
    if trainer.is_world_process_zero():
        raw_logits = (
            prediction_output.predictions[0]
            if isinstance(prediction_output.predictions, tuple)
            else prediction_output.predictions
        )
        logits = np.asarray(raw_logits, dtype=np.float32)
        labels = np.asarray(prediction_output.label_ids, dtype=np.int64)
        predictions = np.asarray(np.argmax(logits, axis=-1), dtype=np.int64)
        shifted_logits = logits - logits.max(axis=-1, keepdims=True)
        unnormalized_probabilities = np.exp(shifted_logits)
        probabilities = np.asarray(
            unnormalized_probabilities / unnormalized_probabilities.sum(axis=-1, keepdims=True),
            dtype=np.float32,
        )
        sample_indices = np.asarray(dataset.test_indices, dtype=np.int64)
        if len(sample_indices) != len(labels):
            raise RuntimeError(
                f"Prediction/sample-index length mismatch for {dataset.label}: "
                f"{len(labels)} predictions versus {len(sample_indices)} test indices."
            )
        matrix = confusion_matrix(labels, predictions, labels=list(range(len(dataset.class_names))))
        per_class = f1_score(
            labels,
            predictions,
            average=None,
            labels=list(range(len(dataset.class_names))),
            zero_division=0,
        )
        accuracy = float(accuracy_score(labels, predictions))
        record = {
            "dataset": dataset.key,
            "dataset_label": dataset.label,
            "protocol": protocol,
            "model": model_key,
            "model_label": model_label,
            "encoder_family": encoder_family,
            "seed": seed,
            "accuracy": accuracy,
            "error": 1.0 - accuracy,
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
            "per_class_f1": [float(value) for value in per_class],
            "confusion_matrix": matrix.tolist(),
            "image_size": image_size,
            "mean": list(mean),
            "std": list(std),
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "effective_global_batch_size": batch_plan.effective_global_batch_size,
            "predictions_artifact": str(predictions_path.relative_to(OUTPUT_DIR)),
            "training_history_artifact": str(history_path.relative_to(OUTPUT_DIR)),
            "training_log_entries": len(trainer.state.log_history),
            "trainer_global_step": int(trainer.state.global_step),
            "trainer_best_metric": trainer.state.best_metric,
        }
        atomic_write_npz(
            predictions_path,
            sample_indices=sample_indices,
            labels=labels,
            predictions=predictions,
            logits=logits,
            probabilities=probabilities,
            class_names=np.asarray(dataset.class_names),
        )
        atomic_write_json(
            history_path,
            {
                "dataset": dataset.key,
                "protocol": protocol,
                "model": model_key,
                "seed": seed,
                "global_step": int(trainer.state.global_step),
                "best_metric": trainer.state.best_metric,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "log_history": trainer.state.log_history,
            },
        )
        atomic_write_json(record_path, record)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        shutil.rmtree(run_directory, ignore_errors=True)
    trainer.accelerator.wait_for_everyone()
    del prediction_output, trainer, model, train_dataset, test_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def paired_test(baseline_runs: list[dict[str, Any]], selected_runs: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    import numpy as np
    from scipy.stats import sem, t, ttest_rel

    baseline_by_seed = {int(run["seed"]): float(run[metric]) for run in baseline_runs}
    selected_by_seed = {int(run["seed"]): float(run[metric]) for run in selected_runs}
    if set(baseline_by_seed) != set(selected_by_seed):
        raise RuntimeError(f"Seed mismatch for paired {metric} test")
    seeds = sorted(baseline_by_seed)
    baseline = np.asarray([baseline_by_seed[seed] for seed in seeds], dtype=float)
    selected = np.asarray([selected_by_seed[seed] for seed in seeds], dtype=float)
    differences = selected - baseline
    mean_difference = float(differences.mean())
    if np.allclose(differences, differences[0]):
        statistic = 0.0 if np.isclose(mean_difference, 0.0) else float("inf")
        p_value = 1.0 if np.isclose(mean_difference, 0.0) else 0.0
        ci_low = ci_high = mean_difference
    else:
        statistic_raw, p_value_raw = ttest_rel(selected, baseline, alternative="two-sided")
        statistic = float(statistic_raw)
        p_value = float(p_value_raw)
        ci_low_raw, ci_high_raw = t.interval(
            0.95,
            df=len(differences) - 1,
            loc=mean_difference,
            scale=sem(differences),
        )
        ci_low = float(ci_low_raw)
        ci_high = float(ci_high_raw)
    return {
        "metric": metric,
        "test": "paired_two_sided_t_test",
        "n_pairs": len(seeds),
        "seeds": seeds,
        "statistic": statistic,
        "p_value": p_value,
        "mean_difference": mean_difference,
        "ci_level": 0.95,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    total = len(ordered)
    for rank, (key, p_value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * float(p_value))
        running_maximum = max(running_maximum, candidate)
        adjusted[key] = running_maximum
    return adjusted


def metric_summary(runs: list[dict[str, Any]], metric: str) -> dict[str, float]:
    import numpy as np

    values = np.asarray([float(run[metric]) for run in runs], dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}


def load_completed_runs(dataset_key: str, protocol: str, model_key: str) -> list[dict[str, Any]]:
    runs = []
    for seed in RUN_SEEDS:
        path = run_record_path(dataset_key, protocol, model_key, seed)
        if not path.is_file():
            raise RuntimeError(f"Missing completed run: {path}")
        runs.append(json.loads(path.read_text()))
    return runs


def format_f1(summary: dict[str, float]) -> str:
    return f"${summary['mean']:.3f}\\pm{summary['std']:.3f}$"


def format_percent(summary: dict[str, float]) -> str:
    return f"${summary['mean'] * 100:.2f}\\pm{summary['std'] * 100:.2f}\\%$"


def format_p_value(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def write_split_outputs(datasets: dict[str, PreparedDataset]) -> None:
    import numpy as np

    split_arrays: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    for key, dataset in datasets.items():
        train_indices = np.asarray(dataset.train_indices, dtype=np.int64)
        test_indices = np.asarray(dataset.test_indices, dtype=np.int64)
        split_arrays[f"{key}_train"] = train_indices
        split_arrays[f"{key}_test"] = test_indices
        all_targets = np.asarray(dataset.all_targets, dtype=np.int64)
        if key == "mirabest":
            train_targets = all_targets[: len(train_indices)]
            test_targets = all_targets[len(train_indices) :]
        else:
            train_targets = all_targets[train_indices]
            test_targets = all_targets[test_indices]
        manifest[key] = {
            "label": dataset.label,
            "split": dataset.split_description,
            "order_sha256": dataset.order_digest,
            "train_indices_sha256": sequence_digest(train_indices.tolist()),
            "test_indices_sha256": sequence_digest(test_indices.tolist()),
            "train_samples": len(train_indices),
            "test_samples": len(test_indices),
            "train_class_counts": dict(sorted(Counter(int(value) for value in train_targets).items())),
            "test_class_counts": dict(sorted(Counter(int(value) for value in test_targets).items())),
            "class_names": dataset.class_names,
        }
    np.savez_compressed(OUTPUT_DIR / "fixed_split_indices.npz", **split_arrays)
    atomic_write_json(OUTPUT_DIR / "split_manifest.json", manifest)


def aggregate_outputs(datasets: dict[str, PreparedDataset]) -> None:
    summaries: dict[str, Any] = {}
    tests: dict[str, Any] = {}
    dinov2_tests: dict[str, Any] = {}
    raw_macro_p_values: dict[str, float] = {}
    raw_dinov2_macro_p_values: dict[str, float] = {}
    for dataset_key in DATASET_KEYS:
        summaries[dataset_key] = {}
        tests[dataset_key] = {}
        dinov2_tests[dataset_key] = {}
        for protocol in PROTOCOLS:
            baseline_runs = load_completed_runs(dataset_key, protocol, "baseline")
            selected_runs = load_completed_runs(dataset_key, protocol, "selected")
            dinov2_baseline_runs = load_completed_runs(dataset_key, protocol, "dinov2_baseline")
            dinov2_adapted_runs = load_completed_runs(dataset_key, protocol, "dinov2_adapted")
            summaries[dataset_key][protocol] = {}
            for model_key, runs in (
                ("baseline", baseline_runs),
                ("selected", selected_runs),
                ("dinov2_baseline", dinov2_baseline_runs),
                ("dinov2_adapted", dinov2_adapted_runs),
            ):
                summaries[dataset_key][protocol][model_key] = {
                    metric: metric_summary(runs, metric)
                    for metric in ("accuracy", "error", "macro_f1", "weighted_f1")
                }
                per_class_count = len(datasets[dataset_key].class_names)
                summaries[dataset_key][protocol][model_key]["per_class_f1"] = [
                    metric_summary(
                        [{"value": run["per_class_f1"][class_index]} for run in runs],
                        "value",
                    )
                    for class_index in range(per_class_count)
                ]
            macro_test = paired_test(baseline_runs, selected_runs, "macro_f1")
            accuracy_test = paired_test(baseline_runs, selected_runs, "accuracy")
            tests[dataset_key][protocol] = {"macro_f1": macro_test, "accuracy": accuracy_test}
            raw_macro_p_values[f"{dataset_key}|{protocol}"] = macro_test["p_value"]
            dinov2_macro_test = paired_test(dinov2_baseline_runs, dinov2_adapted_runs, "macro_f1")
            dinov2_accuracy_test = paired_test(dinov2_baseline_runs, dinov2_adapted_runs, "accuracy")
            dinov2_tests[dataset_key][protocol] = {
                "macro_f1": dinov2_macro_test,
                "accuracy": dinov2_accuracy_test,
            }
            raw_dinov2_macro_p_values[f"{dataset_key}|{protocol}"] = dinov2_macro_test["p_value"]
    adjusted = holm_adjust(raw_macro_p_values)
    for key, adjusted_p in adjusted.items():
        dataset_key, protocol = key.split("|", 1)
        tests[dataset_key][protocol]["macro_f1"]["holm_adjusted_p_value"] = adjusted_p
    dinov2_adjusted = holm_adjust(raw_dinov2_macro_p_values)
    for key, adjusted_p in dinov2_adjusted.items():
        dataset_key, protocol = key.split("|", 1)
        dinov2_tests[dataset_key][protocol]["macro_f1"]["holm_adjusted_p_value"] = adjusted_p
    payload = {
        "protocol": {
            "datasets": list(DATASET_KEYS),
            "models": {key: {"path": path, "encoder_family": family} for key, _, path, family in MODEL_SPECS},
            "seeds": list(RUN_SEEDS),
            "evaluation_modes": list(PROTOCOLS),
            "epochs": EPOCHS,
            "target_global_batch_size": TARGET_GLOBAL_BATCH_SIZE,
            "backbone_lr": BACKBONE_LR,
            "linear_probe_lr": LINEAR_PROBE_LR,
            "head_lr_multiplier": HEAD_LR_MULTIPLIER,
            "layerwise_lr_decay": LAYERWISE_LR_DECAY,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio_lp": LINEAR_PROBE_WARMUP_RATIO,
            "warmup_ratio_ft": FINE_TUNING_WARMUP_RATIO,
            "primary_endpoint": "macro_f1",
            "multiplicity_correction": "Holm within each six-test comparison family",
        },
        "summaries": summaries,
        "paired_tests": tests,
        "dinov2_paired_tests": dinov2_tests,
    }
    atomic_write_json(OUTPUT_DIR / "significance_results.json", payload)
    atomic_write_json(OUTPUT_DIR / "paired_tests.json", tests)
    atomic_write_json(OUTPUT_DIR / "dinov2_paired_tests.json", dinov2_tests)
    write_csv_outputs(datasets, summaries, tests)
    write_dinov2_csv(datasets, summaries, dinov2_tests)
    write_main_table_rows(datasets, summaries, tests)
    write_dinov2_table_rows(datasets, summaries, dinov2_tests)
    write_mirabest_appendix_rows(datasets, summaries, tests)


def write_csv_outputs(datasets: dict[str, PreparedDataset], summaries: dict[str, Any], tests: dict[str, Any]) -> None:
    run_fields = [
        "dataset",
        "protocol",
        "model",
        "seed",
        "accuracy",
        "error",
        "macro_f1",
        "weighted_f1",
        "per_class_f1",
    ]
    with (OUTPUT_DIR / "per_run_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        for dataset_key in DATASET_KEYS:
            for protocol in PROTOCOLS:
                for model_key, _, _, _ in MODEL_SPECS:
                    for run in load_completed_runs(dataset_key, protocol, model_key):
                        writer.writerow(
                            {
                                **{field: run[field] for field in run_fields if field != "per_class_f1"},
                                "per_class_f1": json.dumps(run["per_class_f1"]),
                            }
                        )

    summary_fields = [
        "dataset",
        "protocol",
        "baseline_macro_f1_mean",
        "baseline_macro_f1_std",
        "selected_macro_f1_mean",
        "selected_macro_f1_std",
        "paired_delta",
        "ci_low",
        "ci_high",
        "p_value",
        "holm_adjusted_p_value",
    ]
    with (OUTPUT_DIR / "main_significance_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for dataset_key in DATASET_KEYS:
            for protocol in PROTOCOLS:
                baseline = summaries[dataset_key][protocol]["baseline"]["macro_f1"]
                selected = summaries[dataset_key][protocol]["selected"]["macro_f1"]
                test = tests[dataset_key][protocol]["macro_f1"]
                writer.writerow(
                    {
                        "dataset": datasets[dataset_key].label,
                        "protocol": protocol,
                        "baseline_macro_f1_mean": baseline["mean"],
                        "baseline_macro_f1_std": baseline["std"],
                        "selected_macro_f1_mean": selected["mean"],
                        "selected_macro_f1_std": selected["std"],
                        "paired_delta": test["mean_difference"],
                        "ci_low": test["ci_low"],
                        "ci_high": test["ci_high"],
                        "p_value": test["p_value"],
                        "holm_adjusted_p_value": test["holm_adjusted_p_value"],
                    }
                )


def write_dinov2_csv(datasets: dict[str, PreparedDataset], summaries: dict[str, Any], tests: dict[str, Any]) -> None:
    fields = [
        "dataset",
        "protocol",
        "baseline_macro_f1_mean",
        "baseline_macro_f1_std",
        "adapted_macro_f1_mean",
        "adapted_macro_f1_std",
        "paired_delta",
        "ci_low",
        "ci_high",
        "p_value",
        "holm_adjusted_p_value",
    ]
    with (OUTPUT_DIR / "dinov2_significance_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset_key in DATASET_KEYS:
            for protocol in PROTOCOLS:
                baseline = summaries[dataset_key][protocol]["dinov2_baseline"]["macro_f1"]
                adapted = summaries[dataset_key][protocol]["dinov2_adapted"]["macro_f1"]
                test = tests[dataset_key][protocol]["macro_f1"]
                writer.writerow(
                    {
                        "dataset": datasets[dataset_key].label,
                        "protocol": protocol,
                        "baseline_macro_f1_mean": baseline["mean"],
                        "baseline_macro_f1_std": baseline["std"],
                        "adapted_macro_f1_mean": adapted["mean"],
                        "adapted_macro_f1_std": adapted["std"],
                        "paired_delta": test["mean_difference"],
                        "ci_low": test["ci_low"],
                        "ci_high": test["ci_high"],
                        "p_value": test["p_value"],
                        "holm_adjusted_p_value": test["holm_adjusted_p_value"],
                    }
                )


def write_main_table_rows(datasets: dict[str, PreparedDataset], summaries: dict[str, Any], tests: dict[str, Any]) -> None:
    rows = []
    for dataset_key in DATASET_KEYS:
        for protocol in PROTOCOLS:
            baseline = summaries[dataset_key][protocol]["baseline"]["macro_f1"]
            selected = summaries[dataset_key][protocol]["selected"]["macro_f1"]
            test = tests[dataset_key][protocol]["macro_f1"]
            protocol_label = "LP" if protocol == "linear_probe" else "FT"
            rows.append(
                f"{datasets[dataset_key].label} & {protocol_label}"
                f" & {format_f1(baseline)} & {format_f1(selected)}"
                f" & ${test['mean_difference']:+.3f}$ [{test['ci_low']:+.3f}, {test['ci_high']:+.3f}]"
                f" & ${format_p_value(test['p_value'])}$"
                f" & ${format_p_value(test['holm_adjusted_p_value'])}$ \\\\"
            )
    (OUTPUT_DIR / "main_significance_table_rows.tex").write_text("\n".join(rows) + "\n")


def write_dinov2_table_rows(datasets: dict[str, PreparedDataset], summaries: dict[str, Any], tests: dict[str, Any]) -> None:
    rows = []
    for dataset_key in DATASET_KEYS:
        for protocol in PROTOCOLS:
            baseline = summaries[dataset_key][protocol]["dinov2_baseline"]["macro_f1"]
            adapted = summaries[dataset_key][protocol]["dinov2_adapted"]["macro_f1"]
            test = tests[dataset_key][protocol]["macro_f1"]
            protocol_label = "LP" if protocol == "linear_probe" else "FT"
            rows.append(
                f"{datasets[dataset_key].label} & {protocol_label}"
                f" & {format_f1(baseline)} & {format_f1(adapted)}"
                f" & ${test['mean_difference']:+.3f}$ [{test['ci_low']:+.3f}, {test['ci_high']:+.3f}]"
                f" & ${format_p_value(test['p_value'])}$"
                f" & ${format_p_value(test['holm_adjusted_p_value'])}$ \\\\"
            )
    (OUTPUT_DIR / "dinov2_significance_table_rows.tex").write_text("\n".join(rows) + "\n")


def write_mirabest_appendix_rows(
    datasets: dict[str, PreparedDataset], summaries: dict[str, Any], tests: dict[str, Any]
) -> None:
    rows = []
    dataset_key = "mirabest"
    for protocol in PROTOCOLS:
        protocol_label = "LP" if protocol == "linear_probe" else "FT"
        macro_test = tests[dataset_key][protocol]["macro_f1"]
        for model_key, model_label in (("baseline", "ViT-MAE baseline"), ("selected", "Selected STRADAViT")):
            model_summary = summaries[dataset_key][protocol][model_key]
            class_summaries = model_summary["per_class_f1"]
            if model_key == "baseline":
                delta_cell = r"\textemdash"
                test_cell = r"\textemdash"
            else:
                delta_cell = (
                    f"${macro_test['mean_difference']:+.3f}$ "
                    f"[{macro_test['ci_low']:+.3f}, {macro_test['ci_high']:+.3f}]"
                )
                test_cell = (
                    f"$t={macro_test['statistic']:.2f};\\,p={format_p_value(macro_test['p_value'])}$"
                )
            rows.append(
                f"{protocol_label} & {model_label}"
                f" & {format_percent(model_summary['accuracy'])}"
                f" & {format_percent(model_summary['error'])}"
                f" & {format_f1(model_summary['macro_f1'])}"
                f" & {format_f1(model_summary['weighted_f1'])}"
                f" & {format_f1(class_summaries[0])} / {format_f1(class_summaries[1])}"
                f" & {delta_cell} & {test_cell} \\\\"
            )
    (OUTPUT_DIR / "mirabest_appendix_table_rows.tex").write_text("\n".join(rows) + "\n")


def write_run_manifest(datasets: dict[str, PreparedDataset]) -> None:
    manifest = {
        "dataset_order": list(DATASET_KEYS),
        "protocol_order": list(PROTOCOLS),
        "model_order": [key for key, _, _, _ in MODEL_SPECS],
        "seeds": list(RUN_SEEDS),
        "total_runs": len(DATASET_KEYS) * len(PROTOCOLS) * len(MODEL_SPECS) * len(RUN_SEEDS),
        "splits": {key: dataset.split_description for key, dataset in datasets.items()},
        "retained_per_run": {
            "summary": "seed_XX.json",
            "predictions": "seed_XX_predictions.npz",
            "training_history": "seed_XX_history.json",
        },
    }
    atomic_write_json(OUTPUT_DIR / "run_manifest.json", manifest)


def main() -> None:
    from downstream_eval.llrd.runtime import configure_runtime_environment

    validate_configuration()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_runtime_environment(str(HF_CACHE_PATH), deterministic_eval=False)
    datasets = prepare_datasets()
    if is_main_process():
        write_split_outputs(datasets)
        write_run_manifest(datasets)
    distributed_barrier()
    for dataset_key in DATASET_KEYS:
        dataset = datasets[dataset_key]
        for protocol in PROTOCOLS:
            for seed in RUN_SEEDS:
                for model_key, model_label, model_path, encoder_family in MODEL_SPECS:
                    if is_main_process():
                        print(f"[{dataset.label}] {protocol} seed={seed:02d} model={model_label}", flush=True)
                    evaluate_once(dataset, protocol, model_key, model_label, model_path, encoder_family, seed)
    distributed_barrier()
    if is_main_process():
        aggregate_outputs(datasets)
        print(f"Completed {len(DATASET_KEYS) * len(PROTOCOLS) * len(MODEL_SPECS) * len(RUN_SEEDS)} runs.")


if __name__ == "__main__":
    main()
