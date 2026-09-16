#!/usr/bin/env python3
"""Generate publication-ready UMAP panels from frozen STRADAViT encoders.

Edit the configuration block below, then run this file directly.  The tool is
deliberately separate from OPS and downstream training: it never trains a
classifier and does not alter evaluation caches.  It applies the same ZScale,
channel replication, resize, and checkpoint normalization used by downstream
evaluation before extracting frozen encoder embeddings.

For the paper diagnostic, the same fixed, class-balanced subset of each dataset
is used for every model. The script saves the selected sample indices,
embeddings, UMAP coordinates, and plotting metadata alongside the figure so
the panels can be reproduced exactly.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Configuration -- edit these values; this script intentionally has no CLI.
# ---------------------------------------------------------------------------

DATASET_KEYS: tuple[Literal["rgz", "mirabest", "lotss_dr2"], ...] = ("rgz", "mirabest", "lotss_dr2")
DATASET_ROOT_OVERRIDES: dict[str, Path | None] = {
    "rgz": None,
    "mirabest": None,
    "lotss_dr2": None,
}
SAMPLE_FRACTIONS = {
    "rgz": 1.0,
    "mirabest": 1.0,
    "lotss_dr2": 1.0,
}
OUTPUT_DIR = PROJECT_ROOT / "revision_outputs" / "umap_all_datasets"
# OPS training runs are written beneath pretraining/Models on the cluster.
# Change this only when the checkpoints have been exported elsewhere.
MODEL_ROOT = PROJECT_ROOT / "pretraining" / "Models"

# Set this after the first complete run to regenerate only the PNG/PDF from
# umap_data.npz, without loading models, extracting embeddings, or fitting UMAP.
PLOT_ONLY = False

BALANCE_PER_CLASS = True
MAX_SAMPLES_PER_CLASS: int | None = None
SUBSET_SEED = 2026

# Set this to False when comparing the raw pooled embeddings used by the linear
# probe.  Cosine UMAP is retained as the visual-distance metric either way.
L2_NORMALIZE_EMBEDDINGS = False
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.10
UMAP_METRIC = "cosine"
UMAP_SEED = 2026

# Use four entries for the manuscript's model columns. Each checkpoint may be a
# local path or an HF identifier.  Update these paths to the exact checkpoints
# selected for the revised manuscript before running the script.
@dataclass(frozen=True)
class ModelPanel:
    label: str
    checkpoint_path: str
    encoder_family: Literal["mae", "dinov2", "auto"] = "auto"


MODEL_PANELS = (
    ModelPanel(
        "ViT-MAE baseline\nImageNet initialization",
        "facebook/vit-mae-base",
        "mae",
    ),
    ModelPanel(
        "Reconstruction-only\nL2+L1, R=4",
        str(MODEL_ROOT / "vitmae-b-mode=mae_then_contrastive--stage=phase1--init=facebook-hf--loss=l2_l1--lr=5e-4--proj=none--regs=4--nodinoenc--patch=16--mask=0.75" / "checkpoints"),
        "mae",
    ),
    ModelPanel(
        "Contrastive-only\nSoft-HCL, R=4",
        str(MODEL_ROOT / "vitmae-b-mode=contrastive_only--stage=main--init=facebook-hf--loss=soft_hcl--lr=5e-4--proj=mlp-L2-H2048-O128--prephase=none--regs=4--nodinoenc--views=2--patch=16--mask=0.0" / "checkpoints"),
        "mae",
    ),
    ModelPanel(
        "Two-stage\nL2+L1 → Soft-HCL, R=4",
        str(MODEL_ROOT / "vitmae-b-mode=mae_then_contrastive--stage=phase2--init=mae-phase1--loss=soft_hcl--lr=5e-4--proj=mlp-L2-H2048-O128--prephase=l2_l1--phase1_loss=l2_l1--phase1_mask=0.75--regs=4--nodinoenc--views=2--patch=16--mask=0.0" / "checkpoints"),
        "mae",
    ),
)

EXTRACTION_BATCH_SIZE = 64
NUM_WORKERS = 8
FIGURE_BASENAME = "representation_umap_panels"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    root: Path
    dataset_class: object
    class_names: tuple[str, ...]
    sample_fraction: float


def configure_matplotlib() -> None:
    from downstream_eval.llrd.settings import apply_matplotlib_style
    import matplotlib as mpl

    apply_matplotlib_style()
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 450,
            "savefig.dpi": 450,
        }
    )


def dataset_specs() -> tuple[DatasetSpec, ...]:
    from utils.strada_datasets import LoTSSDR2HortonDataset, MiraBestDatasetV2, RadioGalaxyDataset

    roots = {
        "rgz": PROJECT_ROOT / "data" / "rgz_dr1_data" / "cutouts_pad1.50_minpix8",
        "mirabest": PROJECT_ROOT / "data" / "mirabestv2",
        "lotss_dr2": PROJECT_ROOT / "data" / "lotss_dr2_horton_hires_cutouts",
    }
    defaults = {
        "rgz": DatasetSpec(
            key="rgz",
            label="RGZ DR1",
            root=DATASET_ROOT_OVERRIDES["rgz"] or roots["rgz"],
            dataset_class=RadioGalaxyDataset,
            class_names=("1C1P", "1C2P", "1C3P", "2C2P", "2C3P", "3C3P"),
            sample_fraction=SAMPLE_FRACTIONS["rgz"],
        ),
        "mirabest": DatasetSpec(
            key="mirabest",
            label="MiraBest",
            root=DATASET_ROOT_OVERRIDES["mirabest"] or roots["mirabest"],
            dataset_class=MiraBestDatasetV2,
            class_names=("FR I", "FR II"),
            sample_fraction=SAMPLE_FRACTIONS["mirabest"],
        ),
        "lotss_dr2": DatasetSpec(
            key="lotss_dr2",
            label="LoTSS DR2",
            root=DATASET_ROOT_OVERRIDES["lotss_dr2"] or roots["lotss_dr2"],
            dataset_class=LoTSSDR2HortonDataset,
            class_names=("FR I", "FR II", "Hybrid", "Relaxed double", "Spiral"),
            sample_fraction=SAMPLE_FRACTIONS["lotss_dr2"],
        ),
    }
    return tuple(defaults[key] for key in DATASET_KEYS)


def labels_from_dataset(dataset) -> list[int]:
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]
    if hasattr(dataset, "targets"):
        return [int(label) for label in dataset.targets]
    raise TypeError("Dataset must expose either samples or targets for deterministic UMAP sampling.")


def build_base_dataset(spec: DatasetSpec, image_size: int, transform=None):
    kwargs = {"root": str(spec.root), "transform": transform, "image_size": image_size}
    if spec.dataset_class.__name__ == "RadioGalaxyDataset":
        kwargs["index_cache_path"] = str(OUTPUT_DIR / f"{spec.key}_index.pkl")
        kwargs["index_num_workers"] = 4
    return spec.dataset_class(**kwargs)


def select_indices(labels: list[int], spec: DatasetSpec) -> list[int]:
    import numpy as np

    labels_array = np.asarray(labels, dtype=np.int64)
    pool = np.arange(len(labels_array))

    rng = np.random.default_rng(SUBSET_SEED)
    by_class = {class_id: pool[labels_array[pool] == class_id] for class_id in sorted(set(labels_array[pool]))}
    desired = {
        class_id: max(1, int(round(len(indices) * spec.sample_fraction)))
        for class_id, indices in by_class.items()
    }
    if BALANCE_PER_CLASS:
        common = min(desired.values())
        desired = {class_id: common for class_id in desired}
    if MAX_SAMPLES_PER_CLASS is not None:
        desired = {class_id: min(count, MAX_SAMPLES_PER_CLASS) for class_id, count in desired.items()}

    selected = []
    for class_id, indices in by_class.items():
        take = min(desired[class_id], len(indices))
        selected.extend(rng.choice(indices, size=take, replace=False).tolist())
    return sorted(int(index) for index in selected)


def summarize_selection(labels: list[int], selected_indices: list[int], spec: DatasetSpec) -> dict:
    import numpy as np

    labels_array = np.asarray(labels, dtype=np.int64)
    selected_labels = labels_array[np.asarray(selected_indices, dtype=np.int64)]
    per_class = {}
    for class_id, class_name in enumerate(spec.class_names):
        available = int((labels_array == class_id).sum())
        fraction_target = max(1, int(round(available * spec.sample_fraction))) if available else 0
        per_class[class_name] = {
            "available": available,
            "fraction_target": fraction_target,
            "selected": int((selected_labels == class_id).sum()),
        }
    return {
        "available_samples": int(labels_array.size),
        "selected_samples": len(selected_indices),
        "per_class": per_class,
    }


def print_selection_summary(summary: dict, spec: DatasetSpec) -> None:
    target_label = f"{spec.sample_fraction:.0%}-target"
    print(
        f"[UMAP][Sampling] dataset={spec.label} available={summary['available_samples']:,} "
        f"fraction_per_class={spec.sample_fraction:.0%} balanced={BALANCE_PER_CLASS}",
        flush=True,
    )
    print(f"[UMAP][Sampling] class  available  {target_label:>10}  selected", flush=True)
    for class_name, counts in summary["per_class"].items():
        print(
            f"[UMAP][Sampling] {class_name:<5}  {counts['available']:>9,}  "
            f"{counts['fraction_target']:>10,}  {counts['selected']:>8,}",
            flush=True,
        )
    print(f"[UMAP][Sampling] selected_total={summary['selected_samples']:,}", flush=True)


def infer_encoder_family(panel: ModelPanel) -> str:
    if panel.encoder_family != "auto":
        return panel.encoder_family
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(panel.checkpoint_path)
    return "dinov2" if "dino" in str(getattr(config, "model_type", "")).lower() else "mae"


def model_input_config(checkpoint_path: str, encoder_family: str) -> tuple[list[float], list[float], int]:
    from downstream_eval.llrd.models import _extract_image_size_from_processor
    from transformers import AutoConfig, AutoImageProcessor

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_size: int | None = None
    try:
        processor = AutoImageProcessor.from_pretrained(checkpoint_path, use_fast=False)
        mean = list(processor.image_mean)
        std = list(processor.image_std)
        image_size = _extract_image_size_from_processor(processor)
    except Exception:
        pass
    if image_size is None:
        try:
            image_size = int(getattr(AutoConfig.from_pretrained(checkpoint_path), "image_size", 0) or 0)
        except Exception:
            image_size = 0
    if not image_size:
        image_size = 518 if encoder_family == "dinov2" else 224
    return mean, std, int(image_size)


def load_encoder(panel: ModelPanel, encoder_family: str):
    import torch

    if encoder_family == "mae":
        from pretraining.stradavit_model import StradaViTModel

        encoder = StradaViTModel.from_pretrained(panel.checkpoint_path)

        def extract(pixel_values: torch.Tensor) -> torch.Tensor:
            return encoder(pixel_values=pixel_values).embedding

    elif encoder_family == "dinov2":
        from downstream_eval.llrd.models import DINOv2PoolerClassifier

        encoder = DINOv2PoolerClassifier._load_backbone(panel.checkpoint_path)

        def extract(pixel_values: torch.Tensor) -> torch.Tensor:
            outputs = encoder(pixel_values=pixel_values)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                raise RuntimeError("DINOv2 backbone did not return pooler_output.")
            return pooled

    else:
        raise ValueError(f"Unsupported encoder family: {encoder_family}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()
    return encoder, extract, device


def extract_embeddings(panel: ModelPanel, selected_indices: list[int], spec: DatasetSpec):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    family = infer_encoder_family(panel)
    mean, std, image_size = model_input_config(panel.checkpoint_path, family)
    evaluation_transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Lambda(lambda tensor: tensor.expand(3, -1, -1) if tensor.shape[0] == 1 else tensor),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    dataset = build_base_dataset(spec, image_size=image_size, transform=evaluation_transform)
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=EXTRACTION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )
    encoder, extract, device = load_encoder(panel, family)
    features, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            embedding = extract(pixels)
            features.append(embedding.detach().cpu().float().numpy())
            labels.append(batch["labels"].cpu().numpy())
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(features), np.concatenate(labels), {"encoder_family": family, "image_size": image_size, "mean": mean, "std": std}


def run_umap(embeddings):
    import numpy as np

    try:
        from umap import UMAP
    except ModuleNotFoundError as exc:
        raise RuntimeError("UMAP requires umap-learn; install the project requirements before running this script.") from exc
    values = np.asarray(embeddings, dtype=np.float32)
    if L2_NORMALIZE_EMBEDDINGS:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.maximum(norms, 1e-12)
    reducer = UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_SEED,
        transform_seed=UMAP_SEED,
    )
    return reducer.fit_transform(values)


def class_colours(class_count: int) -> list[str]:
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999"]
    if class_count > len(palette):
        raise ValueError(f"No fixed paper palette configured for {class_count} classes.")
    return palette[:class_count]


def load_saved_plot_inputs(specs: tuple[DatasetSpec, ...]):
    import numpy as np

    archive_path = OUTPUT_DIR / "umap_data.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Plot-only mode requires existing UMAP data: {archive_path}")
    coordinates_by_dataset = {}
    labels_by_dataset = {}
    with np.load(archive_path) as archive:
        for spec in specs:
            labels_key = f"labels_{spec.key}"
            if labels_key not in archive:
                raise KeyError(f"Saved UMAP data do not contain {labels_key!r}; run once with PLOT_ONLY=False.")
            labels_by_dataset[spec.key] = archive[labels_key].copy()
            coordinates_by_dataset[spec.key] = []
            for panel_index, panel in enumerate(MODEL_PANELS):
                key = f"coordinates_{spec.key}_panel_{panel_index}"
                if key not in archive:
                    raise KeyError(f"Saved UMAP data do not contain {key!r}; run once with PLOT_ONLY=False.")
                coordinates_by_dataset[spec.key].append((panel, archive[key].copy()))
    return coordinates_by_dataset, labels_by_dataset


def plot_panels(coordinates_by_dataset, labels_by_dataset, specs: tuple[DatasetSpec, ...]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    configure_matplotlib()
    model_count = len(MODEL_PANELS)
    row_count = len(specs)
    figure = plt.figure(figsize=(7.25, 6.6))
    grid = figure.add_gridspec(
        row_count * 2,
        model_count,
        height_ratios=[value for _ in specs for value in (1.0, 0.12)],
        hspace=0.18,
        wspace=0.18,
    )
    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    for row_index, spec in enumerate(specs):
        labels = labels_by_dataset[spec.key]
        colours = class_colours(len(spec.class_names))
        for column_index, (panel, coordinates) in enumerate(coordinates_by_dataset[spec.key]):
            axis = figure.add_subplot(grid[row_index * 2, column_index])
            for class_id, colour in enumerate(colours):
                mask = labels == class_id
                axis.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    s=1.5,
                    alpha=0.52,
                    linewidths=0.0,
                    color=colour,
                    rasterized=True,
                )
            if row_index == 0:
                axis.set_title(panel.label, pad=5, fontsize=8)
            if row_index == row_count - 1:
                axis.set_xlabel("UMAP 1", fontsize=8)
            if column_index == 0:
                axis.set_ylabel("UMAP 2", fontsize=8)
            axis.tick_params(labelbottom=False, labelleft=False, length=3)
            axis.minorticks_off()
            axis.set_box_aspect(1)
            panel_index = row_index * model_count + column_index
            axis.text(
                0.03,
                0.97,
                f"({panel_letters[panel_index]})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                fontweight="bold",
            )
        legend_axis = figure.add_subplot(grid[row_index * 2 + 1, :])
        legend_axis.axis("off")
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colour, markersize=4, label=class_name)
            for class_name, colour in zip(spec.class_names, colours)
        ]
        legend_axis.text(0.01, 0.5, spec.label, ha="left", va="center", fontsize=8, fontweight="bold")
        legend_axis.legend(
            handles=handles,
            loc="center",
            bbox_to_anchor=(0.58, 0.5),
            ncol=len(handles),
            fontsize=7,
            frameon=True,
            borderpad=0.25,
            handletextpad=0.35,
            columnspacing=0.8,
        )
    figure.subplots_adjust(left=0.06, right=0.995, top=0.91, bottom=0.03)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"{FIGURE_BASENAME}.png", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{FIGURE_BASENAME}.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    import numpy as np
    import torch

    random.seed(SUBSET_SEED)
    np.random.seed(SUBSET_SEED)
    torch.manual_seed(SUBSET_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SUBSET_SEED)
    if not MODEL_PANELS:
        raise ValueError("MODEL_PANELS must contain at least one checkpoint.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = dataset_specs()
    for spec in specs:
        if not 0 < spec.sample_fraction <= 1:
            raise ValueError(f"Sample fraction for {spec.key} must lie in (0, 1].")
    if PLOT_ONLY:
        coordinates_by_dataset, labels_by_dataset = load_saved_plot_inputs(specs)
        plot_panels(coordinates_by_dataset, labels_by_dataset, specs)
        print(f"Regenerated UMAP figure from saved coordinates in {OUTPUT_DIR}")
        return

    dataset_states = {}
    arrays = {}
    for spec in specs:
        base_dataset = build_base_dataset(spec, image_size=224, transform=None)
        source_labels = labels_from_dataset(base_dataset)
        selected_indices = select_indices(source_labels, spec)
        selection_summary = summarize_selection(source_labels, selected_indices, spec)
        print_selection_summary(selection_summary, spec)
        np.save(OUTPUT_DIR / f"selected_indices_{spec.key}.npy", np.asarray(selected_indices, dtype=np.int64))
        dataset_states[spec.key] = {
            "selected_indices": selected_indices,
            "selection_summary": selection_summary,
            "coordinates": [],
            "labels": None,
            "panel_metadata": [],
        }

    for spec in specs:
        state = dataset_states[spec.key]
        for panel_index, panel in enumerate(MODEL_PANELS):
            print(f"[UMAP] dataset={spec.label} panel={panel.label.replace(chr(10), ' | ')}", flush=True)
            embeddings, labels, model_metadata = extract_embeddings(panel, state["selected_indices"], spec)
            if state["labels"] is None:
                state["labels"] = labels
            elif not np.array_equal(state["labels"], labels):
                raise RuntimeError(f"Model panels did not preserve sample order for {spec.label}.")
            coordinates = run_umap(embeddings)
            state["coordinates"].append((panel, coordinates))
            state["panel_metadata"].append(
                {**asdict(panel), **model_metadata, "embedding_dimension": int(embeddings.shape[1])}
            )
            arrays[f"embeddings_{spec.key}_panel_{panel_index}"] = embeddings
            arrays[f"coordinates_{spec.key}_panel_{panel_index}"] = coordinates
        arrays[f"labels_{spec.key}"] = state["labels"]
        arrays[f"indices_{spec.key}"] = np.asarray(state["selected_indices"], dtype=np.int64)

    np.savez_compressed(OUTPUT_DIR / "umap_data.npz", **arrays)
    coordinates_by_dataset = {spec.key: dataset_states[spec.key]["coordinates"] for spec in specs}
    labels_by_dataset = {spec.key: dataset_states[spec.key]["labels"] for spec in specs}
    plot_panels(coordinates_by_dataset, labels_by_dataset, specs)
    metadata = {
        "datasets": [
            {
                "key": spec.key,
                "label": spec.label,
                "root": str(spec.root),
                "sampling": {
                    "sample_fraction_per_class": spec.sample_fraction,
                    "balance_per_class": BALANCE_PER_CLASS,
                    "max_samples_per_class": MAX_SAMPLES_PER_CLASS,
                    "subset_seed": SUBSET_SEED,
                    **dataset_states[spec.key]["selection_summary"],
                },
            }
            for spec in specs
        ],
        "umap": {
            "n_neighbors": UMAP_N_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "metric": UMAP_METRIC,
            "seed": UMAP_SEED,
            "l2_normalize_embeddings": L2_NORMALIZE_EMBEDDINGS,
            "fitted_independently_per_panel": True,
        },
        "panels_by_dataset": {spec.key: dataset_states[spec.key]["panel_metadata"] for spec in specs},
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote UMAP figure and reproducibility artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
