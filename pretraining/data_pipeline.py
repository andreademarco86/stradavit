import os
import random
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch
# Prefer TF32 on Ampere/Ada for faster GEMMs while retaining fp32 API numerics
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
import torchvision.transforms.v2 as tv2
import torchvision.transforms.v2.functional as F
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image
from transformers import AutoImageProcessor, ViTImageProcessor
from transformers.image_processing_base import ImageProcessingMixin

from pretraining.dataset_registry import dataset_alias_slug
from utils.strada_datasets import BinaryFileDataset, DEFAULT_MEGA_CHUNK_BYTES, RECORD_BYTES

class UninformativeCutoutError(RuntimeError):
    """Raised by transforms that choose to skip an uninformative cutout."""


@dataclass(frozen=True)
class DataPolicy:
    """Controls optional data-path behavior shared across SSL transforms."""

    use_roi_crop: bool = True
    enable_augmentations: bool = True

    def run_tokens(self) -> list[str]:
        tokens: list[str] = []
        if not self.use_roi_crop:
            tokens.append("noroi")
        if not self.enable_augmentations:
            tokens.append("noaug")
        return tokens

    def manifest_fields(self) -> dict[str, object]:
        return {
            "use_roi_crop": bool(self.use_roi_crop),
            "enable_augmentations": bool(self.enable_augmentations),
            "mae_crop_mode": ("roi_crop" if bool(self.use_roi_crop) else "fullcutout"),
            "mae_aug_mode": ("enabled" if bool(self.enable_augmentations) else "disabled"),
            "contrastive_crop_mode": ("roi_crop" if bool(self.use_roi_crop) else "fullcutout"),
            "contrastive_aug_mode": ("full" if bool(self.enable_augmentations) else "first_raw_rest_aug"),
        }


class BinaryDatasetMode(Enum):
    DEFAULT = "default"
    CURATED = "curated"

DEFAULT_BINARY_BLOCK_SIZE = 2048

@dataclass(frozen=True)
class BinaryDatasetConfig:
    mode: BinaryDatasetMode = BinaryDatasetMode.DEFAULT
    block_size: int = DEFAULT_BINARY_BLOCK_SIZE
    bin_path_override: str | None = None
    label: str | None = None

    @property
    def bin_path(self) -> str:
        if self.bin_path_override:
            return str(self.bin_path_override)
        raise ValueError(
            f"Binary dataset path is not configured for mode={self.mode.value!r}; "
            "set BinaryDatasetConfig(bin_path_override=...) explicitly."
        )

    def run_tokens(self) -> list[str]:
        label_slug = dataset_alias_slug(str(self.label or self.mode.value), str(self.bin_path_override or ""))
        if label_slug == BinaryDatasetMode.DEFAULT.value:
            return []
        return [f"data={label_slug}"]

    def manifest_fields(self) -> dict[str, object]:
        return {
            "dataset_mode": self.mode.value.lower(),
            "dataset_label": str(self.label or self.mode.value),
            "dataset_bin_path": self.bin_path,
            "dataset_block_size": int(self.block_size),
        }


def resolve_binary_block_size(bin_path: str, configured_block_size: int) -> tuple[int, str, int]:
    """Return the configured binary block size.

    Mega-chunk training provides the large-file locality layer. Block size stays
    under the recipe/config so large files behave like smaller files inside each
    active mega-chunk.
    """
    configured = int(configured_block_size)
    file_bytes = int(Path(bin_path).stat().st_size)
    return configured, "configured", file_bytes


def resolve_binary_dataloader_workers(bin_path: str, configured_workers: int) -> tuple[int, str, int]:
    """Return the configured per-rank loader worker count for binary memmaps."""
    configured = max(0, int(configured_workers))
    file_bytes = int(Path(bin_path).stat().st_size)
    return configured, "configured", file_bytes


@dataclass(frozen=True)
class DataInputConfig:
    dataset: BinaryDatasetConfig = field(default_factory=BinaryDatasetConfig)
    policy: DataPolicy = field(default_factory=DataPolicy)


@dataclass
class DataBundle:
    processor: ImageProcessingMixin
    transform: Callable[[object], object]
    train_dataset: torch.utils.data.Dataset
    preview_dataset: torch.utils.data.Dataset | None
    mean: list[float]
    std: list[float]

class NonEmptyResamplingDataset(torch.utils.data.Dataset):
    """
    Wrapper around an existing Dataset that rejects "effectively empty"
    samples and resamples a different index instead.

    This keeps the Trainer's batch size and loss computation completely
    standard, while implementing an online policy to down-weight empty
    cutouts without touching the underlying BinaryFileDataset/AugmentedRepeatDataset.
    """
    def __init__(self, base_dataset, max_resamples: int = 5):
        super().__init__()
        self.base = base_dataset
        self.max_resamples = int(max(1, max_resamples))

    def __len__(self):
        return len(self.base)

    def _extract_pixel_values(self, sample):
        if sample is None or not hasattr(sample, "get"):
            return None
        pv = sample.get("pixel_values")
        if isinstance(pv, dict):
            pv = pv.get("pixel_values", pv)
        # Multi-view case: return all views so callers can enforce strict policies.
        if isinstance(pv, (list, tuple)):
            if len(pv) == 0:
                return None
            return list(pv)
        return pv

    def _resample_index(self, anchor_idx: int) -> int:
        n = int(len(self.base))
        if n <= 1:
            return 0

        # For binary mega-chunk training, keep rejection/resampling local to the
        # same balanced mega-chunk as the original sampled index. A global random
        # fallback would punch reads into cold regions of multi-TiB files and
        # defeat the sampler's cache-locality contract.
        try:
            chunk_samples = max(1, int(DEFAULT_MEGA_CHUNK_BYTES // int(RECORD_BYTES)))
            chunk_count = max(1, int(math.ceil(n / chunk_samples)))
            if chunk_count > 1:
                base = int(n // chunk_count)
                remainder = int(n % chunk_count)
                idx = max(0, min(int(anchor_idx), n - 1))
                large_prefix = (base + 1) * remainder
                if idx < large_prefix:
                    chunk_id = idx // (base + 1)
                    start = chunk_id * (base + 1)
                    end = start + base + 1
                else:
                    chunk_id = remainder + ((idx - large_prefix) // base)
                    start = large_prefix + (chunk_id - remainder) * base
                    end = start + base
                if end > start:
                    return random.randint(int(start), int(end) - 1)
        except Exception:
            pass

        return random.randint(0, n - 1)

    def __getitem__(self, idx):
        anchor_idx = int(idx)
        last_sample = None
        last_error: Exception | None = None
        tries = 0
        max_total_tries = int(self.max_resamples) + 50  # hard cap to avoid returning None

        while tries < max_total_tries:
            tries += 1
            try:
                sample = self.base[idx]
            except UninformativeCutoutError as e:
                last_error = e
                idx = self._resample_index(anchor_idx)
                continue
            except Exception as e:
                # Be robust to transient dataset/transform failures; resample.
                last_error = e
                idx = self._resample_index(anchor_idx)
                continue

            if sample is None:
                last_error = RuntimeError("base dataset returned None")
                idx = self._resample_index(anchor_idx)
                continue

            last_sample = sample
            pv = self._extract_pixel_values(sample)
            if pv is None:
                last_error = RuntimeError("sample missing/invalid pixel_values")
                idx = self._resample_index(anchor_idx)
                continue

            if isinstance(pv, (list, tuple)):
                ok = True
                for vv in pv:
                    if not isinstance(vv, torch.Tensor):
                        ok = True
                        break
                    if is_effectively_empty_view(vv):
                        ok = False
                        break
                if ok:
                    return sample
            else:
                if not isinstance(pv, torch.Tensor):
                    # Unexpected type; don't risk breaking the pipeline
                    return sample
                if not is_effectively_empty_view(pv):
                    return sample

            # Otherwise, draw a new random index and try again
            idx = self._resample_index(anchor_idx)

        # Final fallback: never return None (collators expect dict-like).
        if last_sample is not None:
            return last_sample
        raise RuntimeError(f"NonEmptyResamplingDataset: failed to fetch a valid sample after {tries} tries; last_error={last_error}")

# -------------------------------------------------------------------------
# Helper pipelines for initial crop, reconstruction, and semantic transforms
# -------------------------------------------------------------------------
def is_effectively_empty_view(x: torch.Tensor,
                              abs_thresh: float = 0.02,
                              rel_frac: float = 0.3,
                              min_pixels: int = 5) -> bool:
    """
    Heuristic check for an effectively empty image/crop.

    Operates on a CHW float tensor. We treat a view as empty if its absolute
    max is very small, or if only a vanishingly small fraction of pixels are
    close to that max.
    """
    if x.numel() == 0:
        return True

    # Clean up NaNs/Infs defensively
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    v = x.abs()
    max_abs = v.max()
    if not torch.isfinite(max_abs):
        return True

    # If everything is very close to zero, treat as empty
    if float(max_abs) <= abs_thresh:
        return True

    # Relative threshold: look at pixels that are a fraction of max
    thr = float(max_abs) * rel_frac
    frac = (v >= thr).float().mean().item()

    # Require at least `min_pixels` above threshold in the view
    min_frac = float(min_pixels) / (v.numel() + 1e-6)
    return frac < min_frac

# --- Shared augmentation helpers ---
def basic_flips(x: torch.Tensor) -> torch.Tensor:
    """
    Shared dihedral (D4) augmentation for square CHW tensors:
      - random rotation by {0, 90, 180, 270} degrees
      - optional horizontal flip (covers all 8 dihedral symmetries when combined with rotation)
    """
    # Rotations are lossless (no interpolation) for 90-degree steps.
    try:
        k = int(torch.randint(0, 4, (1,), device=x.device).item())
    except Exception:
        k = int(torch.randint(0, 4, (1,)).item())
    if k:
        x = torch.rot90(x, k=k, dims=[-2, -1])

    try:
        do_flip = bool((torch.rand((), device=x.device) < 0.5).item())
    except Exception:
        do_flip = bool((torch.rand(()) < 0.5).item())
    if do_flip:
        x = torch.flip(x, dims=[-1])  # horizontal after rotation
    return x

def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Shared additive Gaussian noise helper."""
    if sigma <= 0.0:
        return x
    noise = torch.randn_like(x) * sigma
    return x + noise

def random_asinh_contrast_stretch(
    x: torch.Tensor,
    *,
    p: float = 0.5,
    alpha_range: tuple[float, float] = (3.0, 20.0),
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Astro-like monotonic contrast stretch using an asinh mapping.

    We operate on a clamped [0, 1] view (values outside this range are treated as out-of-band),
    and apply:

        y = asinh(alpha * x) / asinh(alpha)

    Higher `alpha` increases compression of highlights and lifts faint structure.
    """
    try:
        pp = float(max(0.0, min(1.0, p)))
    except Exception:
        pp = 0.0
    if pp <= 0.0:
        return x

    try:
        r = torch.rand((), device=x.device)
    except Exception:
        r = torch.rand(())
    if float(r.item()) >= pp:
        return x

    lo, hi = float(alpha_range[0]), float(alpha_range[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return x
    if hi < lo:
        lo, hi = hi, lo
    lo = max(lo, eps)
    hi = max(hi, lo + eps)

    try:
        alpha = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(lo, hi)
    except Exception:
        alpha = torch.tensor([(lo + hi) * 0.5], device=x.device, dtype=x.dtype)

    # Contrast stretch in [0, 1] to avoid amplifying out-of-band noise too aggressively.
    xc = x.clamp(0.0, 1.0)
    denom = torch.asinh(alpha).clamp_min(eps)
    y = torch.asinh(alpha * xc) / denom
    return y.to(dtype=x.dtype)


def resize_full_cutout_view(x: torch.Tensor, img_size: int) -> torch.Tensor:
    """Resize a full CHW cutout to the model size without ROI selection."""
    if x.shape[-2] == img_size and x.shape[-1] == img_size:
        return x
    try:
        return F.resize(
            x,
            size=[int(img_size), int(img_size)],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    except TypeError:
        return F.resize(x, size=[int(img_size), int(img_size)])

def find_topk_bright_tiles(
    x: torch.Tensor,
    tiles_per_side: int,
    k: int = 5,
    score_mode: str = "mean",
) -> list[tuple[int, int, float]]:
    """
    Find top-K brightest tiles, return list of (cy, cx, score).

    score_mode:
      - "mean": mean absolute value within each tile (robust for extended sources)
      - "max":  max absolute value within each tile (better for tiny/sparse sources)
    """
    C, H, W = x.shape
    grid = int(max(1, tiles_per_side))
    tile_h = max(1, H // grid)
    tile_w = max(1, W // grid)

    mode = str(score_mode or "mean").lower()
    candidates: list[tuple[float, int, int]] = []
    for gy in range(grid):
        y0 = gy * tile_h
        if y0 >= H:
            break
        y1 = min(H, y0 + tile_h)
        for gx in range(grid):
            x0 = gx * tile_w
            if x0 >= W:
                break
            x1 = min(W, x0 + tile_w)

            patch = x[:, y0:y1, x0:x1]
            if patch.numel() == 0:
                continue
            if mode == "max":
                score = patch.abs().max().item()
            else:
                score = patch.abs().mean().item()
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            candidates.append((score, cy, cx))

    if not candidates:
        return []

    candidates.sort(reverse=True, key=lambda t: t[0])
    topk = candidates[:min(k, len(candidates))]
    return [(cy, cx, score) for score, cy, cx in topk]


class DinoEMAMultiViewTransform:
    """
    Multi-view generator tailored for DINO-style EMA distillation.

    Key differences vs the legacy bright-tile contrastive transform:
    - Deterministic view ordering: the first `n_global_views` are always "global" crops, suitable
      for the EMA teacher, so `teacher_views=2` can reliably use views [0, 1].
    - "Global" is defined relative to an object-centric ROI around a bright tile center, not the
      full 512x512 field, to reduce object-switch positives in multi-object cutouts.
    - Global crops use continuous jitter around the object center; local crops can remain more
      tile-locked if desired.
    """

    def __init__(
        self,
        img_size: int,
        n_views: int,
        n_global_views: int,
        mean,
        std,
        apply_norm: bool,
        data_policy: DataPolicy | None = None,
        # Hard minimum crop side in ROI pixel space (before resize to model img_size).
        # Prevents tiny crops that become overly interpolated after upsampling.
        min_crop_side_px: int = 16,
        tiles_per_side: int = 8,
        center_topk: int = 5,
        roi_scale: tuple[float, float] = (0.55, 0.80),
        global_tight_scale: tuple[float, float] = (0.50, 0.70),
        global_wide_scale: tuple[float, float] = (0.30, 0.50),
        local_scale: tuple[float, float] = (0.20, 0.35),
        global_jitter_frac: float = 0.15,
        local_jitter_frac: float = 0.05,
        nonempty_max_resamples: int = 4,
        # Optional overlap enforcement to reduce object-switch positives.
        # - `min_global_iou`: require the teacher global views to overlap spatially within the ROI.
        # - `min_local_ioa`: require each local crop to lie mostly within at least one global crop
        #   (intersection-over-area of the local crop).
        min_global_iou: float = 0.0,
        min_local_ioa: float = 0.0,
        overlap_max_resamples: int = 12,
        # Optional "sparse field" mode: when the image contains only tiny bright blobs,
        # zoom in more aggressively so the instance occupies a meaningful fraction of pixels.
        # Disabled by default (`sparse_area_frac_threshold <= 0`).
        sparse_area_frac_threshold: float = 0.0,
        sparse_top_frac: float = 0.85,
        sparse_detect_mode: str = "topfrac",
        sparse_z: float = 3.0,
        sparse_p: float = 1.0,
        # Optional "extended structure" detection: if a large/extended object exists anywhere in the cutout,
        # prefer centering the ROI on it (prevents accidentally centering on tiny point sources).
        # Disabled by default (`extent_area_frac_threshold <= 0`).
        extent_area_frac_threshold: float = 0.0,
        extent_eval_frac: float = 0.75,
        extent_top_frac: float = 0.60,
        extent_keep_frac_of_max: float = 0.70,
        extent_strong_mult: float = 1.5,
        # Anchor-local evaluation window for sparse/extent heuristics (fraction of side).
        anchor_eval_frac: float = 0.40,
        # Optional alternate anchors inside the ROI for local views.
        local_anchor_tiles_per_side: int | None = None,
        local_anchor_k: int = 4,
        local_anchor_min_tile_dist: int | None = None,
        mean_anchor_p: float = 0.5,
        anchor_peak_ratio_min: float = 0.75,
        # Small-instance handling (ambiguity reduction):
        # If the connected component ("island") around the anchor is very small relative to the ROI,
        # generate teacher globals by expanding outward from that island bbox so the teacher target
        # stays instance-conditioned (reduces object-switch positives in dense sparse fields).
        small_instance_frac_thresh: float = 0.12,
        teacher_instance_expand_mult: float = 2.5,
        teacher_small_min_iou: float = 0.60,
        teacher_small_jitter_cap: float = 0.05,
        # Minimum teacher-global crop size (as fraction of ROI side) when in small-instance mode.
        # This must stay "global" (larger than locals), otherwise view ordering can invert.
        teacher_small_min_side_frac: float = 0.25,
        # When in small-instance mode, tighten local crops relative to the teacher-global crop,
        # so locals remain smaller (more zoomed-in) than teacher globals.
        small_instance_local_min_mult: float = 0.55,
        small_instance_local_max_mult: float = 0.75,
        teacher_small_margin_frac: float = 0.10,
        sparse_roi_scale: tuple[float, float] | None = None,
        sparse_global_tight_scale: tuple[float, float] | None = None,
        sparse_global_wide_scale: tuple[float, float] | None = None,
        sparse_local_scale: tuple[float, float] | None = None,
        sparse_global_jitter_frac: float | None = None,
        sparse_local_jitter_frac: float | None = None,
        sparse_roi_min_pixels: int = 6,
        sparse_global_min_pixels: int = 6,
        sparse_local_min_pixels: int = 4,
        # --- Anchor proposal (tile-based) ---
        anchor_tiles_per_side: int | None = None,
        anchor_k: int = 3,
        anchor_min_tile_dist: int | None = None,
        anchor_z_floor: float = 2.5,
        peak_refine_window: int = 21,
    ) -> None:
        data_policy = data_policy or DataPolicy()
        self.img_size = int(img_size)
        self.n_views = int(n_views)
        self.n_global_views = int(max(1, min(int(n_global_views), int(n_views))))
        self.data_policy = data_policy
        self.use_roi_crop = bool(data_policy.use_roi_crop)
        self.enable_augmentations = bool(data_policy.enable_augmentations)
        self.min_crop_side_px = int(max(1, min_crop_side_px))
        self.tiles_per_side = int(max(1, tiles_per_side))
        self.center_topk = int(max(1, center_topk))

        self.roi_scale = (float(roi_scale[0]), float(roi_scale[1]))
        self.global_tight_scale = (float(global_tight_scale[0]), float(global_tight_scale[1]))
        self.global_wide_scale = (float(global_wide_scale[0]), float(global_wide_scale[1]))
        self.local_scale = (float(local_scale[0]), float(local_scale[1]))
        self.global_jitter_frac = float(global_jitter_frac)
        self.local_jitter_frac = float(local_jitter_frac)
        self.nonempty_max_resamples = int(max(0, nonempty_max_resamples))

        self.min_global_iou = float(min_global_iou)
        self.min_local_ioa = float(min_local_ioa)
        self.overlap_max_resamples = int(max(1, overlap_max_resamples))

        self.sparse_area_frac_threshold = float(sparse_area_frac_threshold)
        self.sparse_top_frac = float(sparse_top_frac)
        self.sparse_detect_mode = str(sparse_detect_mode or "topfrac").lower()
        self.sparse_z = float(sparse_z)
        self.sparse_p = float(sparse_p)
        self.extent_area_frac_threshold = float(extent_area_frac_threshold)
        self.extent_eval_frac = float(extent_eval_frac)
        self.extent_top_frac = float(extent_top_frac)
        self.extent_keep_frac_of_max = float(extent_keep_frac_of_max)
        self.extent_strong_mult = float(extent_strong_mult)
        self.anchor_eval_frac = float(anchor_eval_frac)
        self.sparse_roi_scale = sparse_roi_scale
        self.sparse_global_tight_scale = sparse_global_tight_scale
        self.sparse_global_wide_scale = sparse_global_wide_scale
        self.sparse_local_scale = sparse_local_scale
        self.sparse_global_jitter_frac = sparse_global_jitter_frac
        self.sparse_local_jitter_frac = sparse_local_jitter_frac
        self.sparse_roi_min_pixels = int(max(1, sparse_roi_min_pixels))
        self.sparse_global_min_pixels = int(max(1, sparse_global_min_pixels))
        self.sparse_local_min_pixels = int(max(1, sparse_local_min_pixels))

        # Anchor proposal defaults: dense grid for better secondary-source coverage.
        if anchor_tiles_per_side is None:
            anchor_tiles_per_side = 16
        self.anchor_tiles_per_side = int(max(1, anchor_tiles_per_side))
        self.anchor_k = int(max(1, anchor_k))
        if anchor_min_tile_dist is None:
            anchor_min_tile_dist = 2 if self.anchor_tiles_per_side >= 16 else 1
        self.anchor_min_tile_dist = int(max(0, anchor_min_tile_dist))
        self.anchor_z_floor = float(anchor_z_floor)
        self.peak_refine_window = int(max(7, peak_refine_window))
        if self.peak_refine_window % 2 == 0:
            self.peak_refine_window += 1
        if local_anchor_tiles_per_side is None:
            local_anchor_tiles_per_side = int(self.tiles_per_side)
        self.local_anchor_tiles_per_side = int(max(1, local_anchor_tiles_per_side))
        self.local_anchor_k = int(max(1, local_anchor_k))
        if local_anchor_min_tile_dist is None:
            local_anchor_min_tile_dist = 1 if self.local_anchor_tiles_per_side >= 8 else 0
        self.local_anchor_min_tile_dist = int(max(0, local_anchor_min_tile_dist))
        self.mean_anchor_p = float(max(0.0, min(1.0, mean_anchor_p)))
        self.anchor_peak_ratio_min = float(max(0.0, min(1.0, anchor_peak_ratio_min)))
        self.small_instance_frac_thresh = float(max(0.0, min(1.0, small_instance_frac_thresh)))
        self.teacher_instance_expand_mult = float(max(1.0, teacher_instance_expand_mult))
        self.teacher_small_min_iou = float(max(0.0, min(0.99, teacher_small_min_iou)))
        self.teacher_small_jitter_cap = float(max(0.0, teacher_small_jitter_cap))
        self.teacher_small_min_side_frac = float(max(0.0, min(1.0, teacher_small_min_side_frac)))
        self.small_instance_local_min_mult = float(max(0.0, small_instance_local_min_mult))
        self.small_instance_local_max_mult = float(max(0.0, small_instance_local_max_mult))
        self.teacher_small_margin_frac = float(max(0.0, min(1.0, teacher_small_margin_frac)))

        # Debug stats (primarily for the augmentation preview helper).
        self._debug_total_calls = 0
        self._debug_sparse_calls = 0
        self._debug_extent_present_calls = 0
        self._debug_last_sparse_mode = False
        self._debug_last_area_frac = None
        self._debug_last_area_frac_z = None
        self._debug_last_extent_max = None
        self._debug_last_extent_present = None
        self._debug_last_n_anchors = None
        self._debug_last_anchor_idx = None

        self.resize_to_model = tv2.Resize(
            size=(self.img_size, self.img_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.model_num_channels = int(len(mean)) if isinstance(mean, (list, tuple)) else int(getattr(mean, "__len__", lambda: 1)() or 1)
        if apply_norm:
            self.final_norm = tv2.Normalize(mean=mean, std=std)
        else:
            self.final_norm = tv2.Identity()

    def _resize_to_model_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            try:
                x = self.resize_to_model(x)
            except Exception:
                x = F.resize(x, size=[self.img_size, self.img_size])
        return x

    def _prepare_full_cutout_view(self, img: torch.Tensor) -> torch.Tensor:
        return self._resize_to_model_if_needed(img)

    def _finalize_view(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        if x.dim() == 3 and x.size(0) == 1 and self.model_num_channels == 3:
            x = x.expand(3, x.size(1), x.size(2)).contiguous()
        x = self.final_norm(x)
        return x

    def _add_banding(self, x: torch.Tensor, max_bands: int = 2) -> torch.Tensor:
        if max_bands <= 0:
            return x
        C, H, W = x.shape
        n_bands = int(torch.randint(1, max_bands + 1, (1,)).item())
        x = x.clone()
        for _ in range(n_bands):
            if torch.rand(()) < 0.5:
                band_h = max(1, H // 64)
                y0 = int(torch.randint(0, max(1, H - band_h + 1), (1,)).item())
                strength = (torch.rand(()) - 0.5) * 0.6
                x[:, y0:y0 + band_h, :] = x[:, y0:y0 + band_h, :] + strength
            else:
                band_w = max(1, W // 64)
                x0 = int(torch.randint(0, max(1, W - band_w + 1), (1,)).item())
                strength = (torch.rand(()) - 0.5) * 0.6
                x[:, :, x0:x0 + band_w] = x[:, :, x0:x0 + band_w] + strength
        return x

    def _random_flux_perturb(self, x: torch.Tensor, heavy: bool = False) -> torch.Tensor:
        if torch.rand(()) >= 0.5:
            return x
        mean = x.mean()
        std = x.std()
        if (not torch.isfinite(std)) or std <= 0:
            return x
        thresh = mean + 0.3 * std
        mask = x < thresh
        if not mask.any():
            return x
        lo, hi = (0.5, 0.9) if heavy else (0.7, 0.95)
        factor = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(lo, hi)
        x = x.clone()
        x[mask] = x[mask] * factor
        return x

    def _small_translate_scale_jitter(self, x: torch.Tensor, max_translate_frac: float = 0.05, max_scale_jitter: float = 0.08) -> torch.Tensor:
        C, H, W = x.shape
        if H <= 0 or W <= 0:
            return x
        tx_pix = int(((torch.rand(()) * 2.0) - 1.0) * max_translate_frac * W)
        ty_pix = int(((torch.rand(()) * 2.0) - 1.0) * max_translate_frac * H)
        scale = 1.0 + float(((torch.rand(()) * 2.0) - 1.0) * max_scale_jitter)
        try:
            return F.affine(
                x,
                angle=0.0,
                translate=[tx_pix, ty_pix],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
        except Exception:
            return x

    @staticmethod
    def _signal_area_frac(x: torch.Tensor, top_frac: float = 0.85) -> float:
        """
        Fraction of pixels in x (CHW) that are within `top_frac` of the max signal.

        In sparse cutouts with tiny sources, this tends to be very small.
        """
        if not isinstance(x, torch.Tensor) or x.dim() != 3 or x.numel() == 0:
            return 1.0
        try:
            s = x.abs().mean(dim=0)  # (H, W)
            m = float(s.max().detach().cpu().item())
            if (not torch.isfinite(s).all()) or m <= 0:
                return 1.0
            thr = float(top_frac) * m
            return float((s >= thr).float().mean().detach().cpu().item())
        except Exception:
            return 1.0

    @staticmethod
    def _signal_area_frac_zscore(x: torch.Tensor, z: float = 3.0) -> float:
        """
        Fraction of pixels above mean + z*std of the per-pixel signal map.

        For point-source-like images, this tends to be extremely small.
        For structured textures, it tends to be larger.
        """
        if not isinstance(x, torch.Tensor) or x.dim() != 3 or x.numel() == 0:
            return 1.0
        try:
            s = x.abs().mean(dim=0)  # (H, W)
            if not bool(torch.isfinite(s).all()):
                return 1.0
            mu = float(s.mean().detach().cpu().item())
            sig = float(s.std(unbiased=False).detach().cpu().item())
            if sig <= 0:
                return 1.0
            thr = mu + float(z) * sig
            return float((s >= thr).float().mean().detach().cpu().item())
        except Exception:
            return 1.0

    def _extract_roi_fixed_frac(self, x: torch.Tensor, center: tuple[int, int], frac: float) -> torch.Tensor:
        """Square ROI with fixed side fraction of the input image size."""
        _, H, W = x.shape
        f = float(max(0.0, min(1.0, frac)))
        side = int(max(16, round(f * min(H, W))))
        cy, cx = int(center[0]), int(center[1])
        y0 = int(max(0, min(H - side, cy - side // 2)))
        x0 = int(max(0, min(W - side, cx - side // 2)))
        return x[:, y0:y0 + side, x0:x0 + side]

    def _anchor_local_stats(
        self,
        x: torch.Tensor,
        anchor_yx: tuple[int, int],
        *,
        eval_frac: float,
    ) -> tuple[float, float, float]:
        """
        Compute local area stats around an anchor within a fixed-size ROI.
        Returns (area_frac_top, area_frac_z, area_frac_extent).
        """
        try:
            roi_eval = self._extract_roi_fixed_frac(x, center=anchor_yx, frac=eval_frac).clamp(0.0, 1.0)
            area_frac_top = float(self._signal_area_frac(roi_eval, top_frac=self.sparse_top_frac))
            area_frac_z = float(self._signal_area_frac_zscore(roi_eval, z=self.sparse_z))
            area_frac_extent = float(self._signal_area_frac(roi_eval, top_frac=self.extent_top_frac))
            return (area_frac_top, area_frac_z, area_frac_extent)
        except Exception:
            return (1.0, 1.0, 1.0)

    @staticmethod
    def _box_xyxy(box: tuple[int, int, int]) -> tuple[int, int, int, int]:
        """Convert (y0, x0, side) -> (x0, y0, x1, y1)."""
        y0, x0, side = box
        return (x0, y0, x0 + side, y0 + side)

    @staticmethod
    def _box_intersection_area(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
        ax0, ay0, ax1, ay1 = DinoEMAMultiViewTransform._box_xyxy(a)
        bx0, by0, bx1, by1 = DinoEMAMultiViewTransform._box_xyxy(b)
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        iw = max(0, ix1 - ix0)
        ih = max(0, iy1 - iy0)
        return int(iw * ih)

    @staticmethod
    def _box_area(a: tuple[int, int, int]) -> int:
        side = int(max(0, a[2]))
        return int(side * side)

    @staticmethod
    def _box_iou(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        inter = float(DinoEMAMultiViewTransform._box_intersection_area(a, b))
        if inter <= 0:
            return 0.0
        area_a = float(DinoEMAMultiViewTransform._box_area(a))
        area_b = float(DinoEMAMultiViewTransform._box_area(b))
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return float(inter / union)

    @staticmethod
    def _box_ioa(small: tuple[int, int, int], big: tuple[int, int, int]) -> float:
        """Intersection over area of `small`."""
        inter = float(DinoEMAMultiViewTransform._box_intersection_area(small, big))
        if inter <= 0:
            return 0.0
        denom = float(DinoEMAMultiViewTransform._box_area(small))
        if denom <= 0:
            return 0.0
        return float(inter / denom)

    def _crop_from_box(self, x: torch.Tensor, box: tuple[int, int, int]) -> torch.Tensor:
        y0, x0, side = box
        crop = x[:, y0:y0 + side, x0:x0 + side]
        return self._resize_to_model_if_needed(crop)

    def _aug_global(self, x: torch.Tensor, heavy: bool = False) -> torch.Tensor:
        x = basic_flips(x)
        # Mild astro-like stretch; keep teacher globals relatively stable.
        x = random_asinh_contrast_stretch(x, p=0.25, alpha_range=(3.0, 12.0))
        x = self._small_translate_scale_jitter(x, max_translate_frac=0.03, max_scale_jitter=0.05)
        # Keep teacher globals relatively stable; diversity should come primarily from framing.
        x = add_gaussian_noise(x, sigma=0.01 if not heavy else 0.02)
        x = self._random_flux_perturb(x, heavy=heavy)
        return x

    def _aug_local(self, x: torch.Tensor) -> torch.Tensor:
        x = basic_flips(x)
        # Stronger stretch on locals to expose faint structure under varying display/survey pipelines.
        x = random_asinh_contrast_stretch(x, p=0.50, alpha_range=(3.0, 20.0))
        x = self._small_translate_scale_jitter(x, max_translate_frac=0.05, max_scale_jitter=0.08)
        x = add_gaussian_noise(x, sigma=0.05)
        x = self._add_banding(x, max_bands=3)
        x = self._random_flux_perturb(x, heavy=True)
        return x

    def _saliency_map(self, img: torch.Tensor) -> torch.Tensor:
        """
        Astro-style saliency map for preview/debugging.

        Uses positive brightness significance after a robust low-tail background estimate,
        which tends to highlight real sources (bright blobs / extended emission) and
        downweight "texture-only" noise.
        """
        if img.dim() != 3:
            raise ValueError("saliency_map expects CHW tensor")
        base = img[0] if int(img.size(0)) == 1 else img.mean(dim=0)
        Zb = self._brightness_z_map_lowtail(base)
        return Zb.clamp_min(0.0)

    @staticmethod
    def _brightness_bg_stats_lowtail(
        x2: torch.Tensor,
        q: float = 0.8,
        eps: float = 1e-6,
        max_samples: int = 8192,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return robust (median, scale) for background using the low/typical part of pixels.

        Computed on a random subset for speed. `scale` corresponds to (MAD*1.4826).
        """
        flat = x2.flatten()
        if flat.numel() < 64:
            med = flat.median()
            mad = (flat - med).abs().median()
            scale = (mad * 1.4826).clamp_min(eps)
            return med, scale

        try:
            if flat.numel() > int(max_samples):
                idx = torch.randint(0, flat.numel(), (int(max_samples),), device=flat.device)
                samp = flat[idx]
            else:
                samp = flat
        except Exception:
            samp = flat

        qq = float(max(0.55, min(0.95, q)))
        try:
            qv = torch.quantile(samp, qq)
            sel = samp[samp <= qv]
        except Exception:
            sel = samp

        if sel.numel() < max(64, int(0.02 * samp.numel())):
            sel = samp

        med = sel.median()
        mad = (sel - med).abs().median()
        scale = (mad * 1.4826).clamp_min(eps)
        return med, scale

    @staticmethod
    def _z_from_stats(x2: torch.Tensor, med: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Compute z-map from precomputed robust stats."""
        return (x2 - med) / scale.clamp_min(1e-6)

    @staticmethod
    def _brightness_z_map_lowtail(x2: torch.Tensor, q: float = 0.8, eps: float = 1e-6) -> torch.Tensor:
        """
        Robust z-score map where background stats come from the low/typical part of the pixels.

        This is a cheap "sigma-clip the low part" approximation: it largely ignores the bright
        source tail without iterative clipping loops.
        """
        med, scale = DinoEMAMultiViewTransform._brightness_bg_stats_lowtail(x2, q=q, eps=eps)
        return DinoEMAMultiViewTransform._z_from_stats(x2, med, scale)

    @staticmethod
    def _flood_fill_island(
        mask: torch.Tensor,
        seed_y: int,
        seed_x: int,
        *,
        connectivity: int = 8,
        max_pixels: int | None = None,
    ) -> torch.Tensor:
        """
        Flood-fill (BFS) on a boolean mask. Returns a boolean mask of the connected component.
        Intended for relatively small windows (<= ~129x129).
        """
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)
        H, W = int(mask.shape[-2]), int(mask.shape[-1])
        sy = int(seed_y)
        sx = int(seed_x)
        if not (0 <= sy < H and 0 <= sx < W):
            return torch.zeros((H, W), dtype=torch.bool)
        if not bool(mask[sy, sx].item()):
            return torch.zeros((H, W), dtype=torch.bool)

        from collections import deque

        out = torch.zeros((H, W), dtype=torch.bool)
        q = deque([(sy, sx)])
        out[sy, sx] = True

        if int(connectivity) == 4:
            neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            neigh = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        limit = None if max_pixels is None else int(max(1, max_pixels))
        filled = 1
        while q:
            y, x = q.popleft()
            for dy, dx in neigh:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                if out[ny, nx]:
                    continue
                if not bool(mask[ny, nx].item()):
                    continue
                out[ny, nx] = True
                q.append((ny, nx))
                filled += 1
                if limit is not None and filled >= limit:
                    return out
        return out

    @staticmethod
    def _integral_image(x2: torch.Tensor) -> torch.Tensor:
        """
        Summed-area table with a 1-pixel zero border.
        Returns tensor of shape (H+1, W+1) such that rectangle sums are O(1).
        """
        if x2.dim() != 2:
            raise ValueError("_integral_image expects (H, W)")
        x = x2.to(dtype=torch.float32)
        sat = x.cumsum(dim=0).cumsum(dim=1)
        return torch.nn.functional.pad(sat, (1, 0, 1, 0), mode="constant", value=0.0)

    @staticmethod
    def _rect_sum(sat: torch.Tensor, y0: int, x0: int, side: int) -> torch.Tensor:
        """Sum over [y0:y0+side, x0:x0+side] using an integral image sat(H+1,W+1)."""
        y1 = int(y0 + side)
        x1 = int(x0 + side)
        y0i = int(y0)
        x0i = int(x0)
        return sat[y1, x1] - sat[y0i, x1] - sat[y1, x0i] + sat[y0i, x0i]

    def _tile_anchor_candidates(
        self,
        img: torch.Tensor,
        *,
        tiles_per_side: int,
        k_anchors: int,
        min_tile_dist: int,
        z_floor: float,
        bg_q: float = 0.8,
        bg_max_samples: int = 8192,
    ) -> tuple[list[tuple[int, int]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fast tile-based anchor proposals.

        Returns (anchors_yx, tile_max_z, bg_med, bg_scale).
        - `anchors_yx` are full-res pixel coordinates (y, x) of the max pixel within each selected tile.
        - `tile_max_z` is (grid, grid) of per-tile peak z-scores.
        """
        if img.dim() != 3:
            raise ValueError("_tile_anchor_candidates expects CHW tensor")
        base = img[0] if int(img.size(0)) == 1 else img.mean(dim=0)
        H, W = int(base.shape[-2]), int(base.shape[-1])
        g = int(max(1, tiles_per_side))
        tile_h = int(max(1, H // g))
        tile_w = int(max(1, W // g))
        Hc = int(tile_h * g)
        Wc = int(tile_w * g)
        base_c = base[:Hc, :Wc]

        bg_med, bg_scale = self._brightness_bg_stats_lowtail(
            base_c, q=float(bg_q), eps=1e-6, max_samples=int(bg_max_samples)
        )

        tiles = base_c.reshape(g, tile_h, g, tile_w)
        tile_max = tiles.amax(dim=(1, 3))
        tile_max_z = self._z_from_stats(tile_max, bg_med, bg_scale)

        zf = float(z_floor)
        scores = (tile_max_z - zf).clamp_min(0.0)
        flat_scores = scores.flatten()
        order = torch.argsort(flat_scores, descending=True)

        keep_tiles: list[tuple[int, int]] = []
        d = int(max(0, min_tile_dist))
        for idx in order.tolist():
            if len(keep_tiles) >= int(max(1, k_anchors)):
                break
            if float(flat_scores[idx].detach().cpu().item()) <= 0.0:
                break
            ty = int(idx // g)
            tx = int(idx % g)
            ok = True
            for ky, kx in keep_tiles:
                if max(abs(ty - ky), abs(tx - kx)) <= d:
                    ok = False
                    break
            if not ok:
                continue
            keep_tiles.append((ty, tx))

        anchors: list[tuple[int, int]] = []
        for ty, tx in keep_tiles:
            y0 = int(ty * tile_h)
            x0 = int(tx * tile_w)
            patch = base_c[y0:y0 + tile_h, x0:x0 + tile_w]
            if patch.numel() == 0:
                continue
            li = int(patch.argmax().detach().cpu().item())
            py = li // int(patch.size(-1))
            px = li % int(patch.size(-1))
            anchors.append((int(y0 + py), int(x0 + px)))

        return anchors, tile_max_z, bg_med, bg_scale

    def _best_box_by_island(
        self,
        roi: torch.Tensor,
        *,
        scale_range: tuple[float, float],
        center: tuple[int, int],
        anchor: tuple[int, int],
        jitter_frac: float,
        sat_mass: torch.Tensor,
        sat_hits: torch.Tensor,
        total_mass: float,
        total_hits: float,
        total_area: float,
        bestof: int = 16,
    ) -> tuple[torch.Tensor, tuple[int, int, int], float]:
        """
        Sample a crop box (including the anchor) and pick the best of N candidates based on
        how well it captures the salient island mass.
        Returns (view, box, concentration_score).
        """
        if total_mass <= 0.0 or total_area <= 0.0:
            box = self._sample_centered_box_include_anchor(
                roi,
                scale_range=scale_range,
                center=center,
                anchor=anchor,
                jitter_frac=jitter_frac,
            )
            view = self._crop_from_box(roi, box)
            return (view, box, 0.0)

        best = None  # (score, box, conc)
        n = int(max(1, bestof))
        for _ in range(n):
            box = self._sample_centered_box_include_anchor(
                roi,
                scale_range=scale_range,
                center=center,
                anchor=anchor,
                jitter_frac=jitter_frac,
            )
            y0, x0, side = box
            mass = float(self._rect_sum(sat_mass, y0=y0, x0=x0, side=side).detach().cpu().item())
            area = float(side * side)
            mass_frac = mass / max(total_mass, 1e-12)
            area_frac = area / max(total_area, 1e-12)
            conc = mass_frac / max(area_frac, 1e-12)
            hit = float(self._rect_sum(sat_hits, y0=y0, x0=x0, side=side).detach().cpu().item())
            hit_frac = hit / max(1.0, total_hits)
            score = float(conc) + 0.25 * float(hit_frac)
            if best is None or score > best[0]:
                best = (score, box, conc)

        assert best is not None
        _, box, conc = best
        view = self._crop_from_box(roi, box)
        return (view, box, float(conc))

    def _sample_centered_box_include_anchor(
        self,
        x: torch.Tensor,
        *,
        scale_range: tuple[float, float],
        center: tuple[int, int],
        anchor: tuple[int, int],
        jitter_frac: float,
    ) -> tuple[int, int, int]:
        """Return (y0, x0, side) in `x` coordinates, guaranteeing `anchor` lies inside the box."""
        _, H, W = x.shape
        frac = float(torch.empty(1, device=x.device).uniform_(scale_range[0], scale_range[1]).item())
        side = int(max(self.min_crop_side_px, frac * min(H, W)))
        side = int(min(side, H, W))

        cy, cx = int(center[0]), int(center[1])
        if jitter_frac and jitter_frac > 0:
            j = int(max(1, round(jitter_frac * side)))
            cy = cy + int(torch.randint(-j, j + 1, (1,), device=x.device).item())
            cx = cx + int(torch.randint(-j, j + 1, (1,), device=x.device).item())
        cy = int(max(0, min(H - 1, cy)))
        cx = int(max(0, min(W - 1, cx)))

        y0 = int(max(0, min(H - side, cy - side // 2)))
        x0 = int(max(0, min(W - side, cx - side // 2)))

        ay, ax = int(anchor[0]), int(anchor[1])
        ay = int(max(0, min(H - 1, ay)))
        ax = int(max(0, min(W - 1, ax)))

        low_y = max(0, ay - side + 1)
        high_y = min(ay, H - side)
        low_x = max(0, ax - side + 1)
        high_x = min(ax, W - side)
        y0 = int(min(max(y0, low_y), high_y))
        x0 = int(min(max(x0, low_x), high_x))
        return (y0, x0, side)

    @staticmethod
    def _bbox_yxxy_from_mask(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
        """Return (y0, x0, y1, x1) bbox of True pixels, or None if empty."""
        try:
            if not isinstance(mask, torch.Tensor) or mask.numel() == 0:
                return None
            if mask.dim() != 2:
                return None
            idx = mask.nonzero(as_tuple=False)
            if idx.numel() == 0:
                return None
            ys = idx[:, 0]
            xs = idx[:, 1]
            y0 = int(ys.min().detach().cpu().item())
            y1 = int(ys.max().detach().cpu().item()) + 1
            x0 = int(xs.min().detach().cpu().item())
            x1 = int(xs.max().detach().cpu().item()) + 1
            return (y0, x0, y1, x1)
        except Exception:
            return None

    def _sample_box_include_subbox(
        self,
        *,
        H: int,
        W: int,
        side: int,
        subbox: tuple[int, int, int, int],
        device: torch.device,
    ) -> tuple[int, int, int]:
        """Sample (y0, x0, side) s.t. `subbox` (y0,x0,y1,x1) lies inside."""
        yb0, xb0, yb1, xb1 = subbox
        yb0 = int(max(0, min(H, yb0)))
        yb1 = int(max(0, min(H, yb1)))
        xb0 = int(max(0, min(W, xb0)))
        xb1 = int(max(0, min(W, xb1)))
        if yb1 <= yb0 or xb1 <= xb0:
            cy = int(max(0, min(H - 1, (yb0 + yb1) // 2)))
            cx = int(max(0, min(W - 1, (xb0 + xb1) // 2)))
            y0 = int(max(0, min(H - side, cy - side // 2)))
            x0 = int(max(0, min(W - side, cx - side // 2)))
            return (y0, x0, side)

        # Ensure side can fit the subbox.
        req_side = int(max(yb1 - yb0, xb1 - xb0))
        side = int(max(side, req_side))
        side = int(min(side, H, W))

        low_y = int(max(0, yb1 - side))
        high_y = int(min(yb0, H - side))
        low_x = int(max(0, xb1 - side))
        high_x = int(min(xb0, W - side))

        if low_y > high_y:
            y0 = int(max(0, min(H - side, yb0)))
        else:
            y0 = int(torch.randint(low_y, high_y + 1, (1,), device=device).item())
        if low_x > high_x:
            x0 = int(max(0, min(W - side, xb0)))
        else:
            x0 = int(torch.randint(low_x, high_x + 1, (1,), device=device).item())
        return (int(y0), int(x0), int(side))

    def _box_island_score(
        self,
        *,
        box: tuple[int, int, int],
        sat_mass: torch.Tensor,
        sat_hits: torch.Tensor,
        total_mass: float,
        total_hits: float,
        total_area: float,
    ) -> tuple[float, float]:
        """Return (score, conc) for a crop box w.r.t. island mass/hits."""
        y0, x0, side = box
        try:
            mass = float(self._rect_sum(sat_mass, y0=y0, x0=x0, side=side).detach().cpu().item())
            hit = float(self._rect_sum(sat_hits, y0=y0, x0=x0, side=side).detach().cpu().item())
        except Exception:
            return (0.0, 0.0)
        area = float(side * side)
        mass_frac = mass / max(total_mass, 1e-12)
        area_frac = area / max(total_area, 1e-12)
        conc = mass_frac / max(area_frac, 1e-12)
        hit_frac = hit / max(1.0, total_hits)
        score = float(conc) + 0.25 * float(hit_frac)
        return (float(score), float(conc))

    def _sample_view_with_box_nonempty_island(
        self,
        roi: torch.Tensor,
        *,
        scale_range: tuple[float, float],
        center: tuple[int, int],
        anchor: tuple[int, int],
        jitter_frac: float,
        island_mask: torch.Tensor,
        island_weight: torch.Tensor,
        sat_mass: torch.Tensor | None = None,
        sat_hits: torch.Tensor | None = None,
        total_mass: float | None = None,
        total_hits: float | None = None,
        total_area: float | None = None,
        bestof: int,
        abs_thresh: float,
        rel_frac: float,
        min_pixels: int,
    ) -> tuple[torch.Tensor, tuple[int, int, int], float]:
        """
        Sample a non-empty view while preferring boxes that capture a non-trivial part of the
        salient island (map-based). The anchor pixel is still guaranteed inside the crop.
        """
        if sat_mass is None or sat_hits is None or total_mass is None or total_hits is None or total_area is None:
            try:
                w2 = island_weight.to(dtype=torch.float32)
                h2 = island_mask.to(dtype=torch.float32)
                sat_mass = self._integral_image(w2)
                sat_hits = self._integral_image(h2)
                total_mass = float(w2.sum().detach().cpu().item())
                total_hits = float(h2.sum().detach().cpu().item())
                total_area = float(island_mask.numel())
            except Exception:
                sat_mass = self._integral_image(torch.ones_like(island_weight, dtype=torch.float32))
                sat_hits = self._integral_image(torch.ones_like(island_weight, dtype=torch.float32))
                total_mass = float(island_weight.numel())
                total_hits = float(island_weight.numel())
                total_area = float(island_weight.numel())

        tries = int(self.nonempty_max_resamples)
        if tries <= 0:
            v, box, conc = self._best_box_by_island(
                roi,
                scale_range=scale_range,
                center=center,
                anchor=anchor,
                jitter_frac=jitter_frac,
                sat_mass=sat_mass,
                sat_hits=sat_hits,
                total_mass=total_mass,
                total_hits=total_hits,
                total_area=total_area,
                bestof=bestof,
            )
            return (v, box, conc)

        last = None
        for _ in range(tries):
            v, box, conc = self._best_box_by_island(
                roi,
                scale_range=scale_range,
                center=center,
                anchor=anchor,
                jitter_frac=jitter_frac,
                sat_mass=sat_mass,
                sat_hits=sat_hits,
                total_mass=total_mass,
                total_hits=total_hits,
                total_area=total_area,
                bestof=bestof,
            )
            last = (v, box, conc)
            try:
                if not is_effectively_empty_view(v, abs_thresh=abs_thresh, rel_frac=rel_frac, min_pixels=min_pixels):
                    return (v, box, conc)
            except Exception:
                return (v, box, conc)

        assert last is not None
        return last

    def _extract_roi_with_box(
        self,
        img: torch.Tensor,
        center_yx: tuple[int, int],
        roi_scale: tuple[float, float],
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Extract ROI and return both the tensor and (y0, x0, side) box."""
        _, H, W = img.shape
        frac = float(torch.empty(1, device=img.device).uniform_(float(roi_scale[0]), float(roi_scale[1])).item())
        side = int(max(16, frac * min(H, W)))
        cy, cx = int(center_yx[0]), int(center_yx[1])
        y0 = int(max(0, min(H - side, cy - side // 2)))
        x0 = int(max(0, min(W - side, cx - side // 2)))
        return (img[:, y0:y0 + side, x0:x0 + side], (y0, x0, side))

    def _refine_center_in_roi(self, S_roi: torch.Tensor, init_center_yx: tuple[int, int]) -> tuple[int, int]:
        """Refine center by local argmax in S_roi around init_center."""
        H, W = int(S_roi.shape[-2]), int(S_roi.shape[-1])
        cy, cx = int(init_center_yx[0]), int(init_center_yx[1])
        w = int(min(self.peak_refine_window, H, W))
        if w < 3:
            return (max(0, min(H - 1, cy)), max(0, min(W - 1, cx)))
        if w % 2 == 0:
            w -= 1
        r = w // 2
        y0 = max(0, cy - r)
        y1 = min(H, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(W, cx + r + 1)
        patch = S_roi[y0:y1, x0:x1]
        idx = int(patch.argmax().detach().cpu().item())
        py = idx // int(patch.size(-1))
        px = idx % int(patch.size(-1))
        return (int(y0 + py), int(x0 + px))

    def _local_peak_2d(self, x2d: torch.Tensor, center_yx: tuple[int, int], window: int) -> float:
        """Return local max around center in a square window from a 2D tensor."""
        try:
            H, W = int(x2d.shape[-2]), int(x2d.shape[-1])
            cy, cx = int(center_yx[0]), int(center_yx[1])
            w = int(max(1, window))
            if w % 2 == 0:
                w -= 1
            r = w // 2
            y0 = max(0, cy - r)
            y1 = min(H, cy + r + 1)
            x0 = max(0, cx - r)
            x1 = min(W, cx + r + 1)
            patch = x2d[y0:y1, x0:x1]
            return float(patch.max().detach().cpu().item())
        except Exception:
            return 0.0

    def __call__(self, img: torch.Tensor):
        if not isinstance(img, torch.Tensor) or img.dim() != 3:
            return img

        H, W = int(img.shape[-2]), int(img.shape[-1])
        fallback_center = (H // 2, W // 2)

        # Base image (for global peak and local peak tests)
        base = img[0] if int(img.size(0)) == 1 else img.mean(dim=0)
        base_abs = base.abs()
        try:
            global_peak = float(base_abs.max().detach().cpu().item())
            if not math.isfinite(global_peak):
                global_peak = 0.0
        except Exception:
            global_peak = 0.0
        if global_peak > 0.0:
            try:
                gi = int(base_abs.argmax().detach().cpu().item())
                global_peak_yx = (gi // int(base_abs.size(-1)), gi % int(base_abs.size(-1)))
            except Exception:
                global_peak_yx = fallback_center
        else:
            global_peak_yx = fallback_center

        # --- 1) Mode selection (dual-mode compatible) ---
        extent_present = False
        extent_strong = False
        extent_max = None
        extent_candidates = None
        if self.extent_area_frac_threshold and self.extent_area_frac_threshold > 0:
            try:
                topk_mean = find_topk_bright_tiles(
                    img,
                    tiles_per_side=self.tiles_per_side,
                    k=self.center_topk,
                    score_mode="mean",
                )
            except Exception:
                topk_mean = []

            if topk_mean:
                scored: list[tuple[float, int, int, float]] = []
                for cy, cx, bscore in topk_mean:
                    try:
                        roi_eval = self._extract_roi_fixed_frac(
                            img, center=(int(cy), int(cx)), frac=self.extent_eval_frac
                        ).clamp(0.0, 1.0)
                        es = float(self._signal_area_frac(roi_eval, top_frac=self.extent_top_frac))
                    except Exception:
                        es = 0.0
                    scored.append((es, int(cy), int(cx), float(bscore)))

                extent_max = max((s[0] for s in scored), default=0.0)
                extent_present = bool(extent_max >= float(self.extent_area_frac_threshold))
                try:
                    strong_thr = float(self.extent_area_frac_threshold) * float(self.extent_strong_mult)
                except Exception:
                    strong_thr = float(self.extent_area_frac_threshold)
                extent_strong = bool(extent_max >= float(strong_thr))
                if extent_present:
                    try:
                        keep_thr = float(self.extent_keep_frac_of_max) * float(extent_max)
                    except Exception:
                        keep_thr = float(extent_max)
                    keep = [(cy, cx) for (es, cy, cx, _b) in scored if float(es) >= float(keep_thr)]
                    extent_candidates = keep if keep else [(cy, cx) for (_es, cy, cx, _b) in scored]

        sparse_mode = False
        auto_dual_mode = bool(self.extent_area_frac_threshold and self.extent_area_frac_threshold > 0)
        # Always compute global sparse stats (used as a weak prior only).
        area_frac = float(self._signal_area_frac(img, top_frac=self.sparse_top_frac))
        area_frac_z = float(self._signal_area_frac_zscore(img, z=self.sparse_z))
        is_sparse = False
        if self.sparse_area_frac_threshold and self.sparse_area_frac_threshold > 0:
            if str(self.sparse_detect_mode).lower() == "either":
                is_sparse = (area_frac < self.sparse_area_frac_threshold) or (area_frac_z < self.sparse_area_frac_threshold)
            elif str(self.sparse_detect_mode).lower() == "zscore":
                is_sparse = (area_frac_z < self.sparse_area_frac_threshold)
            else:
                is_sparse = (area_frac < self.sparse_area_frac_threshold)

        extent_mode = bool(extent_strong)

        # --- 2) Anchor selection (fast tile proposals; uniform sampling among top-K) ---
        fallback_global_all = False
        try:
            if extent_mode and extent_candidates:
                if global_peak > 0.0:
                    win = int(max(7, self.peak_refine_window))
                    extent_candidates = [
                        c for c in extent_candidates
                        if self._local_peak_2d(base_abs, c, win) >= (global_peak * self.anchor_peak_ratio_min)
                    ]
                if not extent_candidates:
                    extent_mode = False

            if extent_mode and extent_candidates:
                pick = int(torch.randint(0, len(extent_candidates), (1,), device=img.device).item())
                peak_yx = extent_candidates[pick]
                try:
                    self._debug_last_n_anchors = int(len(extent_candidates))
                    self._debug_last_anchor_idx = int(pick)
                except Exception:
                    pass
            else:
                anchors_yx, _, _, _ = self._tile_anchor_candidates(
                    img,
                    tiles_per_side=int(self.anchor_tiles_per_side),
                    k_anchors=int(self.anchor_k),
                    min_tile_dist=int(self.anchor_min_tile_dist),
                    z_floor=float(self.anchor_z_floor),
                )
                mean_anchors: list[tuple[int, int]] = []
                try:
                    mean_tiles = find_topk_bright_tiles(
                        img,
                        tiles_per_side=self.tiles_per_side,
                        k=self.center_topk,
                        score_mode="mean",
                    )
                    mean_anchors = [(int(cy), int(cx)) for (cy, cx, _s) in mean_tiles]
                except Exception:
                    mean_anchors = []
                if global_peak > 0.0:
                    win = int(max(7, self.peak_refine_window))
                    anchors_yx = [
                        a for a in anchors_yx
                        if self._local_peak_2d(base_abs, a, win) >= (global_peak * self.anchor_peak_ratio_min)
                    ]
                    mean_anchors = [
                        a for a in mean_anchors
                        if self._local_peak_2d(base_abs, a, win) >= (global_peak * self.anchor_peak_ratio_min)
                    ]
                use_mean = False
                if not anchors_yx and not mean_anchors:
                    peak_yx = global_peak_yx
                    pick = 0
                    use_mean = False
                    fallback_global_all = True
                else:
                    fallback_global_all = False
                    use_mean = bool(mean_anchors) and (float(torch.rand(()).item()) < float(self.mean_anchor_p))
                    if use_mean:
                        pick = int(torch.randint(0, len(mean_anchors), (1,), device=img.device).item())
                        peak_yx = mean_anchors[pick]
                    else:
                        pick = int(torch.randint(0, len(anchors_yx), (1,), device=img.device).item())
                        peak_yx = anchors_yx[pick]
                try:
                    self._debug_last_n_anchors = int(len(mean_anchors) if (use_mean and mean_anchors) else len(anchors_yx))
                    self._debug_last_anchor_idx = int(pick)
                except Exception:
                    pass
        except Exception:
            peak_yx = fallback_center
            fallback_global_all = True

        if "fallback_global_all" not in locals():
            fallback_global_all = False

        # Anchor-local stats determine sparse vs normal scaling for this cutout.
        a_top, a_z, a_ext = self._anchor_local_stats(
            img,
            peak_yx,
            eval_frac=self.anchor_eval_frac,
        )
        anchor_is_sparse = False
        if self.sparse_area_frac_threshold and self.sparse_area_frac_threshold > 0:
            if str(self.sparse_detect_mode).lower() == "either":
                anchor_is_sparse = (a_top < self.sparse_area_frac_threshold) or (a_z < self.sparse_area_frac_threshold)
            elif str(self.sparse_detect_mode).lower() == "zscore":
                anchor_is_sparse = (a_z < self.sparse_area_frac_threshold)
            else:
                anchor_is_sparse = (a_top < self.sparse_area_frac_threshold)

        if auto_dual_mode:
            # Prefer sparse scaling when local anchor looks point-like,
            # unless the extent signal is strong enough to override it.
            if bool(extent_mode):
                sparse_mode = False
                extent_present = True
            elif bool(anchor_is_sparse):
                sparse_mode = (float(torch.rand(()).item()) < float(max(0.0, min(1.0, self.sparse_p))))
                extent_present = False
            else:
                sparse_mode = False
        else:
            # Legacy-style sparse detection if dual-mode is disabled (anchor-local).
            sparse_mode = bool(anchor_is_sparse) and (float(torch.rand(()).item()) < float(max(0.0, min(1.0, self.sparse_p))))

        try:
            self._debug_total_calls += 1
            self._debug_last_area_frac = a_top
            self._debug_last_area_frac_z = a_z
            self._debug_last_sparse_mode = bool(sparse_mode)
            self._debug_last_extent_max = (None if extent_max is None else float(extent_max))
            self._debug_last_extent_present = bool(extent_mode)
            if sparse_mode:
                self._debug_sparse_calls += 1
            if extent_mode:
                self._debug_extent_present_calls += 1
        except Exception:
            pass

        # --- 3) Pick ROI + center inside ROI ---
        roi_scale_used = self.roi_scale
        global_tight_scale_used = self.global_tight_scale
        global_wide_scale_used = self.global_wide_scale
        local_scale_used = self.local_scale
        global_jitter_used = self.global_jitter_frac
        local_jitter_used = self.local_jitter_frac
        roi_min_pixels_used = 12
        global_min_pixels_used = 12
        local_min_pixels_used = 8

        roi_scale_base = roi_scale_used
        global_tight_scale_base = global_tight_scale_used
        global_wide_scale_base = global_wide_scale_used
        local_scale_base = local_scale_used
        global_jitter_base = global_jitter_used
        local_jitter_base = local_jitter_used
        roi_min_pixels_base = roi_min_pixels_used
        global_min_pixels_base = global_min_pixels_used
        local_min_pixels_base = local_min_pixels_used

        roi_scale_sparse = roi_scale_base
        global_tight_scale_sparse = global_tight_scale_base
        global_wide_scale_sparse = global_wide_scale_base
        local_scale_sparse = local_scale_base
        global_jitter_sparse = global_jitter_base
        local_jitter_sparse = local_jitter_base
        roi_min_pixels_sparse = roi_min_pixels_base
        global_min_pixels_sparse = global_min_pixels_base
        local_min_pixels_sparse = local_min_pixels_base

        if sparse_mode:
            if self.sparse_roi_scale is not None:
                roi_scale_sparse = self.sparse_roi_scale
            if self.sparse_global_tight_scale is not None:
                global_tight_scale_sparse = self.sparse_global_tight_scale
            if self.sparse_global_wide_scale is not None:
                global_wide_scale_sparse = self.sparse_global_wide_scale
            if self.sparse_local_scale is not None:
                local_scale_sparse = self.sparse_local_scale
            if self.sparse_global_jitter_frac is not None:
                global_jitter_sparse = float(self.sparse_global_jitter_frac)
            if self.sparse_local_jitter_frac is not None:
                local_jitter_sparse = float(self.sparse_local_jitter_frac)
            roi_min_pixels_sparse = int(self.sparse_roi_min_pixels)
            global_min_pixels_sparse = int(self.sparse_global_min_pixels)
            local_min_pixels_sparse = int(self.sparse_local_min_pixels)

        if sparse_mode:
            roi_scale_used = roi_scale_sparse
            global_tight_scale_used = global_tight_scale_sparse
            global_wide_scale_used = global_wide_scale_sparse
            local_scale_used = local_scale_sparse
            global_jitter_used = global_jitter_sparse
            local_jitter_used = local_jitter_sparse
            roi_min_pixels_used = roi_min_pixels_sparse
            global_min_pixels_used = global_min_pixels_sparse
            local_min_pixels_used = local_min_pixels_sparse

        roi, roi_box = self._extract_roi_with_box(img, center_yx=peak_yx, roi_scale=roi_scale_used)
        y0, x0, side = roi_box
        roi_center = (int(peak_yx[0] - y0), int(peak_yx[1] - x0))

        # Refine center using brightness significance within ROI.
        # Background stats once per cutout; compute z-maps only where needed (ROI/window), not on full 512x512.
        bg_med, bg_scale = self._brightness_bg_stats_lowtail(base, q=0.8, eps=1e-6, max_samples=8192)
        try:
            Zb_roi = self._z_from_stats(base[y0:y0 + side, x0:x0 + side], bg_med, bg_scale)
            roi_center = self._refine_center_in_roi(Zb_roi.clamp_min(0.0), roi_center)
        except Exception:
            pass

        # Build an island mask/weight in ROI coordinates for crop selection.
        try:
            Zb_roi = self._z_from_stats(base[y0:y0 + side, x0:x0 + side], bg_med, bg_scale)
            cy, cx = int(roi_center[0]), int(roi_center[1])
            peak_z_roi = float(Zb_roi[cy, cx].detach().cpu().item())
            thr_roi = float(max(2.0, peak_z_roi - 2.0))

            win = int(min(129, side))
            if win % 2 == 0:
                win -= 1
            r = max(1, win // 2)
            wy0 = max(0, cy - r)
            wy1 = min(side, cy + r + 1)
            wx0 = max(0, cx - r)
            wx1 = min(side, cx + r + 1)
            patch = Zb_roi[wy0:wy1, wx0:wx1]
            m_patch = patch >= thr_roi
            island_patch = self._flood_fill_island(m_patch, cy - wy0, cx - wx0, connectivity=8, max_pixels=None)
            island_mask_roi = torch.zeros((side, side), dtype=torch.bool)
            island_mask_roi[wy0:wy1, wx0:wx1] = island_patch
            island_weight_roi = ((Zb_roi - thr_roi).clamp_min(0.0) * island_mask_roi.float()).to(dtype=Zb_roi.dtype)
        except Exception:
            island_mask_roi = torch.ones((side, side), dtype=torch.bool)
            island_weight_roi = torch.ones((side, side), dtype=torch.float32)

        # Instance bbox from the anchor island; used to tighten teacher globals when the instance is tiny.
        instance_box = None  # (y0,x0,y1,x1) in ROI coords
        instance_side = None
        small_instance = False
        try:
            bb = self._bbox_yxxy_from_mask(island_mask_roi)
            if bb is not None:
                yb0, xb0, yb1, xb1 = bb
                h = int(max(1, yb1 - yb0))
                w = int(max(1, xb1 - xb0))
                inst_side = int(max(h, w))
                margin = int(max(1, round(float(self.teacher_small_margin_frac) * float(inst_side))))
                yb0 = max(0, yb0 - margin)
                xb0 = max(0, xb0 - margin)
                yb1 = min(side, yb1 + margin)
                xb1 = min(side, xb1 + margin)
                instance_box = (int(yb0), int(xb0), int(yb1), int(xb1))
                instance_side = int(max(yb1 - yb0, xb1 - xb0))
                frac = float(instance_side) / float(max(1, side))
                small_instance = (frac > 0.0) and (frac < float(self.small_instance_frac_thresh))
        except Exception:
            instance_box = None
            instance_side = None
            small_instance = False

        # Precompute the target teacher-global crop side for small-instance mode (in ROI coords).
        # This is reused for (a) the teacher globals themselves and (b) tightening locals so ordering
        # remains global -> local (globals larger context, locals more zoomed-in).
        teacher_small_target_side = None
        teacher_small_target_frac = None
        try:
            if bool(small_instance and instance_side is not None and instance_box is not None):
                H_roi = int(roi.shape[-2])
                W_roi = int(roi.shape[-1])
                base_side = int(min(H_roi, W_roi))
                target_side = int(round(float(self.teacher_instance_expand_mult) * float(instance_side)))
                min_side = int(max(16, round(float(self.teacher_small_min_side_frac) * float(max(1, base_side)))))
                target_side = int(max(min_side, target_side))
                target_side = int(min(target_side, base_side))
                teacher_small_target_side = int(target_side)
                teacher_small_target_frac = float(target_side) / float(max(1, base_side))
        except Exception:
            teacher_small_target_side = None
            teacher_small_target_frac = None
        # Integral images for fast box scoring (reused across all views for this cutout).
        try:
            w2 = island_weight_roi.to(dtype=torch.float32)
            h2 = island_mask_roi.to(dtype=torch.float32)
            sat_mass_roi = self._integral_image(w2)
            sat_hits_roi = self._integral_image(h2)
            total_mass_roi = float(w2.sum().detach().cpu().item())
            total_hits_roi = float(h2.sum().detach().cpu().item())
            total_area_roi = float(island_mask_roi.numel())
        except Exception:
            sat_mass_roi = self._integral_image(torch.ones_like(island_weight_roi, dtype=torch.float32))
            sat_hits_roi = self._integral_image(torch.ones_like(island_weight_roi, dtype=torch.float32))
            total_mass_roi = float(island_weight_roi.numel())
            total_hits_roi = float(island_weight_roi.numel())
            total_area_roi = float(island_weight_roi.numel())

        # --- 3b) Local anchor candidates inside ROI (for per-view adaptation) ---
        roi_anchor_stats = None
        try:
            roi_anchors, _, _, _ = self._tile_anchor_candidates(
                roi,
                tiles_per_side=int(self.local_anchor_tiles_per_side),
                k_anchors=int(self.local_anchor_k),
                min_tile_dist=int(self.local_anchor_min_tile_dist),
                z_floor=float(self.anchor_z_floor),
            )
            if roi_anchors:
                seen = set()
                stats = []
                roi_base_abs = base_abs[y0:y0 + side, x0:x0 + side]
                win = int(max(7, self.peak_refine_window))
                for ayx in roi_anchors:
                    # Keep local anchors on the same connected island as the global anchor.
                    try:
                        if not bool(island_mask_roi[int(ayx[0]), int(ayx[1])]):
                            continue
                    except Exception:
                        pass
                    if global_peak > 0.0:
                        lp = self._local_peak_2d(roi_base_abs, ayx, win)
                        if lp < (global_peak * self.anchor_peak_ratio_min):
                            continue
                    if ayx in seen:
                        continue
                    seen.add(ayx)
                    lt, lz, _le = self._anchor_local_stats(
                        roi,
                        ayx,
                        eval_frac=self.anchor_eval_frac,
                    )
                    is_sparse_local = False
                    if self.sparse_area_frac_threshold and self.sparse_area_frac_threshold > 0:
                        if str(self.sparse_detect_mode).lower() == "either":
                            is_sparse_local = (lt < self.sparse_area_frac_threshold) or (lz < self.sparse_area_frac_threshold)
                        elif str(self.sparse_detect_mode).lower() == "zscore":
                            is_sparse_local = (lz < self.sparse_area_frac_threshold)
                        else:
                            is_sparse_local = (lt < self.sparse_area_frac_threshold)
                    stats.append({"yx": ayx, "sparse": bool(is_sparse_local)})
                roi_anchor_stats = stats if stats else None
        except Exception:
            roi_anchor_stats = None

        # --- 4) Generate views (globals first, then locals) ---
        views: list[torch.Tensor] = []
        global_boxes: list[tuple[int, int, int]] = []

        for i in range(self.n_global_views):
            # Teacher globals: if the instance is tiny inside the ROI (common for sparse fields),
            # expand outward from the island bbox so both teacher globals stay instance-conditioned.
            use_small_teacher = bool(teacher_small_target_side is not None and instance_box is not None)
            min_global_iou_eff = float(self.min_global_iou)
            jitter_eff = float(global_jitter_used)

            if use_small_teacher:
                min_global_iou_eff = max(min_global_iou_eff, float(self.teacher_small_min_iou))
                jitter_eff = min(jitter_eff, float(self.teacher_small_jitter_cap))

            ref_box = global_boxes[0] if (global_boxes and min_global_iou_eff > 0) else None
            best = None  # (score, view, box)
            tries = int(self.overlap_max_resamples if ref_box is not None else 1)

            for _ in range(max(1, tries)):
                if use_small_teacher:
                    H = int(roi.shape[-2])
                    W = int(roi.shape[-1])
                    # Target teacher crop side: ~expand_mult * instance_side (clamped).
                    target_side = int(min(int(teacher_small_target_side), H, W))
                    box0 = self._sample_box_include_subbox(
                        H=H,
                        W=W,
                        side=target_side,
                        subbox=instance_box,
                        device=roi.device,
                    )
                    v0 = self._crop_from_box(roi, box0)
                    try:
                        if is_effectively_empty_view(v0, abs_thresh=0.02, rel_frac=0.3, min_pixels=global_min_pixels_used):
                            continue
                    except Exception:
                        pass
                    score0, conc0 = self._box_island_score(
                        box=box0,
                        sat_mass=sat_mass_roi,
                        sat_hits=sat_hits_roi,
                        total_mass=total_mass_roi,
                        total_hits=total_hits_roi,
                        total_area=total_area_roi,
                    )
                else:
                    scale = global_tight_scale_used if (i % 2 == 0) else global_wide_scale_used
                    v0, box0, conc0 = self._sample_view_with_box_nonempty_island(
                        roi,
                        scale_range=scale,
                        center=roi_center,
                        anchor=roi_center,
                        jitter_frac=jitter_eff,
                        island_mask=island_mask_roi,
                        island_weight=island_weight_roi,
                        sat_mass=sat_mass_roi,
                        sat_hits=sat_hits_roi,
                        total_mass=total_mass_roi,
                        total_hits=total_hits_roi,
                        total_area=total_area_roi,
                        bestof=16,
                        abs_thresh=0.02,
                        rel_frac=0.3,
                        min_pixels=global_min_pixels_used,
                    )
                    score0 = float(conc0)

                if ref_box is None:
                    v, box = v0, box0
                    break

                iou = float(self._box_iou(box0, ref_box))
                score = float(score0) + 0.5 * float(iou)
                if best is None or score > best[0]:
                    best = (score, v0, box0)
                if iou >= min_global_iou_eff:
                    v, box = v0, box0
                    break
            else:
                assert best is not None
                _, v, box = best

            v = self._aug_global(v, heavy=False)
            v = v.clamp(0.0, 1.0)
            v = self.final_norm(v)
            views.append(v)
            global_boxes.append(box)

        for li in range(self.n_views - self.n_global_views):
            anchor_local = roi_center
            local_scale_view = local_scale_base
            local_jitter_view = local_jitter_base
            local_min_pixels_view = local_min_pixels_base
            if fallback_global_all:
                local_scale_view = global_tight_scale_used if (li % 2 == 0) else global_wide_scale_used
                local_jitter_view = global_jitter_used
                local_min_pixels_view = global_min_pixels_used
            elif teacher_small_target_frac is not None:
                # Ensure locals remain smaller (more zoomed-in) than teacher globals when the instance is tiny.
                # Otherwise the perceived "global/local" order can invert in the preview.
                try:
                    tfrac = float(teacher_small_target_frac)
                    max_frac = float(min(float(local_scale_view[1]), tfrac * float(self.small_instance_local_max_mult)))
                    min_frac = float(min(float(local_scale_view[0]), tfrac * float(self.small_instance_local_min_mult)))
                    # Safety clamps and ordering.
                    max_frac = float(max(0.05, min(0.95, max_frac)))
                    min_frac = float(max(0.04, min(max_frac - 1e-3, min_frac)))
                    if not (min_frac < max_frac):
                        min_frac = float(max(0.04, max_frac * 0.7))
                    local_scale_view = (min_frac, max_frac)
                except Exception:
                    pass
            elif roi_anchor_stats:
                pick = int(torch.randint(0, len(roi_anchor_stats), (1,), device=roi.device).item())
                anchor_local = roi_anchor_stats[pick]["yx"]
                if roi_anchor_stats[pick].get("sparse", False):
                    local_scale_view = local_scale_sparse
                    local_jitter_view = local_jitter_sparse
                    local_min_pixels_view = local_min_pixels_sparse
            require_local = bool(global_boxes) and (self.min_local_ioa > 0)
            best = None  # (score, view, box)
            for _ in range(self.overlap_max_resamples if require_local else 1):
                v0, box0, conc0 = self._sample_view_with_box_nonempty_island(
                    roi,
                    scale_range=local_scale_view,
                    center=roi_center,
                    anchor=anchor_local,
                    jitter_frac=local_jitter_view,
                    island_mask=island_mask_roi,
                    island_weight=island_weight_roi,
                    sat_mass=sat_mass_roi,
                    sat_hits=sat_hits_roi,
                    total_mass=total_mass_roi,
                    total_hits=total_hits_roi,
                    total_area=total_area_roi,
                    bestof=16,
                    abs_thresh=0.02,
                    rel_frac=0.3,
                    min_pixels=local_min_pixels_view,
                )
                if not require_local:
                    v, box = v0, box0
                    break

                ioa = 0.0
                for gb in global_boxes:
                    ioa = max(ioa, float(self._box_ioa(box0, gb)))
                score = float(conc0) + 0.75 * float(ioa)
                if best is None or score > best[0]:
                    best = (score, v0, box0)
                if ioa >= self.min_local_ioa:
                    v, box = v0, box0
                    break
            else:
                assert best is not None
                _, v, box = best

            v = self._aug_local(v)
            v = v.clamp(0.0, 1.0)
            v = self.final_norm(v)
            views.append(v)

        if self.n_views == 1:
            return views[0]
        return views


class SimCLRMultiViewTransform(DinoEMAMultiViewTransform):
    """
    Multi-view generator for SimCLR-style contrastive learning, reusing the ROI/anchor/island
    machinery from `DinoEMAMultiViewTransform`.

    Design goals for sparse astronomical fields:
      - reduce "false positive" pairs where two views contain different objects
      - avoid empty/noise-only crops via non-empty + peak-ratio gating
      - guarantee shared content by anchoring all views to the same object/anchor

    Output ordering is fixed: [wide_view, view_2, ..., view_N], where all additional
    views are anchored to the same object/anchor as the wide view.

    Flag behavior:
      - `use_roi_crop=False`: bypass ROI search and operate on the full cutout.
      - `enable_augmentations=False`: keep view 0 raw, while view 1+ remain augmented.
    """

    def __init__(
        self,
        *,
        img_size: int,
        n_views: int = 2,
        mean,
        std,
        apply_norm: bool,
        data_policy: DataPolicy | None = None,
        # SimCLR pair structure: by default, use a DINO-like global+local positive pair.
        # (n_global_views=1 means: view0 is a "global" crop; view1+ are "local" crops constrained
        # to lie within the global crop via `min_local_ioa`.)
        n_global_views: int | None = None,
        # View geometry (relative to ROI side)
        roi_scale: tuple[float, float] = (0.55, 0.80),
        # For SimCLR, both views are typically sampled from the same crop distribution; we therefore
        # prefer "global-like" crops for both views (avoid one tiny local view vs one huge cluttered view).
        wide_scale: tuple[float, float] = (0.50, 0.70),
        other_scale: tuple[float, float] = (0.20, 0.35),
        overlap_max_resamples: int = 12,
        nonempty_max_resamples: int = 4,
        # Anchor / sparsity
        tiles_per_side: int = 8,
        center_topk: int = 5,
        sparse_area_frac_threshold: float = 0.0,
        sparse_top_frac: float = 0.85,
        sparse_detect_mode: str = "either",
        sparse_z: float = 3.0,
        sparse_p: float = 1.0,
        anchor_peak_ratio_min: float = 0.80,
        # Prevent tiny mushy crops before resize.
        min_crop_side_px: int = 16,
        # Optional small-instance tightening (reuses the same knobs as DINO teacher-tightening,
        # but applies to the *wide* view in SimCLR).
        small_instance_frac_thresh: float = 0.12,
        wide_instance_expand_mult: float = 2.5,
        wide_small_min_side_frac: float = 0.25,
        wide_small_jitter_cap: float = 0.05,
        wide_small_margin_frac: float = 0.10,
        # Require locals to stay mostly inside the global crop (IOA of local wrt global).
        min_local_ioa: float = 0.70,
    ) -> None:
        data_policy = data_policy or DataPolicy()
        n_views = int(n_views)
        if n_views < 2:
            raise ValueError(f"SimCLRMultiViewTransform requires n_views>=2 (got n_views={n_views}).")
        if n_global_views is None:
            n_global_views = 1
        n_global_views = int(max(1, min(int(n_global_views), n_views - 1)))
        super().__init__(
            img_size=int(img_size),
            n_views=n_views,
            n_global_views=n_global_views,
            mean=mean,
            std=std,
            apply_norm=apply_norm,
            data_policy=data_policy,
            min_crop_side_px=int(min_crop_side_px),
            tiles_per_side=int(tiles_per_side),
            center_topk=int(center_topk),
            roi_scale=roi_scale,
            global_tight_scale=wide_scale,
            global_wide_scale=wide_scale,
            local_scale=other_scale,
            global_jitter_frac=0.12,
            local_jitter_frac=0.05,
            nonempty_max_resamples=int(nonempty_max_resamples),
            min_global_iou=0.0,
            min_local_ioa=float(min_local_ioa),
            overlap_max_resamples=int(overlap_max_resamples),
            sparse_area_frac_threshold=float(sparse_area_frac_threshold),
            sparse_top_frac=float(sparse_top_frac),
            sparse_detect_mode=str(sparse_detect_mode),
            sparse_z=float(sparse_z),
            sparse_p=float(sparse_p),
            extent_area_frac_threshold=0.0,  # SimCLR: keep it simple; ROI is anchor-driven.
            anchor_peak_ratio_min=float(anchor_peak_ratio_min),
            small_instance_frac_thresh=float(small_instance_frac_thresh),
            teacher_instance_expand_mult=float(wide_instance_expand_mult),
            teacher_small_min_iou=0.0,
            teacher_small_jitter_cap=float(wide_small_jitter_cap),
            teacher_small_min_side_frac=float(wide_small_min_side_frac),
            teacher_small_margin_frac=float(wide_small_margin_frac),
        )

    def _aug_simclr(self, x: torch.Tensor, heavy: bool) -> torch.Tensor:
        # Geometry-free transforms (plus small framing jitter) that remain DINO/SimCLR-friendly.
        x = basic_flips(x)
        x = random_asinh_contrast_stretch(
            x,
            p=0.50 if heavy else 0.35,
            alpha_range=(3.0, 20.0) if heavy else (3.0, 12.0),
        )
        x = self._small_translate_scale_jitter(
            x,
            max_translate_frac=0.05 if heavy else 0.03,
            max_scale_jitter=0.08 if heavy else 0.05,
        )
        x = add_gaussian_noise(x, sigma=0.04 if heavy else 0.02)
        if heavy:
            x = self._add_banding(x, max_bands=3)
        x = self._random_flux_perturb(x, heavy=heavy)
        return x

    def _aug_global(self, x: torch.Tensor, heavy: bool = False) -> torch.Tensor:
        # SimCLR: keep the first/global view a bit more stable; put stronger corruption
        # pressure on the local views (DINO-like).
        if not self.enable_augmentations:
            return x
        return self._aug_simclr(x, heavy=False)

    def _aug_local(self, x: torch.Tensor) -> torch.Tensor:
        return self._aug_simclr(x, heavy=True)

    def __call__(self, img: torch.Tensor):
        if self.use_roi_crop:
            return super().__call__(img)
        if not isinstance(img, torch.Tensor) or img.dim() != 3:
            return img

        base = self._prepare_full_cutout_view(img)
        views: list[torch.Tensor] = []

        global_view = self._aug_global(base.clone(), heavy=False)
        views.append(self._finalize_view(global_view))

        for _ in range(self.n_views - 1):
            local_view = self._aug_local(base.clone())
            views.append(self._finalize_view(local_view))

        if self.n_views == 1:
            return views[0]
        return views


class MAESingleViewRoiAlignedTransform(DinoEMAMultiViewTransform):
    """
    Single-view MAE transform that reuses the ROI/anchor/island selection machinery from
    `DinoEMAMultiViewTransform` (the same "good cutout" logic used by `SimCLRMultiViewTransform`),
    while keeping MAE as a *single-view* reconstruction objective.

    Policy:
      - With probability `p_global_full`, take a global crop from the *full cutout* (RandomResizedCrop).
      - Otherwise, take an object-centric ROI crop using the DinoEMA anchor/ROI selection, then
        sample a "global-tight" crop inside that ROI (n_views=1, n_global_views=1).

    Flag behavior:
      - `use_roi_crop=False`: bypass ROI search/cropping and use the full cutout directly.
      - `enable_augmentations=False`: disable MAE-side augmentation entirely.

    Augmentations aim to reduce phase-1 → phase-2 domain shift by using the same *family* of
    photometric/geometry-free transforms as the contrastive pipeline (flips, asinh stretch,
    small translate/scale jitter, noise, banding, flux perturb).
    """

    def __init__(
        self,
        *,
        img_size: int,
        mean,
        std,
        apply_norm: bool,
        data_policy: DataPolicy | None = None,
        # Mixture: full-field global crops vs ROI-anchored crops
        p_global_full: float = 0.20,
        global_full_scale: tuple[float, float] = (0.80, 1.00),
        # ROI/anchor geometry (inherited from DinoEMAMultiViewTransform)
        tiles_per_side: int = 8,
        center_topk: int = 5,
        roi_scale: tuple[float, float] = (0.55, 0.80),
        roi_global_scale: tuple[float, float] = (0.50, 0.70),
        nonempty_max_resamples: int = 4,
        overlap_max_resamples: int = 12,
        anchor_peak_ratio_min: float = 0.80,
        min_crop_side_px: int = 16,
        # Aug knobs (kept mild by default; MAE reconstructs the augmented view)
        asinh_p: float = 0.35,
        asinh_alpha_range: tuple[float, float] = (3.0, 12.0),
        max_translate_frac: float = 0.03,
        max_scale_jitter: float = 0.05,
        noise_sigma: float = 0.02,
        banding_max_bands: int = 2,
    ) -> None:
        data_policy = data_policy or DataPolicy()
        self.p_global_full = float(max(0.0, min(1.0, p_global_full)))
        self.asinh_p = float(max(0.0, min(1.0, asinh_p)))
        self.asinh_alpha_range = (float(asinh_alpha_range[0]), float(asinh_alpha_range[1]))
        self.max_translate_frac = float(max(0.0, max_translate_frac))
        self.max_scale_jitter = float(max(0.0, max_scale_jitter))
        self.noise_sigma = float(max(0.0, noise_sigma))
        self.banding_max_bands = int(max(0, banding_max_bands))

        self.global_crop_full = tv2.RandomResizedCrop(
            size=int(img_size),
            scale=(float(global_full_scale[0]), float(global_full_scale[1])),
            ratio=(1.0, 1.0),
            interpolation=InterpolationMode.BILINEAR,
        )

        # Reuse DinoEMA ROI selection, but produce a single "global" view inside the ROI.
        super().__init__(
            img_size=int(img_size),
            n_views=1,
            n_global_views=1,
            mean=mean,
            std=std,
            apply_norm=apply_norm,
            data_policy=data_policy,
            min_crop_side_px=int(min_crop_side_px),
            tiles_per_side=int(tiles_per_side),
            center_topk=int(center_topk),
            roi_scale=roi_scale,
            global_tight_scale=roi_global_scale,
            global_wide_scale=roi_global_scale,
            local_scale=roi_global_scale,
            global_jitter_frac=0.12,
            local_jitter_frac=0.05,
            nonempty_max_resamples=int(nonempty_max_resamples),
            min_global_iou=0.0,
            min_local_ioa=0.0,
            overlap_max_resamples=int(overlap_max_resamples),
            extent_area_frac_threshold=0.0,
            sparse_area_frac_threshold=0.0,
            anchor_peak_ratio_min=float(anchor_peak_ratio_min),
        )

    def _aug_global(self, x: torch.Tensor, heavy: bool = False) -> torch.Tensor:
        # Align with the contrastive family of augs (mild setting).
        if not self.enable_augmentations:
            return x
        x = basic_flips(x)
        x = random_asinh_contrast_stretch(
            x,
            p=self.asinh_p,
            alpha_range=self.asinh_alpha_range,
        )
        x = self._small_translate_scale_jitter(
            x,
            max_translate_frac=self.max_translate_frac,
            max_scale_jitter=self.max_scale_jitter,
        )
        x = add_gaussian_noise(x, sigma=self.noise_sigma)
        x = self._add_banding(x, max_bands=self.banding_max_bands)
        x = self._random_flux_perturb(x, heavy=False)
        return x

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if not isinstance(img, torch.Tensor) or img.dim() != 3:
            return img

        if not self.use_roi_crop:
            x = self._prepare_full_cutout_view(img)
            x = self._aug_global(x, heavy=False)
            return self._finalize_view(x)

        # Occasionally include a full-field crop to retain global statistics.
        if self.enable_augmentations:
            try:
                r = float(torch.rand((), device=img.device).item())
            except Exception:
                r = float(torch.rand(()).item())
            if r < self.p_global_full:
                x = self.global_crop_full(img)
                x = self._aug_global(x, heavy=False)
                return self._finalize_view(x)

        # Otherwise: ROI-anchored crop via DinoEMA pipeline (returns a single tensor because n_views=1).
        return super().__call__(img)


def make_ssl_pipeline(
    img_size: int,
    mean,
    std,
    apply_norm: bool,
    n_views: int,
    loss_mode,
    data_policy: DataPolicy | None = None,
):
    """Single source of truth for SSL data transforms."""
    data_policy = data_policy or DataPolicy()
    recon_modes = ("l2", "l2_l1", "l2_bl1")
    loss_value = getattr(loss_mode, "value", loss_mode)

    if loss_value in recon_modes:
        return MAESingleViewRoiAlignedTransform(
            img_size=img_size,
            mean=mean,
            std=std,
            apply_norm=apply_norm,
            data_policy=data_policy,
            p_global_full=0.20,
            global_full_scale=(0.80, 1.00),
            tiles_per_side=8,
        )

    return SimCLRMultiViewTransform(
        img_size=img_size,
        n_views=int(n_views),
        mean=mean,
        std=std,
        apply_norm=apply_norm,
        data_policy=data_policy,
        tiles_per_side=8,
    )


def build_transform_for_loss(
    loss_mode,
    img_size: int,
    mean,
    std,
    apply_norm: bool,
    n_views: int = 2,
    data_policy: DataPolicy | None = None,
):
    recon_modes = ("l2", "l2_l1", "l2_bl1")
    loss_value = getattr(loss_mode, "value", loss_mode)
    eff_n_views = 1 if (loss_value in recon_modes) else n_views
    return make_ssl_pipeline(
        img_size=img_size,
        mean=mean,
        std=std,
        apply_norm=apply_norm,
        n_views=eff_n_views,
        loss_mode=loss_mode,
        data_policy=data_policy,
    )


def build_processor_and_transform(
    config,
    model_name_or_path: str,
    apply_norm: bool,
    loss_mode,
    n_views: int,
    patch_size: int,
    data_policy: DataPolicy | None = None,
    log_fn=print,
):
    data_policy = data_policy or DataPolicy()
    try:
        processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            size={"height": config.image_size, "width": config.image_size},
            patch_size=patch_size,
        )
    except Exception:
        processor = ViTImageProcessor.from_pretrained(
            model_name_or_path,
            size={"height": config.image_size, "width": config.image_size},
            patch_size=patch_size,
        )
    if not apply_norm:
        processor.image_mean = [0.0, 0.0, 0.0]
        processor.image_std = [1.0, 1.0, 1.0]
    mean = processor.image_mean
    std = processor.image_std
    if log_fn is not None:
        log_fn(f"[Norm] mean={mean}, std={std}, apply_norm={apply_norm}")

    transform = build_transform_for_loss(
        loss_mode=loss_mode,
        img_size=config.image_size,
        mean=mean,
        std=std,
        apply_norm=apply_norm,
        n_views=n_views,
        data_policy=data_policy,
    )
    return processor, transform, mean, std


def make_train_dataset(
    dataset_config: BinaryDatasetConfig,
    transform,
    *,
    verbose_debug: bool = True,
    log_fn=print,
):
    bin_path = dataset_config.bin_path
    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"Binary dataset path for mode '{dataset_config.mode.value}' does not exist: {bin_path}"
        )
    effective_block_size, block_size_policy, file_bytes = resolve_binary_block_size(
        bin_path,
        int(dataset_config.block_size),
    )

    train_dataset = BinaryFileDataset(
        bin_path=bin_path,
        transform=transform,
        copy_tensor=False,
        block_size=int(effective_block_size),
        cold_start_drop_cache=False,
        verbose_debug=bool(verbose_debug),
        to_float32_before_transform=False,
        expand_to_three=False,
    )
    if log_fn is not None:
        log_fn(
            f"[Dataset] mode={dataset_config.mode.value} bin_path={bin_path} "
            f"block_size={effective_block_size} "
            f"(configured={dataset_config.block_size}, policy={block_size_policy}, "
            f"file={file_bytes / (1024 ** 4):.2f} TiB, "
            f"block≈{effective_block_size * RECORD_BYTES / (1024 ** 3):.1f} GiB)"
        )
        log_fn("Dataset loaded successfully.")
    return train_dataset


def preview_augmentations_from_dataset(
    dataset,
    transform,
    mean,
    std,
    apply_norm: bool,
    num_images: int = 16,
    repeats_per_image: int = 10,
    out_dir: str = "viz",
    filename: str | None = None,
    log_fn=print,
) -> None:
    """Save a grid of original images plus repeated augmented views."""
    if num_images <= 0:
        return
    if dataset is None or len(dataset) == 0:
        return

    ns = min(int(num_images), len(dataset))
    idxs = random.sample(range(len(dataset)), ns)

    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)

    tiles = []
    ncols = None

    for idx in idxs:
        sample = dataset[idx]
        img0 = sample["pixel_values"]

        if not isinstance(img0, torch.Tensor) or img0.dim() != 3:
            continue

        orig_vis = img0 if img0.size(0) != 1 else img0.repeat(3, 1, 1)
        orig_vis = orig_vis.clamp(0, 1)

        view_tiles = []
        reps = max(1, int(repeats_per_image))

        for _ in range(reps):
            out = transform(img0)
            views = list(out) if isinstance(out, (list, tuple)) else [out]
            for v in views:
                v_vis = (v * std_t + mean_t) if apply_norm else v
                v_vis = v_vis.clamp(0, 1)
                if v_vis.size(0) == 1:
                    v_vis = v_vis.repeat(3, 1, 1)
                view_tiles.append(v_vis)

        if not view_tiles:
            continue

        tH, tW = view_tiles[0].shape[-2], view_tiles[0].shape[-1]
        try:
            orig_resized = F.resize(
                orig_vis,
                size=[tH, tW],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        except TypeError:
            orig_resized = F.resize(orig_vis, size=[tH, tW])

        norm_view_tiles = []
        for vv in view_tiles:
            if vv.shape[-2:] != (tH, tW):
                try:
                    vv = F.resize(
                        vv,
                        size=[tH, tW],
                        interpolation=InterpolationMode.BILINEAR,
                        antialias=True,
                    )
                except TypeError:
                    vv = F.resize(vv, size=[tH, tW])
            norm_view_tiles.append(vv)

        row = [orig_resized] + norm_view_tiles
        tiles.extend(row)
        if ncols is None:
            ncols = len(row)

    if not tiles or ncols is None:
        return

    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = f"aug_preview_{ns}x{ncols}.png"
    save_path = os.path.join(out_dir, filename)

    grid_tensor = torch.stack(tiles, dim=0)
    save_image(grid_tensor, save_path, nrow=ncols)
    if log_fn is not None:
        log_fn(f"[AugPreview] saved {save_path} (each row: original + {ncols - 1} views)")


class SSLDataModule:
    """Single owner for dataset selection, transforms, processor setup, and preview data."""

    def __init__(self, input_config: DataInputConfig, *, log_fn=print):
        self.input_config = input_config
        self.log_fn = log_fn

    @property
    def dataset_config(self) -> BinaryDatasetConfig:
        return self.input_config.dataset

    @property
    def data_policy(self) -> DataPolicy:
        return self.input_config.policy

    def build_bundle(
        self,
        *,
        model_config,
        model_name_or_path: str,
        apply_norm: bool,
        loss_mode,
        n_views: int,
        patch_size: int,
        build_preview_dataset: bool = False,
    ) -> DataBundle:
        processor, transform, mean, std = build_processor_and_transform(
            config=model_config,
            model_name_or_path=model_name_or_path,
            apply_norm=apply_norm,
            loss_mode=loss_mode,
            n_views=n_views,
            patch_size=patch_size,
            data_policy=self.data_policy,
            log_fn=self.log_fn,
        )
        train_dataset = make_train_dataset(
            dataset_config=self.dataset_config,
            transform=transform,
            verbose_debug=True,
            log_fn=self.log_fn,
        )
        preview_dataset = None
        if build_preview_dataset:
            preview_dataset = make_train_dataset(
                dataset_config=self.dataset_config,
                transform=tv2.Identity(),
                verbose_debug=False,
                log_fn=None,
            )
        return DataBundle(
            processor=processor,
            transform=transform,
            train_dataset=train_dataset,
            preview_dataset=preview_dataset,
            mean=list(mean),
            std=list(std),
        )


# Archived names retained as compatibility aliases only.
ContrastiveMultiViewTransform = SimCLRMultiViewTransform
MAESingleViewMultiScaleTransform = MAESingleViewRoiAlignedTransform
