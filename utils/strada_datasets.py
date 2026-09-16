import numpy as np
from torch.utils.data import Dataset
import pickle
import glob
import json
from torchvision import transforms
from PIL import Image
from astropy.io import fits
from enum import Enum
import warnings
from astropy.io.fits.verify import VerifyWarning
import matplotlib.pyplot as plt
from pathlib import Path
from utils.radioastroprocessor import RadioFoundationProcessor
import torch
import random
import os, ctypes, math
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.append('../')


# Ignore  warnings:
warnings.filterwarnings("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", message=".*NumPy array is not writable.*")

class DatasetType(Enum):
    RADIO_GALAXY_ZOO = "rgz"
    MIRABEST = "mirabest"
    SSL_IMAGENET_GRAY = "imagenet_gray"
    SSL_ASKAP_EMU = "askap_emu"
    SSL_MeerKAT = "meerkat"
    LOTSS_DR2_HORTON = "lotss_dr2_horton"


# ───────────────────────────────────────────────────────────────────────────────
# Datasets for linear probing
# ───────────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────────────────
# Updated MiraBestDatasetV2 and MiraBestDataset for batched Zenodo structure
# ───────────────────────────────────────────────────────────────────────────────
class MiraBestDatasetV2(Dataset):
    """
    MiraBest batched dataset loader for the Zenodo release (e.g. 5588282).

    Expected directory structure::

        root/
            F_batches/
                batches.meta
                data_batch_1
                ...
                test_batch
            N_batches/
                batches.meta
                data_batch_1
                ...
                test_batch

    Either ``F_batches`` or ``N_batches`` (or both) may be present. All
    available batches are read and concatenated.

    Labels in the batched files follow the convention described in
    Porter & Scaife (2023):

        0: FRI,  confident standard
        1: FRI,  confident wide-angle tail
        2: FRI,  confident head-tail
        3: FRI,  uncertain standard
        4: FRI,  uncertain wide-angle tail
        5: FRII, confident standard
        6: FRII, confident double-double
        7: FRII, uncertain standard
        8: Hybrid, confident standard
        9: Hybrid, uncertain standard

    This dataset exposes a binary FRI/FRII classification and always
    drops hybrids. If ``confident_only=True`` (default), only confident
    labels are kept.
    """

    # Mapping: label_id -> (coarse_type, is_confident)
    _LABEL_INFO = {
        0: ("FRI", True),
        1: ("FRI", True),
        2: ("FRI", True),
        3: ("FRI", False),
        4: ("FRI", False),
        5: ("FRII", True),
        6: ("FRII", True),
        7: ("FRII", False),
        8: ("Hybrid", True),
        9: ("Hybrid", False),
    }

    def __init__(self, root, transform=None, image_size=224, confident_only=True):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.transform = transform
        self.processor = RadioFoundationProcessor(image_size=image_size)
        self.confident_only = bool(confident_only)

        # Storage for raw image arrays (flattened) and binary labels
        self.data = []      # list of (n_i, D) arrays; will be vstack'ed
        self.targets = []   # binary labels: 0 = FRI, 1 = FRII

        # Discover batch directories
        candidate_subdirs = ["F_batches", "N_batches"]
        self.batch_dirs = []
        for sub in candidate_subdirs:
            d = os.path.join(self.root, sub)
            if os.path.isdir(d):
                self.batch_dirs.append(d)

        if not self.batch_dirs:
            raise RuntimeError(
                f"MiraBestDatasetV2: no batch directories found under {self.root} "
                f"(expected one of {candidate_subdirs})."
            )

        # Optional meta info (label_names etc.)
        self.label_names = None
        self._load_all_batches()

        if len(self.targets) == 0:
            raise RuntimeError(
                "MiraBestDatasetV2: after applying FRI/FRII + confidence filtering "
                "no samples remain. Check 'confident_only' or dataset integrity."
            )

        # Stack all kept samples into a single (N, D) array
        self.data = np.vstack(self.data)
        self.classes = ["FRI", "FRII"]
        self.class_to_idx = {"FRI": 0, "FRII": 1}

    def _interpret_label(self, lbl):
        """Return (coarse_type, is_confident) or (None, None) if unknown."""
        try:
            lbl_int = int(lbl)
        except Exception:
            warnings.warn(f"MiraBestDatasetV2: could not interpret label '{lbl}', skipping.")
            return None, None
        info = self._LABEL_INFO.get(lbl_int)
        if info is None:
            warnings.warn(f"MiraBestDatasetV2: unknown label id {lbl_int}, skipping.")
            return None, None
        return info  # (coarse_type, is_confident)

    def _keep_and_map(self, lbl):
        """
        Decide whether to keep a sample and map its label to 0/1.

        Returns:
            0 for FRI, 1 for FRII, or None to drop (hybrid / unwanted).
        """
        coarse, is_confident = self._interpret_label(lbl)
        if coarse is None:
            return None
        # Always drop hybrids for this task
        if coarse == "Hybrid":
            return None
        # Optionally drop uncertain objects
        if self.confident_only and not is_confident:
            return None
        # Map FRI/FRII -> 0/1
        return 0 if coarse == "FRI" else 1

    def _load_all_batches(self):
        meta_loaded = False

        for batch_dir in self.batch_dirs:
            # Meta file (optional but useful)
            meta_path = os.path.join(batch_dir, "batches.meta")
            if os.path.isfile(meta_path):
                if not meta_loaded:
                    try:
                        with open(meta_path, "rb") as f:
                            meta = pickle.load(f, encoding="latin1")
                        self.label_names = meta.get("label_names", None)
                        meta_loaded = True
                    except Exception as e:
                        warnings.warn(
                            f"MiraBestDatasetV2: failed to read meta file {meta_path}: {e}"
                        )
                else:
                    # Ignore additional meta files, but don't crash if unreadable
                    try:
                        with open(meta_path, "rb") as f:
                            _ = pickle.load(f, encoding="latin1")
                    except Exception:
                        pass

            # Collect batch pickles
            batch_files = []
            batch_files.extend(sorted(glob.glob(os.path.join(batch_dir, "data_batch_*"))))
            test_path = os.path.join(batch_dir, "test_batch")
            if os.path.isfile(test_path):
                batch_files.append(test_path)

            if not batch_files:
                warnings.warn(
                    f"MiraBestDatasetV2: no batch files found in {batch_dir}; "
                    "expected 'data_batch_*' and optionally 'test_batch'."
                )
                continue

            for file_path in batch_files:
                try:
                    with open(file_path, "rb") as f:
                        entry = pickle.load(f, encoding="latin1")
                except Exception as e:
                    warnings.warn(f"MiraBestDatasetV2: failed to read {file_path}: {e}")
                    continue

                if "data" not in entry:
                    warnings.warn(f"MiraBestDatasetV2: 'data' missing in {file_path}, skipping.")
                    continue

                batch_data = entry["data"]
                labels = entry.get("labels", entry.get("fine_labels", None))
                if labels is None:
                    warnings.warn(
                        f"MiraBestDatasetV2: neither 'labels' nor 'fine_labels' in {file_path}, "
                        "skipping."
                    )
                    continue

                if len(batch_data) != len(labels):
                    warnings.warn(
                        f"MiraBestDatasetV2: data/label length mismatch in {file_path} "
                        f"({len(batch_data)} vs {len(labels)}); skipping this file."
                    )
                    continue

                # Filter and remap labels to FRI/FRII
                keep_any = False
                for i, lbl in enumerate(labels):
                    mapped = self._keep_and_map(lbl)
                    if mapped is None:
                        continue
                    # Keep this sample (preserve 2D shape for vstack)
                    self.data.append(batch_data[i : i + 1])
                    self.targets.append(int(mapped))
                    keep_any = True

                if not keep_any:
                    warnings.warn(
                        f"MiraBestDatasetV2: no usable samples from {file_path} "
                        "(all filtered out as hybrids / uncertain)."
                    )

    def __getitem__(self, index):
        image_flat, label = self.data[index], self.targets[index]

        # 1) Process via RadioFoundationProcessor; expects 1D or 2D numpy
        tensor = self.processor.process_numpy(image_flat, resize=True)

        # 2) Apply external transform (must accept and return a tensor)
        if self.transform is not None:
            tensor = self.transform(tensor)

        return {"pixel_values": tensor, "labels": int(label)}

    def __len__(self):
        return len(self.targets)

class MiraBestDataset(MiraBestDatasetV2):
    """
    Backwards-compatible alias for MiraBestDatasetV2.

    Existing code using ``MiraBestDataset`` will now automatically
    use the updated loader for the Zenodo batched release.
    """

    def __init__(self, root, transform=None, image_size=224, confident_only=True):
        super().__init__(
            root=root,
            transform=transform,
            image_size=image_size,
            confident_only=confident_only,
        )

class RadioGalaxyDataset(Dataset):
    """
    PyTorch Dataset for Radio Galaxy Zoo FITS cutouts, organized as:

    /data
    ├── 1C-1P/
    │   ├ RGZ_Jxxxx.fits
    │   ├ RGZ_Jyyyy.fits
    │   └ …
    ├── 1C-2P/
    │   ├ …
    │   └ …
    …
    └── 3C-3P/
        ├ …
        └ …

    Each top-level folder name (e.g. "1C-1P") is treated as a class label.
    FITS files are found directly inside each class folder (no cutout subfolder).
    Files are loaded on the fly, converted via RadioFoundationProcessor,
    then passed through an optional torchvision transform.

    Args:
        root (str): Path to "/data" (the folder containing "1C-1P", "1C-2P", …).
        train (bool): Ignored here (kept only for signature compatibility).
        transform (callable): A torchvision-style transform (e.g. Resize, ToTensor).
        augmenter: Ignored here (signature compatibility only).
        image_size (int): Image size for RadioFoundationProcessor.
    """

    def __init__(
        self,
        root,
        augmenter=None,
        transform=None,
        image_size=224,
        *,
        samples: Optional[List[Tuple[str, int]]] = None,
        classes: Optional[List[str]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        index_cache_path: Optional[str] = None,
        skip_all_zero: bool = True,
        index_num_workers: int = 0,
        wait_for_cache_seconds: int = 120,
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.augmenter = augmenter  # not used
        self.transform = transform
        self.processor = RadioFoundationProcessor(image_size=image_size)
        self.skip_all_zero = bool(skip_all_zero)
        self.index_num_workers = max(0, int(index_num_workers))

        if samples is not None:
            self.samples = list(samples)
            self.classes = list(classes) if classes is not None else None
            self.class_to_idx = dict(class_to_idx) if class_to_idx is not None else None
            if self.classes is None or self.class_to_idx is None:
                raise ValueError(
                    "RadioGalaxyDataset: when providing `samples`, you must also provide "
                    "`classes` and `class_to_idx`."
                )
        else:
            self.classes, self.class_to_idx, self.samples = self._load_or_build_index(
                index_cache_path=index_cache_path,
                wait_for_cache_seconds=int(wait_for_cache_seconds),
            )

        if len(self.samples) == 0:
            raise RuntimeError(f"No FITS files found under {self.root}/*/*.fits")

    @staticmethod
    def _is_main_process() -> bool:
        return os.getenv("RANK", "0") in ("0", "") and os.getenv("LOCAL_RANK", "0") in ("0", "")

    def _load_or_build_index(
        self,
        *,
        index_cache_path: Optional[str],
        wait_for_cache_seconds: int,
    ) -> Tuple[List[str], Dict[str, int], List[Tuple[str, int]]]:
        """
        Build an index of (fits_path, class_idx) once and optionally cache to disk.

        Motivation: RGZ indexing is expensive because it may open many FITS files
        to filter all-zero images. This dataset is frequently re-instantiated with
        different transforms/image_size; caching avoids repeating that scan.
        """
        cache_path = None
        if index_cache_path:
            cache_path = os.path.abspath(os.path.expanduser(index_cache_path))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if cache_path and os.path.isfile(cache_path):
            loaded = self._try_load_index(cache_path)
            if loaded is not None:
                return loaded

        # In multi-process settings (DDP), let rank 0 build+write the cache and
        # have other ranks wait briefly.
        if cache_path and (not self._is_main_process()):
            import time
            deadline = time.time() + max(0, int(wait_for_cache_seconds))
            while time.time() < deadline:
                if os.path.isfile(cache_path):
                    loaded = self._try_load_index(cache_path)
                    if loaded is not None:
                        return loaded
                time.sleep(1.0)
            # Fallback: build locally (but don't write to avoid races).
            return self._build_index()

        classes, class_to_idx, samples = self._build_index()
        if cache_path and self._is_main_process():
            self._try_save_index(cache_path, classes=classes, class_to_idx=class_to_idx, samples=samples)
        return classes, class_to_idx, samples

    def _build_index(self) -> Tuple[List[str], Dict[str, int], List[Tuple[str, int]]]:
        # 1) List all class folders under root, sorted lexicographically
        classes = sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        )
        if len(classes) == 0:
            raise RuntimeError(f"No class subdirectories found in {self.root}")

        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        tasks: List[Tuple[str, int]] = []
        for cls_name in classes:
            class_dir = os.path.join(self.root, cls_name)
            if not os.path.isdir(class_dir):
                continue
            lbl = int(class_to_idx[cls_name])
            for fits_path in glob.glob(os.path.join(class_dir, "*.fits")):
                tasks.append((fits_path, lbl))

        if not self.skip_all_zero:
            return classes, class_to_idx, tasks

        def _keep(task: Tuple[str, int]) -> Optional[Tuple[str, int]]:
            fits_path, lbl = task
            try:
                with fits.open(fits_path, memmap=True) as hdul:
                    data = hdul[0].data
                    if data is None:
                        return None
                    data = np.asarray(data, dtype=np.float32)
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                    if np.all(data == 0):
                        return None
            except Exception:
                return None
            return fits_path, lbl

        workers = min(self.index_num_workers, 32)
        if workers <= 1:
            samples = [res for res in (_keep(t) for t in tasks) if res is not None]
        else:
            samples = []
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for res in ex.map(_keep, tasks):
                    if res is not None:
                        samples.append(res)

        return classes, class_to_idx, samples

    def _try_load_index(
        self, cache_path: str
    ) -> Optional[Tuple[List[str], Dict[str, int], List[Tuple[str, int]]]]:
        try:
            if cache_path.endswith(".json"):
                with open(cache_path, "r") as f:
                    payload = json.load(f)
            else:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None
        if payload.get("type") != "RadioGalaxyDatasetIndexV1":
            return None
        if os.path.abspath(os.path.expanduser(payload.get("root", ""))) != os.path.abspath(self.root):
            return None
        if bool(payload.get("skip_all_zero", True)) != bool(self.skip_all_zero):
            return None

        classes = payload.get("classes")
        class_to_idx = payload.get("class_to_idx")
        samples = payload.get("samples")
        if not isinstance(classes, list) or not isinstance(class_to_idx, dict) or not isinstance(samples, list):
            return None
        try:
            samples_typed = [(str(p), int(lbl)) for (p, lbl) in samples]
        except Exception:
            return None
        return list(classes), {str(k): int(v) for k, v in class_to_idx.items()}, samples_typed

    def _try_save_index(
        self,
        cache_path: str,
        *,
        classes: List[str],
        class_to_idx: Dict[str, int],
        samples: List[Tuple[str, int]],
    ) -> None:
        payload: Dict[str, Any] = {
            "type": "RadioGalaxyDatasetIndexV1",
            "root": os.path.abspath(self.root),
            "skip_all_zero": bool(self.skip_all_zero),
            "classes": list(classes),
            "class_to_idx": dict(class_to_idx),
            "samples": [(p, int(lbl)) for (p, lbl) in samples],
        }
        tmp_path = cache_path + ".tmp"
        try:
            if cache_path.endswith(".json"):
                with open(tmp_path, "w") as f:
                    json.dump(payload, f)
            else:
                with open(tmp_path, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fits_path, label = self.samples[index]

        # 1) Load FITS & process via RadioFoundationProcessor
        tensor = self.processor(fits_path=fits_path)

        # 2) Apply your transform (must accept a Tensor and return a Tensor)
        if self.transform is not None:
            tensor = self.transform(tensor)

        return {"pixel_values": tensor, "labels": int(label)}

# ───────────────────────────────────────────────────────────────────────────────
# LoTSS DR2 Horton+ 2025 cutouts (flat dir: FITS + JSON sidecars)
# ───────────────────────────────────────────────────────────────────────────────
class LoTSSDR2HortonDataset(Dataset):
    """
    PyTorch Dataset for LoTSS DR2 high-res FITS cutouts downloaded by
    preprocessing/lotss_download.py.

    Expected directory layout (flat)::

        root/
            ILTJ..._sXX.XXarcmin.fits
            ILTJ..._sXX.XXarcmin.json
            ...

    Labels are derived from the JSON sidecar initial classification. Samples
    with no initial label (or ambiguous multiple initial labels) are skipped.

    Args:
        root (str): Directory containing FITS + JSON sidecars.
        augmenter: Ignored here (signature compatibility only).
        transform (callable): Transform applied to the processed tensor.
        image_size (int): Image size for RadioFoundationProcessor.
        skip_all_zero (bool): Optionally drop all-zero FITS images.
        skip_bad_fits (bool): Skip unreadable/corrupt FITS files during indexing.
        log_bad_fits (bool): Emit a warning with paths of unreadable FITS files.
        allow_multi_initial (bool): If True and multiple initial flags are set,
            pick the first in canonical order; otherwise skip.
    """

    INITIAL_ORDER = [
        ("fri", "FRI", "INIT-1"),
        ("frii", "FRII", "INIT-2"),
        ("hybrid", "Hybrid", "INIT-3"),
        ("spiral", "Spiral", "INIT-4"),
        ("relaxed", "Relaxed double", "INIT-5"),
    ]

    def __init__(
        self,
        root,
        augmenter=None,
        transform=None,
        image_size=224,
        skip_all_zero=False,
        skip_bad_fits=True,
        log_bad_fits=True,
        allow_multi_initial=False,
    ):
        super().__init__()
        self.root = os.path.abspath(os.path.expanduser(root))
        self.augmenter = augmenter  # not used
        self.transform = transform
        self.processor = RadioFoundationProcessor(image_size=image_size)
        self.skip_all_zero = bool(skip_all_zero)
        self.skip_bad_fits = bool(skip_bad_fits)
        self.log_bad_fits = bool(log_bad_fits)
        self.allow_multi_initial = bool(allow_multi_initial)

        self.classes = [text for (_, text, _) in self.INITIAL_ORDER]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self._init_code_to_text = {code: text for (_, text, code) in self.INITIAL_ORDER}

        root_path = Path(self.root)
        if not root_path.is_dir():
            raise RuntimeError(f"LoTSSDR2HortonDataset: root is not a directory: {self.root}")

        self.samples = []
        bad_fits = 0
        self.bad_fits = []
        for fits_path in sorted(root_path.glob("*.fits")):
            json_path = fits_path.with_suffix(".json")
            if not json_path.exists():
                continue

            label_text = self._label_from_json(json_path)
            if label_text is None:
                continue

            data_zero = None
            if self.skip_bad_fits or self.skip_all_zero:
                try:
                    with fits.open(fits_path, memmap=True) as hdul_zero:
                        data_zero = hdul_zero[0].data
                        if data_zero is None:
                            raise OSError("FITS primary HDU has no data")
                except Exception:
                    bad_fits += 1
                    if self.log_bad_fits:
                        self.bad_fits.append(str(fits_path))
                    continue

            if self.skip_all_zero:
                try:
                    data_zero = data_zero.astype(np.float32)
                    data_zero = np.nan_to_num(data_zero, nan=0.0, posinf=0.0, neginf=0.0)
                    if np.all(data_zero == 0):
                        continue
                except Exception:
                    continue

            self.samples.append((str(fits_path), int(self.class_to_idx[label_text])))

        if len(self.samples) == 0:
            raise RuntimeError(f"LoTSSDR2HortonDataset: no labeled FITS/JSON pairs found in {self.root}")
        if bad_fits > 0:
            if self.log_bad_fits and self.bad_fits:
                preview = 20
                shown = self.bad_fits[:preview]
                suffix = ""
                if len(self.bad_fits) > preview:
                    suffix = f"\n  ... (+{len(self.bad_fits) - preview} more)"
                warnings.warn(
                    "LoTSSDR2HortonDataset: skipped unreadable FITS files:\n  "
                    + "\n  ".join(shown)
                    + suffix,
                    RuntimeWarning,
                )
            else:
                warnings.warn(
                    f"LoTSSDR2HortonDataset: skipped {bad_fits} unreadable FITS file(s)",
                    RuntimeWarning,
                )

    def _label_from_json(self, json_path: Path):
        try:
            payload = json.loads(json_path.read_text())
        except Exception:
            return None

        # Prefer raw_flags (lossless)
        raw_flags = payload.get("raw_flags")
        if isinstance(raw_flags, dict):
            positives = []
            for key, text, _ in self.INITIAL_ORDER:
                try:
                    val = int(raw_flags.get(key, 0))
                except Exception:
                    val = 0
                if val == 1:
                    positives.append(text)
            if len(positives) == 1:
                return positives[0]
            if len(positives) > 1 and self.allow_multi_initial:
                return positives[0]
            return None

        # Fallback: labels.initial list of {"code": "...", "text": "..."}
        labels = payload.get("labels")
        initial = labels.get("initial") if isinstance(labels, dict) else None
        if not isinstance(initial, list):
            return None

        positives = []
        for entry in initial:
            if not isinstance(entry, dict):
                continue
            code = entry.get("code")
            if code in self._init_code_to_text:
                positives.append(self._init_code_to_text[code])

        if len(positives) == 1:
            return positives[0]
        if len(positives) > 1 and self.allow_multi_initial:
            return positives[0]
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fits_path, label = self.samples[index]

        try:
            tensor = self.processor(fits_path=fits_path)
        except Exception as e:
            msg = f"LoTSSDR2HortonDataset: failed to read FITS '{fits_path}': {e}"
            print(msg, file=sys.stderr, flush=True)
            raise OSError(msg) from e
        if self.transform is not None:
            tensor = self.transform(tensor)

        return {"pixel_values": tensor, "labels": int(label)}

# ───────────────────────────────────────────────────────────────────────────────
# Datasets for SSL tasks are supplied another  way
# ───────────────────────────────────────────────────────────────────────────────
class FITSDataset(Dataset):
    def __init__(self, root_dir, grayscale=True, transform=None, debug=False):
        self.root_dir = root_dir
        self.transform = transform
        self.gray = grayscale
        self.debug = debug
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith(".fits") and not fname.startswith('.')
        ]
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        fits_path = self.image_paths[idx]
        try:
            with fits.open(fits_path) as hdul:
                data = hdul[0].data.astype(np.float32)
                # ... [NaN handling and normalization] ...
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)  # uint8 image
                # Ensure data is float and handle potential NaNs before scaling
                data = np.interp(data, (data.min(), data.max()), (0, 1))

            # Convert to PIL Image
            data_uint8 = (data * 255).astype(np.uint8)
            pil_img = Image.fromarray(data_uint8)
            if not self.gray:
                pil_img = pil_img.convert("RGB")
            else:
                pil_img = pil_img.convert("L")

            # --- Debug: show raw PIL image ---------------------------------
            if self.debug:
                plt.figure(figsize=(4, 4))
                plt.title(f"Idx {idx} – PIL")
                plt.imshow(np.array(pil_img), cmap='gray')
                plt.axis('off')
                plt.show()

            # Apply transforms
            if self.transform:
                views = self.transform(pil_img)  # Returns list of tensors
            else:
                views = [transforms.ToTensor()(pil_img)]

            # --- Debug: show first transformed view ------------------------
            if self.debug:
                first_view = views[0]
                if first_view.ndim == 3 and first_view.shape[0] == 1:
                    first_view = first_view.squeeze(0)
                plt.figure(figsize=(4, 4))
                plt.title(f"Idx {idx} – after transform")
                plt.imshow(first_view.numpy(), cmap='gray')
                plt.axis('off')
                plt.show()

            return {"pixel_values": views}

        except Exception as e:
            print(f"Skipping corrupt file {fits_path}: {e}")
            return self[(idx + 1) % len(self)]  # Skip to next

class PTDataset(Dataset):
    """
    Dataset that loads pre-saved torch tensors (*.pt) instead of FITS files.

    Each .pt file must contain a single tensor of shape (C, H, W),
    dtype float32, range [0, 1].  (E.g. the output of your fits_convert
    script, now stored as RGB => C=3.)

    Args
    ----
    root_dir : str or Path
        Folder containing *.pt files.
    transform : callable, optional
        A transform (or Multi-View wrapper) that accepts a tensor and
        returns *either* a single tensor *or* a list of tensors.
    debug : bool, optional
        If True, prints the file being loaded (handy for sanity checks).
    """

    def __init__(self, root_dir, transform=None, debug=False):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.debug = debug

        self.image_paths = sorted(p for p in self.root_dir.glob("*.pt"))
        if not self.image_paths:
            raise RuntimeError(f"No .pt files found in {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        pt_path = self.image_paths[idx]
        if self.debug:
            print(f"Loading {pt_path.name}")

        # (C, H, W) float32 [0,1]
        tensor = torch.load(pt_path, map_location="cpu")

        # Apply transform (could return list-of-views)
        if self.transform:
            views = self.transform(tensor)
        else:
            views = [tensor]  # wrap in list for consistency

        return {"pixel_values": views}

class ImageNet1kGrayscaleDataset(Dataset):
    def __init__(self, root_dir, transform=None):  # Remove vit_processor
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')

        # Recursive search for images
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_extensions) and not file.startswith('.'):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            pixel_values = self.transform(image)
        else:
            pixel_values = [transforms.ToTensor()(image)]

        return {"pixel_values": pixel_values}

C, H, W = 1, 512, 512
DTYPE = np.float32
RECORD_ELEMS = C * H * W
RECORD_BYTES = RECORD_ELEMS * np.dtype(DTYPE).itemsize
DEFAULT_MEGA_CHUNK_BYTES = 512 << 30
DEFAULT_EPOCHS_PER_MEGA_CHUNK = 1

if sys.platform.startswith("linux"):
    class _PosixReadahead:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _libc.posix_fadvise.argtypes = [
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_int,
        ]
        _POSIX_FADV_WILLNEED = 3
        _POSIX_FADV_DONTNEED = 4

        @classmethod
        def willneed(cls, fd, off, length):
            cls._libc.posix_fadvise(fd, off, length, cls._POSIX_FADV_WILLNEED)

        @classmethod
        def dontneed(cls, fd, off, length):
            cls._libc.posix_fadvise(fd, off, length, cls._POSIX_FADV_DONTNEED)
else:
    class _PosixReadahead:
        """No-op implementation on non-Linux platforms (e.g. macOS, Windows)."""

        @classmethod
        def willneed(cls, fd, off, length):
            return

        @classmethod
        def dontneed(cls, fd, off, length):
            return

class BinaryFileDataset(Dataset):
    """
    Big, single-file, zero-copy dataset backed by a memory-mapped binary file.

    Overview
    --------
    This dataset memory-maps a single contiguous binary file on disk that stores
    `N` images as raw floats in CHW order:

        shape: (N, C, H, W)
        dtype: DTYPE (default float32, see module constants)

    Key properties:
    * **Zero-copy reads**: `np.memmap(..., mode="r")` produces a read-only view that
      does not allocate RAM up-front. `torch.from_numpy()` wraps it without copying.
      If you need a writable tensor, set `copy_tensor=True` (this clones per-sample).
    * **Block-level shuffling (sampler-driven)**: for DDP + persistent workers, do
      shuffling in the sampler (e.g. `BlockShuffleDistributedSampler`) so worker
      processes don't need to observe dataset mutations each epoch.
    * **Warm-ahead (sampler-driven)**: for best performance with DDP + persistent
      workers, warm the next K blocks from the sampler (see `BlockShuffleDistributedSampler`),
      because the dataset itself does not mutate per epoch and does not know the
      permuted block order.
    * **Mega-chunk locality (sampler-driven)**: very large files are read through
      fixed-size contiguous windows by `BlockShuffleDistributedSampler`. The
      trainer keeps one window active for the configured number of epochs before
      moving on, while this dataset remains a simple map from index to image.
    * **Optional cold-start eviction**: `cold_start_drop_cache=True` evicts the
      whole file from page cache once (rank 0 only) for cold benchmarks.

    Distributed / Dataloader behavior
    ---------------------------------
    * This is a **map-style** dataset (`__len__` and `__getitem__` are defined).
      Use it with a standard `DataLoader`. In DDP/Accelerate/🤗Trainer, a
      `DistributedSampler` or `DataLoaderShard` will shard indices per-process.
    * The dataset itself does **not** do any per-rank sharding or epoch shuffle.
      For DDP + persistent workers, use a sampler (e.g. `BlockShuffleDistributedSampler`)
      to provide block-level shuffling and contiguous per-rank slices.
    * For best overlap, enable in the DataLoader:
        - `num_workers` ≥ 2 per rank (start with 2–8),
        - `prefetch_factor` 2–8 (PyTorch default is 2),
        - `pin_memory=True`, `persistent_workers=True`.

    Transform safety
    ----------------
    * Because the underlying storage is read-only, **avoid in-place transforms**.
      By default we upcast to float32 before transforms (`to_float32_before_transform=True`)
      to reduce the chance of inadvertent in-place writes on a view.
    * If your model expects 3 channels but the file stores 1 (C=1), you can set
      `expand_to_three=True` to expand 1→3 channels on the fly, or handle it
      upstream (e.g., inside your collator or loss function).

    Tuning knobs
    ------------
    * `block_size` (int):
        - The number of samples per locality block for permutation.
        - Trade-off: larger blocks ⇒ fewer random seeks and better sequential IO,
          but slightly less mixing. Good starting point:
              `block_size ≈ per_device_batch_size * num_workers * prefetch_factor`
          Examples: 1024, 2048, 4096, 8192.
    * `cold_start_drop_cache` (bool):
        - Rank-0 only: evict all pages of this file at dataset construction. Use
          for “cold” benchmarks. Disable for real training.
    * `copy_tensor` (bool):
        - Clone the per-sample tensor to get a writable copy. Off by default for
          maximum throughput and minimal allocations.
    * `to_float32_before_transform` (bool):
        - Upcast to float32 before applying transforms; recommended unless your
          transforms are guaranteed non-inplace and fp16-safe.
    * `expand_to_three` (bool):
        - If the file stores one channel (C=1) but your model expects RGB, expand
          to (3,H,W). If you do this elsewhere, leave False.

    Sampler locality policy
    -----------------------
    `BlockShuffleDistributedSampler` is the intended sampler for large binary
    training runs. With mega-chunk mode, one trainer epoch is one pass over the
    active mega-chunk, not one pass over the whole file. The configured
    training epoch count is treated by the trainer as epochs per mega-chunk; for
    a file with 6 mega-chunks and 35 configured epochs, training runs 210
    chunk-local trainer epochs. This preserves the full-file-equivalent sample
    budget while improving page-cache locality for multi-TiB files.

    How it actually runs
    --------------------
    * **Construction**:
        1) Validates file size and computes `N`.
        2) Optionally evicts page cache (cold benchmark).
        3) Memory-maps the file: `self.mm = np.memmap(..., mode="r", dtype=DTYPE, shape=(N,C,H,W))`.
        4) Shuffling + warm-ahead are provided by the sampler.
    * **Each epoch** (`set_epoch(e)`):
        - No-op; samplers handle epoch changes.
    * **Each sample** (`__getitem__(idx)`):
        - Returns a (zero-copy) tensor view on `self.mm[idx]`, optionally
          upcast/expanded/cloned, and applies `transform(t)` if provided.

    Practical recipes
    -----------------
    * **Local NVMe, strong single-drive bandwidth**:
        - `num_workers=2–4`, `prefetch_factor=4–8`, `pin_memory=True`,
          `block_size=2048–4096`, `warm_ahead_blocks=2–4`,
          `readahead_chunk_bytes=128<<20`.
    * **Network-attached or shared storage**:
        - Reduce parallelism (`num_workers=2`), increase `block_size` (4096–8192),
          keep `warm_ahead_blocks` small (1–2).
    * **Cold benchmark only**:
        - Set `cold_start_drop_cache=True`, run a short timed loop. For real training,
          **turn off** `cold_start_drop_cache`.

    Caveats & gotchas
    -----------------
    * Warm-ahead is advisory; the kernel may ignore hints under memory pressure.
    * Excessive warm-ahead can backfire by evicting hot pages prematurely.
    * Do not perform in-place transforms on the returned tensor when zero-copy is used.
    * When changing `C,H,W,DTYPE`, update the module constants and regenerate the
      binary file accordingly; the class validates `RECORD_BYTES` alignment.

    """
    def __init__(self, bin_path: str, transform=None,
                 copy_tensor: bool = False,
                 block_size: int = 8192,
                 verbose_debug: bool = False,
                 to_float32_before_transform: bool = True,
                 expand_to_three: bool = False,
                 cold_start_drop_cache: bool = False,
                 ):
        """
        bin_path    : path to the single .bin file
        copy_tensor : True => returns writable tensors (adds .copy()); False => zero-copy read-only
        block_size  : locality block size (used by block-shuffle samplers)
        to_float32_before_transform : upcast to float32 before transforms (default True)
        expand_to_three : expand 1→3 channels if True (default False)
        cold_start_drop_cache : if True, evict the file from the OS page cache once at dataset construction (rank 0 only).
        """
        self.path = str(bin_path)
        self.transform = transform
        self.copy_tensor = copy_tensor
        self.block_size = int(block_size)
        self.verbose = verbose_debug
        self.to_float32_before_transform = to_float32_before_transform
        self.expand_to_three = expand_to_three
        self.cold_start_drop_cache = bool(cold_start_drop_cache)

        # Infer N from file size
        file_bytes = Path(self.path).stat().st_size
        if file_bytes % RECORD_BYTES != 0:
            raise ValueError(f"File size {file_bytes} not multiple of RECORD_BYTES={RECORD_BYTES}")
        self.N = file_bytes // RECORD_BYTES

        # Optional one-time cold-start eviction (only once by rank 0)
        if self.cold_start_drop_cache and self._is_rank0():
            if self.verbose:
                print("[BinaryFileDataset] cold-start cache eviction…")
            self._drop_cache_all()

        # Map the entire file as (N,C,H,W); memmap is lazy (no RAM blowup)
        self.mm = np.memmap(self.path, mode="r", dtype=DTYPE, shape=(self.N, C, H, W))

        self._epoch = -1
        # Shuffling is sampler-driven for DDP + persistent workers.
        # (Keeping no per-epoch mutation avoids divergence between parent/worker dataset copies.)

    def _is_rank0(self) -> bool:
        try:
            return int(os.getenv("RANK", "0")) == 0
        except Exception:
            return True

    def _drop_cache_all(self):
        """Evict the whole file from the OS page cache once (rank0 only)."""
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except OSError:
            return
        try:
            file_bytes = int(self.N * RECORD_BYTES)
            step = 64 << 20
            off = 0
            while off < file_bytes:
                _PosixReadahead.dontneed(fd, off, min(step, file_bytes - off))
                off += step
        finally:
            try:
                os.close(fd)
            except OSError:
                pass


    def __len__(self):
        return self.N

    def set_epoch(self, epoch: int):
        """No-op: shuffling is sampler-driven (see BlockShuffleDistributedSampler)."""
        try:
            self._epoch = int(epoch)
        except Exception:
            self._epoch = -1

    def __getitem__(self, idx):
        # Apply modulo FIRST to ensure idx is always within valid range
        idx = idx % self.N
        real_idx = int(idx)

        # Zero-copy tensor from memmap view
        np_img = self.mm[real_idx]  # shape: (1, 512, 512) in float32
        t = torch.from_numpy(np_img)

        # Upcast before transforms if requested (avoids in-place writes on read-only storage)
        if self.to_float32_before_transform:
            t = t.float()

        # Optionally expand to three channels for models expecting RGB
        if self.expand_to_three and t.shape[0] == 1:
            t = t.expand(3, H, W).contiguous()

        # Make a writable copy only if the caller really needs to mutate
        if self.copy_tensor:
            t = t.clone()

        # Apply user transform (should be non-inplace)
        if self.transform is not None:
            t = self.transform(t)

        return {"pixel_values": t}

class BlockShuffleDistributedSampler(DistributedSampler):
    """
    Block-locality sampler for BinaryFileDataset-like map datasets.

    Key goals:
      - Shuffle at the *block* level (contiguous runs of indices of size `block_size`)
      - Partition work across DDP ranks without striding (keeps locality per rank)
      - Keep all yielded indices inside the active mega-chunk
      - Compatible with persistent DataLoader workers (shuffle lives in the sampler)

    Mega-chunk mode:
      - Split the file into balanced contiguous windows up to `mega_chunk_bytes`.
      - Keep one mega-chunk active for `epochs_per_mega_chunk` trainer epochs.
      - Yield one rank-local pass over the active mega-chunk per trainer epoch.
      - Shuffle blocks independently for each `(cycle, chunk, local_epoch)`.

    The trainer expands configured full-file-equivalent epochs into
    `configured_epochs * mega_chunk_count` chunk-local epochs. The sampler only
    maps each trainer epoch to its active mega-chunk and local epoch.

    In the main binary training path this sampler owns DDP rank sharding. Do
    not pass its DataLoader through an additional dataloader sharder afterward.
    """

    def __init__(
        self,
        dataset: Dataset,
        block_size: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_workers: int = 0,
        interleave_workers: bool = True,
        warm_ahead_blocks: int = 0,
        readahead_chunk_bytes: int = 512 << 20,
        warm_start_blocks: int = 0,
        warm_bytes_at_init: int = 0,
        warm_local_rank0_only: bool = True,
        mega_chunk_bytes: int = DEFAULT_MEGA_CHUNK_BYTES,
        mega_chunk_order: str = "rotate",
        mega_chunk_order_period_epochs: int = 1,
        epochs_per_mega_chunk: int | None = None,
        binary_dataset: Dataset | None = None,
    ):
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.block_size = int(block_size)
        self.num_workers = max(0, int(num_workers))
        self.interleave_workers = bool(interleave_workers)
        self.warm_ahead_blocks = max(0, int(warm_ahead_blocks))
        self.readahead_chunk_bytes = int(readahead_chunk_bytes)
        self.warm_start_blocks = max(0, int(warm_start_blocks))
        self.warm_bytes_at_init = max(0, int(warm_bytes_at_init))
        self.warm_local_rank0_only = bool(warm_local_rank0_only)
        self.binary_dataset = binary_dataset if binary_dataset is not None else dataset
        self.mega_chunk_bytes = max(int(RECORD_BYTES), int(mega_chunk_bytes))
        self.mega_chunk_samples = max(1, int(self.mega_chunk_bytes // int(RECORD_BYTES)))
        order = str(mega_chunk_order or "rotate").strip().lower()
        if order in ("none", "sequential", "fixed"):
            order = "sequential"
        if order not in ("rotate", "sequential"):
            raise ValueError("mega_chunk_order must be one of: rotate, sequential")
        self.mega_chunk_order = order
        self.mega_chunk_order_period_epochs = max(1, int(mega_chunk_order_period_epochs))
        if epochs_per_mega_chunk is None:
            epochs_per_mega_chunk = DEFAULT_EPOCHS_PER_MEGA_CHUNK
        self.epochs_per_mega_chunk = max(1, int(epochs_per_mega_chunk))
        self._did_init_warm = False
        self.num_samples = self._active_chunk_rank_num_samples(len(self.dataset), int(self.epoch))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self.num_samples = self._active_chunk_rank_num_samples(len(self.dataset), int(self.epoch))
        self.total_size = self.num_samples * self.num_replicas

    def _mega_chunks(self, n: int) -> list[tuple[int, int, int]]:
        if n <= 0:
            return []
        chunk_count = max(1, int(math.ceil(n / self.mega_chunk_samples)))
        base = int(n // chunk_count)
        remainder = int(n % chunk_count)
        chunks: list[tuple[int, int, int]] = []
        start = 0
        for chunk_id in range(chunk_count):
            chunk_len = base + (1 if chunk_id < remainder else 0)
            end = min(n, start + chunk_len)
            chunks.append((chunk_id, start, end))
            start = end
        return chunks

    def _ordered_mega_chunks(self, n: int, cycle_id: int | None = None) -> list[tuple[int, int, int]]:
        chunks = self._mega_chunks(n)
        if len(chunks) <= 1 or self.mega_chunk_order == "sequential":
            return chunks
        if cycle_id is None:
            cycle_id = int(self.epoch)
        rotation = (int(cycle_id) // self.mega_chunk_order_period_epochs) % len(chunks)
        return chunks[rotation:] + chunks[:rotation]

    def _active_chunk(self, n: int, epoch: int) -> tuple[int, int, int, int, int] | None:
        chunks = self._mega_chunks(n)
        if not chunks:
            return None
        cycle_span = max(1, len(chunks) * self.epochs_per_mega_chunk)
        cycle_id = int(epoch) // cycle_span
        epoch_in_cycle = int(epoch) % cycle_span
        chunk_pos = int(epoch_in_cycle // self.epochs_per_mega_chunk)
        local_epoch = int(epoch_in_cycle % self.epochs_per_mega_chunk)
        ordered = self._ordered_mega_chunks(n, cycle_id=cycle_id)
        chunk_id, chunk_start, chunk_end = ordered[chunk_pos]
        return int(chunk_id), int(chunk_start), int(chunk_end), int(cycle_id), int(local_epoch)

    def _chunk_rank_num_samples(self, chunk_len: int) -> int:
        if chunk_len <= 0:
            return 0
        if self.drop_last:
            return (chunk_len // self.num_replicas)
        return int(math.ceil(chunk_len / self.num_replicas))

    def _active_chunk_rank_num_samples(self, n: int, epoch: int) -> int:
        active = self._active_chunk(n, epoch)
        if active is None:
            return 0
        _, start, end, _, _ = active
        return self._chunk_rank_num_samples(int(end - start))

    def _chunk_blocks(
        self,
        chunk_id: int,
        chunk_start: int,
        chunk_end: int,
        cycle_id: int,
        local_epoch: int,
    ) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        start = int(chunk_start)
        while start < int(chunk_end):
            end = min(int(chunk_end), start + self.block_size)
            blocks.append((start, end))
            start = end
        if self.shuffle and len(blocks) > 1:
            g = torch.Generator()
            g.manual_seed(
                int(self.seed)
                + int(chunk_id) * 1000003
                + int(cycle_id) * 10007
                + int(local_epoch) * 9176
            )
            perm = torch.randperm(len(blocks), generator=g).tolist()
            blocks = [blocks[i] for i in perm]
        return blocks

    def active_chunk_summary(self, trace_blocks: int = 0) -> dict[str, int | list[int] | None]:
        active = self._active_chunk(len(self.dataset), int(self.epoch))
        if active is None:
            return {
                "chunk_id": None,
                "chunk_start": None,
                "chunk_end": None,
                "cycle_id": None,
                "local_epoch": None,
                "rank_start_pos": None,
                "rank_end_pos": None,
                "rank_samples": 0,
                "first_block_ids": [],
            }
        chunk_id, chunk_start, chunk_end, cycle_id, local_epoch = active
        chunk_len = int(chunk_end - chunk_start)
        rank_samples = self._chunk_rank_num_samples(chunk_len)
        rank_start_pos = int(self.rank * rank_samples)
        rank_end_pos = int(rank_start_pos + rank_samples)
        first_block_ids: list[int] = []
        if trace_blocks > 0:
            blocks = self._chunk_blocks(chunk_id, chunk_start, chunk_end, cycle_id, local_epoch)
            first_block_ids = [int(start // self.block_size) for start, _ in blocks[: int(trace_blocks)]]
        return {
            "chunk_id": int(chunk_id),
            "chunk_start": int(chunk_start),
            "chunk_end": int(chunk_end),
            "cycle_id": int(cycle_id),
            "local_epoch": int(local_epoch),
            "rank_start_pos": int(rank_start_pos),
            "rank_end_pos": int(rank_end_pos),
            "rank_samples": int(rank_samples),
            "first_block_ids": first_block_ids,
        }

    def __len__(self) -> int:
        return self._active_chunk_rank_num_samples(len(self.dataset), int(self.epoch))

    @property
    def mega_chunk_count(self) -> int:
        return len(self._mega_chunks(len(self.dataset)))

    def mega_chunk_summary(self) -> dict[str, int | str]:
        return {
            "mega_chunk_bytes": int(self.mega_chunk_bytes),
            "mega_chunk_samples": int(self.mega_chunk_samples),
            "mega_chunk_count": int(self.mega_chunk_count),
            "mega_chunk_order_policy": str(self.mega_chunk_order),
            "mega_chunk_order_period_epochs": int(self.mega_chunk_order_period_epochs),
            "epochs_per_mega_chunk": int(self.epochs_per_mega_chunk),
            "epoch_semantics": "mega_chunk_local",
        }

    def __iter__(self):
        n = len(self.dataset)
        if n <= 0:
            return

        fd = None
        if (self.warm_ahead_blocks > 0 or self.warm_start_blocks > 0 or self.warm_bytes_at_init > 0) and sys.platform.startswith("linux"):
            try:
                path = getattr(self.binary_dataset, "path", None)
                if path:
                    # Optional: only warm once per node by restricting to local rank 0 (if available).
                    do_warm = True
                    if self.warm_local_rank0_only:
                        lr = os.getenv("LOCAL_RANK")
                        if lr is not None and lr != "" and lr != "0":
                            do_warm = False
                    if do_warm:
                        fd = os.open(str(path), os.O_RDONLY)
            except OSError:
                fd = None

        def _advise(off: int, length: int) -> None:
            if fd is None:
                return
            try:
                step = max(64 << 20, int(self.readahead_chunk_bytes))
                end = int(off + length)
                cur = int(off)
                while cur < end:
                    nbytes = min(step, end - cur)
                    _PosixReadahead.willneed(fd, cur, nbytes)
                    cur += nbytes
            except Exception:
                return

        def _warm_from_pos(blocks: list[tuple[int, int]], chunk_len: int, pos0: int, count_blocks: int) -> None:
            if fd is None or count_blocks <= 0 or len(blocks) <= 0 or chunk_len <= 0:
                return
            try:
                off = int(pos0 % chunk_len)
                b_idx = 0
                while b_idx < len(blocks):
                    block_start, block_end = blocks[b_idx]
                    bl = max(0, int(block_end - block_start))
                    if off < bl:
                        break
                    off -= bl
                    b_idx += 1
                for j in range(b_idx, min(b_idx + int(count_blocks), len(blocks))):
                    start, end = blocks[j]
                    if end > start:
                        _advise(start * RECORD_BYTES, (end - start) * RECORD_BYTES)
            except Exception:
                pass

        def _yield_from_offset(blocks: list[tuple[int, int]], off: int, count: int):
            # off is relative to the permuted chunk sequence.
            remaining = int(count)
            if remaining <= 0:
                return
            # Seek to block containing the offset.
            b_idx = 0
            while b_idx < len(blocks):
                block_start, block_end = blocks[b_idx]
                bl = max(0, int(block_end - block_start))
                if off < bl:
                    break
                off -= bl
                b_idx += 1
            # Yield within blocks.
            last_warm_b_idx = -1
            while remaining > 0 and b_idx < len(blocks):
                block_start, block_end = blocks[b_idx]
                bl = max(0, int(block_end - block_start))
                if bl <= 0:
                    b_idx += 1
                    off = 0
                    continue
                # Warm the next K blocks in *permuted* order when entering a new block.
                if self.warm_ahead_blocks > 0 and b_idx != last_warm_b_idx:
                    for j in range(b_idx + 1, min(b_idx + 1 + self.warm_ahead_blocks, len(blocks))):
                        start, end = blocks[j]
                        if end > start:
                            _advise(start * RECORD_BYTES, (end - start) * RECORD_BYTES)
                    last_warm_b_idx = b_idx
                take = min(remaining, bl - off)
                block_start, _ = blocks[b_idx]
                base = block_start + off
                for j in range(int(take)):
                    yield base + j
                remaining -= int(take)
                b_idx += 1
                off = 0

        def _yield_padded(blocks: list[tuple[int, int]], chunk_len: int, pos0: int, count: int):
            # Positions are in the padded chunk sequence; map by repeating the
            # first `chunk_len` elements of that chunk's permuted order.
            if count <= 0:
                return
            if chunk_len <= 0:
                return
            pos0_mod = int(pos0 % chunk_len)
            first = min(int(count), int(chunk_len - pos0_mod))
            yield from _yield_from_offset(blocks, pos0_mod, first)
            rem = int(count) - int(first)
            if rem > 0:
                yield from _yield_from_offset(blocks, 0, rem)

        def _yield_active_chunk(
            chunk_id: int,
            chunk_start: int,
            chunk_end: int,
            cycle_id: int,
            local_epoch: int,
        ):
            chunk_len = int(chunk_end - chunk_start)
            num_samples = self._chunk_rank_num_samples(chunk_len)
            if chunk_len <= 0 or num_samples <= 0:
                return
            start_pos = int(self.rank * num_samples)
            end_pos = int(start_pos + num_samples)
            blocks = self._chunk_blocks(chunk_id, chunk_start, chunk_end, cycle_id, local_epoch)

            _warm_from_pos(blocks, chunk_len, start_pos, self.warm_start_blocks)

            if self.num_workers > 1 and self.interleave_workers:
                per_worker = int(math.ceil(num_samples / self.num_workers))
                iters = []
                alive = []
                active_workers = 0
                for w in range(self.num_workers):
                    worker_start = start_pos + w * per_worker
                    worker_end = min(start_pos + (w + 1) * per_worker, end_pos)
                    if worker_end > worker_start:
                        active_workers += 1
                        iters.append(iter(_yield_padded(blocks, chunk_len, worker_start, worker_end - worker_start)))
                        alive.append(True)
                    else:
                        iters.append(iter(()))
                        alive.append(False)

                remaining_workers = active_workers
                while remaining_workers > 0:
                    for w in range(self.num_workers):
                        if not alive[w]:
                            continue
                        try:
                            yield next(iters[w])
                        except StopIteration:
                            alive[w] = False
                            remaining_workers -= 1
            else:
                yield from _yield_padded(blocks, chunk_len, start_pos, num_samples)

        try:
            active = self._active_chunk(n, int(self.epoch))
            if active is None:
                return

            chunk_id, chunk_start, chunk_end, cycle_id, local_epoch = active
            if fd is not None and self.warm_bytes_at_init > 0 and not self._did_init_warm:
                try:
                    first_bytes = max(0, int(chunk_end - chunk_start) * int(RECORD_BYTES))
                    _advise(
                        int(chunk_start) * int(RECORD_BYTES),
                        min(int(self.warm_bytes_at_init), first_bytes),
                    )
                    self._did_init_warm = True
                except Exception:
                    pass

            yield from _yield_active_chunk(chunk_id, chunk_start, chunk_end, cycle_id, local_epoch)
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

class AugmentedRepeatDataset(torch.utils.data.Dataset):
    """
    Cycle a base map-style dataset up to a large virtual length (epoch_size)
    by simple modulo indexing. No remapping tables, no retries, no shuffling.

    • __getitem__(i) -> base[i % len(base)]
    • set_epoch is forwarded to the base dataset (if present), but the mapping
      stays fixed so indices are stable across epochs.

    This guarantees we never pass an out-of-range index to the base dataset.
    """
    def __init__(self, base_dataset, epoch_size: int):
        self.base = base_dataset
        self.base_len = int(len(base_dataset))
        if self.base_len <= 0:
            raise ValueError("AugmentedRepeatDataset: base dataset length must be > 0")
        self.epoch_size = int(epoch_size)
        if self.epoch_size <= 0:
            raise ValueError("AugmentedRepeatDataset: epoch_size must be > 0")

    def __len__(self):
        return self.epoch_size

    def set_epoch(self, epoch: int):
        # Forward to base if it supports epoch-based reshuffling etc.
        try:
            if hasattr(self.base, "set_epoch"):
                self.base.set_epoch(int(epoch))
        except Exception:
            pass

    def __getitem__(self, idx: int):
        base_len = int(len(self.base))
        if base_len <= 0:
            raise IndexError("AugmentedRepeatDataset: base dataset is empty")
        return self.base[idx % base_len]
